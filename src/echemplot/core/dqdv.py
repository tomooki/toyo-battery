"""dQ/dV computation via interpolation + Savitzky-Golay derivative.

For each ``(cycle, side)`` segment in a ``chdis_df`` (produced by
:func:`echemplot.core.chdis.get_chdis_df`), this module:

1. Extracts the voltage and capacity series, drops NaN rows, sorts by V,
   and removes duplicate V values (``keep="last"`` to preserve CC-CV tail
   capacity, which accumulates at ``V_max``/``V_min`` plateaus).
2. Linearly interpolates Q onto a uniform V grid
   (``np.linspace(V.min(), V.max(), ipnum)`` with
   ``ipnum = max(int(inter_num * (V.max() - V.min())), 2)``).
3. Applies :func:`scipy.signal.savgol_filter` with ``deriv=1`` and
   ``delta=dx`` to compute ``dQ/dV`` in physical units on the resampled
   curve.

Output shape: a wide DataFrame whose column MultiIndex mirrors ``chdis_df``
(levels ``cycle``, ``side``, ``quantity``). The ``quantity`` level holds
``{"電圧", "dQ/dV"}`` when ``column_lang="ja"`` and
``{"voltage", "dq_dv"}`` when ``column_lang="en"``. The row axis is a
``RangeIndex`` shared across all segments; shorter segments are NaN-padded
to the longest.

Numerical caveats:

- The leading and trailing ``window_length // 2`` samples of each
  ``dQ/dV`` column are edge-approximated (savgol ``mode="nearest"``) and
  should be trimmed when precise slopes matter.
- Discharge segments enter with V decreasing but Q still monotone
  non-decreasing per the chdis invariant. After the ascending V sort, Q
  becomes decreasing along the grid, so ``dQ/dV`` for discharge is
  **negative** by construction. Callers plotting charge and discharge on
  the same axis commonly take ``abs(dQ/dV)`` for a magnitude comparison;
  the raw signed value is retained here so the direction of accumulation
  is not lost.

Rewrite notes (vs. legacy TOYO_Origin_2.01 ``plot_dQdV_curve`` L378-453):

- v2.01 read the voltage bounds via ``chdis_df[cycle].at[1, "電圧"]`` and
  ``chdis_df[cycle].at[len(...)-1, "電圧"]``. ``.at[1, ...]`` reads the
  *second* row, not the first, so the original implicitly skipped the
  first sample and was sensitive to whichever row landed at position 1
  before sorting. This rewrite uses ``V.min()``/``V.max()`` after an
  explicit ascending sort with duplicate removal — deterministic,
  monotone, and works whether the source segment is charge or discharge.
- v2.01's ``savgol_filter`` call (L423-429) omitted ``delta=``, so its
  output was ``d(Q)/d(sample-index)`` rather than ``d(Q)/d(V)`` — the
  legacy plots carried a sample-count-scaled axis labeled ``dQ/dV``.
  This rewrite passes ``delta=dx`` so the output is in physical units
  (mAh/g per V).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from echemplot.io.schema import JA_TO_EN, ColumnLang

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # Per-segment result: (V-grid, dQ/dV samples), both same length.
    _SegPair = tuple[NDArray[np.float64], NDArray[np.float64]]

_JA_COLS: dict[str, str] = {
    "voltage": "電圧",
    "capacity": "電気量",
    "dqdv": "dQ/dV",
}


def _resolve_cols(column_lang: ColumnLang) -> dict[str, str]:
    if column_lang == "ja":
        return _JA_COLS
    # Input voltage/capacity follow JA_TO_EN; the dqdv output name is fixed EN.
    return {
        "voltage": JA_TO_EN[_JA_COLS["voltage"]],
        "capacity": JA_TO_EN[_JA_COLS["capacity"]],
        "dqdv": "dq_dv",
    }


def _empty_result() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([], names=["cycle", "side", "quantity"])
    return pd.DataFrame(columns=idx)


def _validate_savgol_params(inter_num: int, window_length: int, polyorder: int) -> None:
    """Validate Savitzky-Golay preconditions up-front for a helpful error.

    scipy raises from deep inside ``savgol_filter`` otherwise, which is
    harder to correlate with a caller's actual input. ``window_length``
    is required to be odd here — scipy 1.13+ accepts even, but the
    project floor is ``scipy>=1.10`` where even values raise. Enforcing
    odd uniformly keeps behaviour consistent across the supported range.
    """
    if inter_num < 1:
        raise ValueError(f"inter_num must be >= 1, got {inter_num}")
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder < 0:
        raise ValueError(f"polyorder must be >= 0, got {polyorder}")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) must be < window_length ({window_length})")


def get_dqdv_df(
    chdis_df: pd.DataFrame,
    *,
    inter_num: int = 100,
    window_length: int = 11,
    polyorder: int = 2,
    column_lang: ColumnLang = "ja",
) -> pd.DataFrame:
    """Compute dQ/dV per (cycle, side) segment from a chdis DataFrame.

    Parameters
    ----------
    chdis_df
        Output of :func:`echemplot.core.chdis.get_chdis_df`. Its column
        MultiIndex must have levels ``cycle``, ``side``, ``quantity`` with
        the ``quantity`` level using the names resolved by ``column_lang``
        (voltage + capacity).
    inter_num
        Interpolation density. The number of samples on the uniform V grid
        is ``max(int(inter_num * (V.max() - V.min())), 2)`` per segment.
        Segments whose interpolated length is below ``window_length`` yield
        all-NaN columns (Savitzky-Golay requires at least ``window_length``
        samples). Must be ``>= 1``.
    window_length
        Savitzky-Golay window length. Must be a positive odd integer
        strictly greater than ``polyorder``. (scipy 1.13+ accepts even,
        but the project floor is ``scipy>=1.10`` where odd is required;
        enforcing odd uniformly avoids a version-dependent failure mode.)
    polyorder
        Savitzky-Golay polynomial order. Must be ``>= 0`` and strictly
        less than ``window_length``.
    column_lang
        Language of the ``quantity`` level on both input and output. When
        ``"ja"`` the output uses ``{"電圧", "dQ/dV"}``; when ``"en"`` it
        uses ``{"voltage", "dq_dv"}``.

    Returns
    -------
    DataFrame
        Columns: MultiIndex with levels ``cycle``, ``side``, ``quantity``.
        Row axis: ``RangeIndex`` sized to the longest segment; shorter
        segments are NaN-padded.

    Raises
    ------
    ValueError
        If ``inter_num`` / ``window_length`` / ``polyorder`` violate the
        Savitzky-Golay preconditions above.
    KeyError
        If ``chdis_df.columns`` is not a 3-level ``(cycle, side, quantity)``
        MultiIndex, or if the ``quantity`` level is missing the voltage /
        capacity labels for the requested ``column_lang``.

    Notes
    -----
    Non-default ``inter_num`` / ``window_length`` / ``polyorder`` must be
    passed by calling this function directly —
    :attr:`echemplot.core.cell.Cell.dqdv_df` uses the defaults.
    """
    _validate_savgol_params(inter_num, window_length, polyorder)

    cols = _resolve_cols(column_lang)
    v_name, q_name, dqdv_name = cols["voltage"], cols["capacity"], cols["dqdv"]

    # Structural check runs *before* the empty short-circuit so a
    # flat-columns empty frame surfaces the same ``KeyError`` as a
    # flat-columns populated frame — the error surface for a given bug
    # should not depend on whether the input happened to contain rows.
    if not isinstance(chdis_df.columns, pd.MultiIndex) or chdis_df.columns.nlevels != 3:
        # Exception: a truly empty frame (no rows, no columns — `pd.DataFrame()`)
        # is the one shape we accept without a 3-level MultiIndex, because
        # ``chdis._empty_result`` is the canonical empty contract and we want
        # ``get_dqdv_df(empty) -> empty`` to round-trip.
        if chdis_df.empty and chdis_df.columns.empty:
            return _empty_result()
        raise KeyError(
            "chdis_df.columns must be a 3-level MultiIndex "
            "(cycle, side, quantity); "
            f"got {chdis_df.columns!r}"
        )

    if chdis_df.empty or chdis_df.columns.empty:
        return _empty_result()

    quantity_values = set(chdis_df.columns.get_level_values("quantity"))
    missing_quantities = sorted({v_name, q_name} - quantity_values)
    if missing_quantities:
        raise KeyError(
            f"input missing required quantity level values {missing_quantities} "
            f"for column_lang={column_lang!r}; got quantities={sorted(quantity_values)}"
        )

    cols_tuples = cast("list[tuple[int, str, str]]", list(chdis_df.columns))
    # Deterministic order: (cycle, side) with explicit side rank.
    # ``ch`` before ``dis``, any future side lands after both (stable).
    _side_rank: dict[str, int] = {"ch": 0, "dis": 1}
    cycle_side_pairs: list[tuple[int, str]] = sorted(
        {(int(c), str(s)) for c, s, _q in cols_tuples},
        key=lambda t: (t[0], _side_rank.get(t[1], 2), t[1]),
    )

    per_segment: dict[tuple[int, str], _SegPair | None] = {}
    for cycle, side in cycle_side_pairs:
        # Both quantities are guaranteed to exist: cycle_side_pairs is
        # derived from chdis_df.columns, and the top-level missing_quantities
        # check already verified v_name and q_name are present in the
        # quantity level. A KeyError here would indicate a chdis contract
        # violation — a per-(cycle,side) quantity asymmetry — which should
        # propagate loudly rather than be silently masked.
        v_series = chdis_df[(cycle, side, v_name)]
        q_series = chdis_df[(cycle, side, q_name)]

        frame = pd.concat({"v": v_series, "q": q_series}, axis=1).dropna()
        if frame.empty:
            per_segment[(cycle, side)] = None
            continue

        # ``keep="last"``: chdis guarantees Q is monotone non-decreasing within
        # each segment. After a stable sort by V, the last row among duplicate
        # V values preserves the highest Q — which is essential for CC-CV
        # plateaus where V saturates at V_max (charge) or V_min (discharge)
        # while Q continues to grow. ``keep="first"`` would silently discard
        # the CV tail capacity.
        frame = (
            frame.sort_values("v", kind="mergesort")
            .drop_duplicates(subset="v", keep="last")
            .reset_index(drop=True)
        )
        if len(frame) < 2:
            per_segment[(cycle, side)] = None
            continue

        v_arr: NDArray[np.float64] = frame["v"].to_numpy(dtype=float)
        q_arr: NDArray[np.float64] = frame["q"].to_numpy(dtype=float)
        v_min, v_max = float(v_arr.min()), float(v_arr.max())
        ipnum = max(int(inter_num * (v_max - v_min)), 2)

        if ipnum < window_length:
            per_segment[(cycle, side)] = None
            continue

        x_latent: NDArray[np.float64] = np.linspace(v_min, v_max, ipnum)
        dx = float(x_latent[1] - x_latent[0])
        # ``np.interp`` replaces legacy ``scipy.interpolate.interp1d`` for
        # 1-D linear interpolation (scipy flags interp1d as legacy in 1.10+).
        # Preconditions: ``v_arr`` strictly increasing (guaranteed by the
        # sort + drop_duplicates above) and ``x_latent`` within
        # ``[v_min, v_max]`` (linspace over the same bounds). Under these
        # conditions ``np.interp``'s default clamping is equivalent to
        # ``interp1d(..., fill_value="extrapolate")`` — no extrapolation
        # actually occurs.
        q_latent: NDArray[np.float64] = np.interp(x_latent, v_arr, q_arr)
        dy: NDArray[np.float64] = np.asarray(
            savgol_filter(
                q_latent,
                window_length=window_length,
                polyorder=polyorder,
                deriv=1,
                delta=dx,
                mode="nearest",
            ),
            dtype=float,
        )
        per_segment[(cycle, side)] = (x_latent, dy)

    longest = max(
        (len(pair[0]) for pair in per_segment.values() if pair is not None),
        default=0,
    )
    # The output MultiIndex is the same shape whether rows are populated
    # or not — build it once and hand it to the DataFrame constructor.
    out_cols = pd.MultiIndex.from_tuples(
        [(c, s, q) for (c, s) in cycle_side_pairs for q in (v_name, dqdv_name)],
        names=["cycle", "side", "quantity"],
    )
    if longest == 0:
        # All segments degenerate — return a 0-row frame with the expected
        # columns so downstream consumers can detect the empty case without
        # KeyErrors.
        return pd.DataFrame(columns=out_cols, dtype=float)

    data_matrix: NDArray[np.float64] = np.full((longest, len(out_cols)), np.nan, dtype=float)
    for col_idx, (c, s, q) in enumerate(out_cols):
        pair = per_segment[(int(c), str(s))]
        if pair is None:
            continue
        x_latent_, dy_ = pair
        n = len(x_latent_)
        data_matrix[:n, col_idx] = x_latent_ if q == v_name else dy_
    return pd.DataFrame(data_matrix, columns=out_cols)
