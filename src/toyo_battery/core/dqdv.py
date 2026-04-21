"""dQ/dV computation via interpolation + Savitzky-Golay derivative.

For each ``(cycle, side)`` segment in a ``chdis_df`` (produced by
:func:`toyo_battery.core.chdis.get_chdis_df`), this module:

1. Extracts the voltage and capacity series, drops NaN rows, sorts by V,
   and removes duplicate V values.
2. Linearly interpolates Q onto a uniform V grid
   (``np.linspace(V.min(), V.max(), ipnum)`` with
   ``ipnum = max(int(inter_num * (V.max() - V.min())), 2)``).
3. Applies :func:`scipy.signal.savgol_filter` with ``deriv=1`` to compute
   ``dQ/dV`` on the resampled curve.

Output shape: a wide DataFrame whose column MultiIndex mirrors ``chdis_df``
(levels ``cycle``, ``side``, ``quantity``). The ``quantity`` level holds
``{"電圧", "dQ/dV"}`` when ``column_lang="ja"`` and
``{"voltage", "dq_dv"}`` when ``column_lang="en"``. The row axis is a
``RangeIndex`` shared across all segments; shorter segments are NaN-padded
to the longest.

Rewrite note (vs. legacy TOYO_Origin_2.01 L378 ``plot_dQdV_curve``): the
original read the voltage bounds via ``chdis_df[cycle].at[1, "電圧"]`` and
``chdis_df[cycle].at[len(...)-1, "電圧"]``. ``.at[1, ...]`` reads the
*second* row, not the first, so the original implicitly skipped the first
sample and was sensitive to whichever row landed at position 1 before
sorting. This rewrite uses ``V.min()`` and ``V.max()`` after an explicit
ascending sort with duplicate removal — deterministic, monotone, and works
whether the source segment is a charge or a discharge.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from toyo_battery.io.schema import JA_TO_EN, ColumnLang

_JA_COLS: dict[str, str] = {
    "voltage": "電圧",
    "capacity": "電気量",
    "dqdv": "dQ/dV",
}

# Per-segment result: (V-grid, dQ/dV samples), both same length.
_SegPair = tuple[NDArray[np.float64], NDArray[np.float64]]


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
        Output of :func:`toyo_battery.core.chdis.get_chdis_df`. Its column
        MultiIndex must have levels ``cycle``, ``side``, ``quantity`` with
        the ``quantity`` level using the names resolved by ``column_lang``
        (voltage + capacity).
    inter_num
        Interpolation density. The number of samples on the uniform V grid
        is ``max(int(inter_num * (V.max() - V.min())), 2)`` per segment.
        Segments whose interpolated length is below ``window_length`` yield
        all-NaN columns (Savitzky-Golay requires at least ``window_length``
        samples).
    window_length
        Savitzky-Golay window length (must be odd and > ``polyorder``; the
        caller is responsible for satisfying the scipy constraints).
    polyorder
        Savitzky-Golay polynomial order.
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

    Notes
    -----
    Non-default ``inter_num`` / ``window_length`` / ``polyorder`` must be
    passed by calling this function directly —
    :attr:`toyo_battery.core.cell.Cell.dqdv_df` uses the defaults.
    """
    cols = _resolve_cols(column_lang)
    v_name, q_name, dqdv_name = cols["voltage"], cols["capacity"], cols["dqdv"]

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
    cycle_side_pairs: list[tuple[int, str]] = sorted(
        {(int(c), str(s)) for c, s, _q in cols_tuples},
        key=lambda t: (t[0], 0 if t[1] == "ch" else 1),
    )

    per_segment: dict[tuple[int, str], _SegPair | None] = {}
    for cycle, side in cycle_side_pairs:
        try:
            v_series = chdis_df[(cycle, side, v_name)]
            q_series = chdis_df[(cycle, side, q_name)]
        except KeyError:
            per_segment[(cycle, side)] = None
            continue

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
        ip = interp1d(v_arr, q_arr, fill_value="extrapolate")
        q_latent: NDArray[np.float64] = np.asarray(ip(x_latent), dtype=float)
        # ``delta=dx`` makes savgol return d(Q)/d(V) in the physical units of
        # the input (mAh/g per V), not d(Q)/d(sample-index). v2.01 omitted
        # this and the legacy plots showed a slope-scaled axis; the rewrite
        # fixes it.
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
    if longest == 0:
        # All segments degenerate — still return a frame whose columns mirror
        # the input so downstream consumers can detect the empty case without
        # KeyErrors. Rows are empty.
        empty_data: dict[tuple[int, str, str], NDArray[np.float64]] = {}
        for cycle, side in cycle_side_pairs:
            empty_data[(cycle, side, v_name)] = np.array([], dtype=float)
            empty_data[(cycle, side, dqdv_name)] = np.array([], dtype=float)
        out = pd.DataFrame(empty_data)
        out.columns = pd.MultiIndex.from_tuples(
            cast("list[tuple[int, str, str]]", list(out.columns)),
            names=["cycle", "side", "quantity"],
        )
        return out

    columns: dict[tuple[int, str, str], NDArray[np.float64]] = {}
    for cycle, side in cycle_side_pairs:
        pair = per_segment[(cycle, side)]
        v_col: NDArray[np.float64] = np.full(longest, np.nan, dtype=float)
        d_col: NDArray[np.float64] = np.full(longest, np.nan, dtype=float)
        if pair is not None:
            x_latent, dy = pair
            v_col[: len(x_latent)] = x_latent
            d_col[: len(dy)] = dy
        columns[(cycle, side, v_name)] = v_col
        columns[(cycle, side, dqdv_name)] = d_col

    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(
        cast("list[tuple[int, str, str]]", list(out.columns)),
        names=["cycle", "side", "quantity"],
    )
    return out
