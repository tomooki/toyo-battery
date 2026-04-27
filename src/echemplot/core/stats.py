"""Per-cell summary statistics (``stat_table``).

Aggregates a list of :class:`echemplot.core.cell.Cell` objects into a
single wide summary table — one row per cell, with a deterministic set of
capacity / efficiency / retention columns keyed off a caller-supplied
``target_cycles``. This is the table that downstream users typically
export as CSV or feed into a plotting pipeline.

Output columns (all float64, English-fixed regardless of each cell's
``column_lang``):

* ``Q_dis_max`` — ``max(cap_df["q_dis"])`` over all cycles for that cell.
* ``cycle_at_{pct}pct`` — first cycle at which ``q_dis`` dips below
  ``retention_threshold * q_dis[1]``. ``NaN`` if the fade threshold is
  never crossed, or if ``q_dis[1]`` is missing / zero. The ``{pct}``
  suffix is derived from the threshold: ``retention_threshold=0.80``
  yields ``cycle_at_80pct``.
* For each ``n`` in ``target_cycles`` (in caller-given order):

  - ``Q_dis@{n}`` — discharge capacity at cycle ``n`` (NaN if missing).
  - ``CE@{n}`` — Coulombic efficiency at cycle ``n`` (NaN if missing).
  - ``V_mean_dis@{n}`` — energy-weighted mean discharge voltage at cycle
    ``n``: ``(∫V dQ) / q_dis[n]`` where the integral runs over the
    discharge segment's ``(V, Q)`` curve.
  - ``EE@{n}`` — energy efficiency in percent:
    ``100 * ∫V dQ_dis / ∫V dQ_ch``. NaN if either integral is NaN or if
    the charge-side integral is zero.
  - ``retention@{n}`` — ``100 * q_dis[n] / q_dis[1]`` in percent. Exactly
    ``100.0`` for ``n == 1`` when ``q_dis[1]`` is non-NaN. NaN when
    ``q_dis[1]`` or ``q_dis[n]`` is missing / zero-denominator.
  - ``CE_mean_1to{n}`` — mean of ``cap_df.loc[1:n, "ce"]`` skipping NaN.
    NaN if cycle 1 is absent or the sub-range contains no finite values.

Integration detail: each per-segment ``∫V dQ`` uses Q as the integration
variable (the :mod:`chdis` contract guarantees Q is monotone non-decreasing
within a segment). After sorting rows by Q ascending and dropping duplicate
Q values (``keep="last"``, same convention as :mod:`dqdv` for CC-CV tail
preservation), the integral is evaluated by:

- :func:`scipy.integrate.simpson` when ``len >= 3``
- :func:`scipy.integrate.trapezoid` when ``len == 2``
- ``NaN`` otherwise

The deprecated ``scipy.integrate.simps`` is intentionally not used;
``numpy.trapz`` was removed in NumPy 2.0 and is likewise avoided in favour
of :func:`scipy.integrate.trapezoid`.

Rewrite note (vs. legacy TOYO_Origin_2.01 ``stat_table`` L579-656): v2.01
passed positional args to ``simpson`` and read segment bounds from
``chdis_df[str(n)+"-dis"].at[1, "電圧"]`` (row 1, not 0), then wrapped the
entire per-n block in a bare ``except Exception`` that silently masked
structural bugs as missing data. This rewrite uses Q as the integration
variable via explicit keyword args (``simpson(y=v, x=q)`` ⇒ unambiguously
``∫V dQ``), sorts + de-duplicates Q before reading bounds, and propagates
NaN only for genuinely-missing data — structural mismatches raise loudly
from upstream (:mod:`chdis`, :mod:`capacity`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.integrate import simpson, trapezoid

from echemplot.io.schema import JA_COLS, JA_TO_EN, ColumnLang

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from echemplot.core.cell import Cell

_NEEDED_KEYS = ("voltage", "capacity")


def _resolve_cols(column_lang: ColumnLang) -> dict[str, str]:
    if column_lang == "ja":
        return {k: JA_COLS[k] for k in _NEEDED_KEYS}
    return {k: JA_TO_EN[JA_COLS[k]] for k in _NEEDED_KEYS}


def _build_column_order(target_cycles: Sequence[int], pct: int) -> list[str]:
    """Canonical column order shared between the populated and empty paths."""
    cols: list[str] = ["Q_dis_max", f"cycle_at_{pct}pct"]
    for n in target_cycles:
        cols.extend(
            [
                f"Q_dis@{n}",
                f"CE@{n}",
                f"V_mean_dis@{n}",
                f"EE@{n}",
                f"retention@{n}",
                f"CE_mean_1to{n}",
            ]
        )
    return cols


def _empty_result(columns: list[str]) -> pd.DataFrame:
    """Empty frame with the right columns / dtypes / index name."""
    idx = pd.Index([], dtype="object", name="cell")
    data = {c: pd.Series([], dtype="float64") for c in columns}
    return pd.DataFrame(data, index=idx)


def _integrate_v_dq(v_arr: NDArray[np.float64], q_arr: NDArray[np.float64]) -> float:
    """Return ``∫V dQ`` over a single segment, using Q as the integration axis.

    The :mod:`chdis` contract guarantees Q is monotone non-decreasing within
    a segment, so a Q-ascending sort is a stable reorder. Duplicate Q values
    — rare in real data because CC-CV plateaus saturate V, not Q — are
    dropped with ``keep="last"`` to match the :mod:`dqdv` convention,
    preserving the latest-recorded V for any repeat.

    Returns ``NaN`` for <2 finite points; trapezoidal for exactly 2;
    Simpson's rule for 3+.
    """
    # Pair up, drop any NaN rows, and sort by Q ascending. A dedicated
    # DataFrame round-trip keeps the sort + dedup logic one-liner-clean
    # and mirrors the :mod:`dqdv` pattern.
    frame = pd.DataFrame({"v": v_arr, "q": q_arr}).dropna()
    if frame.empty:
        return float("nan")
    frame = (
        frame.sort_values("q", kind="mergesort")
        .drop_duplicates(subset="q", keep="last")
        .reset_index(drop=True)
    )
    n = len(frame)
    if n < 2:
        return float("nan")
    v: NDArray[np.float64] = frame["v"].to_numpy(dtype=float)
    q: NDArray[np.float64] = frame["q"].to_numpy(dtype=float)
    if n == 2:
        return float(trapezoid(y=v, x=q))
    # simpson() returns a numpy scalar; explicit float() cast pins the
    # return contract for mypy under the scipy.* ignore_missing_imports
    # override so callers see a plain float, not an ``Any``.
    return float(simpson(y=v, x=q))


def _segment_arrays(
    chdis_df: pd.DataFrame,
    cycle: int,
    side: Literal["ch", "dis"],
    v_name: str,
    q_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Pull (V, Q) arrays for one ``(cycle, side)`` segment, or None if absent.

    Returns ``None`` when the segment's columns are not present at all
    (e.g. ``target_cycles`` references a cycle the cell never reached).
    When the columns exist but hold only NaN, returns the raw arrays and
    lets the integrator handle the degenerate case — keeps the "cycle
    exists but reversal-filter wiped it" and "cycle doesn't exist" cases
    distinguishable upstream if needed.
    """
    key_v = (cycle, side, v_name)
    key_q = (cycle, side, q_name)
    if key_v not in chdis_df.columns or key_q not in chdis_df.columns:
        return None
    v: NDArray[np.float64] = chdis_df[key_v].to_numpy(dtype=float)
    q: NDArray[np.float64] = chdis_df[key_q].to_numpy(dtype=float)
    return v, q


def _safe_lookup(cap_df: pd.DataFrame, cycle: int, col: str) -> float:
    """Return ``cap_df.loc[cycle, col]`` as float, or NaN if the row is absent."""
    if cycle not in cap_df.index:
        return float("nan")
    val = cap_df.at[cycle, col]
    return float("nan") if pd.isna(val) else float(val)


def _cycle_at_threshold(
    q_dis_series: pd.Series, q_dis_first: float, retention_threshold: float
) -> float:
    """First cycle where ``q_dis < retention_threshold * q_dis[1]``, else NaN.

    ``q_dis_first`` is passed in (rather than re-derived) so callers that
    already decided cycle 1 is unusable (missing / zero) can short-circuit
    without a redundant lookup.
    """
    if not np.isfinite(q_dis_first) or q_dis_first == 0.0:
        return float("nan")
    threshold = retention_threshold * q_dis_first
    fade_mask = q_dis_series < threshold
    if not fade_mask.any():
        return float("nan")
    # idxmax on a boolean Series returns the label of the first True;
    # safe here because we've confirmed at least one True exists.
    return float(fade_mask.idxmax())


def _ce_mean_1_to_n(cap_df: pd.DataFrame, n: int) -> float:
    """``cap_df.loc[1:n, "ce"].mean(skipna=True)``, NaN-safe.

    Returns NaN when cycle 1 is absent from the cap frame or when the
    sliced sub-series contains no finite values (so downstream consumers
    don't conflate "no data" with an artificial zero).
    """
    if 1 not in cap_df.index:
        return float("nan")
    # ``.loc[1:n]`` is label-inclusive for both bounds on a sorted int
    # index; capacity guarantees the index is sorted int64 ascending.
    sub = cap_df.loc[1:n, "ce"]
    if sub.empty:
        return float("nan")
    mean = sub.mean(skipna=True)
    return float("nan") if pd.isna(mean) else float(mean)


def _per_cell_row(
    cell: Cell,
    target_cycles: Sequence[int],
    retention_threshold: float,
    pct: int,
) -> dict[str, float]:
    """Compute one row of the stat table for a single Cell."""
    cap_df = cell.cap_df
    chdis_df = cell.chdis_df
    cols = _resolve_cols(cell.column_lang)
    v_name, q_name = cols["voltage"], cols["capacity"]

    # :func:`capacity.get_cap_df` unconditionally emits ``q_ch`` / ``q_dis``
    # / ``ce`` columns (see capacity.py's empty-path branch), so no column
    # presence guards are needed here — the frame may be row-empty, but the
    # schema is fixed.
    q_dis_series = cap_df["q_dis"]
    q_dis_max = float(q_dis_series.max()) if not q_dis_series.dropna().empty else float("nan")

    # cycle-1 discharge capacity — the reference for retention + fade.
    q_dis_first = _safe_lookup(cap_df, 1, "q_dis")

    row: dict[str, float] = {
        "Q_dis_max": q_dis_max,
        f"cycle_at_{pct}pct": _cycle_at_threshold(q_dis_series, q_dis_first, retention_threshold),
    }

    for n in target_cycles:
        q_dis_n = _safe_lookup(cap_df, n, "q_dis")
        ce_n = _safe_lookup(cap_df, n, "ce")

        # V_mean_dis@n and EE@n need per-segment integrals.
        v_mean_dis = float("nan")
        ee = float("nan")
        dis_pair = _segment_arrays(chdis_df, n, "dis", v_name, q_name)
        ch_pair = _segment_arrays(chdis_df, n, "ch", v_name, q_name)

        if dis_pair is not None:
            int_v_dq_dis = _integrate_v_dq(*dis_pair)
            if np.isfinite(int_v_dq_dis):
                if np.isfinite(q_dis_n) and q_dis_n != 0.0:
                    v_mean_dis = int_v_dq_dis / q_dis_n
                # Charge-side integral is only needed when the discharge
                # side yielded a finite value — otherwise EE is NaN
                # regardless of the charge integral.
                if ch_pair is not None:
                    int_v_dq_ch = _integrate_v_dq(*ch_pair)
                    if np.isfinite(int_v_dq_ch) and int_v_dq_ch != 0.0:
                        ee = 100.0 * int_v_dq_dis / int_v_dq_ch

        # Retention@n: 100.0 exactly for n=1 iff q_dis[1] is finite; else
        # 100 * q_dis[n] / q_dis[1] with NaN propagation through both
        # numerator and denominator.
        if not np.isfinite(q_dis_first) or q_dis_first == 0.0:
            retention = float("nan")
        elif n == 1:
            retention = 100.0
        elif np.isfinite(q_dis_n):
            retention = 100.0 * q_dis_n / q_dis_first
        else:
            retention = float("nan")

        row[f"Q_dis@{n}"] = q_dis_n
        row[f"CE@{n}"] = ce_n
        row[f"V_mean_dis@{n}"] = v_mean_dis
        row[f"EE@{n}"] = ee
        row[f"retention@{n}"] = retention
        row[f"CE_mean_1to{n}"] = _ce_mean_1_to_n(cap_df, n)

    return row


def stat_table(
    cells: Sequence[Cell],
    *,
    target_cycles: Sequence[int] = (10, 50),
    retention_threshold: float = 0.80,
) -> pd.DataFrame:
    """Summarize a list of cells into a per-cell wide statistics table.

    Parameters
    ----------
    cells
        Sequence of :class:`echemplot.core.cell.Cell`. Each cell
        contributes one row keyed by ``cell.name``. An empty sequence
        returns an empty DataFrame with the correct column schema.
    target_cycles
        Cycle numbers at which to emit ``Q_dis@n`` / ``CE@n`` /
        ``V_mean_dis@n`` / ``EE@n`` / ``retention@n`` / ``CE_mean_1to{n}``.
        Order is preserved in the output columns so callers can pin column
        layout by tuple order. Cycles beyond a cell's maximum produce NaN
        for that cell, but the row itself is still emitted. Duplicate
        entries raise ``ValueError`` to avoid silently-duplicated columns.
    retention_threshold
        Fade threshold as a fraction of ``q_dis[1]``, in the open interval
        ``(0, 1)`` (values at or outside the bounds raise ``ValueError``).
        The derived column name is
        ``f"cycle_at_{round(retention_threshold * 100)}pct"`` so a
        threshold of ``0.80`` yields ``cycle_at_80pct`` (default) and
        ``0.50`` yields ``cycle_at_50pct``. ``round`` (not ``int``) avoids
        the IEEE-754 truncation trap where e.g. ``0.29 * 100`` evaluates
        to ``28.999999999999996`` and would otherwise mislabel the column
        ``cycle_at_28pct``. Python's ``round`` uses banker's rounding (tie
        to even) — irrelevant for realistic thresholds but worth noting
        if a caller passes an exact-half value like ``0.125``. The value
        in that column is the first cycle where ``q_dis`` dips below
        ``retention_threshold * q_dis[1]``; NaN if the threshold is never
        crossed or if ``q_dis[1]`` is unusable.

    Returns
    -------
    DataFrame
        Index ``cell`` (``cell.name`` strings). Columns in deterministic
        order (see the module docstring). All values are float64; empty
        positions are NaN.

    Raises
    ------
    ValueError
        If ``retention_threshold`` is not in the open interval ``(0, 1)``,
        or if ``target_cycles`` contains duplicate entries.

    Notes
    -----
    Integration over ``∫V dQ`` uses :func:`scipy.integrate.simpson` when
    the segment has 3+ distinct-Q points, :func:`scipy.integrate.trapezoid`
    for exactly 2, and yields NaN for fewer. The deprecated
    ``scipy.integrate.simps`` is not used; ``numpy.trapz`` (removed in
    NumPy 2.0) is avoided in favour of ``scipy.integrate.trapezoid``.
    """
    if not (0.0 < retention_threshold < 1.0):
        raise ValueError(
            f"retention_threshold must lie in the open interval (0, 1); got {retention_threshold!r}"
        )
    target_cycles_list = list(target_cycles)
    if len(set(target_cycles_list)) != len(target_cycles_list):
        raise ValueError(f"target_cycles must not contain duplicates; got {target_cycles_list!r}")

    # ``round`` (not ``int``) because ``0.29 * 100 == 28.999999999999996``
    # under IEEE-754 and ``int`` would truncate to ``28``, silently
    # mislabelling the column. See the ``retention_threshold`` docstring.
    pct = round(retention_threshold * 100)
    columns = _build_column_order(target_cycles_list, pct)

    if not cells:
        return _empty_result(columns)

    # Seed the accumulator with the full column set so every row has the
    # same key set and ``DataFrame.from_records`` doesn't need a manual
    # reindex after the fact. The dict preserves insertion order (py 3.7+)
    # so column order round-trips cleanly from ``_build_column_order``.
    names: list[str] = []
    rows: list[dict[str, float]] = []
    for cell in cells:
        names.append(cell.name)
        rows.append(_per_cell_row(cell, target_cycles_list, retention_threshold, pct))

    # ``dtype="object"`` for the populated path too, matching ``_empty_result``
    # so ``pd.concat`` of an empty + populated stat_table doesn't trigger
    # dtype-coercion warnings on the index.
    out = pd.DataFrame(rows, index=pd.Index(names, dtype="object", name="cell"))
    # Enforce column order + float64 dtype end-to-end. ``reindex`` also
    # guards against any runtime dict-mutation slip that silently dropped
    # a key — an absent column would surface here as a silent all-NaN
    # column, which is the correct degraded behaviour anyway. The explicit
    # ``pd.DataFrame`` wrap pins the mypy return type against pandas-stubs
    # occasionally inferring ``.astype`` as ``Any``.
    return pd.DataFrame(out.reindex(columns=columns).astype("float64"))
