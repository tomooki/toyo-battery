"""Charge/discharge segmentation.

Splits a raw DataFrame (from :func:`echemplot.io.reader.read_cell_dir`)
into per-cycle charge/discharge segments. The output is a wide DataFrame
with a 3-level column MultiIndex:

    level 0  cycle    — int cycle number (1, 2, 3, ...). Sourced from
                        ``総サイクル`` (the global cycle counter) when the
                        column is present, otherwise from ``サイクル`` (per-
                        mode counter). Multi-mode TOYO programs reset
                        ``サイクル`` at every mode boundary so two physically
                        distinct cycles can share ``サイクル=1`` — preferring
                        ``総サイクル`` keeps them as separate groups so the
                        running-max filter below does not have to disentangle
                        sub-runs across mode boundaries.
    level 1  side     — "ch" or "dis"
    level 2  quantity — "電気量"/"電圧" (or EN equivalents) as in the input

Capacity sign: real TOYO data has ``電気量`` reset to 0 at the start of each
segment and accumulate monotone-non-decreasing within it — the native
``連続データ.csv`` supplies such values inline, and the raw-6-digit path
reproduces the same convention because its ``経過時間[Sec]`` resets at
state transitions and its ``電流[mA]`` is unsigned. The reversal filter
nonetheless takes ``|電気量|`` so that *any* signed input (e.g. a
hand-crafted dataset with polarity applied) segments correctly. Rows
whose ``|電気量|`` falls below the segment's running maximum are dropped.
This guards against single-row tester glitches (e.g. 500→400→600, where
the 400 is dropped). It is *not* relied on to handle constant-voltage
(CV) holds: the raw-6-digit reader accumulates ``電気量`` as a true
cumulative integral ``∫I dt`` per step (see
:func:`echemplot.io.reader._ensure_capacity`), so a CC→CV charge is
already monotone and its CV tail is preserved here rather than clipped.
Earlier revisions computed an instantaneous ``I·t`` product that peaked
at the CC→CV transition and would have been silently dropped by this
filter, undercounting the charge capacity.

First-cycle-is-charge normalization: if cycle 1 begins with 放電 (discharge),
*all* 状態 labels in the frame are swapped (充電 ↔ 放電). This is the TOYO
convention for half-cells characterized discharge-first. Callers with an
unusual dataset where only cycle 1 is discharge-first (e.g. a one-off
formation discharge followed by normal charge-first cycles) must relabel
before calling this function. A test pins the global-swap behavior so
regressions are visible.

Rewrite note (vs. legacy TOYO_Origin_2.01): the original used string
MultiIndex labels like ``"1-ch"`` and checked the first-cycle direction
from ``df.at[1, "状態"]`` (row *1*, not row 0). This implementation uses
tuple labels ``(1, "ch", "電気量")`` and checks the first row of the first
cycle, which is also more robust against a rest row at position 0 that
v2.01 could misread.
"""

from __future__ import annotations

import warnings

import pandas as pd

from echemplot.core import DataIntegrityWarning
from echemplot.io.schema import (
    COL_TOTAL_CYCLE_JA,
    JA_COLS,
    JA_TO_EN,
    STATE_JA_TO_EN,
    ColumnLang,
)

_CHARGE_JA = "充電"
_DISCHARGE_JA = "放電"
_CHARGE_EN = STATE_JA_TO_EN[_CHARGE_JA]
_DISCHARGE_EN = STATE_JA_TO_EN[_DISCHARGE_JA]

_NEEDED_KEYS = ("cycle", "state", "voltage", "capacity")

_TOTAL_CYCLE_EN = JA_TO_EN[COL_TOTAL_CYCLE_JA]


def _resolve_cols(column_lang: ColumnLang) -> dict[str, str]:
    if column_lang == "ja":
        return {k: JA_COLS[k] for k in _NEEDED_KEYS}
    return {k: JA_TO_EN[JA_COLS[k]] for k in _NEEDED_KEYS}


def _resolve_cycle_col(df: pd.DataFrame, column_lang: ColumnLang, fallback: str) -> str:
    """Return the column name used as the cycle key for groupby.

    Prefers the global counter (``総サイクル`` / ``total_cycle``) when present
    in ``df.columns``, otherwise returns ``fallback`` (the per-mode
    ``サイクル`` / ``cycle`` column resolved by :func:`_resolve_cols`). The
    global counter is the right key for multi-mode TOYO programs where the
    per-mode counter resets at mode boundaries — see the module docstring.
    """
    total_col = COL_TOTAL_CYCLE_JA if column_lang == "ja" else _TOTAL_CYCLE_EN
    if total_col in df.columns:
        return total_col
    return fallback


def _resolve_states(column_lang: ColumnLang) -> tuple[str, str, dict[str, str]]:
    """Return (charge, discharge, side_map) labels for the active language.

    The reader translates state values when ``column_lang='en'``, so
    chdis must filter on EN literals in EN mode. ``side_map`` keys are the
    state literals; values are the canonical "ch"/"dis" side codes used in
    the output MultiIndex.
    """
    if column_lang == "ja":
        return _CHARGE_JA, _DISCHARGE_JA, {_CHARGE_JA: "ch", _DISCHARGE_JA: "dis"}
    return _CHARGE_EN, _DISCHARGE_EN, {_CHARGE_EN: "ch", _DISCHARGE_EN: "dis"}


def _empty_result() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([], names=["cycle", "side", "quantity"])
    return pd.DataFrame(columns=idx)


def get_chdis_df(df: pd.DataFrame, *, column_lang: ColumnLang = "ja") -> pd.DataFrame:
    """Segment a raw cell DataFrame into per-cycle ch/dis pairs.

    Parameters
    ----------
    df
        Output of :func:`echemplot.io.reader.read_cell_dir`. Must contain
        サイクル/状態/電圧/電気量 columns (or their EN equivalents when
        ``column_lang="en"``). ``状態`` cell values follow ``column_lang``:
        JA literals (``充電``/``放電``/``休止``/``充電休止``/``放電休止``/
        ``中断``) when ``column_lang='ja'``, EN equivalents (``charge``/
        ``discharge``/``rest``/``charge_rest``/``discharge_rest``/``abort``)
        when ``column_lang='en'``.
    column_lang
        Language of the input column names and the ``quantity`` level of
        the returned MultiIndex.

    Returns
    -------
    DataFrame
        Columns: MultiIndex with levels ``cycle``, ``side``, ``quantity``.
        For missing pairs (e.g. a cycle that only has a charge segment),
        rows are padded with NaN so all segments share a single row axis.

    Raises
    ------
    KeyError
        If the input frame is missing any of the required columns for the
        requested ``column_lang``.
    """
    cols = _resolve_cols(column_lang)
    required = [cols[k] for k in ("cycle", "state", "voltage", "capacity")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"input missing required columns {missing} "
            f"for column_lang={column_lang!r}; got columns={list(df.columns)}"
        )

    cap_col, v_col, state_col = (
        cols["capacity"],
        cols["voltage"],
        cols["state"],
    )
    cycle_col = _resolve_cycle_col(df, column_lang, fallback=cols["cycle"])
    charge_lbl, discharge_lbl, state_to_side = _resolve_states(column_lang)

    working = df.loc[df[state_col].isin([charge_lbl, discharge_lbl])].reset_index(drop=True)
    if working.empty:
        return _empty_result()

    first_cycle = working[cycle_col].min()
    first_state = working.loc[working[cycle_col] == first_cycle, state_col].iloc[0]
    if first_state == discharge_lbl:
        working = working.assign(
            **{
                state_col: working[state_col].map(
                    {charge_lbl: discharge_lbl, discharge_lbl: charge_lbl}
                )
            }
        )

    pieces: dict[tuple[int, str], pd.DataFrame] = {}
    total_dropped = 0
    segments_with_drops = 0
    for (cycle_val, state_val), g in working.groupby([cycle_col, state_col], sort=True):
        side = state_to_side[str(state_val)]
        # Drop rows whose |電気量| falls below the segment's running maximum.
        # Local diff()<0 was insufficient: in raw-6-digit data 経過時間[Sec]
        # resets at CC→CV sub-step boundaries (not just state boundaries),
        # so capacity drops by ~200 mAh/g and a row whose t*I happens to
        # inch up vs. its immediate predecessor would leak through and
        # produce a spurious connecting line in plot_chdis.
        cap = g[cap_col].abs().reset_index(drop=True)
        keep = (cap == cap.cummax()).to_numpy()
        dropped = int(len(keep) - int(keep.sum()))
        if dropped > 0:
            total_dropped += dropped
            segments_with_drops += 1
        segment = g[[cap_col, v_col]].iloc[keep].reset_index(drop=True)
        pieces[(int(cycle_val), side)] = segment

    if total_dropped > 0:
        warnings.warn(
            DataIntegrityWarning(
                f"chdis: dropped {total_dropped} rows below segment running-max "
                f"across {segments_with_drops} segments"
            ),
            stacklevel=2,
        )

    out = pd.concat(pieces, axis=1)
    out.columns = out.columns.set_names(["cycle", "side", "quantity"])
    return out
