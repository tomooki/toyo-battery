"""Charge/discharge segmentation.

Splits a raw DataFrame (from :func:`toyo_battery.io.reader.read_cell_dir`)
into per-cycle charge/discharge segments. The output is a wide DataFrame
with a 3-level column MultiIndex:

    level 0  cycle    — int cycle number (1, 2, 3, ...)
    level 1  side     — "ch" or "dis"
    level 2  quantity — "電気量"/"電圧" (or EN equivalents) as in the input

Capacity sign: real TOYO data has ``電気量`` reset to 0 at the start of each
segment and accumulate monotone-non-decreasing within it — the native
``連続データ.csv`` supplies such values inline, and the raw-6-digit path
reproduces the same convention because its ``経過時間[Sec]`` resets at
state transitions and its ``電流[mA]`` is unsigned. The reversal filter
nonetheless takes ``|電気量|`` so that *any* signed input (e.g. a
hand-crafted dataset with polarity applied) segments correctly. Rows
where ``|電気量|.diff() < 0`` are dropped as tester-side glitches.

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

import pandas as pd

from toyo_battery.io.schema import JA_TO_EN, ColumnLang

_CHARGE = "充電"
_DISCHARGE = "放電"
_STATE_TO_SIDE = {_CHARGE: "ch", _DISCHARGE: "dis"}

_JA_COLS: dict[str, str] = {
    "cycle": "サイクル",
    "state": "状態",
    "voltage": "電圧",
    "capacity": "電気量",
}


def _resolve_cols(column_lang: ColumnLang) -> dict[str, str]:
    if column_lang == "ja":
        return _JA_COLS
    return {k: JA_TO_EN[v] for k, v in _JA_COLS.items()}


def _empty_result() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([], names=["cycle", "side", "quantity"])
    return pd.DataFrame(columns=idx)


def get_chdis_df(df: pd.DataFrame, *, column_lang: ColumnLang = "ja") -> pd.DataFrame:
    """Segment a raw cell DataFrame into per-cycle ch/dis pairs.

    Parameters
    ----------
    df
        Output of :func:`toyo_battery.io.reader.read_cell_dir`. Must contain
        サイクル/状態/電圧/電気量 columns (or their EN equivalents when
        ``column_lang="en"``). ``状態`` cell values are always JA
        (``充電``/``放電``/``休止``) regardless of ``column_lang``.
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

    cap_col, v_col, cycle_col, state_col = (
        cols["capacity"],
        cols["voltage"],
        cols["cycle"],
        cols["state"],
    )

    working = df.loc[df[state_col].isin([_CHARGE, _DISCHARGE])].reset_index(drop=True)
    if working.empty:
        return _empty_result()

    first_cycle = working[cycle_col].min()
    first_state = working.loc[working[cycle_col] == first_cycle, state_col].iloc[0]
    if first_state == _DISCHARGE:
        working = working.assign(
            **{state_col: working[state_col].map({_CHARGE: _DISCHARGE, _DISCHARGE: _CHARGE})}
        )

    pieces: dict[tuple[int, str], pd.DataFrame] = {}
    for (cycle_val, state_val), g in working.groupby([cycle_col, state_col], sort=True):
        side = _STATE_TO_SIDE[str(state_val)]
        # abs() is defensive: real TOYO data (both 連続データ and raw-6-digit
        # paths) is already monotone non-decreasing per segment; abs() keeps
        # the filter correct under any hand-crafted signed input.
        segment = g[[cap_col, v_col]].loc[~(g[cap_col].abs().diff() < 0)].reset_index(drop=True)
        pieces[(int(cycle_val), side)] = segment

    out = pd.concat(pieces, axis=1)
    out.columns = out.columns.set_names(["cycle", "side", "quantity"])
    return out
