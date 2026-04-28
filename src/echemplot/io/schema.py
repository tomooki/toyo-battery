"""Column name mapping between Japanese (TOYO native) and English.

The canonical in-memory column set after reading is the JA list below. When
`column_lang="en"` is requested, columns are renamed to EN at the end of the
read pipeline.
"""

from __future__ import annotations

from typing import Literal

ColumnLang = Literal["ja", "en"]

CANONICAL_COLUMNS_JA: tuple[str, ...] = (
    "サイクル",
    "モード",
    "状態",
    "電圧",
    "電気量",
)

# Non-canonical TOYO source columns referenced by the reader. Held as named
# constants here so reader code does not have to hard-code the JP literals.
COL_ELAPSED_S: str = "経過時間[Sec]"
COL_CURRENT_MA: str = "電流[mA]"
COL_CAPACITY: str = "電気量"

# Optional global cycle counter that the TOYO firmware emits alongside the
# per-mode ``サイクル`` column. Cyclers running multi-mode test programs
# (e.g. mode 1 = formation, mode 2 = regular cycling) reset ``サイクル`` at
# every mode boundary while ``総サイクル`` keeps counting monotonically; chdis
# prefers ``総サイクル`` as the cycle key when present so cycles from
# different modes do not get conflated. Half-width ``総ｻｲｸﾙ`` is the source
# spelling and is canonicalized to this full-width form by the reader.
COL_TOTAL_CYCLE_JA: str = "総サイクル"

CANONICAL_COLUMNS_EN: tuple[str, ...] = (
    "cycle",
    "mode",
    "state",
    "voltage",
    "capacity",
)

JA_TO_EN: dict[str, str] = {
    "サイクル": "cycle",
    "総サイクル": "total_cycle",
    "モード": "mode",
    "状態": "state",
    "電圧": "voltage",
    "電流": "current",
    "電気量": "capacity",
    "経過時間": "elapsed_time",
    "経過時間[Sec]": "elapsed_time_s",
    "電圧[V]": "voltage_v",
    "電流[mA]": "current_ma",
}

EN_TO_JA: dict[str, str] = {v: k for k, v in JA_TO_EN.items()}

# Canonical-key → JA column literal. Single source of truth used by every
# core/plotting module that needs to address TOYO source columns by a
# language-neutral logical name. Callers select either ``JA_COLS[k]`` (when
# the in-memory frame is ``column_lang="ja"``) or
# ``JA_TO_EN[JA_COLS[k]]`` (when ``column_lang="en"``).
#
# Keys are kept tight: only the columns at least one ``core`` / ``plotting``
# module dereferences by logical name today. Do not add a key without a
# real consumer — non-canonical TOYO source columns (経過時間[Sec], 電流[mA],
# モード, …) live as ``COL_*`` named constants above and are addressed
# directly by the reader, not via this map.
#
# * ``cycle``     — per-row cycle index
# * ``state``     — JP state literal (``充電``/``放電``/``休止`` etc.)
# * ``voltage``   — terminal voltage column
# * ``capacity``  — accumulated charge/discharge capacity column
JA_COLS: dict[str, str] = {
    "cycle": "サイクル",
    "state": "状態",
    "voltage": "電圧",
    "capacity": COL_CAPACITY,
}

# State-code mapping for the raw 6-digit format. Codes {0, 1, 2} are
# documented; code 9 is empirically observed across multiple TOYO cyclers
# (No1 / No2 / No6) as an end-of-test "abort" signal — typically a single
# trailing row, sometimes with non-zero 経過時間/電流 values left from the
# moment of interruption. We do not have official TOYO documentation for
# code 9, but it has been validated against real cell directories. Do not
# strip the 9-entry without re-checking against current data.
STATE_CODE_TO_JA: dict[int, str] = {0: "休止", 1: "充電", 2: "放電", 9: "中断"}
STATE_CODE_TO_EN: dict[int, str] = {
    0: "rest",
    1: "charge",
    2: "discharge",
    9: "abort",
}
# Native ``連続データ.csv`` exports surface ``充電休止`` / ``放電休止`` as
# distinct labels alongside the basic 3-state set (and the 中断 abort
# sentinel), so the JA→EN map carries six entries. The raw 6-digit path
# only emits the basic codes {0, 1, 2, 9}, so ``STATE_CODE_TO_JA`` keeps
# its 4-entry shape.
STATE_JA_TO_EN: dict[str, str] = {
    "休止": "rest",
    "充電": "charge",
    "放電": "discharge",
    "中断": "abort",
    "充電休止": "charge_rest",
    "放電休止": "discharge_rest",
}
STATE_EN_TO_JA: dict[str, str] = {v: k for k, v in STATE_JA_TO_EN.items()}


def rename(columns: list[str], target: ColumnLang) -> list[str]:
    """Translate a list of column names to the target language."""
    if target == "ja":
        return [EN_TO_JA.get(c, c) for c in columns]
    return [JA_TO_EN.get(c, c) for c in columns]
