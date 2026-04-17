"""Column name mapping between Japanese (TOYO native) and English."""

from __future__ import annotations

from typing import Literal

ColumnLang = Literal["ja", "en"]

JA_TO_EN: dict[str, str] = {
    "サイクル": "cycle",
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

STATE_CODE_TO_JA: dict[int, str] = {0: "休止", 1: "充電", 2: "放電"}
STATE_CODE_TO_EN: dict[int, str] = {0: "rest", 1: "charge", 2: "discharge"}


def rename(columns: list[str], target: ColumnLang) -> list[str]:
    """Translate a list of column names to the target language."""
    if target == "ja":
        return [EN_TO_JA.get(c, c) for c in columns]
    return [JA_TO_EN.get(c, c) for c in columns]
