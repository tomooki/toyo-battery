"""Cell: the unit of analysis — one TOYO cell's raw + derived DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from toyo_battery.io.schema import ColumnLang


@dataclass
class Cell:
    """Single-cell measurement + derived quantities.

    Stubbed in P0. Real logic lands in P1 (reader, chdis, capacity, dqdv).
    """

    name: str
    mass_g: float
    raw_df: pd.DataFrame
    column_lang: ColumnLang = "ja"

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        mass: float | None = None,
        encoding: str = "shift_jis",
        column_lang: ColumnLang = "ja",
    ) -> Cell:
        raise NotImplementedError("Cell.from_dir will be implemented in P1")
