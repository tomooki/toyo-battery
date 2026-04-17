"""Cell: the unit of analysis — one TOYO cell's raw + derived DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from toyo_battery.io.reader import read_cell_dir
from toyo_battery.io.schema import ColumnLang


@dataclass
class Cell:
    """Single-cell measurement + derived quantities.

    P1: holds the normalized raw DataFrame. Derived tables (chdis, capacity,
    dQ/dV, stats) land in subsequent P1 branches.
    """

    name: str
    mass_g: float
    raw_df: pd.DataFrame
    column_lang: ColumnLang = "ja"

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        mass: float | None = None,
        encoding: str = "shift_jis",
        column_lang: ColumnLang = "ja",
    ) -> Cell:
        p = Path(path)
        df, mass_g = read_cell_dir(p, mass=mass, encoding=encoding, column_lang=column_lang)
        return cls(name=p.name, mass_g=mass_g, raw_df=df, column_lang=column_lang)
