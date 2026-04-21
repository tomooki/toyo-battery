"""Cell: the unit of analysis — one TOYO cell's raw + derived DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from toyo_battery.core.capacity import get_cap_df
from toyo_battery.core.chdis import get_chdis_df
from toyo_battery.core.dqdv import get_dqdv_df
from toyo_battery.io.reader import read_cell_dir
from toyo_battery.io.schema import ColumnLang


@dataclass
class Cell:
    """Single-cell measurement + derived quantities.

    ``raw_df`` is the normalized source frame. Derived tables (``chdis_df``
    and, in subsequent branches, capacity / dQ-dV / stats) are exposed as
    ``cached_property`` so they materialize on first access and are reused.

    ``raw_df`` is treated as immutable after construction. Mutating it in
    place (e.g. ``cell.raw_df.drop(...)``) will leave any already-accessed
    cached properties stale. Build a new ``Cell`` from the modified frame
    instead.
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

    @cached_property
    def chdis_df(self) -> pd.DataFrame:
        return get_chdis_df(self.raw_df, column_lang=self.column_lang)

    @cached_property
    def cap_df(self) -> pd.DataFrame:
        return get_cap_df(self.chdis_df, column_lang=self.column_lang)

    @cached_property
    def dqdv_df(self) -> pd.DataFrame:
        """dQ/dV with default parameters. For non-default ``inter_num`` /
        ``window_length`` / ``polyorder``, call
        :func:`toyo_battery.core.dqdv.get_dqdv_df` directly on
        ``cell.chdis_df``.
        """
        return get_dqdv_df(self.chdis_df, column_lang=self.column_lang)
