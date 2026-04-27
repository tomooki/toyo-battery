"""Cell: the unit of analysis — one TOYO cell's raw + derived DataFrames."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import cast

import pandas as pd

from echemplot.core.capacity import get_cap_df
from echemplot.core.chdis import get_chdis_df
from echemplot.core.dqdv import get_dqdv_df
from echemplot.io.reader import read_cell_dir
from echemplot.io.schema import ColumnLang


class Cell:
    """Single-cell measurement + derived quantities.

    ``raw_df`` is the normalized source frame. Derived tables (``chdis_df``,
    ``cap_df``, ``dqdv_df``) materialize on first access via
    ``cached_property`` and are reused thereafter.

    Immutability (イミュータブル志向)
    --------------------------------
    The constructor deep-copies its input frame so the caller's DataFrame is
    fully isolated from the ``Cell``. Each of the four public DataFrame
    properties (``raw_df``, ``chdis_df``, ``cap_df``, ``dqdv_df``) returns a
    *defensive copy* of its internal state, so mutating a returned frame
    cannot corrupt the cached derivatives. The cached value itself is
    computed exactly once and reused as the source for subsequent copies.

    Tradeoff: the constructor copy roughly doubles peak memory for the raw
    frame at construction time, and every public read allocates a fresh
    copy. For typical TOYO-cell sizes (≤10⁶ rows) this is negligible; for
    very large frames or hot inner loops, hold the returned frame in a
    local variable rather than re-reading the property.

    Mutating the original frame *after* construction has no effect on the
    ``Cell``. Mutating a previously-returned copy from a property has no
    effect on subsequent reads. To produce a ``Cell`` from a transformed
    frame, build a new ``Cell`` from the new frame.
    """

    def __init__(
        self,
        name: str,
        mass_g: float,
        raw_df: pd.DataFrame,
        column_lang: ColumnLang = "ja",
    ) -> None:
        self.name = name
        self.mass_g = mass_g
        # Deep-copy on ingest so the caller can keep using their frame
        # without affecting (or being affected by) this Cell.
        self._raw_df: pd.DataFrame = cast("pd.DataFrame", raw_df.copy(deep=True))
        self.column_lang: ColumnLang = column_lang

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

    @property
    def raw_df(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized raw frame.

        Each call yields a fresh ``DataFrame``; mutations on the returned
        object do not propagate back into the ``Cell``.
        """
        return cast("pd.DataFrame", self._raw_df.copy())

    # ------------------------------------------------------------------
    # Derived frames. ``_<name>_cached`` holds the materialized result;
    # the public ``<name>`` property returns a defensive copy on every
    # read. The compute call runs exactly once per ``Cell`` instance.
    # ------------------------------------------------------------------

    @cached_property
    def _chdis_cached(self) -> pd.DataFrame:
        return get_chdis_df(self._raw_df, column_lang=self.column_lang)

    @property
    def chdis_df(self) -> pd.DataFrame:
        """Charge/discharge V-Q segments per cycle (defensive copy)."""
        return cast("pd.DataFrame", self._chdis_cached.copy())

    @cached_property
    def _cap_cached(self) -> pd.DataFrame:
        return get_cap_df(self._chdis_cached, column_lang=self.column_lang)

    @property
    def cap_df(self) -> pd.DataFrame:
        """Per-cycle capacity + Coulombic efficiency (defensive copy)."""
        return cast("pd.DataFrame", self._cap_cached.copy())

    @cached_property
    def _dqdv_cached(self) -> pd.DataFrame:
        return get_dqdv_df(self._chdis_cached, column_lang=self.column_lang)

    @property
    def dqdv_df(self) -> pd.DataFrame:
        """dQ/dV with default parameters (defensive copy).

        For non-default ``inter_num`` / ``window_length`` / ``polyorder``,
        call :func:`echemplot.core.dqdv.get_dqdv_df` directly on
        ``cell.chdis_df``.
        """
        return cast("pd.DataFrame", self._dqdv_cached.copy())
