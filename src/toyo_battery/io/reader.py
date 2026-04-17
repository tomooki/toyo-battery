"""Reader for TOYO cycler output files. P1 scope — scaffold only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_cell_dir(path: str | Path, encoding: str = "shift_jis") -> pd.DataFrame:
    """Read a single cell directory and return a normalized DataFrame.

    Delegated to P1. This stub exists so downstream imports resolve.
    """
    raise NotImplementedError("read_cell_dir will be implemented in P1")
