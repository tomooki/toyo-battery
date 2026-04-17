"""Reader for TOYO cycler output directories.

Discovery priority for a given cell directory ``path``:

1. ``連続データ.csv``              — native export from the tester (header=3, skiprows=[4,5,6])
2. ``連続データ_py.csv``           — already-normalized output from a previous run
3. 6-digit raw file(s) + ``*.PTN`` — factory raw; 電気量 is computed from mass

All three paths converge on the same canonical schema (canonical columns first;
any extra source columns such as 経過時間[Sec] / 電流[mA] are preserved after
them so downstream P1 phases can use them):

    Canonical columns: サイクル, モード, 状態, 電圧, 電気量
    状態 values: {"休止", "充電", "放電"} (or NaN where the source had no value)

The column count therefore differs from legacy v2.01, which dropped the raw
measurement columns after computing 電気量. Numerical parity with v2.01 holds
on the canonical 5-column subset; tests that compare whole DataFrames must
project to that subset first.

When ``column_lang="en"`` is requested, only column *names* are translated.
状態 cell values stay JP — translate via ``schema.STATE_JA_TO_EN`` downstream
if needed.

The active-material mass (grams) is required to compute 電気量 when the source
file does not already contain it. It comes from (in order): the ``mass=``
argument, or a single top-level ``*.PTN`` file in the directory (third
whitespace token on line 0). If multiple ``.PTN`` files are present, the
caller must disambiguate via ``mass=`` — the reader refuses to guess.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import cast

import pandas as pd

from toyo_battery.io.schema import (
    CANONICAL_COLUMNS_JA,
    COL_CAPACITY,
    COL_CURRENT_MA,
    COL_ELAPSED_S,
    JA_TO_EN,
    STATE_CODE_TO_JA,
    ColumnLang,
)

RAW_FILENAME_RE = re.compile(r"[0-9]{6}")
PTN_SUFFIX = ".ptn"  # matched case-insensitively (Linux CI is case-sensitive)
RENZOKU_DATA = "連続データ.csv"
RENZOKU_DATA_PY = "連続データ_py.csv"

_RAW_TO_CANONICAL = {"ｻｲｸﾙ": "サイクル", "ﾓｰﾄﾞ": "モード", "電圧[V]": "電圧"}


def read_ptn_mass(ptn_path: str | Path) -> float:
    """Extract active-material mass (grams) from a ``.PTN`` file.

    TOYO convention: the mass is the third whitespace-separated token on line 0.
    """
    path = Path(ptn_path)
    with path.open(encoding="shift_jis", errors="replace") as f:
        first_line = f.readline()
    tokens = first_line.split()
    if len(tokens) < 3:
        raise ValueError(
            f"{path} line 0 has fewer than 3 tokens; cannot extract active-material mass"
        )
    try:
        return float(tokens[2])
    except ValueError as e:
        raise ValueError(f"{path} line 0 token[2]={tokens[2]!r} is not a valid float mass") from e


def read_cell_dir(
    path: str | Path,
    *,
    mass: float | None = None,
    encoding: str = "shift_jis",
    column_lang: ColumnLang = "ja",
) -> tuple[pd.DataFrame, float]:
    """Read a single cell directory into a normalized DataFrame.

    Returns
    -------
    df : DataFrame
        Canonical columns first (see :data:`schema.CANONICAL_COLUMNS_JA` or
        the EN equivalent when ``column_lang="en"``), followed by any extra
        columns present in the source file (e.g. 経過時間[Sec], 電流[mA]).
    mass_g : float
        Active-material mass used for the capacity calculation, in grams.
        ``math.nan`` if the source already had 電気量 precomputed and no
        ``.PTN`` was found at the top level of the directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    native = p / RENZOKU_DATA
    if native.exists():
        resolved_mass = _resolve_mass(mass, p)
        df = _read_renzoku_data(native, resolved_mass, encoding, column_lang)
        return df, _nan_if_none(resolved_mass)

    py_version = p / RENZOKU_DATA_PY
    if py_version.exists():
        resolved_mass = _resolve_mass(mass, p)
        df = _read_renzoku_data_py(py_version, encoding, column_lang)
        return df, _nan_if_none(resolved_mass)

    raw_files = _find_raw_files(p)
    if raw_files:
        resolved_mass = _resolve_mass(mass, p)
        if resolved_mass is None:
            raise ValueError(
                "6-digit raw files found but no mass available. "
                "Pass `mass=<grams>` or place a `.PTN` file in the directory."
            )
        return _read_raw_6digit(raw_files, resolved_mass, encoding, column_lang)

    raise FileNotFoundError(
        f"no TOYO data found under {p}: expected "
        f"'{RENZOKU_DATA}', '{RENZOKU_DATA_PY}', or 6-digit raw files"
    )


def _read_renzoku_data(
    path: Path, mass: float | None, encoding: str, column_lang: ColumnLang
) -> pd.DataFrame:
    df = pd.read_csv(path, header=3, skiprows=[4, 5, 6], encoding=encoding)
    df = _clean_columns(df)
    df = _ensure_capacity(df, mass)
    return _finalize(df, column_lang)


def _read_renzoku_data_py(path: Path, encoding: str, column_lang: ColumnLang) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, encoding=encoding)
    df = _clean_columns(df)
    return _finalize(df, column_lang)


def _read_raw_6digit(
    raw_files: list[Path], mass: float, encoding: str, column_lang: ColumnLang
) -> tuple[pd.DataFrame, float]:
    frames = [pd.read_csv(f, header=1, encoding=encoding) for f in raw_files]
    base_cols = list(frames[0].columns)
    for f, frame in zip(raw_files[1:], frames[1:]):
        if list(frame.columns) != base_cols:
            raise ValueError(
                f"raw file {f.name} has columns differing from {raw_files[0].name}: "
                f"{list(frame.columns)} vs {base_cols}"
            )
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = _clean_columns(df)
    df = _ensure_capacity(df, mass)
    return _finalize(df, column_lang), mass


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip BOM (which Excel-saved files sometimes prepend) before whitespace,
    # so a column like "\ufeffｻｲｸﾙ" still maps via _RAW_TO_CANONICAL.
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return cast("pd.DataFrame", df.rename(columns=_RAW_TO_CANONICAL))


def _ensure_capacity(df: pd.DataFrame, mass: float | None) -> pd.DataFrame:
    """Add 電気量 = elapsed_s / 3600 * current_mA / mass, if missing."""
    if COL_CAPACITY in df.columns:
        return df
    if mass is None or not math.isfinite(mass) or mass <= 0:
        raise ValueError(
            f"cannot compute {COL_CAPACITY}: source lacks the column and mass is missing "
            "or non-positive. Pass `mass=<grams>` or provide a valid `.PTN`."
        )
    missing = [c for c in (COL_ELAPSED_S, COL_CURRENT_MA) if c not in df.columns]
    if missing:
        raise ValueError(
            f"cannot compute {COL_CAPACITY}: source missing columns {missing} "
            f"and has no precomputed {COL_CAPACITY}"
        )
    out = df.copy()
    out[COL_CAPACITY] = out[COL_ELAPSED_S] / 3600.0 * out[COL_CURRENT_MA] / mass
    return cast("pd.DataFrame", out)


def _finalize(df: pd.DataFrame, column_lang: ColumnLang) -> pd.DataFrame:
    missing = [c for c in CANONICAL_COLUMNS_JA if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns after read: {missing}")
    extras = [c for c in df.columns if c not in CANONICAL_COLUMNS_JA]
    out = df.loc[:, [*CANONICAL_COLUMNS_JA, *extras]].copy()
    if pd.api.types.is_numeric_dtype(out["状態"]):
        mapped = out["状態"].map(STATE_CODE_TO_JA)
        unmapped_mask = mapped.isna() & out["状態"].notna()
        if unmapped_mask.any():
            bad = sorted(out.loc[unmapped_mask, "状態"].unique().tolist())
            raise ValueError(
                f"unknown 状態 codes in source: {bad} (known codes: {sorted(STATE_CODE_TO_JA)})"
            )
        out["状態"] = mapped
    out = out.reset_index(drop=True)
    if column_lang == "en":
        out = out.rename(columns={c: JA_TO_EN.get(c, c) for c in out.columns})
    return cast("pd.DataFrame", out)


def _find_raw_files(cell_dir: Path) -> list[Path]:
    """Find 6-digit raw files at the top level only.

    Mirrors the top-level-only rule for ``.PTN``: stale files in subdirs (e.g.
    ``backup/``) must not be silently absorbed into the read.
    """
    return sorted(
        q for q in cell_dir.iterdir() if q.is_file() and RAW_FILENAME_RE.fullmatch(q.name)
    )


def _resolve_mass(explicit: float | None, cell_dir: Path) -> float | None:
    if explicit is not None:
        return explicit
    ptn_files = sorted(
        p for p in cell_dir.iterdir() if p.is_file() and p.suffix.lower() == PTN_SUFFIX
    )
    if not ptn_files:
        return None
    if len(ptn_files) > 1:
        names = [p.name for p in ptn_files]
        raise ValueError(
            f"multiple .PTN files found in {cell_dir} ({names}); "
            "pass `mass=<grams>` to disambiguate"
        )
    return read_ptn_mass(ptn_files[0])


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else value
