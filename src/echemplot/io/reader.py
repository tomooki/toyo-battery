"""Reader for TOYO cycler output directories.

Discovery priority for a given cell directory ``path``:

1. ``連続データ.csv``              — native export from the tester
   (7-line header: 3 metadata rows, then column-name row, channel row,
   separator row, units row; ``header=3, skiprows=[4,5,6]``)
2. ``連続データ_py.csv``           — already-normalized output from a previous run
3. 6-digit raw file(s) + ``*.PTN`` — factory raw; 電気量 is computed from mass

All three paths converge on the same canonical schema (canonical columns first;
any extra source columns such as 経過時間[Sec] / 電流[mA] / 日付 / 時刻 /
総ｻｲｸﾙ are preserved after them so downstream P1 phases can use them):

    Canonical columns: サイクル, モード, 状態, 電圧, 電気量
    状態 values:
      - Native 連続データ.csv:  {"充電", "放電", "充電休止", "放電休止"}
      - Raw 6-digit (int→JA):   {"充電", "放電", "休止"}
      - 連続データ_py.csv:       whatever was persisted (usually the 3-value set)

When ``column_lang="en"`` is requested, only column *names* are translated.
状態 cell values stay JP — translate via ``schema.STATE_JA_TO_EN`` downstream
if needed.

The active-material mass (grams) is resolved in this priority order:

1. Explicit ``mass=`` argument (grams)
2. ``重量[mg]`` from ``連続データ.csv`` metadata row 3 (converted mg → g)
3. A ``*.PTN`` file whose first line's 3rd whitespace-separated token parses
   as float (grams). Other ``.PTN`` files (e.g. ``*_OPTION.PTN`` config
   files shipped alongside the main pattern file) are skipped automatically.

On the raw 6-digit path the formula is::

    電気量 = 経過時間[Sec] / 3600  * 電流[mA] / mass

Real TOYO raw files set ``経過時間[Sec]`` to reset at each state transition
and emit ``電流[mA]`` as an unsigned magnitude, so this formula produces
per-segment monotone-non-decreasing 電気量 (matching the convention that
``連続データ.csv`` already uses inline).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import cast

import pandas as pd

from echemplot.io.schema import (
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

# Half-width → full-width canonical rename. The raw 6-digit header row uses
# half-width katakana for the cycle/mode columns.
_RAW_TO_CANONICAL = {"ｻｲｸﾙ": "サイクル", "ﾓｰﾄﾞ": "モード", "電圧[V]": "電圧"}

# Metadata key used by the native 連続データ.csv to carry active-material mass.
_METADATA_MASS_KEY = "重量[mg]"

# Number of metadata rows to scan when looking for 重量[mg]. The real file
# has exactly 3 metadata rows before the column-header row (index 3), so
# a scan of the first 4 rows is generous.
_METADATA_SCAN_ROWS = 4


def read_ptn_mass(ptn_path: str | Path) -> float:
    """Extract active-material mass (grams) from a ``.PTN`` file.

    TOYO ships at least two PTN dialects that differ in how the mass field
    is rendered on line 0. Both encode the field as a 9-byte composite of
    ``<flag><mass>``:

    * Older dialect (e.g. cycler "No6"): ``"0 0.00116"`` — flag, space,
      ``%.5f`` mass. ``str.split()`` yields the flag at ``tokens[2]`` and
      the mass at ``tokens[3]``.
    * Newer dialect (e.g. cyclers "No5"/"No1"): ``"00.000358"`` — flag
      glued to a ``%.6f`` mass. ``tokens[2]`` parses directly as the mass.

    Match the legacy ``TOYO_Origin_2.01`` heuristic: take ``tokens[2]``,
    and if it parses as exactly zero, fall back to ``tokens[3]``.

    Auxiliary ``.PTN`` files that TOYO ships alongside the main one
    (``*_OPTION.PTN``, ``*_Option2.PTN``, etc.) carry INI- or CSV-style
    configuration, not a mass; calling this on them will raise
    ``ValueError``, which :func:`_resolve_mass_from_ptn` relies on to
    skip them.
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
        candidate = float(tokens[2])
    except ValueError as e:
        raise ValueError(f"{path} line 0 token[2]={tokens[2]!r} is not a valid float mass") from e
    if candidate != 0.0:
        return candidate
    if len(tokens) >= 4:
        try:
            return float(tokens[3])
        except ValueError:
            pass
    raise ValueError(
        f"{path} line 0: token[2]=0 and token[3] absent or non-numeric; "
        "cannot extract active-material mass"
    )


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
        columns present in the source file (e.g. 経過時間[Sec], 電流[mA],
        日付, 時刻, 総ｻｲｸﾙ).
    mass_g : float
        Active-material mass used for the capacity calculation, in grams.
        ``math.nan`` if the source already had 電気量 precomputed and no
        mass information was available anywhere.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    native = p / RENZOKU_DATA
    if native.exists():
        resolved_mass = _resolve_mass(mass, p, native_csv=native, encoding=encoding)
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
    # Real-format raw files have: line 0 = "0,0,0,..." summary marker,
    # blank line(s), then the column-header row. pandas' default
    # skip_blank_lines=True collapses the blanks, so header=1 selects the
    # real header (the summary line is row 0 post-blank-skip).
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
    df = _drop_unnamed(df)
    df = _ensure_capacity(df, mass)
    return _finalize(df, column_lang), mass


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip BOM (which Excel-saved files sometimes prepend) before whitespace,
    # so a column like "\ufeffｻｲｸﾙ" still maps via _RAW_TO_CANONICAL.
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return cast("pd.DataFrame", df.rename(columns=_RAW_TO_CANONICAL))


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns pandas auto-labels ``Unnamed: N``.

    Raw TOYO 6-digit files have 6-7 empty separator columns between the
    sensor block (``経過時間[Sec]/電圧[V]/電流[mA]``) and the per-row
    metadata block (``状態/ﾓｰﾄﾞ/ｻｲｸﾙ/総ｻｲｸﾙ``). pandas names them
    ``Unnamed: 5``, ``Unnamed: 6``, ... — noise, never useful downstream.
    """
    keep = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    return cast("pd.DataFrame", df.loc[:, keep])


def _ensure_capacity(df: pd.DataFrame, mass: float | None) -> pd.DataFrame:
    """Add 電気量 = elapsed_s / 3600 * current_mA / mass, if missing.

    The TOYO raw 6-digit format emits ``経過時間[Sec]`` reset at each
    state transition and ``電流[mA]`` as an unsigned magnitude, so the
    formula produces per-segment monotone-non-decreasing 電気量 matching
    the convention that 連続データ.csv uses inline.
    """
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


def _resolve_mass(
    explicit: float | None,
    cell_dir: Path,
    *,
    native_csv: Path | None = None,
    encoding: str = "shift_jis",
) -> float | None:
    """Resolve active-material mass (grams) with the documented priority order.

    1. Explicit ``mass=`` argument.
    2. ``重量[mg]`` from native 連続データ.csv metadata (if ``native_csv`` given).
    3. First ``.PTN`` whose first line carries a parseable mass at token[2].
    """
    if explicit is not None:
        return explicit
    if native_csv is not None:
        metadata_mass = _extract_mass_from_renzoku_metadata(native_csv, encoding)
        if metadata_mass is not None:
            return metadata_mass
    return _resolve_mass_from_ptn(cell_dir)


def _resolve_mass_from_ptn(cell_dir: Path) -> float | None:
    """Scan top-level ``*.PTN`` files and return the single parseable mass.

    Raises ``ValueError`` if *more than one* ``.PTN`` yields a parseable
    mass — the caller must disambiguate with ``mass=``. A ``.PTN`` whose
    first line is INI-style (``[BaseCellCapacity]``), CSV-style, or simply
    doesn't have a float at token[2] is silently skipped; this is how a
    dir with both a main pattern and ``*_OPTION.PTN`` / ``*_Option2.PTN``
    resolves unambiguously.
    """
    ptn_files = sorted(
        p for p in cell_dir.iterdir() if p.is_file() and p.suffix.lower() == PTN_SUFFIX
    )
    masses: list[tuple[Path, float]] = []
    for ptn in ptn_files:
        try:
            masses.append((ptn, read_ptn_mass(ptn)))
        except ValueError:
            continue
    if len(masses) == 1:
        return masses[0][1]
    if len(masses) > 1:
        names = [p.name for p, _ in masses]
        raise ValueError(
            f"multiple .PTN files with parseable mass in {cell_dir} ({names}); "
            "pass `mass=<grams>` to disambiguate"
        )
    return None


def _extract_mass_from_renzoku_metadata(path: Path, encoding: str) -> float | None:
    """Return mass (grams) from the ``重量[mg]`` metadata row, or ``None``.

    Native 連続データ.csv has 3 metadata rows before the column-name row;
    the 3rd carries ``,重量[mg],<value>`` where <value> is in milligrams.
    """
    with path.open(encoding=encoding, errors="replace") as f:
        for _ in range(_METADATA_SCAN_ROWS):
            line = f.readline()
            if not line:
                return None
            fields = [c.strip() for c in line.rstrip("\r\n").split(",")]
            for i, cell in enumerate(fields):
                if cell == _METADATA_MASS_KEY and i + 1 < len(fields):
                    try:
                        mg = float(fields[i + 1])
                    except ValueError:
                        return None
                    return mg * 1e-3
    return None


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else value
