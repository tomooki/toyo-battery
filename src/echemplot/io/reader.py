"""Reader for TOYO cycler output directories.

Discovery priority for a given cell directory ``path``:

1. ``連続データ.csv``              — native export from the tester.
   Real TOYO firmware emits a 7-line preamble: 3 metadata rows, then
   column-name row, channel row, separator row, units row (so historically
   ``header=3, skiprows=[4,5,6]``). The reader now finds the column-name
   row by content (first cell == ``サイクル``) so an extra/missing
   metadata row in a future firmware revision does not silently shift the
   header. See :func:`_detect_renzoku_header`.
2. ``連続データ_py.csv``           — already-normalized output from a previous run
3. 6-digit raw file(s) + ``*.PTN`` — factory raw; 電気量 is computed from mass

All three paths converge on the same canonical schema (canonical columns first;
any extra source columns such as 経過時間[Sec] / 電流[mA] / 日付 / 時刻 /
総サイクル are preserved after them so downstream P1 phases can use them.
The half-width source spelling ``総ｻｲｸﾙ`` is canonicalized to the full-width
``総サイクル`` here so :mod:`echemplot.core.chdis` can prefer it over
``サイクル`` as the cycle key without re-asking which form the reader emitted):

    Canonical columns: サイクル, モード, 状態, 電圧, 電気量
    状態 values:
      - Native 連続データ.csv:  {"充電", "放電", "充電休止", "放電休止"}
      - Raw 6-digit (int→JA):   {"充電", "放電", "休止", "中断"}
                                ("中断" = state code 9, the TOYO end-of-test
                                 / abort sentinel; usually a single trailing
                                 row, sometimes with non-zero 経過時間/電流.)
      - 連続データ_py.csv:       whatever was persisted (usually the 3-value set)

When ``column_lang="en"`` is requested, both column *names* and 状態 cell
values are translated. State values are mapped via ``schema.STATE_JA_TO_EN``
(e.g. ``充電休止`` → ``charge_rest``) so EN-mode consumers receive a fully
EN frame. Unknown JA-string state literals raise ``ValueError`` regardless
of ``column_lang`` — same strictness as the unknown-numeric-state branch.

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

import logging
import math
import os
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from echemplot.io.schema import (
    CANONICAL_COLUMNS_JA,
    COL_CAPACITY,
    COL_CURRENT_MA,
    COL_ELAPSED_S,
    JA_TO_EN,
    STATE_CODE_TO_JA,
    STATE_JA_TO_EN,
    ColumnLang,
)

logger = logging.getLogger(__name__)


class EncodingError(ValueError):
    """Raised when a TOYO file cannot be decoded with the configured encoding.

    Subclasses :class:`ValueError` so callers that already catch
    ``ValueError`` keep working. The message includes the file path, the
    encoding being attempted, the byte position of the first bad sequence
    (from :class:`UnicodeDecodeError`), and a hint for overriding the
    encoding via :func:`read_cell_dir`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_encoding: str,
        original: UnicodeDecodeError,
    ) -> None:
        self.path = Path(path)
        self.expected_encoding = expected_encoding
        self.original = original
        msg = (
            f"failed to decode {self.path} with encoding={expected_encoding!r}: "
            f"invalid byte(s) at position {original.start}-{original.end} "
            f"({original.reason}); "
            f"pass `encoding=<other>` to `read_cell_dir` to override "
            f"(default 'shift_jis')."
        )
        super().__init__(msg)


class RawConcatError(ValueError):
    """Raised when a constituent 6-digit raw file fails row-continuity validation.

    A single ``000NNN`` raw file is expected to be internally consistent before
    being concatenated with its siblings — specifically, ``経過時間[Sec]`` must
    be monotone-non-decreasing within each run of identical ``状態`` values
    (state transitions reset elapsed_time). A negative diff inside a single
    state segment indicates the file was truncated mid-cycle and resumed,
    silently joining two distinct runs into one DataFrame.

    Attributes
    ----------
    file_path : Path
        The offending 6-digit raw file.
    segment_index : int | None
        0-based index of the state-run segment that failed the check, counting
        from the top of the file. ``None`` for whole-file failures (e.g. an
        empty file).
    reason : str
        Human-readable explanation of the failure mode.
    """

    def __init__(
        self,
        file_path: Path,
        segment_index: int | None,
        reason: str,
    ) -> None:
        self.file_path = file_path
        self.segment_index = segment_index
        self.reason = reason
        seg_str = "n/a" if segment_index is None else str(segment_index)
        super().__init__(
            f"raw file {file_path.name} failed row-continuity validation "
            f"(segment_index={seg_str}): {reason}"
        )


RAW_FILENAME_RE = re.compile(r"[0-9]{6}")
PTN_SUFFIX = ".ptn"  # matched case-insensitively (Linux CI is case-sensitive)
RENZOKU_DATA = "連続データ.csv"
RENZOKU_DATA_PY = "連続データ_py.csv"

# Number of leading lines to scan when looking for the column-header row in
# 連続データ.csv. Real TOYO firmware ships with the header at line index 3
# (3 metadata rows above it); the scan window is sized to absorb a small
# amount of future drift (extra metadata rows, optional summary lines)
# while still failing fast on a totally unexpected file.
_HEADER_SCAN_ROWS = 20

# Legacy positional layout used as a fallback when content-based detection
# cannot find the header row in the first ``_HEADER_SCAN_ROWS`` lines.
_LEGACY_HEADER_ROW = 3
_LEGACY_SKIPROWS = [4, 5, 6]

# First-cell literals identifying the header row of 連続データ.csv. The
# native export uses the JP literal; the EN form is only present if a user
# has manually pre-renamed columns, but is cheap to also accept here.
_HEADER_FIRST_CELL_CANDIDATES: tuple[str, ...] = ("サイクル", "cycle")

# Patterns that mark a row immediately following the header as a unit /
# channel-id / separator metadata row rather than a data row. We strip the
# header-following rows greedily until the first row whose first cell parses
# as a numeric cycle index (a plausible data row).
_UNIT_ROW_FIRST_CELL_PATTERNS = (
    re.compile(r"^\[.*\]$"),  # e.g. "[V]", "[mAh/g]"
    re.compile(r"^-+$"),  # e.g. "-" separator row
    re.compile(r"^\d+\s*ch$", re.IGNORECASE),  # channel id e.g. "1ch"
)

# Half-width → full-width canonical rename. The raw 6-digit header row uses
# half-width katakana for the cycle/mode columns. ``総ｻｲｸﾙ`` is the global
# cycle counter (does not reset across mode boundaries); chdis prefers it as
# the cycle key when present so multi-mode programs (e.g. formation in mode 1
# followed by regular cycling in mode 2) do not collapse two physically
# distinct cycles into a single chdis_df group.
_RAW_TO_CANONICAL = {
    "ｻｲｸﾙ": "サイクル",
    "総ｻｲｸﾙ": "総サイクル",
    "ﾓｰﾄﾞ": "モード",
    "電圧[V]": "電圧",
}

# Metadata key used by the native 連続データ.csv to carry active-material mass.
_METADATA_MASS_KEY = "重量[mg]"

# Number of metadata rows to scan when looking for 重量[mg]. The real file
# has exactly 3 metadata rows before the column-header row (index 3), so
# a scan of the first 4 rows is generous.
_METADATA_SCAN_ROWS = 4

# Fixed-column layout for the PTN mass field on line 0. The TOYO PTN format
# always places a 9-character ``<flag><mass>`` composite at character offset
# 44 (after a 42-char operator field + the literal ``"2 "`` electrode-count
# prefix). The 9-char composite is one of two known dialects:
#
# * "concat" (cyclers No5 / No1):  ``flag(1) + mass(%.6f, 8 chars)``
#                                  e.g. ``"00.000358"`` — flag at index 0,
#                                  mass at index 1..8.
# * "spaced" (cycler No6):         ``flag(1) + " "(1) + mass(%.5f, 7 chars)``
#                                  e.g. ``"0 0.00116"`` — mass at index 2..8.
#
# Detection: examine the byte at index 1 of the 9-char composite. ``" "`` →
# spaced dialect; otherwise concat. ``tests/conftest.py:write_ptn_main`` is
# the synthetic-fixture truth source; real-data validation against No1 / No5
# / No6 cyclers is recorded in PR #90 (issue #90). Character indexing is safe
# even when the operator field contains multi-byte JP names because the file
# is decoded Shift-JIS first and ``str.ljust`` pads in characters.
_PTN_MASS_FIELD_START = 44
_PTN_MASS_FIELD_LEN = 9
_PTN_MASS_FIELD_END = _PTN_MASS_FIELD_START + _PTN_MASS_FIELD_LEN  # 53

# Escape hatch: when this env var is set to a truthy value the legacy
# ``TOYO_Origin_2.01`` heuristic (``str.split()`` + ``token[2] / token[3]``
# fallback) is used instead of the fixed-column parser. Provided for users
# who hit a third PTN dialect in the wild — file an issue if you need this.
_PTN_LEGACY_ENV_VAR = "ECHEMPLOT_PTN_LEGACY"


def _is_legacy_ptn_mode() -> bool:
    val = os.environ.get(_PTN_LEGACY_ENV_VAR, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _parse_ptn_mass_fixed_column(first_line: str, path: Path) -> float:
    """Parse the 9-char fixed-column mass field on line 0.

    Detects the dialect by inspecting index 1 of the 9-char composite. Raises
    ``ValueError`` on a short line, a non-numeric mass, or a non-positive
    parsed value — :func:`_resolve_mass_from_ptn` relies on this to skip
    auxiliary PTN files (e.g. ``*_OPTION.PTN``).
    """
    if len(first_line) < _PTN_MASS_FIELD_END:
        raise ValueError(
            f"{path} line 0 is too short for the fixed-column PTN format "
            f"(need {_PTN_MASS_FIELD_END} chars, got {len(first_line)}); "
            f"set {_PTN_LEGACY_ENV_VAR}=1 to fall back to the legacy "
            "whitespace-split parser if your file uses a different layout."
        )
    composite = first_line[_PTN_MASS_FIELD_START:_PTN_MASS_FIELD_END]
    if composite[1] == " ":
        dialect = "spaced"
        mass_str = composite[2:].strip()
    else:
        dialect = "concat"
        mass_str = composite[1:].strip()
    try:
        mass = float(mass_str)
    except ValueError as e:
        raise ValueError(
            f"{path} line 0 fixed-column mass field {composite!r} (dialect={dialect!r}) "
            f"is not a valid float; "
            f"set {_PTN_LEGACY_ENV_VAR}=1 to fall back to the legacy parser if needed."
        ) from e
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError(
            f"{path} line 0 fixed-column mass field {composite!r} (dialect={dialect!r}) "
            f"parsed as non-positive value {mass!r}"
        )
    return mass


def _parse_ptn_mass_legacy(first_line: str, path: Path) -> float:
    """Legacy ``TOYO_Origin_2.01`` whitespace-split heuristic.

    Take ``tokens[2]``; if it parses as exactly zero, fall back to
    ``tokens[3]``. Preserved for the ``ECHEMPLOT_PTN_LEGACY=1`` escape hatch
    and for the V2.01 parity tests under ``tests/legacy_v201/``.
    """
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


def read_ptn_mass(ptn_path: str | Path) -> float:
    """Extract active-material mass (grams) from a ``.PTN`` file.

    Uses a **fixed-column parser** by default: the TOYO PTN format places a
    9-char ``<flag><mass>`` composite at character offset 44 of line 0, in
    one of two known dialects:

    * "concat" (cyclers No5 / No1): flag glued to a ``%.6f`` mass —
      ``"00.000358"``.
    * "spaced" (cycler No6): flag, space, then a ``%.5f`` mass —
      ``"0 0.00116"``.

    Dialect is auto-detected by the byte at index 1 of the composite (a
    space implies spaced; otherwise concat). Both decode to the same numeric
    mass.

    The previous :data:`TOYO_Origin_2.01`-derived whitespace-split heuristic
    (``tokens[2]``, falling back to ``tokens[3]`` when ``tokens[2]==0``) is
    available behind the ``ECHEMPLOT_PTN_LEGACY=1`` environment variable as
    an escape hatch for users who hit a third dialect in the wild.

    Auxiliary ``.PTN`` files that TOYO ships alongside the main one
    (``*_OPTION.PTN``, ``*_Option2.PTN``, etc.) carry INI- or CSV-style
    configuration whose first line is too short for the fixed-column layout
    and whose tokens don't form a positive mass; calling this on them will
    raise ``ValueError``, which :func:`_resolve_mass_from_ptn` relies on to
    skip them.
    """
    path = Path(ptn_path)
    encoding = "shift_jis"
    try:
        with path.open(encoding=encoding) as f:
            first_line = f.readline()
    except UnicodeDecodeError as e:
        raise EncodingError(path, expected_encoding=encoding, original=e) from e
    if _is_legacy_ptn_mode():
        return _parse_ptn_mass_legacy(first_line, path)
    return _parse_ptn_mass_fixed_column(first_line, path)


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
        日付, 時刻, 総サイクル — half-width ``総ｻｲｸﾙ`` from the raw 6-digit
        header is canonicalized to the full-width spelling).
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
    header_row, skiprows = _detect_renzoku_header(path, encoding)
    df = pd.read_csv(path, header=header_row, skiprows=skiprows, encoding=encoding)
    df = _clean_columns(df)
    df = _ensure_capacity(df, mass)
    return _finalize(df, column_lang)


def _detect_renzoku_header(path: Path, encoding: str) -> tuple[int, list[int]]:
    """Find the column-header row of 連続データ.csv by content.

    Reads the first :data:`_HEADER_SCAN_ROWS` lines and locates the row
    whose first non-empty cell matches one of
    :data:`_HEADER_FIRST_CELL_CANDIDATES` (``サイクル`` for the native JP
    export; ``cycle`` accepted defensively for hand-renamed sources).
    Returns ``(header_row, skiprows)`` where ``skiprows`` lists the
    contiguous unit / channel-id / separator rows that immediately follow
    the header and should be skipped before the first data row.

    On detection failure (no candidate row in the scan window) emits a
    ``logger.warning`` and falls back to the historical fixed layout
    ``(3, [4, 5, 6])``, preserving behaviour on every TOYO file shipped
    so far while leaving the failure visible in the log stream.
    """
    with path.open(encoding=encoding, errors="replace") as f:
        lines = [f.readline() for _ in range(_HEADER_SCAN_ROWS)]
    rows: list[list[str]] = [
        [cell.strip() for cell in line.rstrip("\r\n").split(",")] for line in lines if line
    ]

    header_row: int | None = None
    for i, fields in enumerate(rows):
        first = next((c for c in fields if c), "")
        # Strip a possible BOM the same way ``_clean_columns`` does, so an
        # Excel-saved file whose first cell starts with U+FEFF still matches.
        first = first.replace("﻿", "")
        if first in _HEADER_FIRST_CELL_CANDIDATES:
            header_row = i
            break

    if header_row is None:
        logger.warning(
            "could not detect header row in %s, falling back to legacy positional skip",
            path,
        )
        return _LEGACY_HEADER_ROW, list(_LEGACY_SKIPROWS)

    skiprows: list[int] = []
    for j in range(header_row + 1, len(rows)):
        first = next((c for c in rows[j] if c), "")
        if _looks_like_unit_row(first):
            skiprows.append(j)
            continue
        # First plausible data row: stop greedy unit-row consumption.
        break
    return header_row, skiprows


def _looks_like_unit_row(first_cell: str) -> bool:
    """Return ``True`` if ``first_cell`` looks like a unit / separator row.

    A row whose first non-empty cell parses as a number is considered a
    data row and is *not* skipped. Anything matching the unit / channel-id
    / separator patterns in :data:`_UNIT_ROW_FIRST_CELL_PATTERNS` is.
    """
    if not first_cell:
        # An entirely blank row immediately after the header is treated as
        # a separator and skipped.
        return True
    try:
        float(first_cell)
    except ValueError:
        pass
    else:
        return False
    return any(p.match(first_cell) for p in _UNIT_ROW_FIRST_CELL_PATTERNS)


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
    # Row-continuity validation: each constituent file must be internally
    # consistent before concat. A truncated-and-resumed file would otherwise
    # silently splice two unrelated runs into a single state segment. See
    # _validate_raw_frame_continuity for the precise rules.
    for f, frame in zip(raw_files, frames):
        _validate_raw_frame_continuity(f, frame)
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = _clean_columns(df)
    df = _drop_unnamed(df)
    df = _ensure_capacity(df, mass)
    return _finalize(df, column_lang), mass


def _validate_raw_frame_continuity(file_path: Path, frame: pd.DataFrame) -> None:
    """Validate one raw 6-digit frame's internal row continuity.

    Checks performed:

    * Frame is non-empty. A 6-digit file with zero data rows is suspicious
      (the on-disk format always carries at least the summary marker plus
      header, but the data section can be empty if the cycler crashed
      before any sample landed). Raises :class:`RawConcatError` with
      ``segment_index=None``.
    * ``経過時間[Sec]`` is monotone-non-decreasing within each contiguous
      run of identical ``状態`` values. State transitions reset elapsed
      time in the TOYO raw format, so we check per-segment, not globally.
      Run-length segments are computed via a state-change cumsum.

    The elapsed-time column is *optional* in the TOYO 6-digit dialect —
    some older firmware revisions emit files without it. In that case we
    skip the continuity check (with a ``logger.debug`` notice) rather
    than raise; the column-equality check upstream still guarantees that
    every constituent file in a given concat shares the same schema, so
    the missing column is consistent across siblings.

    Edge case: if ``状態`` itself is missing we cannot RLE the segments,
    so we treat the entire frame as one segment for the diff check.
    """
    if len(frame) == 0:
        raise RawConcatError(
            file_path,
            None,
            "frame has 0 data rows after header skip; file is empty or truncated before "
            "any sample landed",
        )

    # The reader runs continuity validation pre-_clean_columns, so at this
    # point ``frame`` carries the source-literal column names. The elapsed
    # column is the canonical JP form ``経過時間[Sec]``; the state column
    # is ``状態``.
    if COL_ELAPSED_S not in frame.columns:
        logger.debug(
            "raw file %s: %s column missing; skipping per-segment continuity check",
            file_path.name,
            COL_ELAPSED_S,
        )
        return

    elapsed = pd.to_numeric(frame[COL_ELAPSED_S], errors="coerce").to_numpy()
    if "状態" in frame.columns:
        # Run-length encode: a new segment starts wherever 状態 changes
        # (using fillna-aware comparison so two consecutive NaN states
        # are treated as a single segment, matching np.diff semantics on
        # the elapsed column itself).
        state_series = frame["状態"]
        # shift().ne(state_series) marks each row where the state differs
        # from its predecessor; cumsum turns that into a 0,1,2,... segment id.
        # We use ne(...) | (both NaN) — pandas' default ne treats NaN!=NaN, so
        # we compensate by also accepting matching NaNs as same-segment.
        prev = state_series.shift()
        changed = state_series.ne(prev) & ~(state_series.isna() & prev.isna())
        # The very first row's "changed" is True by definition (no
        # predecessor); cumsum gives segment ids starting at 1.
        segment_ids = changed.cumsum().to_numpy()
    else:
        segment_ids = np.ones(len(frame), dtype=np.int64)

    unique_segments = np.unique(segment_ids)
    for seg_idx, seg_id in enumerate(unique_segments):
        mask = segment_ids == seg_id
        seg_elapsed = elapsed[mask]
        if seg_elapsed.size < 2:
            continue
        diffs = np.diff(seg_elapsed)
        # NaN diffs are not negative — treat them as inconclusive and skip.
        # Any strictly-negative diff inside a single state segment is the
        # truncation/resume signature.
        bad = diffs < 0
        if bool(np.any(bad)) and not bool(np.all(np.isnan(diffs[bad]))):
            first_bad = int(np.argmax(bad))
            raise RawConcatError(
                file_path,
                seg_idx,
                f"{COL_ELAPSED_S} decreased within state segment "
                f"(prev={seg_elapsed[first_bad]!r}, "
                f"next={seg_elapsed[first_bad + 1]!r}); file likely truncated "
                "and resumed mid-segment",
            )


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip BOM (which Excel-saved files sometimes prepend) before whitespace,
    # so a column like "\ufeffｻｲｸﾙ" still maps via _RAW_TO_CANONICAL.
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return cast("pd.DataFrame", df.rename(columns=_RAW_TO_CANONICAL))


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns pandas auto-labels ``Unnamed: N``.

    Raw TOYO 6-digit files have 6-7 empty separator columns between the
    sensor block (``経過時間[Sec]/電圧[V]/電流[mA]``) and the per-row
    metadata block (``状態/ﾓｰﾄﾞ/ｻｲｸﾙ/総ｻｲｸﾙ``, which ``_clean_columns``
    later canonicalizes to ``状態/モード/サイクル/総サイクル``). pandas names them
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


def _drop_trailing_sentinel_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Drop a contiguous tail block of *unknown-code* sentinel rows.

    Defensive backstop for any future TOYO firmware that emits an
    end-of-test sentinel with a state code we have not yet catalogued
    in :data:`STATE_CODE_TO_JA`. A row is treated as a sentinel iff its
    状態 is numeric and unknown AND ``経過時間[Sec] == 0`` AND
    ``電流[mA] == 0``. State code ``9`` (``中断``) is *not* a sentinel
    here — it is now a known state and is mapped to its JA label like
    any other.

    The scan is strictly trailing/contiguous; an unknown-code row in
    the middle of the file (or with non-zero flow) is *not* dropped —
    those still surface as the ``unknown 状態 codes`` error so
    genuinely surprising data is not silently swallowed.
    """
    if not pd.api.types.is_numeric_dtype(df["状態"]):
        return df, []
    if COL_ELAPSED_S not in df.columns or COL_CURRENT_MA not in df.columns:
        return df, []
    drop_idx: list[int] = []
    for i in range(len(df) - 1, -1, -1):
        state = df["状態"].iat[i]
        if pd.isna(state) or int(state) in STATE_CODE_TO_JA:
            break
        if df[COL_ELAPSED_S].iat[i] == 0 and df[COL_CURRENT_MA].iat[i] == 0:
            drop_idx.append(int(df.index[i]))
        else:
            break
    if not drop_idx:
        return df, []
    return cast("pd.DataFrame", df.drop(index=drop_idx).reset_index(drop=True)), drop_idx


def _finalize(df: pd.DataFrame, column_lang: ColumnLang) -> pd.DataFrame:
    missing = [c for c in CANONICAL_COLUMNS_JA if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns after read: {missing}")
    extras = [c for c in df.columns if c not in CANONICAL_COLUMNS_JA]
    out = df.loc[:, [*CANONICAL_COLUMNS_JA, *extras]].copy()
    out, dropped = _drop_trailing_sentinel_rows(out)
    if dropped:
        logger.debug(
            "Dropped %d trailing TOYO sentinel row(s) at indices %s",
            len(dropped),
            dropped,
        )
    # Step 1: numeric → JA. Both branches (numeric source and already-JA
    # string source) converge on a JA-string 状態 column before validation.
    if pd.api.types.is_numeric_dtype(out["状態"]):
        mapped = out["状態"].map(STATE_CODE_TO_JA)
        unmapped_mask = mapped.isna() & out["状態"].notna()
        if unmapped_mask.any():
            bad = sorted(out.loc[unmapped_mask, "状態"].unique().tolist())
            first_bad_idx = int(out.index[unmapped_mask][0])
            raise ValueError(
                f"unknown 状態 codes in source: {bad} "
                f"(known codes: {sorted(STATE_CODE_TO_JA)}); "
                f"first seen at row {first_bad_idx}. If this is a TOYO "
                "sentinel code, please file an issue at "
                "https://github.com/tomooki/toyo-battery/issues "
                "with a sample file."
            )
        out["状態"] = mapped
    # Step 2: validate JA-string 状態 strictly. The numeric branch is
    # already exhaustive (mapped values are by definition in
    # STATE_CODE_TO_JA's value set, all of which are STATE_JA_TO_EN keys);
    # this guard primarily catches unknown literals from native
    # 連続データ.csv / 連続データ_py.csv sources that were never mapped
    # through STATE_CODE_TO_JA.
    state_notna = out["状態"].notna()
    unknown_mask = state_notna & ~out["状態"].isin(STATE_JA_TO_EN)
    if unknown_mask.any():
        bad_labels = sorted(out.loc[unknown_mask, "状態"].unique().tolist())
        first_bad_idx = int(out.index[unknown_mask][0])
        raise ValueError(
            f"unknown 状態 labels in source: {bad_labels} "
            f"(known labels: {sorted(STATE_JA_TO_EN)}); "
            f"first seen at row {first_bad_idx}. If this is a TOYO "
            "label we have not yet catalogued, please file an issue at "
            "https://github.com/tomooki/toyo-battery/issues "
            "with a sample file."
        )
    out = out.reset_index(drop=True)
    # Step 3: JA → EN translation when requested. Column names always
    # translate; state values translate so EN-mode callers receive a fully
    # EN frame. NaN rows are preserved by ``Series.map`` (na_action='ignore'
    # default in modern pandas, but we pass it explicitly via the dict
    # which leaves missing values untouched).
    if column_lang == "en":
        out["状態"] = out["状態"].map(lambda v: STATE_JA_TO_EN[v] if pd.notna(v) else v)
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
    try:
        with path.open(encoding=encoding) as f:
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
    except UnicodeDecodeError as e:
        raise EncodingError(path, expected_encoding=encoding, original=e) from e
    return None


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else value
