"""Shared fixtures for the test suite.

``make_cell_dir`` builds a synthetic TOYO-style cell directory in ``tmp_path``
for each of the three supported on-disk layouts. Fixture layouts mirror what
real TOYO testers emit (column order, state value convention, mass
location) so tests actually exercise the same codepaths as real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import pytest

Layout = Literal["renzoku", "renzoku_py", "raw_6digit"]
PtnDialect = Literal["concat", "spaced"]

# Row template shared between layouts so the three fixtures stay in sync.
# Each tuple: (cycle, mode, state_int, voltage, elapsed_s, current_mA,
# capacity_mAh_per_g). state_int uses the raw-6-digit convention (0=rest,
# 1=charge, 2=discharge); the renzoku fixture translates these to JP state
# strings including the 充電休止/放電休止 substate distinction.
#
# 経過時間 resets at each state transition and 電流 is unsigned (both match
# real TOYO raw files), so capacity = elapsed/3600 * current / mass_g gives
# the per-segment monotone-non-decreasing values in the 7th tuple slot for
# the default mass of 1 mg (0.001 g).
_DEFAULT_MASS_G: float = 0.001  # 1 mg
_ROWS_SHARED: tuple[tuple[int, str, int, float, float, float, float], ...] = (
    (1, "1", 1, 3.50, 0.0, 1.0, 0.0),  # charge start
    (1, "1", 1, 3.60, 3600.0, 1.0, 1000.0),  # charge end
    (1, "1", 0, 3.61, 0.0, 0.0, 0.0),  # rest (elapsed resets at transition)
    (1, "1", 2, 3.40, 0.0, 1.0, 0.0),  # discharge start (elapsed resets)
    (1, "1", 2, 3.20, 3600.0, 1.0, 1000.0),  # discharge end
)

_STATE_INT_TO_RENZOKU_JA: dict[int, tuple[str, str]] = {
    # state_int -> (start-of-segment JP string, continuation JP string).
    # In real 連続データ.csv the tester relabels the 0=rest state as either
    # 充電休止 (rest after a charge) or 放電休止 (rest after a discharge).
    # This helper just maps by integer; the caller tracks which preceded.
    1: ("充電", "充電"),
    2: ("放電", "放電"),
}


@pytest.fixture
def make_cell_dir(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes a synthetic cell directory."""

    def _make(
        layout: Layout,
        *,
        name: str = "cell_A",
        mass: float = _DEFAULT_MASS_G,
        ptn_dialect: PtnDialect = "concat",
        n_metadata_rows: int = 3,
    ) -> Path:
        d = tmp_path / name
        d.mkdir()
        if layout == "renzoku":
            _write_renzoku(d, mass_g=mass, n_metadata_rows=n_metadata_rows)
        elif layout == "renzoku_py":
            _write_renzoku_py(d)
        elif layout == "raw_6digit":
            _write_raw_6digit(d, mass_g=mass, ptn_dialect=ptn_dialect)
        else:
            raise ValueError(f"unknown layout: {layout}")
        return d

    return _make


def _build_renzoku_metadata(n_metadata_rows: int, mass_mg: float) -> list[str]:
    """Build the metadata block for the synthetic 連続データ.csv.

    The mass row is always included (so the reader can resolve mass without
    needing a .PTN); extra rows are filler ``,メモ<i>,`` lines that look
    like the genuine product's free-form remarks. ``n_metadata_rows=0``
    yields an empty block — the header lands at row 0.
    """
    if n_metadata_rows < 0:
        raise ValueError(f"n_metadata_rows must be >= 0, got {n_metadata_rows}")
    base = [
        ",試験名,C:¥synthetic¥test¥path,,,,開始日時,2026-01-01 00:00:00",
        ",測定備考,",
        f",重量[mg],{mass_mg:.3f}",
    ]
    if n_metadata_rows <= len(base):
        return base[:n_metadata_rows]
    extras = [f",メモ{i},comment-{i}" for i in range(n_metadata_rows - len(base))]
    return [*base, *extras]


def _write_renzoku(
    cell_dir: Path,
    *,
    mass_g: float,
    n_metadata_rows: int = 3,
) -> None:
    """Write a native-format 連続データ.csv.

    Real format:
      - Row 0: test-name + directory + start-time metadata
      - Row 1: 測定備考 (remarks)
      - Row 2: ``,重量[mg],<float>`` — mass source
      - Row 3: header ``サイクル,モード,状態,電圧,電気量`` (full-width JP)
      - Row 4: channel identifier (e.g. ``72ch,72ch,...``)
      - Row 5: ``-`` separator row
      - Row 6: units (``[],[],[],[V],[mAh/g]``)
      - Row 7+: data rows with state as JP strings including
        ``充電休止``/``放電休止`` substates and pre-computed 電気量.

    The ``n_metadata_rows`` knob lets tests exercise hypothetical TOYO
    firmware layouts that drift from the historical 3-row preamble. The
    mass row stays inside the metadata block (so the reader's metadata
    scan still finds it); extra padding rows are inserted before/after as
    needed.
    """
    mass_mg = mass_g * 1e3
    metadata_lines = _build_renzoku_metadata(n_metadata_rows, mass_mg)
    lines = [
        *metadata_lines,
        "サイクル,モード,状態,電圧,電気量",
        "1ch,1ch,1ch,1ch,1ch",
        "-,-,-,-,-",
        "[],[],[],[V],[mAh/g]",
    ]
    # State-string bookkeeping: translate 0 into either 充電休止 or 放電休止
    # based on which non-rest state preceded it.
    last_active_state = 1
    for cycle, mode, state_int, voltage, _elapsed, _current, capacity in _ROWS_SHARED:
        if state_int == 0:
            state_ja = "充電休止" if last_active_state == 1 else "放電休止"
        else:
            state_ja = _STATE_INT_TO_RENZOKU_JA[state_int][0]
            last_active_state = state_int
        lines.append(f"{cycle},{mode},{state_ja},{voltage:.4f},{capacity:.6f}")
    (cell_dir / "連続データ.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")


def _write_renzoku_py(cell_dir: Path) -> None:
    """Write an already-normalized 連続データ_py.csv (header=0).

    The legacy v2.01 pipeline writes this file with the simpler 3-state
    convention (state ∈ {充電, 放電, 休止}) — no substate distinction.
    """
    state_int_to_simple_ja = {0: "休止", 1: "充電", 2: "放電"}
    lines = ["サイクル,モード,状態,電圧,電気量"]
    for cycle, mode, state_int, voltage, _elapsed, _current, capacity in _ROWS_SHARED:
        state_ja = state_int_to_simple_ja[state_int]
        lines.append(f"{cycle},{mode},{state_ja},{voltage:.4f},{capacity:.6f}")
    (cell_dir / "連続データ_py.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")


def _write_raw_6digit(
    cell_dir: Path,
    *,
    mass_g: float,
    ptn_dialect: PtnDialect = "concat",
) -> None:
    """Write a real-format 6-digit raw file plus ``.PTN`` mass files.

    Real format:
      - Line 0: ``0,0,0,0,0,0,0`` (summary marker, not data)
      - Lines 1-2: blank (pandas ``skip_blank_lines=True`` collapses these)
      - Line 3: header ``日付,時刻,経過時間[Sec],電圧[V],電流[mA],<6 empty>,
        状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ``
      - Line 4+: data rows; 経過時間 resets per state transition, 電流 is
        an unsigned magnitude.

    Also writes *two* ``.PTN`` files: the main pattern with the mass, and a
    ``*_OPTION.PTN`` companion mirroring what real TOYO dirs ship. The
    reader must pick the main one automatically (the OPTION file's first
    line starts with ``[`` so ``read_ptn_mass`` raises and it is skipped).
    """
    empty_sep = ",,,,,,"  # 6 empty columns
    lines = [
        "0,0,0,0,0,0,0",
        "",
        "",
        f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ",
    ]
    date = "2026/01/01"
    time = "00:00:00"
    total_cycle = 1
    for cycle, mode, state_int, voltage, elapsed, current, _capacity in _ROWS_SHARED:
        lines.append(
            f"{date},{time},{int(elapsed):08d},+{voltage:.4f},{current:.6f}{empty_sep}"
            f",{state_int:d}, {mode},  {cycle:d},     {total_cycle:d}"
        )
    (cell_dir / "000001").write_text("\n".join(lines) + "\n", encoding="shift_jis")

    write_ptn_main(cell_dir / "pattern.PTN", mass_g=mass_g, dialect=ptn_dialect)

    # Companion PTN: INI-style header, no mass. `read_ptn_mass` raises on this
    # and the reader falls through to the main PTN.
    (cell_dir / "pattern_OPTION.PTN").write_text(
        "[BaseCellCapacity]\nCapacity=0.1\n", encoding="shift_jis"
    )


def write_renzoku_with_states(
    cell_dir: Path,
    *,
    rows: list[tuple[int, str, str, float, float]],
    mass_g: float = _DEFAULT_MASS_G,
) -> None:
    """Write a native ``連続データ.csv`` with caller-supplied state literals.

    Mirrors :func:`_write_renzoku`'s 7-row header but lets each test pin
    the exact 状態 string per row — useful for exercising substate
    coverage (``充電休止``/``放電休止``) and unknown-label rejection in
    :func:`echemplot.io.reader._finalize`.

    Parameters
    ----------
    rows
        ``[(cycle, mode, state_ja, voltage, capacity), ...]``.
    """
    mass_mg = mass_g * 1e3
    lines = [
        ",試験名,C:¥synthetic¥test¥path,,,,開始日時,2026-01-01 00:00:00",
        ",測定備考,",
        f",重量[mg],{mass_mg:.3f}",
        "サイクル,モード,状態,電圧,電気量",
        "1ch,1ch,1ch,1ch,1ch",
        "-,-,-,-,-",
        "[],[],[],[V],[mAh/g]",
    ]
    for cycle, mode, state_ja, voltage, capacity in rows:
        lines.append(f"{cycle},{mode},{state_ja},{voltage:.4f},{capacity:.6f}")
    (cell_dir / "連続データ.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")


def write_ptn_main(
    path: Path,
    *,
    mass_g: float,
    dialect: PtnDialect = "concat",
    operator: str = "Synthetic",
    sample: str = "TestCell",
) -> None:
    """Write a single main ``.PTN`` first-line in the requested TOYO dialect.

    ``concat`` is the newer dialect (cyclers No5/No1) where the per-electrode
    flag is glued to the mass: ``"00.001000"``. ``spaced`` is the older
    dialect (cycler No6) where flag and mass are space-separated and the
    mass uses ``%.5f``: ``"0 0.00100"``. Both occupy a 9-char composite
    field and the surrounding fixed-width layout is otherwise identical.
    """
    operator_field = f" 1{operator}".ljust(42)
    if dialect == "concat":
        field1 = f"0{mass_g:.6f}".rjust(9)
        field2 = f"1{mass_g:.6f}".rjust(9)
    elif dialect == "spaced":
        field1 = f"0 {mass_g:.5f}".rjust(9)
        field2 = f"1 {mass_g:.5f}".rjust(9)
    else:
        raise ValueError(f"unknown dialect: {dialect}")
    line0 = f"{operator_field}2 {field1}       {field2}{sample}"
    path.write_text(line0 + "\n", encoding="shift_jis")
