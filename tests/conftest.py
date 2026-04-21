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
    ) -> Path:
        d = tmp_path / name
        d.mkdir()
        if layout == "renzoku":
            _write_renzoku(d, mass_g=mass)
        elif layout == "renzoku_py":
            _write_renzoku_py(d)
        elif layout == "raw_6digit":
            _write_raw_6digit(d, mass_g=mass)
        else:
            raise ValueError(f"unknown layout: {layout}")
        return d

    return _make


def _write_renzoku(cell_dir: Path, *, mass_g: float) -> None:
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


def _write_raw_6digit(cell_dir: Path, *, mass_g: float) -> None:
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

    # Main PTN: fixed-column. ``split()[2]`` extracts the mass in grams.
    ptn_main = (
        f" 1Synthetic                                 2 {mass_g:09.6f}       1{mass_g:09.6f}"
        f"TestCell                                 24 00000"
    )
    (cell_dir / "pattern.PTN").write_text(ptn_main + "\n", encoding="shift_jis")

    # Companion PTN: INI-style header, no mass. `read_ptn_mass` raises on this
    # and the reader falls through to the main PTN.
    (cell_dir / "pattern_OPTION.PTN").write_text(
        "[BaseCellCapacity]\nCapacity=0.1\n", encoding="shift_jis"
    )
