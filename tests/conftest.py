"""Shared fixtures for the test suite.

`make_cell_dir` builds a synthetic TOYO-style cell directory in tmp_path for
each of the three supported on-disk layouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import pytest

Layout = Literal["renzoku", "renzoku_py", "raw_6digit"]

# Used by the renzoku fixture when include_capacity=True. Picked to be
# distinguishable from any plausible recompute output (formula would give
# values like 0, 500, 0, -527.78, -1027.78 for the row sequence below).
SENTINEL_CAPACITY_BASE: float = 100.0


@pytest.fixture
def make_cell_dir(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes a synthetic cell directory."""

    def _make(
        layout: Layout,
        *,
        name: str = "cell_A",
        mass: float = 0.002,  # 2 mg
        include_capacity_col: bool = True,
    ) -> Path:
        d = tmp_path / name
        d.mkdir()
        if layout == "renzoku":
            _write_renzoku(d, mass=mass, include_capacity=include_capacity_col)
        elif layout == "renzoku_py":
            _write_renzoku_py(d)
        elif layout == "raw_6digit":
            _write_raw_6digit(d, mass=mass)
        else:
            raise ValueError(f"unknown layout: {layout}")
        return d

    return _make


def _write_renzoku(cell_dir: Path, *, mass: float, include_capacity: bool) -> None:
    """Write a 連続データ.csv with header=3, skiprows=[4,5,6]."""
    lines = [
        "# metadata line 0",
        "# metadata line 1",
        "# metadata line 2",
    ]
    if include_capacity:
        lines.append("ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA],電気量")
    else:
        lines.append("ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]")
    lines.extend(["# skip 4", "# skip 5", "# skip 6"])

    rows = [
        (1, "CC", 1, 3.50, 0.0, 1.0),
        (1, "CC", 1, 3.60, 3600.0, 1.0),
        (1, "CC", 0, 3.61, 3700.0, 0.0),
        (1, "CC", 2, 3.40, 3800.0, -1.0),
        (1, "CC", 2, 3.20, 7400.0, -1.0),
    ]
    for idx, (cycle, mode, state, voltage, t_sec, current_ma) in enumerate(rows):
        fields = [
            str(cycle),
            mode,
            str(state),
            f"{voltage:.3f}",
            f"{t_sec:.1f}",
            f"{current_ma:.3f}",
        ]
        if include_capacity:
            # Distinct sentinel per row, deliberately NOT matching the recompute
            # formula t/3600·I/mass. A regression where the reader recomputes
            # an already-present 電気量 column would be visible as the sentinel
            # being overwritten.
            sentinel_capacity = SENTINEL_CAPACITY_BASE + idx
            fields.append(f"{sentinel_capacity:.6f}")
        lines.append(",".join(fields))
    (cell_dir / "連続データ.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")


def _write_renzoku_py(cell_dir: Path) -> None:
    """Write an already-normalized 連続データ_py.csv (header=0)."""
    lines = [
        "サイクル,モード,状態,電圧,電気量",
        "1,CC,充電,3.50,0.000",
        "1,CC,充電,3.60,500.000",
        "1,CC,休止,3.61,500.000",
        "1,CC,放電,3.40,0.000",
        "1,CC,放電,3.20,-500.000",
    ]
    (cell_dir / "連続データ_py.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")


def _write_raw_6digit(cell_dir: Path, *, mass: float) -> None:
    """Write one 6-digit raw file (header=1) plus a .PTN mass file."""
    raw_lines = [
        "# summary line 0",
        "ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]",
        "1,CC,1,3.50,0.0,1.0",
        "1,CC,1,3.60,3600.0,1.0",
        "1,CC,0,3.61,3700.0,0.0",
        "1,CC,2,3.40,3800.0,-1.0",
        "1,CC,2,3.20,7400.0,-1.0",
    ]
    (cell_dir / "000001").write_text("\n".join(raw_lines) + "\n", encoding="shift_jis")
    (cell_dir / "pattern.PTN").write_text(f"ACTIVE_MATERIAL WEIGHT {mass}\n", encoding="shift_jis")
