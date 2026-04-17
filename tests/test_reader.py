"""Tests for toyo_battery.io.reader."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from toyo_battery.core.cell import Cell
from toyo_battery.io.reader import read_cell_dir, read_ptn_mass
from toyo_battery.io.schema import CANONICAL_COLUMNS_EN, CANONICAL_COLUMNS_JA

# Mirrored from conftest.SENTINEL_CAPACITY_BASE. We do not `from conftest import`
# because pytest does not add the conftest directory to sys.path (the project
# uses no tests/__init__.py and no pythonpath= config). Change both if you
# change either.
SENTINEL_CAPACITY_BASE: float = 100.0


def test_read_renzoku_with_capacity_column(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("renzoku", mass=0.002, include_capacity_col=True)
    df, mass_g = read_cell_dir(d)
    assert list(df.columns[: len(CANONICAL_COLUMNS_JA)]) == list(CANONICAL_COLUMNS_JA)
    # extras (経過時間[Sec], 電流[mA]) preserved after the canonical block
    assert "経過時間[Sec]" in df.columns
    assert "電流[mA]" in df.columns
    assert len(df) == 5
    # value-level state mapping: rows are written as integer codes 1,1,0,2,2
    assert df["状態"].tolist() == ["充電", "充電", "休止", "放電", "放電"]
    # precomputed sentinel survives — reader must NOT recompute when 電気量 is present
    expected_sentinels = [SENTINEL_CAPACITY_BASE + i for i in range(5)]
    assert df["電気量"].tolist() == pytest.approx(expected_sentinels)
    assert math.isnan(mass_g)  # no .PTN for this layout


def test_read_renzoku_without_capacity_requires_mass(
    make_cell_dir: Callable[..., Path],
) -> None:
    d = make_cell_dir("renzoku", include_capacity_col=False)
    with pytest.raises(ValueError, match="電気量"):
        read_cell_dir(d)


def test_read_renzoku_without_capacity_computes_with_mass(
    make_cell_dir: Callable[..., Path],
) -> None:
    d = make_cell_dir("renzoku", mass=0.002, include_capacity_col=False)
    df, mass_g = read_cell_dir(d, mass=0.002)
    assert mass_g == 0.002
    # Row at t=3600s, I=1mA, mass=0.002g → capacity = 1*1/0.002 = 500 mAh/g
    charge_row = df[(df["電圧"] == 3.60) & (df["状態"] == "充電")].iloc[0]
    assert charge_row["電気量"] == pytest.approx(500.0)


def test_read_renzoku_py(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("renzoku_py")
    df, mass_g = read_cell_dir(d)
    assert list(df.columns) == list(CANONICAL_COLUMNS_JA)
    assert math.isnan(mass_g)
    assert df["状態"].iloc[0] == "充電"


def test_read_raw_6digit_with_ptn(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.002)
    df, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.002)
    assert list(df.columns[: len(CANONICAL_COLUMNS_JA)]) == list(CANONICAL_COLUMNS_JA)
    assert set(df["状態"].unique()) <= {"休止", "充電", "放電"}
    # Same capacity formula as above
    assert df.loc[df["電圧"] == 3.60, "電気量"].iloc[0] == pytest.approx(500.0)


def test_multiple_ptn_files_raises(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.002)
    (d / "pattern2.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.003\n", encoding="shift_jis")
    with pytest.raises(ValueError, match=r"multiple \.PTN"):
        read_cell_dir(d)


def test_multiple_ptn_files_disambiguated_by_explicit_mass(
    make_cell_dir: Callable[..., Path],
) -> None:
    d = make_cell_dir("raw_6digit", mass=0.002)
    (d / "pattern2.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.003\n", encoding="shift_jis")
    df, mass_g = read_cell_dir(d, mass=0.002)
    assert mass_g == 0.002
    assert df.loc[df["電圧"] == 3.60, "電気量"].iloc[0] == pytest.approx(500.0)


def test_ptn_in_subdir_is_ignored(make_cell_dir: Callable[..., Path]) -> None:
    """A stale .PTN in a subdirectory must not silently determine mass."""
    d = make_cell_dir("renzoku", mass=0.002, include_capacity_col=False)
    sub = d / "backup"
    sub.mkdir()
    (sub / "stale.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.999\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="電気量"):
        read_cell_dir(d)


def test_unknown_state_code_raises(tmp_path: Path) -> None:
    """An unrecognized integer 状態 code must not silently pass through."""
    cell_dir = tmp_path / "bad_state"
    cell_dir.mkdir()
    raw = [
        "# summary line 0",
        "ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]",
        "1,CC,1,3.50,0.0,1.0",
        "1,CC,9,3.60,3600.0,1.0",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    (cell_dir / "pattern.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.002\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="unknown 状態 codes"):
        read_cell_dir(cell_dir)


def test_nan_state_is_preserved(tmp_path: Path) -> None:
    """A row with empty 状態 should survive — NaN preserved, not raised."""
    cell_dir = tmp_path / "nan_state"
    cell_dir.mkdir()
    raw = [
        "# summary line 0",
        "ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]",
        "1,CC,1,3.50,0.0,1.0",
        "1,CC,,3.55,1800.0,1.0",
        "1,CC,1,3.60,3600.0,1.0",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    (cell_dir / "pattern.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.002\n", encoding="shift_jis")
    df, _ = read_cell_dir(cell_dir)
    assert df["状態"].iloc[0] == "充電"
    assert pd.isna(df["状態"].iloc[1])
    assert df["状態"].iloc[2] == "充電"


def test_renzoku_py_honors_ptn_mass(make_cell_dir: Callable[..., Path]) -> None:
    """The _py branch must also resolve mass from a top-level .PTN, like the native branch."""
    d = make_cell_dir("renzoku_py")
    (d / "pattern.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.005\n", encoding="shift_jis")
    _, mass_g = read_cell_dir(d)
    assert mass_g == 0.005


def test_renzoku_py_with_multiple_ptn_raises(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("renzoku_py")
    (d / "p1.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.005\n", encoding="shift_jis")
    (d / "p2.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.003\n", encoding="shift_jis")
    with pytest.raises(ValueError, match=r"multiple \.PTN"):
        read_cell_dir(d)


def test_lowercase_ptn_extension_is_discovered(make_cell_dir: Callable[..., Path]) -> None:
    """Linux CI is case-sensitive; `.ptn` (lowercase) must still resolve mass."""
    d = make_cell_dir("renzoku_py")
    (d / "pattern.ptn").write_text("ACTIVE_MATERIAL WEIGHT 0.007\n", encoding="shift_jis")
    _, mass_g = read_cell_dir(d)
    assert mass_g == 0.007


def test_path_is_a_file_raises_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir.csv"
    f.write_text("x", encoding="shift_jis")
    with pytest.raises(NotADirectoryError):
        read_cell_dir(f)


def test_6digit_in_subdir_is_ignored(tmp_path: Path) -> None:
    """A stale 6-digit raw file in a subdirectory must not be auto-discovered."""
    cell_dir = tmp_path / "with_backup"
    cell_dir.mkdir()
    sub = cell_dir / "old_run"
    sub.mkdir()
    (sub / "999999").write_text(
        "# summary\nｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]\n1,CC,1,3.50,0.0,1.0\n",
        encoding="shift_jis",
    )
    (cell_dir / "pattern.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.002\n", encoding="shift_jis")
    with pytest.raises(FileNotFoundError, match="no TOYO data"):
        read_cell_dir(cell_dir)


def test_read_raw_6digit_multi_file(tmp_path: Path) -> None:
    cell_dir = tmp_path / "multi"
    cell_dir.mkdir()
    header = "ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]"
    file_a = ["# summary line 0", header, "1,CC,1,3.50,0.0,1.0", "1,CC,1,3.55,1800.0,1.0"]
    file_b = ["# summary line 0", header, "2,CC,2,3.40,3600.0,-1.0", "2,CC,2,3.20,5400.0,-1.0"]
    (cell_dir / "000001").write_text("\n".join(file_a) + "\n", encoding="shift_jis")
    (cell_dir / "000002").write_text("\n".join(file_b) + "\n", encoding="shift_jis")
    (cell_dir / "pattern.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.002\n", encoding="shift_jis")
    df, _ = read_cell_dir(cell_dir)
    assert len(df) == 4
    assert df["サイクル"].tolist() == [1, 1, 2, 2]


def test_read_raw_6digit_mismatched_columns_raises(tmp_path: Path) -> None:
    cell_dir = tmp_path / "mismatch"
    cell_dir.mkdir()
    file_a = [
        "# summary line 0",
        "ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V],経過時間[Sec],電流[mA]",
        "1,CC,1,3.50,0.0,1.0",
    ]
    file_b = [
        "# summary line 0",
        "ｻｲｸﾙ,ﾓｰﾄﾞ,状態,電圧[V]",
        "1,CC,1,3.50",
    ]
    (cell_dir / "000001").write_text("\n".join(file_a) + "\n", encoding="shift_jis")
    (cell_dir / "000002").write_text("\n".join(file_b) + "\n", encoding="shift_jis")
    (cell_dir / "pattern.PTN").write_text("ACTIVE_MATERIAL WEIGHT 0.002\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="columns differing"):
        read_cell_dir(cell_dir)


def test_read_raw_6digit_without_mass_and_no_ptn(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    d = make_cell_dir("raw_6digit", mass=0.002)
    (d / "pattern.PTN").unlink()
    with pytest.raises(ValueError, match="mass"):
        read_cell_dir(d)


def test_read_raw_6digit_with_explicit_mass_overrides_ptn(
    make_cell_dir: Callable[..., Path],
) -> None:
    d = make_cell_dir("raw_6digit", mass=0.002)
    df, mass_g = read_cell_dir(d, mass=0.004)
    assert mass_g == 0.004
    # capacity halves when mass doubles
    assert df.loc[df["電圧"] == 3.60, "電気量"].iloc[0] == pytest.approx(250.0)


def test_column_lang_en(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.002)
    df, _ = read_cell_dir(d, column_lang="en")
    assert list(df.columns[: len(CANONICAL_COLUMNS_EN)]) == list(CANONICAL_COLUMNS_EN)
    # extras translated via schema.JA_TO_EN
    assert "elapsed_time_s" in df.columns
    assert "current_ma" in df.columns
    # state values are NOT translated here — that is schema.STATE_JA_TO_EN's job
    # when needed by downstream consumers
    assert set(df["state"].unique()) <= {"休止", "充電", "放電"}


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_cell_dir("/nonexistent/cell/dir")


def test_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty_cell"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no TOYO data"):
        read_cell_dir(empty)


def test_read_ptn_mass_happy(tmp_path: Path) -> None:
    ptn = tmp_path / "x.PTN"
    ptn.write_text("ACTIVE_MATERIAL WEIGHT 1.234\n", encoding="shift_jis")
    assert read_ptn_mass(ptn) == 1.234


def test_read_ptn_mass_malformed_raises(tmp_path: Path) -> None:
    ptn = tmp_path / "x.PTN"
    ptn.write_text("too short\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="fewer than 3 tokens"):
        read_ptn_mass(ptn)


def test_read_ptn_mass_non_numeric_raises(tmp_path: Path) -> None:
    ptn = tmp_path / "x.PTN"
    ptn.write_text("A B NOT_A_NUMBER\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="not a valid float"):
        read_ptn_mass(ptn)


@pytest.mark.parametrize("layout", ["renzoku", "renzoku_py", "raw_6digit"])
def test_cell_from_dir_wires_reader(make_cell_dir: Callable[..., Path], layout: str) -> None:
    d = make_cell_dir(layout, mass=0.002, name="mycell")
    cell = Cell.from_dir(d)
    assert cell.name == "mycell"
    assert isinstance(cell.raw_df, pd.DataFrame)
    assert len(cell.raw_df) == 5
    # only raw_6digit ships a .PTN by default; the renzoku layouts return NaN
    if layout == "raw_6digit":
        assert cell.mass_g == pytest.approx(0.002)
    else:
        assert math.isnan(cell.mass_g)
