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


def _write_fixed_column_ptn(path: Path, mass_g: float) -> None:
    """Write a TOYO-style fixed-column PTN whose first line has the mass at token[2]."""
    line = (
        f" 1TestName                                 2 {mass_g:09.6f}       1{mass_g:09.6f}"
        f"TestCell                                 24 00000"
    )
    path.write_text(line + "\n", encoding="shift_jis")


# ---- renzoku (native 連続データ.csv) path ----------------------------------


def test_read_renzoku_uses_precomputed_capacity(make_cell_dir: Callable[..., Path]) -> None:
    """連続データ.csv carries 電気量 inline. Reader must pass it through, not recompute."""
    d = make_cell_dir("renzoku", mass=0.001)
    df, mass_g = read_cell_dir(d)
    assert list(df.columns) == list(CANONICAL_COLUMNS_JA)  # no extras in native CSV
    assert len(df) == 5
    # Fixture capacity values (per-segment monotone non-decreasing, as in real data)
    assert df["電気量"].tolist() == pytest.approx([0.0, 1000.0, 0.0, 0.0, 1000.0])
    # Mass comes from the 重量[mg] metadata row (1.0 mg → 0.001 g)
    assert mass_g == pytest.approx(0.001)


def test_read_renzoku_exposes_four_state_substates(make_cell_dir: Callable[..., Path]) -> None:
    """Real 連続データ.csv distinguishes 充電休止 / 放電休止 from each other."""
    d = make_cell_dir("renzoku", mass=0.001)
    df, _ = read_cell_dir(d)
    # Row after charge (before discharge) is 充電休止, not just 休止
    assert df["状態"].tolist() == ["充電", "充電", "充電休止", "放電", "放電"]


def test_read_renzoku_metadata_mass_is_in_milligrams(make_cell_dir: Callable[..., Path]) -> None:
    """``重量[mg]`` in metadata is milligrams; reader must convert to grams."""
    d = make_cell_dir("renzoku", mass=0.002)  # 2 mg
    _, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.002)


def test_explicit_mass_overrides_renzoku_metadata(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("renzoku", mass=0.001)
    _, mass_g = read_cell_dir(d, mass=0.005)
    assert mass_g == pytest.approx(0.005)


def test_read_renzoku_falls_back_to_ptn_when_metadata_missing(tmp_path: Path) -> None:
    """If 重量[mg] is not in metadata, reader falls back to a .PTN file."""
    d = tmp_path / "cell"
    d.mkdir()
    # Write a renzoku CSV with metadata rows that do NOT carry 重量[mg].
    lines = [
        ",試験名,synthetic,,,,開始日時,-",
        ",測定備考,",
        ",備考2,-",
        "サイクル,モード,状態,電圧,電気量",
        "1ch,1ch,1ch,1ch,1ch",
        "-,-,-,-,-",
        "[],[],[],[V],[mAh/g]",
        "1,1,充電,3.5000,0.000000",
        "1,1,放電,3.4000,10.000000",
    ]
    (d / "連続データ.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(d / "pattern.PTN", 0.004)
    _, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.004)


# ---- renzoku_py (normalized output) path ----------------------------------


def test_read_renzoku_py(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("renzoku_py")
    df, mass_g = read_cell_dir(d)
    assert list(df.columns) == list(CANONICAL_COLUMNS_JA)
    assert math.isnan(mass_g)  # no mass source available
    assert df["状態"].tolist() == ["充電", "充電", "休止", "放電", "放電"]


def test_renzoku_py_honors_ptn_mass(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("renzoku_py")
    _write_fixed_column_ptn(d / "pattern.PTN", 0.005)
    _, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.005)


def test_renzoku_py_with_two_valid_ptn_masses_raises(make_cell_dir: Callable[..., Path]) -> None:
    """Two .PTN files both yielding parseable masses must force disambiguation."""
    d = make_cell_dir("renzoku_py")
    _write_fixed_column_ptn(d / "p1.PTN", 0.005)
    _write_fixed_column_ptn(d / "p2.PTN", 0.003)
    with pytest.raises(ValueError, match=r"multiple \.PTN"):
        read_cell_dir(d)


def test_lowercase_ptn_extension_is_discovered(make_cell_dir: Callable[..., Path]) -> None:
    """Linux CI is case-sensitive; `.ptn` (lowercase) must still resolve mass."""
    d = make_cell_dir("renzoku_py")
    _write_fixed_column_ptn(d / "pattern.ptn", 0.007)
    _, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.007)


# ---- 6-digit raw path -----------------------------------------------------


def test_read_raw_6digit_computes_per_segment_positive_capacity(
    make_cell_dir: Callable[..., Path],
) -> None:
    """Real raw convention: 経過時間 resets per segment, 電流 is unsigned.

    The formula ``elapsed/3600 * current / mass`` therefore produces
    per-segment monotone-non-decreasing 電気量.
    """
    d = make_cell_dir("raw_6digit", mass=0.001)
    df, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.001)
    assert list(df.columns[: len(CANONICAL_COLUMNS_JA)]) == list(CANONICAL_COLUMNS_JA)
    # Extras preserved (date, time, elapsed, current, total cycle)
    assert "経過時間[Sec]" in df.columns
    assert "電流[mA]" in df.columns
    # No 'Unnamed: *' noise from pandas' handling of empty separator columns
    assert not any(str(c).startswith("Unnamed:") for c in df.columns)
    # Per-segment positive: [0, 1000, 0, 0, 1000] for this fixture
    assert df["電気量"].tolist() == pytest.approx([0.0, 1000.0, 0.0, 0.0, 1000.0])
    # State codes mapped to simple 3-value set (no substates in raw format)
    assert df["状態"].tolist() == ["充電", "充電", "休止", "放電", "放電"]


def test_raw_6digit_main_ptn_picked_when_option_ptn_present(
    make_cell_dir: Callable[..., Path],
) -> None:
    """conftest ships both ``pattern.PTN`` and ``pattern_OPTION.PTN``.

    The option file starts with ``[BaseCellCapacity]`` which `read_ptn_mass`
    cannot parse, so `_resolve_mass_from_ptn` skips it and picks the main.
    """
    d = make_cell_dir("raw_6digit", mass=0.0008)
    _, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.0008)
    # Sanity: the OPTION file really is there, so the test is meaningful
    assert (d / "pattern_OPTION.PTN").exists()


def test_raw_6digit_multiple_valid_ptn_masses_raises(
    make_cell_dir: Callable[..., Path],
) -> None:
    d = make_cell_dir("raw_6digit", mass=0.001)
    _write_fixed_column_ptn(d / "pattern2.PTN", 0.003)
    with pytest.raises(ValueError, match=r"multiple \.PTN"):
        read_cell_dir(d)


def test_raw_6digit_explicit_mass_overrides_ptn(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.001)
    df, mass_g = read_cell_dir(d, mass=0.004)
    assert mass_g == pytest.approx(0.004)
    # Row at elapsed=3600, I=1mA, mass=0.004 → capacity = 1*1/0.004 = 250 mAh/g
    assert df.loc[1, "電気量"] == pytest.approx(250.0)


def test_raw_6digit_without_mass_and_no_ptn(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.001)
    (d / "pattern.PTN").unlink()
    (d / "pattern_OPTION.PTN").unlink()
    with pytest.raises(ValueError, match="mass"):
        read_cell_dir(d)


def test_read_raw_6digit_multi_file(tmp_path: Path) -> None:
    """Multiple 000NNN files are concatenated in sorted order."""
    cell_dir = tmp_path / "multi"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    file_a = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.5500,1.000000{empty_sep},1, 1,  1,     1",
    ]
    file_b = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,01:00:00,00000000,+3.4000,1.000000{empty_sep},2, 1,  2,     2",
        f"2025/01/01,01:30:00,00001800,+3.2000,1.000000{empty_sep},2, 1,  2,     2",
    ]
    (cell_dir / "000001").write_text("\n".join(file_a) + "\n", encoding="shift_jis")
    (cell_dir / "000002").write_text("\n".join(file_b) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert len(df) == 4
    assert df["サイクル"].tolist() == [1, 1, 2, 2]


def test_read_raw_6digit_mismatched_columns_raises(tmp_path: Path) -> None:
    cell_dir = tmp_path / "mismatch"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header_full = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    header_short = "日付,時刻,経過時間[Sec],電圧[V]"
    file_a = [
        "0,0,0,0,0,0,0",
        "",
        header_full,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
    ]
    file_b = [
        "0,0,0,0,0,0,0",
        "",
        header_short,
        "2025/01/01,00:00:00,00000000,+3.5000",
    ]
    (cell_dir / "000001").write_text("\n".join(file_a) + "\n", encoding="shift_jis")
    (cell_dir / "000002").write_text("\n".join(file_b) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    with pytest.raises(ValueError, match="columns differing"):
        read_cell_dir(cell_dir)


def test_unknown_state_code_raises(tmp_path: Path) -> None:
    """An unrecognized integer 状態 code must not silently pass through."""
    cell_dir = tmp_path / "bad_state"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.6000,1.000000{empty_sep},9, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    with pytest.raises(ValueError, match="unknown 状態 codes"):
        read_cell_dir(cell_dir)


def test_nan_state_is_preserved(tmp_path: Path) -> None:
    """A row with empty 状態 should survive — NaN preserved, not raised."""
    cell_dir = tmp_path / "nan_state"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        # Middle row: empty state field (no character between commas, not a
        # space). pandas reads this as NaN; see also the non-space column
        # alignment required to keep the numeric dtype of the whole column.
        f"2025/01/01,00:15:00,00000900,+3.5500,1.000000{empty_sep},, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.6000,1.000000{empty_sep},1, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert df["状態"].iloc[0] == "充電"
    assert pd.isna(df["状態"].iloc[1])
    assert df["状態"].iloc[2] == "充電"


# ---- discovery and file-system plumbing -----------------------------------


def test_ptn_in_subdir_is_ignored(make_cell_dir: Callable[..., Path]) -> None:
    """A stale .PTN in a subdirectory must not silently determine mass."""
    d = make_cell_dir("renzoku_py")  # no top-level mass source
    sub = d / "backup"
    sub.mkdir()
    _write_fixed_column_ptn(sub / "stale.PTN", 0.999)
    _, mass_g = read_cell_dir(d)
    # No top-level .PTN, no renzoku metadata → mass_g is NaN (not 0.999)
    assert math.isnan(mass_g)


def test_6digit_in_subdir_is_ignored(tmp_path: Path) -> None:
    """A stale 6-digit raw file in a subdirectory must not be auto-discovered."""
    cell_dir = tmp_path / "with_backup"
    cell_dir.mkdir()
    sub = cell_dir / "old_run"
    sub.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    (sub / "999999").write_text(
        "\n".join(
            [
                "0,0,0,0,0,0,0",
                "",
                header,
                f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
            ]
        )
        + "\n",
        encoding="shift_jis",
    )
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    with pytest.raises(FileNotFoundError, match="no TOYO data"):
        read_cell_dir(cell_dir)


def test_path_is_a_file_raises_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir.csv"
    f.write_text("x", encoding="shift_jis")
    with pytest.raises(NotADirectoryError):
        read_cell_dir(f)


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_cell_dir("/nonexistent/cell/dir")


def test_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty_cell"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no TOYO data"):
        read_cell_dir(empty)


# ---- read_ptn_mass unit tests --------------------------------------------


def test_read_ptn_mass_happy(tmp_path: Path) -> None:
    ptn = tmp_path / "x.PTN"
    _write_fixed_column_ptn(ptn, 1.234)
    assert read_ptn_mass(ptn) == pytest.approx(1.234)


def test_read_ptn_mass_simple_whitespace_format(tmp_path: Path) -> None:
    """Legacy tests used to write ``ACTIVE_MATERIAL WEIGHT <float>``; that
    format also works because ``split()[2]`` still picks the float."""
    ptn = tmp_path / "x.PTN"
    ptn.write_text("ACTIVE_MATERIAL WEIGHT 1.234\n", encoding="shift_jis")
    assert read_ptn_mass(ptn) == pytest.approx(1.234)


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


def test_read_ptn_mass_ini_style_raises(tmp_path: Path) -> None:
    """An `[BaseCellCapacity]` INI-header PTN (TOYO OPTION files) is unparseable.

    `_resolve_mass_from_ptn` relies on this raising to skip companion files.
    """
    ptn = tmp_path / "OPTION.PTN"
    ptn.write_text("[BaseCellCapacity]\nCapacity=0.1\n", encoding="shift_jis")
    with pytest.raises(ValueError):
        read_ptn_mass(ptn)


# ---- column_lang translation ---------------------------------------------


def test_column_lang_en(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.001)
    df, _ = read_cell_dir(d, column_lang="en")
    assert list(df.columns[: len(CANONICAL_COLUMNS_EN)]) == list(CANONICAL_COLUMNS_EN)
    # extras translated via schema.JA_TO_EN
    assert "elapsed_time_s" in df.columns
    assert "current_ma" in df.columns
    # state values are NOT translated here — that is schema.STATE_JA_TO_EN's job
    # when needed by downstream consumers
    assert set(df["state"].unique()) <= {"休止", "充電", "放電"}


# ---- Cell.from_dir wiring -----------------------------------------------


@pytest.mark.parametrize("layout", ["renzoku", "renzoku_py", "raw_6digit"])
def test_cell_from_dir_wires_reader(make_cell_dir: Callable[..., Path], layout: str) -> None:
    d = make_cell_dir(layout, mass=0.001, name="mycell")
    cell = Cell.from_dir(d)
    assert cell.name == "mycell"
    assert isinstance(cell.raw_df, pd.DataFrame)
    assert len(cell.raw_df) == 5
    # renzoku carries mass inline (metadata); raw_6digit has main PTN; _py has nothing
    if layout == "renzoku_py":
        assert math.isnan(cell.mass_g)
    else:
        assert cell.mass_g == pytest.approx(0.001)
