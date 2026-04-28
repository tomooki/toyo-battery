"""Tests for echemplot.io.reader."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from echemplot.core.cell import Cell
from echemplot.io import EncodingError as EncodingErrorReexport
from echemplot.io import RawConcatError as RawConcatErrorReexport
from echemplot.io.reader import (
    EncodingError,
    RawConcatError,
    _extract_mass_from_renzoku_metadata,
    read_cell_dir,
    read_ptn_mass,
)
from echemplot.io.schema import CANONICAL_COLUMNS_EN, CANONICAL_COLUMNS_JA


def _write_fixed_column_ptn(path: Path, mass_g: float) -> None:
    """Write a TOYO-style fixed-column PTN whose first line has the mass at token[2].

    Layout matches ``conftest.write_ptn_main`` (concat dialect): a 42-char
    operator/electrode field, the literal ``"2 "`` electrode-count prefix,
    a 9-char ``<flag><mass>`` composite at columns 44..52, 7 spaces, then a
    9-char companion electrode block.
    """
    operator_field = " 1TestName".ljust(42)
    field1 = f"0{mass_g:.6f}".rjust(9)
    field2 = f"1{mass_g:.6f}".rjust(9)
    line = f"{operator_field}2 {field1}       {field2}TestCell"
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


# ---- renzoku header detection (issue #101) -------------------------------


def test_read_renzoku_data_detects_header_at_default_position(
    make_cell_dir: Callable[..., Path],
) -> None:
    """Default fixture (3 metadata rows, header at row 3) parses cleanly.

    Sanity-anchors the existing-behaviour-unchanged contract from issue
    #101: content-based detection must produce the same output as the
    historical ``header=3, skiprows=[4,5,6]`` positional skip on every
    file the reader has ever supported.
    """
    d = make_cell_dir("renzoku", mass=0.001)
    df, mass_g = read_cell_dir(d)
    assert list(df.columns) == list(CANONICAL_COLUMNS_JA)
    assert len(df) == 5
    assert df["電気量"].tolist() == pytest.approx([0.0, 1000.0, 0.0, 0.0, 1000.0])
    assert mass_g == pytest.approx(0.001)


def test_read_renzoku_data_detects_header_with_extra_metadata_row(
    make_cell_dir: Callable[..., Path],
) -> None:
    """A future TOYO firmware shipping 4 metadata rows (header at row 4)
    must still be parsed correctly thanks to content-based detection."""
    d = make_cell_dir("renzoku", mass=0.001, n_metadata_rows=4)
    df, mass_g = read_cell_dir(d)
    assert list(df.columns) == list(CANONICAL_COLUMNS_JA)
    assert len(df) == 5
    assert df["電気量"].tolist() == pytest.approx([0.0, 1000.0, 0.0, 0.0, 1000.0])
    # Mass row is still inside the metadata-scan window (row index 2).
    assert mass_g == pytest.approx(0.001)


def test_read_renzoku_data_detects_header_with_fewer_metadata_rows(
    make_cell_dir: Callable[..., Path],
) -> None:
    """A hypothetical layout with only 2 metadata rows (header at row 2)
    must also parse. The 重量[mg] row is dropped from this fixture, so
    mass falls through to NaN — the parse itself is what we assert."""
    d = make_cell_dir("renzoku", mass=0.001, n_metadata_rows=2)
    df, _ = read_cell_dir(d)
    assert list(df.columns) == list(CANONICAL_COLUMNS_JA)
    assert len(df) == 5
    assert df["電気量"].tolist() == pytest.approx([0.0, 1000.0, 0.0, 0.0, 1000.0])


def test_read_renzoku_data_falls_back_with_warning_on_no_detection(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No row in the first 20 lines starts with サイクル → reader emits
    a logger.warning and falls back to the legacy positional skip.

    The fallback layout (``header=3, skiprows=[4,5,6]``) is unlikely to
    parse the synthetic worst-case file into a canonical schema, so we
    expect a downstream ``ValueError`` from ``_finalize`` after the
    warning fires. The contract under test is: warning emitted +
    fallback attempted, not "fallback succeeds" — the latter would
    require a file whose first 20 lines bury the header but whose
    line-3 happens to look canonical, which is contrived."""
    cell_dir = tmp_path / "no_header"
    cell_dir.mkdir()
    # 25 lines of random metadata-looking junk, no サイクル anywhere.
    lines = [f",ﾒﾓ{i},junk-row-{i}" for i in range(25)]
    (cell_dir / "連続データ.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")
    with (
        caplog.at_level(logging.WARNING, logger="echemplot.io.reader"),
        pytest.raises(ValueError),
    ):
        read_cell_dir(cell_dir)
    assert any(
        "could not detect header row" in rec.message and rec.name == "echemplot.io.reader"
        for rec in caplog.records
    )


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


def test_read_raw_6digit_canonicalizes_total_cycle_column(tmp_path: Path) -> None:
    """Raw 6-digit ``総ｻｲｸﾙ`` (half-width) is canonicalized to ``総サイクル``.

    chdis.py prefers the global cycle counter as the cycle key; the reader
    has to surface it under the full-width canonical spelling so chdis can
    look it up without knowing the source dialect.
    """
    cell_dir = tmp_path / "total_cycle"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    rows = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.5500,1.000000{empty_sep},1, 1,  1,     2",
    ]
    (cell_dir / "000001").write_text("\n".join(rows) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert "総サイクル" in df.columns
    assert "総ｻｲｸﾙ" not in df.columns
    assert df["総サイクル"].tolist() == [1, 2]
    # EN-mode rename: 総サイクル → total_cycle via JA_TO_EN.
    df_en, _ = read_cell_dir(cell_dir, column_lang="en")
    assert "total_cycle" in df_en.columns
    assert df_en["total_cycle"].tolist() == [1, 2]


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


# ---- 6-digit raw row-continuity validation (#97) -------------------------


def test_raw_concat_error_re_exported_from_io_package() -> None:
    """`RawConcatError` is part of the public ``echemplot.io`` surface."""
    assert RawConcatErrorReexport is RawConcatError
    assert issubclass(RawConcatError, ValueError)


def test_read_raw_6digit_concat_normal_two_segment_pass(tmp_path: Path) -> None:
    """Two consistent 6-digit files concatenate without raising.

    File A: a charge segment with monotone-increasing 経過時間.
    File B: a discharge segment with monotone-increasing 経過時間.
    State transitions occur cleanly at file boundaries; no per-segment
    backwards diff anywhere; row-continuity validation must pass.
    """
    from tests.conftest import write_ptn_main, write_raw_6digit_file

    cell_dir = tmp_path / "two_seg"
    cell_dir.mkdir()
    write_raw_6digit_file(
        cell_dir / "000001",
        [
            (1, "1", 1, 3.50, 0.0, 1.0),
            (1, "1", 1, 3.55, 1800.0, 1.0),
            (1, "1", 1, 3.60, 3600.0, 1.0),
        ],
    )
    write_raw_6digit_file(
        cell_dir / "000002",
        [
            (1, "1", 2, 3.40, 0.0, 1.0),
            (1, "1", 2, 3.30, 1800.0, 1.0),
            (1, "1", 2, 3.20, 3600.0, 1.0),
        ],
    )
    write_ptn_main(cell_dir / "pattern.PTN", mass_g=0.001)
    df, mass_g = read_cell_dir(cell_dir)
    assert mass_g == pytest.approx(0.001)
    assert len(df) == 6
    # Two charge rows from each file's last sample land at indices 2 and 5.
    assert df["状態"].tolist() == ["充電", "充電", "充電", "放電", "放電", "放電"]
    # Per-segment monotone capacity (3 rows each, last hits 1000 mAh/g).
    assert df["電気量"].tolist() == pytest.approx([0.0, 500.0, 1000.0, 0.0, 500.0, 1000.0])


def test_read_raw_6digit_raises_on_truncated_segment(tmp_path: Path) -> None:
    """An elapsed-time backstep within a state segment surfaces as RawConcatError.

    Simulates a tester that crashed mid-charge and resumed: the second
    "row 1" of the charge segment sees ``経過時間`` go from 1800 back to
    300 — the truncation/resume signature this validation is designed
    to catch. The error message must include the segment index.
    """
    from tests.conftest import write_ptn_main, write_raw_6digit_file

    cell_dir = tmp_path / "truncated"
    cell_dir.mkdir()
    write_raw_6digit_file(
        cell_dir / "000001",
        [
            (1, "1", 1, 3.50, 0.0, 1.0),
            (1, "1", 1, 3.55, 1800.0, 1.0),
            # Backstep: tester crashed and resumed mid-charge.
            (1, "1", 1, 3.52, 300.0, 1.0),
            (1, "1", 1, 3.60, 3600.0, 1.0),
        ],
    )
    write_ptn_main(cell_dir / "pattern.PTN", mass_g=0.001)
    with pytest.raises(RawConcatError) as exc_info:
        read_cell_dir(cell_dir)
    err = exc_info.value
    assert err.file_path.name == "000001"
    assert err.segment_index == 0  # The first (and only) state segment.
    assert "segment_index=0" in str(err)
    assert "経過時間[Sec]" in str(err)


def test_read_raw_6digit_skips_validation_when_elapsed_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Files lacking 経過時間[Sec] read successfully with a debug notice.

    Some older TOYO firmware revisions emit raw files without the
    elapsed-time column. The continuity check must be skipped (not
    raise) for that schema, and the skip must be observable at debug
    level so a user investigating downstream weirdness can find it.

    Also verifies that the absence of the elapsed column does not
    prevent ``_ensure_capacity`` from raising: the fixture has no
    pre-computed 電気量 either, so we expect the existing
    "missing columns" ValueError. The test asserts that we reach that
    point — i.e. continuity validation did NOT short-circuit with a
    RawConcatError first.
    """
    from tests.conftest import write_ptn_main, write_raw_6digit_file

    cell_dir = tmp_path / "no_elapsed"
    cell_dir.mkdir()
    write_raw_6digit_file(
        cell_dir / "000001",
        [
            (1, "1", 1, 3.50, 0.0, 1.0),
            (1, "1", 1, 3.60, 0.0, 1.0),
        ],
        include_elapsed=False,
    )
    write_ptn_main(cell_dir / "pattern.PTN", mass_g=0.001)
    caplog.set_level(logging.DEBUG, logger="echemplot.io.reader")
    # Without 経過時間 the capacity column cannot be derived, so the
    # downstream _ensure_capacity raises. We catch that to assert we
    # got past validation; the validation step is the focus of this test.
    with pytest.raises(ValueError, match="cannot compute"):
        read_cell_dir(cell_dir)
    assert any(
        "skipping per-segment continuity check" in rec.getMessage() for rec in caplog.records
    )
    # Negative assertion: no RawConcatError was raised on the way to that ValueError.
    assert not any(
        rec.exc_info and isinstance(rec.exc_info[1], RawConcatError) for rec in caplog.records
    )


def test_read_raw_6digit_raises_on_empty_data_section(tmp_path: Path) -> None:
    """A 6-digit file with header but zero data rows raises RawConcatError.

    Covers the ``segment_index=None`` branch — the empty-frame check
    fires before per-segment RLE could even be computed.
    """
    from tests.conftest import write_ptn_main

    cell_dir = tmp_path / "empty_data"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    (cell_dir / "000001").write_text(
        "\n".join(["0,0,0,0,0,0,0", "", "", header]) + "\n",
        encoding="shift_jis",
    )
    write_ptn_main(cell_dir / "pattern.PTN", mass_g=0.001)
    with pytest.raises(RawConcatError) as exc_info:
        read_cell_dir(cell_dir)
    err = exc_info.value
    assert err.file_path.name == "000001"
    assert err.segment_index is None
    assert "segment_index=n/a" in str(err)


def test_unknown_state_code_raises(tmp_path: Path) -> None:
    """An unrecognized integer 状態 code with non-zero flow must raise.

    Code ``9`` is now a known state (``中断``); use ``8`` to exercise
    the unknown-code branch. The sentinel-drop guard in
    :func:`_finalize` only swallows trailing rows whose 経過時間 *and*
    電流 are both zero; this fixture has non-zero values, so it must
    still surface the unknown code.
    """
    cell_dir = tmp_path / "bad_state"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.6000,1.000000{empty_sep},8, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    with pytest.raises(ValueError, match="unknown 状態 codes"):
        read_cell_dir(cell_dir)


def test_unknown_state_code_error_message_includes_row_index(tmp_path: Path) -> None:
    """The unknown-code ValueError must include the offending row index."""
    cell_dir = tmp_path / "bad_state_msg"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.6000,1.000000{empty_sep},8, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    with pytest.raises(ValueError, match=r"first seen at row \d+"):
        read_cell_dir(cell_dir)


def test_raw_6digit_trailing_state_9_preserved_as_中断(tmp_path: Path) -> None:
    """State code 9 maps to ``中断`` and is preserved as a real row.

    Mirrors the No1/No6-style end-of-test marker: state=9 with zero
    elapsed and zero current at the file tail.
    """
    cell_dir = tmp_path / "sentinel"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.6000,1.000000{empty_sep},1, 1,  1,     1",
        # Trailing TOYO end-of-test row: state=9, elapsed=0, current=0
        f"2025/01/01,00:30:00,00000000,+3.6000,0.000000{empty_sep},9, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert len(df) == 3
    assert df["状態"].tolist() == ["充電", "充電", "中断"]


def test_raw_6digit_trailing_state_9_with_nonzero_flow_preserved(
    tmp_path: Path,
) -> None:
    """Reproduces the No2 case (Issue #91 follow-up): trailing state=9
    with **non-zero** elapsed/current. v0.1.7 raised here; v0.1.8 must
    map the row to ``中断`` and preserve it."""
    cell_dir = tmp_path / "trailing_nonzero"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        # Trailing state=9 with non-zero current — exactly the No2 No.13
        # cell pattern that triggered the v0.1.7 regression.
        f"2025/01/01,00:30:00,00000900,+3.6000,0.500000{empty_sep},9, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert len(df) == 2
    assert df["状態"].tolist() == ["充電", "中断"]


def test_raw_6digit_state_9_midfile_preserved_as_中断(tmp_path: Path) -> None:
    """A state-9 row in the *middle* of the file is also valid and
    maps to ``中断``. Downstream ``chdis`` filters it out cleanly
    because it is neither 充電 nor 放電."""
    cell_dir = tmp_path / "midfile_9"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        f"2025/01/01,00:15:00,00000000,+3.5500,0.000000{empty_sep},9, 1,  1,     1",
        f"2025/01/01,00:30:00,00001800,+3.6000,1.000000{empty_sep},1, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert df["状態"].tolist() == ["充電", "中断", "充電"]


def test_raw_6digit_trailing_unknown_code_with_zero_flow_still_dropped(
    tmp_path: Path,
) -> None:
    """The defensive backstop in :func:`_drop_trailing_sentinel_rows`
    still drops a *trailing* row with an unknown code AND zero flow,
    e.g. a hypothetical future TOYO sentinel code ``8``."""
    cell_dir = tmp_path / "future_sentinel"
    cell_dir.mkdir()
    empty_sep = ",,,,,,"
    header = f"日付,時刻,経過時間[Sec],電圧[V],電流[mA]{empty_sep},状態,ﾓｰﾄﾞ,ｻｲｸﾙ,総ｻｲｸﾙ"
    raw = [
        "0,0,0,0,0,0,0",
        "",
        header,
        f"2025/01/01,00:00:00,00000000,+3.5000,1.000000{empty_sep},1, 1,  1,     1",
        # Trailing unknown code 8, zero elapsed, zero current → backstop drops it.
        f"2025/01/01,00:30:00,00000000,+3.6000,0.000000{empty_sep},8, 1,  1,     1",
    ]
    (cell_dir / "000001").write_text("\n".join(raw) + "\n", encoding="shift_jis")
    _write_fixed_column_ptn(cell_dir / "pattern.PTN", 0.001)
    df, _ = read_cell_dir(cell_dir)
    assert len(df) == 1
    assert df["状態"].tolist() == ["充電"]


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


def test_read_ptn_mass_simple_whitespace_format_under_legacy_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``ECHEMPLOT_PTN_LEGACY=1``, the V2.01 whitespace-split heuristic
    accepts hand-crafted formats like ``ACTIVE_MATERIAL WEIGHT <float>`` that
    the new fixed-column parser rejects. This pins the escape-hatch path."""
    monkeypatch.setenv("ECHEMPLOT_PTN_LEGACY", "1")
    ptn = tmp_path / "x.PTN"
    ptn.write_text("ACTIVE_MATERIAL WEIGHT 1.234\n", encoding="shift_jis")
    assert read_ptn_mass(ptn) == pytest.approx(1.234)


def test_read_ptn_mass_too_short_raises(tmp_path: Path) -> None:
    """A PTN whose first line is shorter than the fixed-column mass field
    raises so ``_resolve_mass_from_ptn`` can skip it (e.g. ``*_OPTION.PTN``)."""
    ptn = tmp_path / "x.PTN"
    ptn.write_text("too short\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="too short for the fixed-column PTN format"):
        read_ptn_mass(ptn)


def test_read_ptn_mass_legacy_short_line_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In legacy mode, a line with fewer than 3 whitespace tokens still raises."""
    monkeypatch.setenv("ECHEMPLOT_PTN_LEGACY", "1")
    ptn = tmp_path / "x.PTN"
    ptn.write_text("too short\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="fewer than 3 tokens"):
        read_ptn_mass(ptn)


def test_read_ptn_mass_non_numeric_field_raises(tmp_path: Path) -> None:
    """A 53+-char line whose 9-char composite field at columns 44..52 contains
    non-numeric content raises ``ValueError`` mentioning the dialect."""
    ptn = tmp_path / "x.PTN"
    # 44 leading chars (canonical operator+"2 " block) followed by a 9-char
    # garbage composite at the mass position.
    junk = "X" * 44 + "0XXNOTANUM" + "extra"
    ptn.write_text(junk + "\n", encoding="shift_jis")
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


# ---- read_ptn_mass fixed-column parser (#102) ----------------------------


def test_read_ptn_mass_concat_dialect(tmp_path: Path) -> None:
    """Concat dialect (cyclers No5/No1): flag glued to ``%.6f`` mass."""
    from .conftest import write_ptn_main

    ptn = tmp_path / "x.PTN"
    write_ptn_main(ptn, mass_g=0.000358, dialect="concat")
    assert read_ptn_mass(ptn) == pytest.approx(0.000358)


def test_read_ptn_mass_spaced_dialect(tmp_path: Path) -> None:
    """Spaced dialect (cycler No6): flag, space, then ``%.5f`` mass."""
    from .conftest import write_ptn_main

    ptn = tmp_path / "x.PTN"
    write_ptn_main(ptn, mass_g=0.00116, dialect="spaced")
    assert read_ptn_mass(ptn) == pytest.approx(0.00116)


def test_read_ptn_mass_japanese_operator_name(tmp_path: Path) -> None:
    """Multi-byte JP operator names don't shift the fixed-column position
    because the file is decoded Shift-JIS first and ``ljust`` pads in
    characters, not bytes."""
    from .conftest import write_ptn_main

    ptn = tmp_path / "x.PTN"
    write_ptn_main(ptn, mass_g=0.00116, dialect="spaced", operator="ともい")
    assert read_ptn_mass(ptn) == pytest.approx(0.00116)


def test_read_ptn_mass_negative_mass_field_raises(tmp_path: Path) -> None:
    """A composite that parses but yields a non-positive mass raises so the
    file is skipped (defensive against malformed flag bytes)."""
    ptn = tmp_path / "x.PTN"
    # 44-char prefix + composite "0-0.00100" (concat-shape, parses to -0.00100).
    junk = "X" * 44 + "0-0.00100" + "tail"
    ptn.write_text(junk + "\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="non-positive"):
        read_ptn_mass(ptn)


def test_read_ptn_mass_legacy_env_falls_back_to_whitespace_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ECHEMPLOT_PTN_LEGACY=1`` enables the V2.01 ``token[2] / token[3]``
    fallback for hypothetical third dialects."""
    monkeypatch.setenv("ECHEMPLOT_PTN_LEGACY", "1")
    ptn = tmp_path / "x.PTN"
    # Line that is too short for the fixed-column parser but valid for
    # whitespace-split with token[2] non-zero.
    ptn.write_text("name id 1.234\n", encoding="shift_jis")
    assert read_ptn_mass(ptn) == pytest.approx(1.234)


# ---- strict-encoding error surface (Issue #96) ----------------------------


def _write_corrupt_shift_jis_ptn(path: Path) -> None:
    """Write a PTN file with a valid Shift-JIS prefix and an invalid byte mid-stream.

    ``0xFF`` is not a legal Shift-JIS lead byte in any common dialect, so
    Python's ``shift_jis`` codec rejects it strictly. A trailing valid
    suffix is included so the failure is genuinely mid-stream rather than
    at EOF.
    """
    valid_prefix = " 1Operator                                 2 ".encode("shift_jis")
    invalid_bytes = b"\xff\xfe"
    valid_suffix = b" 0.001000       1 0.001000TestCell\n"
    path.write_bytes(valid_prefix + invalid_bytes + valid_suffix)


def test_encoding_error_is_reexported_from_io_package() -> None:
    """``echemplot.io.EncodingError`` must be the same class as the reader's."""
    assert EncodingErrorReexport is EncodingError
    assert issubclass(EncodingError, ValueError)


def test_read_ptn_mass_raises_encoding_error_on_corrupt_bytes(tmp_path: Path) -> None:
    """A PTN with bytes invalid for Shift-JIS must raise EncodingError, not silently
    return ``?`` replacements that break downstream float parsing with a misleading
    error pointing at the replacement character."""
    ptn = tmp_path / "corrupt_shift_jis_ptn.PTN"
    _write_corrupt_shift_jis_ptn(ptn)
    with pytest.raises(EncodingError) as exc_info:
        read_ptn_mass(ptn)
    # Path is part of the message so the failure is actionable
    assert str(ptn) in str(exc_info.value)
    # Byte position from the underlying UnicodeDecodeError surfaces in the message
    assert "position" in str(exc_info.value)
    # The invalid bytes are in the prefix region; assert a numeric byte offset shows
    assert any(ch.isdigit() for ch in str(exc_info.value))


def test_read_ptn_mass_strict_message_includes_hint(tmp_path: Path) -> None:
    """Users must see how to override the encoding from the error message."""
    ptn = tmp_path / "corrupt.PTN"
    _write_corrupt_shift_jis_ptn(ptn)
    with pytest.raises(EncodingError) as exc_info:
        read_ptn_mass(ptn)
    msg = str(exc_info.value)
    assert "encoding=" in msg
    # Hint mentions the default so users know what is being overridden
    assert "shift_jis" in msg


def test_read_ptn_mass_encoding_error_is_value_error(tmp_path: Path) -> None:
    """EncodingError must be a ValueError so existing handlers keep working."""
    ptn = tmp_path / "corrupt2.PTN"
    _write_corrupt_shift_jis_ptn(ptn)
    with pytest.raises(ValueError):
        read_ptn_mass(ptn)


def test_extract_mass_from_renzoku_metadata_raises_encoding_error(tmp_path: Path) -> None:
    """The renzoku-metadata reader must surface invalid bytes the same way."""
    csv = tmp_path / "連続データ.csv"
    valid_prefix = ",試験名,synthetic,,,,開始日時,2026-01-01\n,測定備考,\n,重量[mg],".encode(
        "shift_jis"
    )
    invalid_bytes = b"\xff\xfe"
    valid_suffix = b"1.0\n"
    csv.write_bytes(valid_prefix + invalid_bytes + valid_suffix)
    with pytest.raises(EncodingError) as exc_info:
        _extract_mass_from_renzoku_metadata(csv, "shift_jis")
    msg = str(exc_info.value)
    assert str(csv) in msg
    assert "encoding=" in msg
    assert "shift_jis" in msg


# ---- column_lang translation ---------------------------------------------


def test_column_lang_en(make_cell_dir: Callable[..., Path]) -> None:
    d = make_cell_dir("raw_6digit", mass=0.001)
    df, _ = read_cell_dir(d, column_lang="en")
    assert list(df.columns[: len(CANONICAL_COLUMNS_EN)]) == list(CANONICAL_COLUMNS_EN)
    # extras translated via schema.JA_TO_EN
    assert "elapsed_time_s" in df.columns
    assert "current_ma" in df.columns
    # State values are translated to EN (issue #94) so EN-mode consumers
    # receive a fully EN frame.
    assert set(df["state"].unique()) <= {"rest", "charge", "discharge"}


def test_finalize_translates_state_values_when_lang_en(tmp_path: Path) -> None:
    """Native 連続データ.csv state literals translate to EN when requested.

    Pins the issue #94 contract: the reader maps every JA literal in the
    extended STATE_JA_TO_EN set (including 充電休止/放電休止 substates) to
    its EN equivalent.
    """
    from tests.conftest import write_renzoku_with_states

    d = tmp_path / "cell"
    d.mkdir()
    write_renzoku_with_states(
        d,
        rows=[
            (1, "1", "充電", 3.50, 0.0),
            (1, "1", "充電休止", 3.60, 100.0),
            (1, "1", "放電", 3.55, 0.0),
            (1, "1", "放電休止", 3.40, 50.0),
            (1, "1", "中断", 3.40, 50.0),
        ],
    )
    df, _ = read_cell_dir(d, column_lang="en")
    assert df["state"].tolist() == [
        "charge",
        "charge_rest",
        "discharge",
        "discharge_rest",
        "abort",
    ]


def test_finalize_raises_on_unknown_ja_state_string(tmp_path: Path) -> None:
    """An unrecognized JA state literal (e.g. ``予期しない``) must raise.

    Mirrors the strictness already enforced for unknown numeric state
    codes — closing the asymmetry called out in issue #94.
    """
    from tests.conftest import write_renzoku_with_states

    d = tmp_path / "cell"
    d.mkdir()
    write_renzoku_with_states(
        d,
        rows=[
            (1, "1", "充電", 3.50, 0.0),
            (1, "1", "予期しない", 3.55, 50.0),
        ],
    )
    with pytest.raises(ValueError, match=r"unknown 状態 labels.*予期しない"):
        read_cell_dir(d)


def test_finalize_passes_charge_rest_through_unchanged_when_lang_ja(tmp_path: Path) -> None:
    """In JA mode, native 充電休止/放電休止 substate labels must round-trip
    untranslated. Belt-and-suspenders against the new validation step
    accidentally normalizing JA literals."""
    from tests.conftest import write_renzoku_with_states

    d = tmp_path / "cell"
    d.mkdir()
    write_renzoku_with_states(
        d,
        rows=[
            (1, "1", "充電", 3.50, 0.0),
            (1, "1", "充電休止", 3.60, 100.0),
            (1, "1", "放電", 3.55, 0.0),
            (1, "1", "放電休止", 3.40, 50.0),
        ],
    )
    df, _ = read_cell_dir(d, column_lang="ja")
    assert df["状態"].tolist() == ["充電", "充電休止", "放電", "放電休止"]


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
