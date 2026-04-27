"""V2.01 PTN-dialect parity tests for :mod:`echemplot.io.reader`.

These tests pin the legacy ``TOYO_Origin_2.01.py`` ``if token[2]==0 then
token[3]`` PTN-mass fallback rule. They are NOT the primary specification
for ``echemplot`` reader behavior — see ``tests/test_reader.py`` at the
parent level for that. When V2.01 parity is consciously broken (e.g. as
part of the 0.2.0 cleanup tracked by issue #105), update or remove these
tests rather than letting them drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from echemplot.io.reader import read_cell_dir, read_ptn_mass

from ..conftest import write_ptn_main

# ---- read_ptn_mass dialect handling --------------------------------------
#
# TOYO ships at least two PTN dialects on real cyclers. Both encode the
# active-material mass in a 9-byte ``<flag><mass>`` composite field on
# line 0:
#   * "concat" (No5 / No1): flag is glued to a %.6f mass — ``"00.000358"``
#     → ``split()[2]`` parses directly as a non-zero float.
#   * "spaced" (No6): flag, space, then a %.5f mass — ``"0 0.00116"``
#     → ``split()[2]`` is the flag ``"0"``; the mass is at ``[3]``.
# The reader collapses both via the v2.01 "if token[2]==0 then token[3]"
# heuristic.


def test_read_ptn_mass_concat_dialect_no5(tmp_path: Path) -> None:
    ptn = tmp_path / "x.PTN"
    write_ptn_main(ptn, mass_g=0.000358, dialect="concat")
    assert read_ptn_mass(ptn) == pytest.approx(0.000358)


def test_read_ptn_mass_spaced_dialect_no6(tmp_path: Path) -> None:
    """Older No6-cycler dialect — token[2] is ``"0"``, real mass at token[3]."""
    ptn = tmp_path / "x.PTN"
    write_ptn_main(ptn, mass_g=0.00116, dialect="spaced")
    assert read_ptn_mass(ptn) == pytest.approx(0.00116)


def test_read_ptn_mass_japanese_operator_name_spaced(tmp_path: Path) -> None:
    """No1 cycler renders the operator name in Japanese (e.g. ``ともい``).

    cp932 encodes each kana as 2 bytes; after shift-jis decode the line
    becomes shorter in characters than in bytes, so any column-slice
    parser would shift. Token-based parsing must stay correct regardless.
    """
    ptn = tmp_path / "x.PTN"
    write_ptn_main(ptn, mass_g=0.00116, dialect="spaced", operator="ともい")
    assert read_ptn_mass(ptn) == pytest.approx(0.00116)


def test_read_ptn_mass_zero_with_unparseable_token3_raises(tmp_path: Path) -> None:
    """token[2]==0 + token[3] not numeric → ValueError so the file is skipped.

    The skip path is essential: ``_resolve_mass_from_ptn`` iterates over
    every ``.PTN`` in a cell dir and treats ``ValueError`` as "not the
    main pattern, skip me". Pathological lines must keep raising.
    """
    ptn = tmp_path / "x.PTN"
    ptn.write_text("name id 0 X\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="token"):
        read_ptn_mass(ptn)


def test_read_ptn_mass_zero_with_no_token3_raises(tmp_path: Path) -> None:
    """token[2]==0 and only 3 tokens total → ValueError."""
    ptn = tmp_path / "x.PTN"
    ptn.write_text("name id 0\n", encoding="shift_jis")
    with pytest.raises(ValueError, match="token"):
        read_ptn_mass(ptn)


# ---- raw_6digit + dialect end-to-end -------------------------------------


def test_raw_6digit_with_no6_spaced_dialect(make_cell_dir: Callable[..., Path]) -> None:
    """End-to-end: a No6-shaped cell dir (older PTN dialect) must produce
    the same mass + 電気量 as the newer dialect. This is the regression
    pin for the v2.01 fallback rule."""
    d = make_cell_dir("raw_6digit", mass=0.00116, ptn_dialect="spaced")
    df, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.00116)
    # capacity = elapsed/3600 * I / mass for the 1-hour@1mA charge segment row
    assert df.loc[1, "電気量"] == pytest.approx(1.0 * 1.0 / 0.00116)


def test_raw_6digit_no1_layout_no_option_file(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    """No1 cycler ships only ``main.PTN`` + ``main_Option2.PTN`` (no OPTION.PTN).

    Option2 starts with comma-separated values (``0,1,1,1,...``) — no
    whitespace, so ``.split()`` yields a single token and the
    ``len(tokens) < 3`` skip path catches it. The main PTN's mass must
    still resolve unambiguously.
    """
    d = make_cell_dir("raw_6digit", mass=0.000784, ptn_dialect="concat")
    (d / "pattern_OPTION.PTN").unlink()
    (d / "pattern_Option2.PTN").write_text(
        "0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,-,-,-,-,-\n",
        encoding="shift_jis",
    )
    _, mass_g = read_cell_dir(d)
    assert mass_g == pytest.approx(0.000784)
