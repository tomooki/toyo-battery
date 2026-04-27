"""P0 smoke tests: package importable, version string present."""

from __future__ import annotations


def test_import_package() -> None:
    import echemplot

    assert echemplot.__version__


def test_cell_reexported_at_top_level() -> None:
    """README Quick Start uses ``from echemplot import Cell`` — keep it working."""
    from echemplot import Cell
    from echemplot.core.cell import Cell as CoreCell

    assert Cell is CoreCell


def test_import_submodules() -> None:
    from echemplot import config  # noqa: F401
    from echemplot.core import cell  # noqa: F401
    from echemplot.io import reader, schema  # noqa: F401


def test_schema_roundtrip() -> None:
    from echemplot.io.schema import rename

    ja = ["サイクル", "電圧", "電流"]
    en = rename(ja, "en")
    assert en == ["cycle", "voltage", "current"]
    assert rename(en, "ja") == ja


def test_state_code_9_maps_to_中断() -> None:
    """State code 9 is the empirical TOYO end-of-test/abort sentinel
    and must round-trip through all three state mappings."""
    from echemplot.io.schema import (
        STATE_CODE_TO_EN,
        STATE_CODE_TO_JA,
        STATE_JA_TO_EN,
    )

    assert STATE_CODE_TO_JA[9] == "中断"
    assert STATE_CODE_TO_EN[9] == "abort"
    assert STATE_JA_TO_EN["中断"] == "abort"


def test_origin_submodule_imports_without_originpro() -> None:
    """origin submodule must be importable even when originpro is missing."""
    from echemplot import origin

    assert hasattr(origin, "push_to_origin")
