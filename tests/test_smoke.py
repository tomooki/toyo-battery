"""P0 smoke tests: package importable, version string present."""

from __future__ import annotations


def test_import_package() -> None:
    import toyo_battery

    assert toyo_battery.__version__


def test_import_submodules() -> None:
    from toyo_battery import config  # noqa: F401
    from toyo_battery.core import cell  # noqa: F401
    from toyo_battery.io import reader, schema  # noqa: F401


def test_schema_roundtrip() -> None:
    from toyo_battery.io.schema import rename

    ja = ["サイクル", "電圧", "電流"]
    en = rename(ja, "en")
    assert en == ["cycle", "voltage", "current"]
    assert rename(en, "ja") == ja


def test_origin_submodule_imports_without_originpro() -> None:
    """origin submodule must be importable even when originpro is missing."""
    from toyo_battery import origin

    assert hasattr(origin, "push_to_origin")
