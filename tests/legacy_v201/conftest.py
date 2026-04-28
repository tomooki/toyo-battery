"""Autouse fixture: enable the V2.01 PTN parser for the legacy parity suite.

The tests in this directory pin ``TOYO_Origin_2.01.py``-era behavior. The
0.2.0 cleanup (issue #102) replaced the default ``read_ptn_mass`` with a
fixed-column parser; the V2.01 ``token[2] / token[3]`` whitespace-split
heuristic survives behind ``ECHEMPLOT_PTN_LEGACY=1``. This fixture sets
that env var for every test under ``tests/legacy_v201/`` so the parity
suite continues to exercise the legacy path it was written against.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_v201_ptn_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECHEMPLOT_PTN_LEGACY", "1")
