"""Tests for the consolidated schema mappings in :mod:`echemplot.io.schema`.

Pins the canonical-key set of ``JA_COLS`` (single source of truth used by
core / plotting modules), the JA↔EN round-trip via ``JA_TO_EN``, and the
extended 6-state ``STATE_JA_TO_EN`` / ``STATE_EN_TO_JA`` shape including
the native-CSV ``充電休止`` / ``放電休止`` substates.
"""

from __future__ import annotations

import pytest

from echemplot.io.schema import (
    COL_CAPACITY,
    JA_COLS,
    JA_TO_EN,
    STATE_EN_TO_JA,
    STATE_JA_TO_EN,
)

# Expected canonical keys → (JA literal, EN literal) for the consolidated map.
_EXPECTED_JA_COLS: dict[str, tuple[str, str]] = {
    "cycle": ("サイクル", "cycle"),
    "state": ("状態", "state"),
    "voltage": ("電圧", "voltage"),
    "capacity": ("電気量", "capacity"),
}


@pytest.mark.parametrize(("key", "expected"), list(_EXPECTED_JA_COLS.items()))
def test_ja_cols_literal_per_key(key: str, expected: tuple[str, str]) -> None:
    """Every canonical key resolves to its expected JA literal."""
    ja_literal, _ = expected
    assert JA_COLS[key] == ja_literal


@pytest.mark.parametrize(("key", "expected"), list(_EXPECTED_JA_COLS.items()))
def test_ja_cols_round_trip_through_ja_to_en(key: str, expected: tuple[str, str]) -> None:
    """JA_TO_EN[JA_COLS[k]] yields the EN equivalent for every canonical key."""
    ja_literal, en_literal = expected
    assert JA_TO_EN[ja_literal] == en_literal


def test_ja_cols_uses_named_constants() -> None:
    """The capacity key reuses the existing ``COL_CAPACITY`` constant —
    guards against a future drift where one is updated and the other isn't.
    """
    assert JA_COLS["capacity"] == COL_CAPACITY


def test_ja_cols_key_set_is_canonical() -> None:
    """The canonical key set is fixed; new entries must be added intentionally
    (and the parametrised tests above updated)."""
    assert set(JA_COLS.keys()) == set(_EXPECTED_JA_COLS.keys())


def test_state_ja_to_en_includes_legacy_three_state() -> None:
    """The legacy 3-state set (charge/discharge/rest) plus the abort sentinel
    must remain present after the substate extension."""
    assert STATE_JA_TO_EN["休止"] == "rest"
    assert STATE_JA_TO_EN["充電"] == "charge"
    assert STATE_JA_TO_EN["放電"] == "discharge"
    assert STATE_JA_TO_EN["中断"] == "abort"


def test_state_ja_to_en_includes_substates() -> None:
    """The 4-state native-CSV substates must map to dedicated EN labels."""
    assert STATE_JA_TO_EN["充電休止"] == "charge_rest"
    assert STATE_JA_TO_EN["放電休止"] == "discharge_rest"


def test_state_ja_to_en_full_set() -> None:
    """Pin the full 5-state set so a regression that drops an entry surfaces."""
    assert STATE_JA_TO_EN == {
        "休止": "rest",
        "充電": "charge",
        "放電": "discharge",
        "中断": "abort",
        "充電休止": "charge_rest",
        "放電休止": "discharge_rest",
    }


def test_state_en_to_ja_round_trip() -> None:
    """STATE_EN_TO_JA is the inverse of STATE_JA_TO_EN with no lossy collisions."""
    # Round-trip JA → EN → JA for every JA label.
    for ja, en in STATE_JA_TO_EN.items():
        assert STATE_EN_TO_JA[en] == ja
    # Inverse direction too: EN → JA → EN.
    for en, ja in STATE_EN_TO_JA.items():
        assert STATE_JA_TO_EN[ja] == en
    # And the reverse map covers every EN value (no silently-dropped entries).
    assert set(STATE_EN_TO_JA.keys()) == set(STATE_JA_TO_EN.values())
