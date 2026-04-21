"""Analytical parity tests for :func:`toyo_battery.core.stats.stat_table`.

These tests pin every column of ``stat_table`` against **closed-form
analytical expected values** derived from the shared ``make_cell_dir``
fixture's own specification — no reference-implementation capture is
involved.

Why analytical ground truth (not a v2.01 captured JSON)
-------------------------------------------------------
The original plan for issue #7 was to compare ``stat_table`` output
against a captured JSON from the legacy v2.01 pipeline. That approach was
rejected because after PR #19 rewrote the ``make_cell_dir`` fixture to
match real TOYO on-disk layouts, running v2.01 on the post-#19 fixtures
would freeze the *new* output shape rather than verify physical
correctness. The revised approach derives expected values directly from
the fixture's ``(V, I, t, mass)`` specification and is therefore
reference-implementation-independent — it would detect a silent
regression in either ``stat_table`` itself or any upstream module
(``chdis``, ``capacity``, reader) that feeds it.

Fixture spec (from ``tests/conftest.py``)
-----------------------------------------
``_ROWS_SHARED`` emits a single cycle with five rows:

- charge:     V = 3.50 → 3.60 V, elapsed 0 → 3600 s, I = 1.0 mA
- rest:       V = 3.61 V (dropped by :mod:`chdis`)
- discharge:  V = 3.40 → 3.20 V, elapsed 0 → 3600 s, I = 1.0 mA

With ``mass_g = 0.001`` (1 mg):

- Q at segment end = ``I * Δt / 3600 / mass = 1.0 * 3600 / 3600 / 0.001
  = 1000 mAh/g`` for both charge and discharge.
- Each segment has exactly two distinct-Q points, so ``stat_table`` falls
  through to the trapezoidal-rule branch for ``∫V dQ``. Trapezoid on a
  linear two-point segment is exact: ``∫V dQ = (V_lo + V_hi) / 2 * ΔQ``.

Expected values (all layouts)
-----------------------------
==================   =====================================================
column               expected (closed form)
==================   =====================================================
Q_dis_max            1000.0
cycle_at_80pct       NaN (single-cycle fixture — never fades)
Q_dis@1              1000.0
CE@1                 100.0
V_mean_dis@1         3.30 = (3.40 + 3.20) / 2
EE@1                 100.0 * 3300 / 3550 ≈ 92.95774647887323
retention@1          100.0 (exact — ``n == 1`` short-circuit in stats.py)
CE_mean_1to1         100.0
==================   =====================================================

Note the ``EE@1`` form: because ``stat_table`` computes
``100 * ∫V dQ_dis / ∫V dQ_ch`` (not through the V_mean ratio), the
numerically-exact output is ``100.0 * 3300.0 / 3550.0``, which is 2 ULPs
away from ``100.0 * 3.30 / 3.55``. The tests use the former (the
integral form) to stay bit-exact; the docstring records both for
readability.

All three on-disk layouts (``renzoku``, ``renzoku_py``, ``raw_6digit``)
produce bit-identical values for this fixture. ``raw_6digit`` would in
principle accumulate FP error in its ``Q = elapsed/3600 * I / mass``
derivation, but for the specific values ``3600 / 3600 * 1.0 / 0.001``
the computation is exact. The tolerances are set accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from toyo_battery.core.cell import Cell
from toyo_battery.core.stats import stat_table

# Fixture parameters — mirrored from ``tests/conftest.py`` so the
# analytical expected values below are derived from the fixture's
# declared physics rather than from a reference implementation.
_FIXTURE_MASS_G: float = 0.001
_FIXTURE_I_MA: float = 1.0
_FIXTURE_DELTA_T_S: float = 3600.0
_FIXTURE_V_CH_LO: float = 3.50
_FIXTURE_V_CH_HI: float = 3.60
_FIXTURE_V_DIS_HI: float = 3.40
_FIXTURE_V_DIS_LO: float = 3.20

# Capacity (mAh/g) at the end of a segment: Q = I * Δt / 3600 / mass.
# For I=1 mA, Δt=3600 s, mass=0.001 g → Q = 1000.
_EXPECTED_Q: float = _FIXTURE_I_MA * _FIXTURE_DELTA_T_S / 3600.0 / _FIXTURE_MASS_G
assert _EXPECTED_Q == 1000.0  # sanity-check the derivation at import time

# Energy-weighted mean voltage over a linear V ramp with monotone-rising
# Q is the arithmetic mean of the V endpoints: V_mean = (V_lo + V_hi)/2.
_EXPECTED_V_MEAN_CH: float = (_FIXTURE_V_CH_LO + _FIXTURE_V_CH_HI) / 2.0  # 3.55
_EXPECTED_V_MEAN_DIS: float = (_FIXTURE_V_DIS_HI + _FIXTURE_V_DIS_LO) / 2.0  # 3.30

# ``stat_table`` computes EE via the integrals (not the V_mean ratio), so
# pin the integral form to stay bit-exact with the implementation under
# test. The trapezoidal rule on a two-point linear ramp gives
# ``∫V dQ = V_mean * ΔQ`` exactly, so each integral is ``V_mean * Q``.
_EXPECTED_INT_V_DQ_CH: float = _EXPECTED_V_MEAN_CH * _EXPECTED_Q  # 3550.0
_EXPECTED_INT_V_DQ_DIS: float = _EXPECTED_V_MEAN_DIS * _EXPECTED_Q  # 3300.0
_EXPECTED_EE: float = 100.0 * _EXPECTED_INT_V_DQ_DIS / _EXPECTED_INT_V_DQ_CH

# Q_dis / Q_ch are equal by fixture construction, so CE = 100 %.
_EXPECTED_CE: float = 100.0 * _EXPECTED_Q / _EXPECTED_Q  # 100.0

# All three layouts preserve the per-segment endpoint values exactly for
# this specific fixture (the raw-6-digit ``elapsed/3600 * I / mass``
# derivation happens to be FP-exact here because every operand is a
# power-of-two-friendly round number). Keep the tolerance tight so any
# future FP regression in the reader or chdis surfaces immediately.
_RTOL: float = 1e-12
_ATOL: float = 1e-12

_LAYOUTS: tuple[str, ...] = ("renzoku", "renzoku_py", "raw_6digit")


def _expected_row_cycle_1() -> dict[str, float]:
    """Closed-form expected ``stat_table`` row for the shared fixture.

    Kept as a function (not a module-level constant) so a future
    multi-cycle parity test can parametrize the cycles without refactor.
    """
    return {
        "Q_dis_max": _EXPECTED_Q,
        "cycle_at_80pct": float("nan"),  # single-cycle → threshold never crossed
        "Q_dis@1": _EXPECTED_Q,
        "CE@1": _EXPECTED_CE,
        "V_mean_dis@1": _EXPECTED_V_MEAN_DIS,
        "EE@1": _EXPECTED_EE,
        "retention@1": 100.0,  # stats.py short-circuits n==1 to exactly 100.0
        "CE_mean_1to1": _EXPECTED_CE,
    }


def _assert_row_matches(row: pd.Series, expected: dict[str, float]) -> None:
    """Assert every (col, expected) pair with NaN-aware comparison.

    ``np.testing.assert_allclose`` treats NaN as unequal by default; the
    NaN columns are split out and asserted with ``pd.isna`` so a silent
    "column suddenly got a value" regression still fires.
    """
    for col, expected_val in expected.items():
        assert col in row.index, f"expected column {col!r} missing from stat_table output"
        actual_val = row[col]
        if np.isnan(expected_val):
            assert pd.isna(actual_val), f"{col!r}: expected NaN, got {actual_val!r}"
        else:
            np.testing.assert_allclose(
                actual_val,
                expected_val,
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=f"{col!r}: expected {expected_val!r}, got {actual_val!r}",
            )


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_stat_table_analytical_parity_per_layout(
    make_cell_dir: Callable[..., Path], layout: str
) -> None:
    """Every ``stat_table`` column matches the closed-form expected value.

    Parametrized over all three on-disk layouts. The fixture's physics
    (Q = I·Δt/mass, linear V ramps, single cycle) fully determines every
    output column analytically, so any FP drift or algorithmic change in
    ``stat_table`` / ``capacity`` / ``chdis`` / reader is detected as a
    tolerance failure on a specific column rather than a vague "numbers
    moved" report.
    """
    cell = Cell.from_dir(make_cell_dir(layout))
    tbl = stat_table([cell], target_cycles=(1,))

    assert tbl.index.tolist() == ["cell_A"]
    _assert_row_matches(tbl.loc["cell_A"], _expected_row_cycle_1())


def test_stat_table_multi_cell_parity(make_cell_dir: Callable[..., Path]) -> None:
    """Two cells built from the same fixture → two bit-identical rows.

    Cross-cell independence check: the per-cell stats pipeline must not
    share mutable state between ``_per_cell_row`` invocations. Also pins
    that the index order follows the input ``cells`` list order (not
    sorted, not reordered).
    """
    cell_a = Cell.from_dir(make_cell_dir("renzoku", name="cell_A"))
    cell_b = Cell.from_dir(make_cell_dir("renzoku", name="cell_B"))
    tbl = stat_table([cell_a, cell_b], target_cycles=(1,))

    assert tbl.index.tolist() == ["cell_A", "cell_B"]

    expected = _expected_row_cycle_1()
    _assert_row_matches(tbl.loc["cell_A"], expected)
    _assert_row_matches(tbl.loc["cell_B"], expected)

    # Row equality: every non-NaN column must agree between A and B to
    # machine precision (NaN columns are expected on both sides).
    for col in tbl.columns:
        a_val = tbl.loc["cell_A", col]
        b_val = tbl.loc["cell_B", col]
        if pd.isna(a_val):
            assert pd.isna(b_val), f"{col!r}: A=NaN but B={b_val!r}"
        else:
            assert a_val == b_val, f"{col!r}: A={a_val!r} != B={b_val!r} for identical fixtures"


def test_stat_table_column_schema_parity(make_cell_dir: Callable[..., Path]) -> None:
    """Schema invariants: column order, column dtypes, index name + dtype.

    ``stat_table`` is positioned as the CSV-export surface for downstream
    consumers, so schema stability (not just numerical correctness) is a
    first-class contract. A silent column-reorder or dtype-drift would
    break fragile downstream CSV readers; this test pins both.
    """
    cell = Cell.from_dir(make_cell_dir("renzoku"))
    tbl = stat_table([cell], target_cycles=(1,))

    # Exact column order for ``target_cycles=(1,)``. Mirrors the order
    # defined in :func:`toyo_battery.core.stats._build_column_order`.
    assert list(tbl.columns) == [
        "Q_dis_max",
        "cycle_at_80pct",
        "Q_dis@1",
        "CE@1",
        "V_mean_dis@1",
        "EE@1",
        "retention@1",
        "CE_mean_1to1",
    ]

    # Every column is float64 (not object, not nullable Int64) — the
    # ``stats.py`` output contract materializes NaN as float NaN so
    # downstream ``.to_csv`` emits "" or a user-chosen ``na_rep``
    # uniformly rather than the pandas nullable-array string form.
    for col in tbl.columns:
        assert tbl[col].dtype == np.float64, f"{col!r}: expected float64, got {tbl[col].dtype}"

    # Index name + dtype: the populated path must match the empty path
    # (see ``_empty_result`` in stats.py) so concatenating an empty +
    # populated stat_table doesn't trigger a dtype-coercion warning.
    assert tbl.index.name == "cell"
    assert tbl.index.dtype == object


def test_retention_and_fade_are_nan_on_single_cycle(
    make_cell_dir: Callable[..., Path],
) -> None:
    """Single-cycle fixture → ``cycle_at_80pct`` is NaN.

    Covers the degenerate case that would otherwise masquerade as "passed
    the fade threshold" if ``stat_table`` misreported a False fade_mask
    as a finite cycle. Tangentially exercised by the per-layout parity
    test too, but isolated here so a regression in the fade-tracking
    code surfaces with a targeted failure message.
    """
    cell = Cell.from_dir(make_cell_dir("renzoku"))
    tbl = stat_table([cell], target_cycles=(1,))

    assert pd.isna(tbl.loc["cell_A", "cycle_at_80pct"])
    # retention@1 is *not* NaN — it's exactly 100.0 (the n==1 short-circuit
    # in stats.py), which is the defining property that separates "never
    # faded" from "can't compute retention because q_dis[1] is unusable".
    assert tbl.loc["cell_A", "retention@1"] == 100.0
