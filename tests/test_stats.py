"""Tests for :mod:`toyo_battery.core.stats`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from toyo_battery.core.cell import Cell
from toyo_battery.core.stats import stat_table


def _linear_cell(
    name: str,
    *,
    cycles_q_ch: dict[int, float],
    cycles_q_dis: dict[int, float],
    v_ch_lo: float = 3.0,
    v_ch_hi: float = 4.2,
    v_dis_lo: float = 3.0,
    v_dis_hi: float = 4.2,
    n_points: int = 50,
) -> Cell:
    """Build a :class:`Cell` whose raw_df has linear V-Q ramps per cycle.

    Each cycle's charge segment rises V from ``v_ch_lo`` to ``v_ch_hi`` while
    Q rises 0 → ``cycles_q_ch[cycle]``. Discharge reverses V (``v_dis_hi`` →
    ``v_dis_lo``) while Q rises 0 → ``cycles_q_dis[cycle]`` — mirroring
    :mod:`chdis`'s invariant that Q is monotone non-decreasing within a
    segment regardless of V direction.

    With linear V-Q ramps the energy-weighted mean voltage equals the
    arithmetic mean ``(V_lo + V_hi) / 2`` (for either direction), which
    lets tests pin ``V_mean_dis`` / ``EE`` values at arbitrarily tight
    tolerances.
    """
    rows: list[tuple[int, str, str, float, float]] = []
    for cycle, q_max_ch in cycles_q_ch.items():
        v_ch = np.linspace(v_ch_lo, v_ch_hi, n_points)
        q_ch = np.linspace(0.0, q_max_ch, n_points)
        rows.extend((cycle, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch))
        if cycle in cycles_q_dis:
            q_max_dis = cycles_q_dis[cycle]
            v_dis = np.linspace(v_dis_hi, v_dis_lo, n_points)
            q_dis = np.linspace(0.0, q_max_dis, n_points)
            rows.extend((cycle, "1", "放電", float(v), float(q)) for v, q in zip(v_dis, q_dis))
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    return Cell(name=name, mass_g=0.001, raw_df=raw)


def test_two_cell_hand_computed_fixture_pinned() -> None:
    """Two synthetic cells, every output column pinned to hand-computed values.

    Fixture design: both cells have linear V-Q ramps (3.0→4.2 V on charge,
    4.2→3.0 V on discharge with Q still rising per chdis invariant). Cell A
    has q_ch=[1000, 900] and q_dis=[990, 880]; Cell B has q_ch=[800, 600]
    and q_dis=[760, 500]. With these numbers every column is either
    analytical (capacity/CE/retention ratios) or derivable from the mean-V
    identity for linear ramps (mean = 3.6 V on both sides).
    """
    cell_a = _linear_cell(
        "A",
        cycles_q_ch={1: 1000.0, 2: 900.0},
        cycles_q_dis={1: 990.0, 2: 880.0},
    )
    cell_b = _linear_cell(
        "B",
        cycles_q_ch={1: 800.0, 2: 600.0},
        cycles_q_dis={1: 760.0, 2: 500.0},
    )
    tbl = stat_table([cell_a, cell_b], target_cycles=(1, 2))

    # Index + shape
    assert tbl.index.name == "cell"
    assert tbl.index.tolist() == ["A", "B"]
    assert (tbl.dtypes == np.float64).all()

    # ------- Cell A -------
    # Q_dis_max = max(990, 880) = 990
    np.testing.assert_allclose(tbl.loc["A", "Q_dis_max"], 990.0, atol=1e-9)
    # q_dis doesn't dip below 0.8 * q_dis[1] = 792 (cycle 2 = 880 > 792).
    assert pd.isna(tbl.loc["A", "cycle_at_80pct"])
    # Cycle 1
    np.testing.assert_allclose(tbl.loc["A", "Q_dis@1"], 990.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "CE@1"], 99.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "V_mean_dis@1"], 3.6, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "EE@1"], 99.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "retention@1"], 100.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "CE_mean_1to1"], 99.0, atol=1e-9)
    # Cycle 2
    np.testing.assert_allclose(tbl.loc["A", "Q_dis@2"], 880.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "CE@2"], 100.0 * 880.0 / 900.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "V_mean_dis@2"], 3.6, atol=1e-9)
    # EE@2 = 100 * (3.6 * 880) / (3.6 * 900) = 100 * 880 / 900
    np.testing.assert_allclose(tbl.loc["A", "EE@2"], 100.0 * 880.0 / 900.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["A", "retention@2"], 100.0 * 880.0 / 990.0, atol=1e-9)
    # CE_mean_1to2 = mean(CE@1, CE@2) = (99 + 100*880/900) / 2
    np.testing.assert_allclose(
        tbl.loc["A", "CE_mean_1to2"],
        (99.0 + 100.0 * 880.0 / 900.0) / 2.0,
        atol=1e-9,
    )

    # ------- Cell B -------
    # Q_dis_max = 760. 0.8 * q_dis[1] = 608. Cycle 2 = 500 < 608 → cycle_at_80pct = 2.
    np.testing.assert_allclose(tbl.loc["B", "Q_dis_max"], 760.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "cycle_at_80pct"], 2.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "Q_dis@1"], 760.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "CE@1"], 95.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "V_mean_dis@1"], 3.6, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "EE@1"], 95.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "retention@1"], 100.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "CE_mean_1to1"], 95.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "Q_dis@2"], 500.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "CE@2"], 100.0 * 500.0 / 600.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "V_mean_dis@2"], 3.6, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "EE@2"], 100.0 * 500.0 / 600.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["B", "retention@2"], 100.0 * 500.0 / 760.0, atol=1e-9)
    np.testing.assert_allclose(
        tbl.loc["B", "CE_mean_1to2"],
        (95.0 + 100.0 * 500.0 / 600.0) / 2.0,
        atol=1e-9,
    )


def test_never_faded_yields_nan_cycle_at_threshold() -> None:
    """A cell whose q_dis never dips below the threshold → cycle_at_80pct is NaN.

    Also exercises the "Q_dis increases slightly cycle-over-cycle" shape,
    which older ``while``-style fade trackers could misreport as "never
    faded" for the wrong reason.
    """
    cell = _linear_cell(
        "flat",
        # Nearly-flat discharge capacity: 1000, 1010, 995. Never < 800.
        cycles_q_ch={1: 1050.0, 2: 1050.0, 3: 1050.0},
        cycles_q_dis={1: 1000.0, 2: 1010.0, 3: 995.0},
    )
    tbl = stat_table([cell], target_cycles=(1, 2, 3))
    assert pd.isna(tbl.loc["flat", "cycle_at_80pct"])
    # Sanity: other columns remain alive.
    np.testing.assert_allclose(tbl.loc["flat", "Q_dis_max"], 1010.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["flat", "retention@3"], 99.5, atol=1e-9)


def test_target_cycles_beyond_max_cycle_yield_nan_in_row() -> None:
    """Asking for cycle 50 on a 2-cycle cell → NaN in that column, row preserved."""
    cell = _linear_cell(
        "short",
        cycles_q_ch={1: 1000.0, 2: 950.0},
        cycles_q_dis={1: 990.0, 2: 940.0},
    )
    tbl = stat_table([cell], target_cycles=(1, 50))

    # Row is still present
    assert "short" in tbl.index
    # Cycle-1 columns are populated
    np.testing.assert_allclose(tbl.loc["short", "Q_dis@1"], 990.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["short", "CE@1"], 99.0, atol=1e-9)
    # Cycle-50 direct look-ups are NaN (no such row in cap_df / chdis_df)
    for col in ("Q_dis@50", "CE@50", "V_mean_dis@50", "EE@50", "retention@50"):
        assert pd.isna(tbl.loc["short", col]), f"{col} should be NaN on a short cell"
    # CE_mean_1to50 is *not* NaN — pandas ``.loc[1:50]`` on a 2-row cap_df
    # returns cycles 1 & 2, and the mean of those two CEs is a finite value.
    # This pins the "take whatever cycles exist up to n" semantics; a caller
    # who wanted a NaN-on-short contract would need a separate helper.
    np.testing.assert_allclose(
        tbl.loc["short", "CE_mean_1to50"],
        (99.0 + 100.0 * 940.0 / 950.0) / 2.0,
        atol=1e-9,
    )
    # Whole-cell columns still populated
    np.testing.assert_allclose(tbl.loc["short", "Q_dis_max"], 990.0, atol=1e-9)


def test_empty_cells_list_returns_empty_frame_with_schema() -> None:
    """``stat_table([])`` → empty frame with the right columns / dtype / index name."""
    tbl = stat_table([])
    assert tbl.empty
    assert tbl.index.name == "cell"
    assert list(tbl.columns) == [
        "Q_dis_max",
        "cycle_at_80pct",
        "Q_dis@10",
        "CE@10",
        "V_mean_dis@10",
        "EE@10",
        "retention@10",
        "CE_mean_1to10",
        "Q_dis@50",
        "CE@50",
        "V_mean_dis@50",
        "EE@50",
        "retention@50",
        "CE_mean_1to50",
    ]
    for col in tbl.columns:
        assert tbl[col].dtype == np.float64


def test_column_order_is_deterministic_and_follows_target_cycles() -> None:
    """Column order: [Q_dis_max, cycle_at_Xpct, then each n's 6-column block]."""
    cell = _linear_cell(
        "one",
        cycles_q_ch={1: 1000.0, 5: 900.0, 3: 950.0},
        cycles_q_dis={1: 990.0, 5: 880.0, 3: 930.0},
    )
    # target_cycles=(5, 3) (intentionally not sorted) to pin that insertion
    # order is preserved rather than sorted.
    tbl = stat_table([cell], target_cycles=(5, 3))
    assert list(tbl.columns) == [
        "Q_dis_max",
        "cycle_at_80pct",
        "Q_dis@5",
        "CE@5",
        "V_mean_dis@5",
        "EE@5",
        "retention@5",
        "CE_mean_1to5",
        "Q_dis@3",
        "CE@3",
        "V_mean_dis@3",
        "EE@3",
        "retention@3",
        "CE_mean_1to3",
    ]


def test_retention_threshold_parametrized_column_name() -> None:
    """``retention_threshold=0.5`` yields a ``cycle_at_50pct`` column."""
    cell = _linear_cell(
        "fade",
        cycles_q_ch={1: 1000.0, 2: 1000.0, 3: 1000.0},
        cycles_q_dis={1: 1000.0, 2: 700.0, 3: 400.0},
    )
    tbl = stat_table([cell], target_cycles=(1,), retention_threshold=0.5)
    assert "cycle_at_50pct" in tbl.columns
    assert "cycle_at_80pct" not in tbl.columns
    # 0.5 * 1000 = 500. q_dis drops below 500 first at cycle 3 (= 400).
    np.testing.assert_allclose(tbl.loc["fade", "cycle_at_50pct"], 3.0, atol=1e-9)


@pytest.mark.parametrize(
    ("threshold", "expected_pct"),
    [
        # 0.29 * 100 == 28.999999999999996 under IEEE-754
        # 0.58 * 100 == 57.99999999999999
        # int(...) would truncate to 28 / 57 — regression guard.
        (0.29, 29),
        (0.58, 58),
        (0.80, 80),
    ],
)
def test_retention_threshold_column_name_rounds_not_truncates(
    threshold: float, expected_pct: int
) -> None:
    """Float-imprecise thresholds round to nearest integer, not truncate.

    Regression for the IEEE-754 trap in ``int(retention_threshold * 100)``:
    callers who build thresholds programmatically (e.g. ``1.0 - fade_pct``)
    would otherwise receive a silently-mislabelled column.
    """
    cell = _linear_cell(
        "fade",
        cycles_q_ch={1: 1000.0, 2: 1000.0},
        cycles_q_dis={1: 1000.0, 2: 100.0},
    )
    tbl = stat_table([cell], target_cycles=(1,), retention_threshold=threshold)
    expected_column = f"cycle_at_{expected_pct}pct"
    assert expected_column in tbl.columns, (
        f"Expected {expected_column!r} for threshold={threshold}; got columns={list(tbl.columns)}"
    )
    # The ``int()`` regression would produce the truncated-down label (e.g.
    # 0.29 → ``cycle_at_28pct``). Assert it's absent — makes the intent
    # of the test explicit even if the ``assert ... in columns`` above
    # already covers it by omission.
    wrong_column = f"cycle_at_{expected_pct - 1}pct"
    assert wrong_column not in tbl.columns, (
        f"Truncation-regression label {wrong_column!r} must not appear"
    )


def test_retention_at_cycle_1_is_100_exactly() -> None:
    """``retention@1`` must be 100.0 exactly (no floating-point drift).

    Guards against a regression that would compute ``q_dis[1]/q_dis[1]`` via
    the general formula and hit a sub-ULP error — harmless numerically, but
    visible in CSV exports and annoying to downstream consumers comparing
    equality.
    """
    cell = _linear_cell(
        "solo",
        cycles_q_ch={1: 1000.0},
        cycles_q_dis={1: 990.0},
    )
    tbl = stat_table([cell], target_cycles=(1,))
    assert tbl.loc["solo", "retention@1"] == 100.0


def test_q_dis_first_zero_yields_nan_retention_and_cycle_at() -> None:
    """Literal ``q_dis[1] == 0.0`` (non-NaN) triggers the zero-denominator guard.

    Distinguishes from the NaN-denominator path: a cell where cycle 1
    *physically* went through a discharge segment that accumulated zero
    capacity (e.g. a mistimed test, a misconfigured mass, or a genuine
    dud first cycle) is a legitimate-shaped input, not missing data.
    Retention / fade must still be NaN (can't divide by zero), but the
    rest of the row must populate normally.
    """
    rows: list[tuple[int, str, str, float, float]] = []
    v_ch = np.linspace(3.0, 4.2, 50)
    q_ch = np.linspace(0.0, 1000.0, 50)
    # Cycle 1: charge fine, but discharge Q stays at 0 throughout.
    rows.extend((1, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch))
    v_dis = np.linspace(4.2, 3.0, 50)
    rows.extend((1, "1", "放電", float(v), 0.0) for v in v_dis)
    # Cycle 2: normal charge + discharge.
    rows.extend((2, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch))
    q_dis_2 = np.linspace(0.0, 500.0, 50)
    rows.extend((2, "1", "放電", float(v), float(q)) for v, q in zip(v_dis, q_dis_2))
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    cell = Cell(name="zero_dis_1", mass_g=0.001, raw_df=raw)
    tbl = stat_table([cell], target_cycles=(1, 2))

    # Zero-denominator guard fires: fade + all retentions NaN.
    assert pd.isna(tbl.loc["zero_dis_1", "cycle_at_80pct"])
    assert pd.isna(tbl.loc["zero_dis_1", "retention@1"])
    assert pd.isna(tbl.loc["zero_dis_1", "retention@2"])
    # Q_dis@1 is literally 0.0, not NaN — the segment existed.
    np.testing.assert_allclose(tbl.loc["zero_dis_1", "Q_dis@1"], 0.0, atol=1e-12)
    # Cycle 2 remains functional.
    np.testing.assert_allclose(tbl.loc["zero_dis_1", "Q_dis@2"], 500.0, atol=1e-9)


def test_q_dis_first_missing_yields_nan_retention_and_cycle_at() -> None:
    """When cycle 1 has no discharge (q_dis[1] NaN), retention + fade are NaN."""
    # Build a cell where cycle 1 has only charge (no discharge).
    rows: list[tuple[int, str, str, float, float]] = []
    v_ch = np.linspace(3.0, 4.2, 50)
    q_ch = np.linspace(0.0, 1000.0, 50)
    rows.extend((1, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch))
    # Cycle 2: full charge + discharge
    rows.extend((2, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch))
    v_dis = np.linspace(4.2, 3.0, 50)
    q_dis = np.linspace(0.0, 900.0, 50)
    rows.extend((2, "1", "放電", float(v), float(q)) for v, q in zip(v_dis, q_dis))
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    cell = Cell(name="no_dis_1", mass_g=0.001, raw_df=raw)
    tbl = stat_table([cell], target_cycles=(1, 2))

    # q_dis[1] is NaN → fade + all retentions are NaN
    assert pd.isna(tbl.loc["no_dis_1", "cycle_at_80pct"])
    assert pd.isna(tbl.loc["no_dis_1", "retention@1"])
    assert pd.isna(tbl.loc["no_dis_1", "retention@2"])
    # Cycle-2 direct look-ups still work
    np.testing.assert_allclose(tbl.loc["no_dis_1", "Q_dis@2"], 900.0, atol=1e-9)


def test_multi_cell_rows_independent() -> None:
    """Two cells in one call must yield two rows with independent values."""
    a = _linear_cell(
        "a",
        cycles_q_ch={1: 1000.0, 2: 1000.0},
        cycles_q_dis={1: 990.0, 2: 990.0},
    )
    b = _linear_cell(
        "b",
        cycles_q_ch={1: 500.0, 2: 500.0},
        cycles_q_dis={1: 450.0, 2: 300.0},
    )
    tbl = stat_table([a, b], target_cycles=(1, 2))
    assert tbl.index.tolist() == ["a", "b"]
    # Cell b's cycle 2 retention = 100 * 300/450 ≈ 66.67%
    np.testing.assert_allclose(tbl.loc["b", "retention@2"], 100.0 * 300.0 / 450.0, atol=1e-9)
    # Cell a's cycle 2 retention = 100% (same q_dis)
    np.testing.assert_allclose(tbl.loc["a", "retention@2"], 100.0, atol=1e-9)


def test_two_point_segment_uses_trapezoidal_fallback() -> None:
    """2-point segment → trapezoid integral (not NaN, not Simpson error).

    Hand-constructed raw_df with exactly 2 points in the cycle-1 discharge
    segment. The trapezoidal integral of V with the x-axis being Q is
    ``(V1 + V2)/2 * (Q2 - Q1)`` → mean V is exactly the arithmetic mean.
    """
    rows = [
        # Cycle 1 charge: full well-formed ramp (3+ points so simpson works).
        (1, "1", "充電", 3.0, 0.0),
        (1, "1", "充電", 3.6, 500.0),
        (1, "1", "充電", 4.2, 1000.0),
        # Cycle 1 discharge: exactly 2 distinct points → trapezoid fallback.
        (1, "1", "放電", 4.2, 0.0),
        (1, "1", "放電", 3.0, 990.0),
    ]
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    cell = Cell(name="tiny", mass_g=0.001, raw_df=raw)
    tbl = stat_table([cell], target_cycles=(1,))

    # Arithmetic-mean V on the 2-point trapezoid: (4.2 + 3.0) / 2 = 3.6
    np.testing.assert_allclose(tbl.loc["tiny", "V_mean_dis@1"], 3.6, atol=1e-9)
    # EE = 100 * ∫V dQ_dis / ∫V dQ_ch
    # ∫V dQ_ch via simpson on (V=3.0/3.6/4.2, Q=0/500/1000) — with uniform Q
    # spacing, simpson reduces to (Q_step/3) * (V[0] + 4 V[1] + V[2])
    # = (500/3) * (3.0 + 14.4 + 4.2) = (500/3) * 21.6 = 3600.
    # ∫V dQ_dis via trapezoid = 0.5 * (4.2 + 3.0) * 990 = 3.6 * 990 = 3564.
    # EE = 100 * 3564 / 3600 = 99.0.
    np.testing.assert_allclose(tbl.loc["tiny", "EE@1"], 99.0, atol=1e-9)


def test_single_point_segment_yields_nan_integrals() -> None:
    """Segment with only 1 distinct Q → V_mean_dis and EE are NaN."""
    rows = [
        # Cycle 1 charge: well-formed.
        (1, "1", "充電", 3.0, 0.0),
        (1, "1", "充電", 3.6, 500.0),
        (1, "1", "充電", 4.2, 1000.0),
        # Cycle 1 discharge: 1 row. Q_dis from cap_df is 500 (the single max).
        (1, "1", "放電", 3.7, 500.0),
    ]
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    cell = Cell(name="degen", mass_g=0.001, raw_df=raw)
    tbl = stat_table([cell], target_cycles=(1,))

    assert pd.isna(tbl.loc["degen", "V_mean_dis@1"])
    assert pd.isna(tbl.loc["degen", "EE@1"])
    # Direct look-ups survive.
    np.testing.assert_allclose(tbl.loc["degen", "Q_dis@1"], 500.0, atol=1e-9)


def test_column_lang_en_input_produces_english_output() -> None:
    """EN-column cell yields the same EN-fixed output columns.

    ``column_lang`` selects which raw-frame labels to read. The output
    schema is EN-fixed regardless, so the column list is identical to the
    JA-input case.
    """
    rows = [
        (1, "1", "充電", 3.0, 0.0),
        (1, "1", "充電", 4.2, 1000.0),
        (1, "1", "放電", 4.2, 0.0),
        (1, "1", "放電", 3.0, 990.0),
    ]
    raw = pd.DataFrame(rows, columns=["cycle", "mode", "state", "voltage", "capacity"])
    cell = Cell(name="en_cell", mass_g=0.001, raw_df=raw, column_lang="en")
    tbl = stat_table([cell], target_cycles=(1,))

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
    np.testing.assert_allclose(tbl.loc["en_cell", "Q_dis@1"], 990.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["en_cell", "V_mean_dis@1"], 3.6, atol=1e-9)


def test_all_nan_cap_df_yields_all_nan_row() -> None:
    """A cell whose chdis is structurally empty produces a row of NaN.

    Build a raw frame with only rest rows → chdis filters everything out →
    cap_df is empty → every stat column is NaN, but the cell still shows up
    as a row so downstream consumers aren't silently missing it.
    """
    rows = [
        (1, "1", "休止", 3.0, 0.0),
        (1, "1", "休止", 3.0, 0.0),
    ]
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    cell = Cell(name="empty", mass_g=0.001, raw_df=raw)
    tbl = stat_table([cell], target_cycles=(1, 2))

    assert "empty" in tbl.index
    # Every column is NaN (including Q_dis_max — empty .max() → NaN).
    assert tbl.loc["empty"].isna().all()


def test_end_to_end_with_cell_from_dir(make_cell_dir: Callable[..., Path]) -> None:
    """Smoke test via ``Cell.from_dir`` — exercises the on-disk read path.

    The shared fixture has a single cycle with q_ch = q_dis = 1000 mAh/g
    and V ranging 3.50→3.60 V on charge, 3.40→3.20 V on discharge. With
    ``target_cycles=(1,)`` we can pin the whole row analytically.
    """
    cell_dir = make_cell_dir("renzoku")
    cell = Cell.from_dir(cell_dir)
    tbl = stat_table([cell], target_cycles=(1,))

    # One row keyed by directory name.
    assert tbl.index.tolist() == ["cell_A"]
    np.testing.assert_allclose(tbl.loc["cell_A", "Q_dis_max"], 1000.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["cell_A", "Q_dis@1"], 1000.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["cell_A", "CE@1"], 100.0, atol=1e-9)
    np.testing.assert_allclose(tbl.loc["cell_A", "retention@1"], 100.0, atol=1e-9)
    # Mean V on linear ramp: (3.50 + 3.60)/2 = 3.55 for charge,
    # (3.40 + 3.20)/2 = 3.30 for discharge.
    np.testing.assert_allclose(tbl.loc["cell_A", "V_mean_dis@1"], 3.30, atol=1e-9)
    # EE = 100 * (3.30 * 1000) / (3.55 * 1000) = 100 * 3.30 / 3.55
    np.testing.assert_allclose(tbl.loc["cell_A", "EE@1"], 100.0 * 3.30 / 3.55, atol=1e-9)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, -1e-12, 1.0 + 1e-12])
def test_retention_threshold_out_of_range_raises(bad: float) -> None:
    """``retention_threshold`` outside the open interval (0, 1) raises ValueError.

    Closed-interval endpoints (0.0 and 1.0) also raise — 0.0 is a
    degenerate "never fade" threshold and 1.0 trivially fires on the
    first sub-unity cycle, neither of which is a meaningful caller
    intent. Callers expressing those extremes should pass a default
    threshold and filter the output themselves.
    """
    cell = _linear_cell(
        "v",
        cycles_q_ch={1: 1000.0},
        cycles_q_dis={1: 990.0},
    )
    with pytest.raises(ValueError, match="retention_threshold"):
        stat_table([cell], target_cycles=(1,), retention_threshold=bad)


@pytest.mark.parametrize("dup", [(1, 1, 2), (2, 1, 1), (1, 2, 1)])
def test_duplicate_target_cycles_raises(dup: tuple[int, ...]) -> None:
    """``target_cycles`` with duplicates raises — regardless of position.

    Pandas will happily emit two columns named ``Q_dis@1``; downstream
    consumers (and ``.to_csv``) then behave surprisingly. Surface the
    sloppy caller intent at the entry point rather than shipping a
    malformed frame. The parametrize pins position-independence: the
    duplicate can sit at start, end, or sandwiched — all fire.
    """
    cell = _linear_cell(
        "v",
        cycles_q_ch={1: 1000.0, 2: 900.0},
        cycles_q_dis={1: 990.0, 2: 880.0},
    )
    with pytest.raises(ValueError, match="target_cycles"):
        stat_table([cell], target_cycles=dup)


@pytest.mark.parametrize("layout", ["renzoku", "renzoku_py", "raw_6digit"])
def test_layouts_produce_equivalent_stats(make_cell_dir: Callable[..., Path], layout: str) -> None:
    """All three on-disk layouts must yield the same stat_table values.

    This is a consistency test across reader paths — not an algorithmic
    test. ``raw_6digit`` derives capacity from elapsed * current / mass,
    introducing float rounding, so the tolerance is relaxed.
    """
    cell = Cell.from_dir(make_cell_dir(layout))
    tbl = stat_table([cell], target_cycles=(1,))
    np.testing.assert_allclose(tbl.loc["cell_A", "Q_dis@1"], 1000.0, rtol=1e-6)
    np.testing.assert_allclose(tbl.loc["cell_A", "V_mean_dis@1"], 3.30, rtol=1e-6)
    np.testing.assert_allclose(tbl.loc["cell_A", "EE@1"], 100.0 * 3.30 / 3.55, rtol=1e-6)
