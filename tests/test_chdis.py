"""Tests for :mod:`echemplot.core.chdis`."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from echemplot.core import DataIntegrityWarning
from echemplot.core.cell import Cell
from echemplot.core.chdis import get_chdis_df


def _raw(
    rows: list[tuple[int, str, float, float]],
    *,
    lang: str = "ja",
) -> pd.DataFrame:
    """Build a minimal canonical raw frame. Row tuple = (cycle, state, voltage, capacity)."""
    if lang == "ja":
        columns = ["サイクル", "状態", "電圧", "電気量"]
    else:
        columns = ["cycle", "state", "voltage", "capacity"]
    return pd.DataFrame(rows, columns=columns)


def test_single_cycle_charge_then_discharge() -> None:
    df = _raw(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 500.0),
            (1, "休止", 3.61, 500.0),
            (1, "放電", 3.60, 0.0),
            (1, "放電", 3.40, 500.0),
        ]
    )
    out = get_chdis_df(df)
    assert list(out.columns.names) == ["cycle", "side", "quantity"]
    assert set(out.columns.get_level_values("cycle")) == {1}
    assert set(out.columns.get_level_values("side")) == {"ch", "dis"}

    ch_cap = out[(1, "ch", "電気量")].dropna().tolist()
    assert ch_cap == [0.0, 500.0]
    dis_cap = out[(1, "dis", "電気量")].dropna().tolist()
    assert dis_cap == [0.0, 500.0]


def test_rest_rows_filtered_out() -> None:
    df = _raw(
        [
            (1, "休止", 3.00, 0.0),
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 500.0),
            (1, "休止", 3.61, 500.0),
            (1, "放電", 3.60, 0.0),
            (1, "放電", 3.40, 500.0),
            (1, "休止", 3.40, 500.0),
        ]
    )
    out = get_chdis_df(df)
    assert out[(1, "ch", "電気量")].notna().sum() == 2
    assert out[(1, "dis", "電気量")].notna().sum() == 2


def test_capacity_reversal_rows_dropped() -> None:
    df = _raw(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 500.0),
            (1, "充電", 3.55, 400.0),  # reversal — drop
            (1, "充電", 3.65, 600.0),
        ]
    )
    out = get_chdis_df(df)
    assert out[(1, "ch", "電気量")].dropna().tolist() == [0.0, 500.0, 600.0]
    assert out[(1, "ch", "電圧")].dropna().tolist() == [3.50, 3.60, 3.65]


def test_capacity_step_boundary_within_charge_state_dropped() -> None:
    """A sustained capacity reset within the same 充電 group (e.g. the
    CC→CV sub-step boundary in raw-6-digit format, where 経過時間[Sec]
    restarts) must be dropped wholesale — including any row in the new
    sub-step that happens to inch up versus its immediate predecessor.
    Regression for the V-Q loop-back artifact reported with the negative
    electrode's first Li-insertion curve."""
    df = _raw(
        [
            (1, "充電", 0.0023, 212.58808),
            (1, "充電", 0.0020, 213.45448),
            (1, "充電", 0.0017, 214.26966),
            (1, "充電", 0.0014, 215.16914),
            (1, "充電", 0.0015, 19.28384),
            (1, "充電", 0.0016, 15.97689),
            (1, "充電", 0.0016, 14.97401),
            (1, "充電", 0.0016, 14.64183),
            (1, "充電", 0.0016, 12.49434),
            (1, "充電", 0.0016, 12.13643),
            (1, "充電", 0.0016, 12.14260),  # diff>0 vs predecessor, but < running max
        ]
    )
    out = get_chdis_df(df)
    cap = out[(1, "ch", "電気量")].dropna().tolist()
    assert cap == [212.58808, 213.45448, 214.26966, 215.16914]


def test_discharge_first_triggers_state_swap() -> None:
    """If the first cycle starts with 放電, all state labels are swapped so
    the cell is treated as charge-first."""
    df = _raw(
        [
            (1, "放電", 3.60, 0.0),
            (1, "放電", 3.40, 500.0),
            (1, "充電", 3.40, 0.0),
            (1, "充電", 3.60, 500.0),
        ]
    )
    out = get_chdis_df(df)
    # After the swap, the original 放電 rows are treated as "ch" (charge).
    ch_v = out[(1, "ch", "電圧")].dropna().tolist()
    assert ch_v == [3.60, 3.40]
    dis_v = out[(1, "dis", "電圧")].dropna().tolist()
    assert dis_v == [3.40, 3.60]


def test_multi_cycle_multiindex_shape() -> None:
    rows = []
    for cycle in (1, 2, 3):
        rows.extend(
            [
                (cycle, "充電", 3.50, 0.0),
                (cycle, "充電", 3.60, 500.0),
                (cycle, "放電", 3.60, 0.0),
                (cycle, "放電", 3.40, 500.0 - cycle * 10),  # mild fade
            ]
        )
    out = get_chdis_df(_raw(rows))

    assert sorted(set(out.columns.get_level_values("cycle"))) == [1, 2, 3]
    for cycle in (1, 2, 3):
        assert (cycle, "ch", "電気量") in out.columns
        assert (cycle, "dis", "電気量") in out.columns
        assert (cycle, "ch", "電圧") in out.columns
        assert (cycle, "dis", "電圧") in out.columns


def test_column_lang_en() -> None:
    """EN-mode input carries EN state values (``charge``/``discharge``).

    The reader translates state values when ``column_lang='en'`` is
    requested (issue #94), so chdis must filter on EN literals in EN
    mode. Passing JA state values with EN columns (the legacy behavior)
    is no longer supported — see ``test_en_mode_filters_on_en_state_labels``
    below for the active EN contract.
    """
    df = _raw(
        [
            (1, "charge", 3.50, 0.0),
            (1, "charge", 3.60, 500.0),
            (1, "discharge", 3.60, 0.0),
            (1, "discharge", 3.40, 500.0),
        ],
        lang="en",
    )
    out = get_chdis_df(df, column_lang="en")
    assert set(out.columns.get_level_values("quantity")) == {"capacity", "voltage"}
    assert (1, "ch", "capacity") in out.columns
    assert out[(1, "dis", "voltage")].dropna().tolist() == [3.60, 3.40]


def test_en_mode_filters_on_en_state_labels() -> None:
    """In EN mode, chdis filters on ``charge``/``discharge`` and ignores
    ``rest``/``charge_rest``/``discharge_rest``/``abort`` rows, mirroring
    the JA-mode 充電/放電 filter."""
    df = _raw(
        [
            (1, "charge", 3.50, 0.0),
            (1, "charge", 3.60, 500.0),
            (1, "charge_rest", 3.61, 500.0),
            (1, "discharge", 3.60, 0.0),
            (1, "discharge", 3.40, 500.0),
            (1, "discharge_rest", 3.40, 500.0),
            (1, "abort", 3.40, 500.0),
        ],
        lang="en",
    )
    out = get_chdis_df(df, column_lang="en")
    assert out[(1, "ch", "capacity")].notna().sum() == 2
    assert out[(1, "dis", "capacity")].notna().sum() == 2


def test_empty_frame_returns_empty_multiindex() -> None:
    out = get_chdis_df(_raw([]))
    assert out.empty
    assert list(out.columns.names) == ["cycle", "side", "quantity"]


def test_only_rest_rows_returns_empty() -> None:
    df = _raw(
        [
            (1, "休止", 3.00, 0.0),
            (1, "休止", 3.01, 0.0),
        ]
    )
    out = get_chdis_df(df)
    assert out.empty


def test_cell_chdis_df_is_cached(make_cell_dir: Callable[..., Path]) -> None:
    """Cell.chdis_df returns equal-but-distinct frames; computation is cached."""
    cell_dir = make_cell_dir("renzoku")
    cell = Cell.from_dir(cell_dir)

    first = cell.chdis_df
    second = cell.chdis_df
    # Defensive-copy contract: each access returns a fresh frame, but the
    # underlying cached computation is reused so the contents are identical.
    assert first is not second
    pd.testing.assert_frame_equal(first, second)
    assert (1, "ch", "電気量") in first.columns
    assert (1, "dis", "電気量") in first.columns


def test_cell_chdis_df_respects_column_lang(make_cell_dir: Callable[..., Path]) -> None:
    cell_dir = make_cell_dir("renzoku")
    cell = Cell.from_dir(cell_dir, column_lang="en")
    out = cell.chdis_df
    assert set(out.columns.get_level_values("quantity")) == {"capacity", "voltage"}


@pytest.mark.parametrize("layout", ["renzoku", "renzoku_py", "raw_6digit"])
def test_cell_chdis_df_from_all_layouts(make_cell_dir: Callable[..., Path], layout: str) -> None:
    """All three reader layouts produce a 2+2 row ch/dis shape for the shared fixture.

    Each fixture writes exactly 2 charge rows + 1 rest + 2 discharge rows (plus 1
    extra leading/trailing row in some variants). The reversal filter runs on
    ``|電気量|`` as a defensive measure: real TOYO data (all three paths) is
    already monotone non-decreasing per segment, so no rows are spuriously
    dropped, and any hand-crafted signed input would still segment correctly.
    """
    cell = Cell.from_dir(make_cell_dir(layout))
    out = cell.chdis_df
    assert out[(1, "ch", "電気量")].notna().sum() == 2
    assert out[(1, "dis", "電気量")].notna().sum() == 2


def test_missing_required_column_raises_helpful_error() -> None:
    df = pd.DataFrame([(1, "充電", 3.5, 0.0)], columns=["サイクル", "状態", "電圧", "まちがい"])
    with pytest.raises(KeyError, match="電気量"):
        get_chdis_df(df)


def test_wrong_column_lang_raises_with_context() -> None:
    """Passing JA columns with ``column_lang='en'`` surfaces a helpful error."""
    df = _raw([(1, "充電", 3.5, 0.0)], lang="ja")
    with pytest.raises(KeyError, match="column_lang='en'"):
        get_chdis_df(df, column_lang="en")


def test_chdis_warns_on_capacity_reversal_drops() -> None:
    """The running-max filter must surface dropped rows via DataIntegrityWarning.

    Real TOYO data legitimately triggers this warning at CC→CV sub-step
    boundaries; for hand-preprocessed inputs it surfaces previously-silent
    data loss.
    """
    df = _raw(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 500.0),
            (1, "充電", 3.55, 400.0),  # reversal — drop, should warn
            (1, "充電", 3.65, 600.0),
        ]
    )
    with pytest.warns(DataIntegrityWarning, match=r"dropped \d+ rows"):
        out = get_chdis_df(df)
    # Behavior preserved — the warning is informational and does not change shape.
    assert out[(1, "ch", "電気量")].dropna().tolist() == [0.0, 500.0, 600.0]


def test_chdis_silent_on_monotone_data() -> None:
    """Strictly monotone-non-decreasing capacity must not emit DataIntegrityWarning."""
    df = _raw(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.55, 100.0),
            (1, "充電", 3.60, 200.0),
            (1, "充電", 3.65, 300.0),
            (1, "放電", 3.60, 0.0),
            (1, "放電", 3.50, 100.0),
            (1, "放電", 3.40, 200.0),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataIntegrityWarning)
        get_chdis_df(df)
    integrity_warnings = [w for w in caught if issubclass(w.category, DataIntegrityWarning)]
    assert integrity_warnings == []


def test_first_cycle_discharge_swaps_all_cycles_globally() -> None:
    """If cycle 1 starts with 放電, *every* cycle's labels are swapped.

    This is the TOYO half-cell convention. Half-cell data where cycles 2+ are
    also naturally "discharge-first" get normalized consistently. Callers with
    a one-off formation-discharge on cycle 1 and normal cycles 2+ must relabel
    before calling get_chdis_df — this test pins the current semantics so
    regressions are loud.
    """
    rows = []
    for cycle in (1, 2, 3):
        rows.extend(
            [
                (cycle, "放電", 3.60, 0.0),
                (cycle, "放電", 3.40, 500.0),
                (cycle, "充電", 3.40, 0.0),
                (cycle, "充電", 3.60, 500.0),
            ]
        )
    out = get_chdis_df(_raw(rows))

    # All cycles: originally-放電 voltages (3.60, 3.40) land under the "ch" side.
    for cycle in (1, 2, 3):
        assert out[(cycle, "ch", "電圧")].dropna().tolist() == [3.60, 3.40]
        assert out[(cycle, "dis", "電圧")].dropna().tolist() == [3.40, 3.60]


def _raw_with_total_cycle(
    rows: list[tuple[int, int, str, float, float]],
    *,
    lang: str = "ja",
) -> pd.DataFrame:
    """Build a frame with both ``サイクル`` and ``総サイクル``.

    Row tuple = (cycle, total_cycle, state, voltage, capacity).
    """
    if lang == "ja":
        columns = ["サイクル", "総サイクル", "状態", "電圧", "電気量"]
    else:
        columns = ["cycle", "total_cycle", "state", "voltage", "capacity"]
    return pd.DataFrame(rows, columns=columns)


def test_total_cycle_column_supersedes_cycle() -> None:
    """When ``総サイクル`` is present it is preferred over ``サイクル`` for grouping.

    Multi-mode TOYO programs reset ``サイクル`` at every mode boundary while
    ``総サイクル`` keeps counting monotonically. Without this preference, two
    physically distinct cycles that share ``サイクル=1`` collapse into a
    single chdis_df group; preferring ``総サイクル`` keeps them separate.
    """
    df = _raw_with_total_cycle(
        [
            # Mode 1, サイクル=1 → 総サイクル=1
            (1, 1, "充電", 3.50, 0.0),
            (1, 1, "充電", 3.60, 500.0),
            (1, 1, "放電", 3.60, 0.0),
            (1, 1, "放電", 3.40, 500.0),
            # Mode 2 boundary: cycler resets サイクル to 1 again
            # but 総サイクル continues to 2.
            (1, 2, "充電", 3.55, 0.0),
            (1, 2, "充電", 3.65, 600.0),
            (1, 2, "放電", 3.65, 0.0),
            (1, 2, "放電", 3.45, 600.0),
        ]
    )
    out = get_chdis_df(df)
    assert sorted(out.columns.get_level_values("cycle").unique().tolist()) == [1, 2]
    # First mode's run lands under cycle 1 only.
    assert out[(1, "ch", "電気量")].dropna().tolist() == [0.0, 500.0]
    assert out[(1, "dis", "電気量")].dropna().tolist() == [0.0, 500.0]
    # Second mode's run lands under cycle 2 only — no cross-contamination.
    assert out[(2, "ch", "電気量")].dropna().tolist() == [0.0, 600.0]
    assert out[(2, "dis", "電気量")].dropna().tolist() == [0.0, 600.0]


def test_no_total_cycle_falls_back_to_cycle() -> None:
    """Without ``総サイクル`` the per-mode ``サイクル`` is the cycle key (legacy)."""
    df = _raw(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 500.0),
            (2, "充電", 3.55, 0.0),
            (2, "充電", 3.65, 600.0),
        ]
    )
    out = get_chdis_df(df)
    assert sorted(out.columns.get_level_values("cycle").unique().tolist()) == [1, 2]
    assert out[(1, "ch", "電気量")].dropna().tolist() == [0.0, 500.0]
    assert out[(2, "ch", "電気量")].dropna().tolist() == [0.0, 600.0]


def test_multi_run_within_cycle_no_leakage_regression() -> None:
    """Regression for the No6 ``98`` cell artifact (cycle 2 leaking into cycle 1).

    Two charge sub-runs share ``サイクル=1`` because the cycler resets the
    per-mode counter at the mode 1→2 boundary. The second sub-run's tail
    capacity (1100) exceeds the first sub-run's max (1000); under the legacy
    ``サイクル``-only grouping the running-max filter would keep the second
    sub-run's tail and stitch it onto cycle 1's curve as a visible V drop.
    With ``総サイクル`` as the cycle key the sub-runs become cycles 1 and 2
    respectively and no leakage is possible.
    """
    df = _raw_with_total_cycle(
        [
            # Sub-run 1: smooth charge to 1000
            (1, 1, "充電", 3.00, 0.0),
            (1, 1, "充電", 3.50, 500.0),
            (1, 1, "充電", 4.00, 1000.0),
            # Sub-run 2: cycler reset capacity but kept サイクル=1
            (1, 2, "充電", 2.50, 0.0),
            (1, 2, "充電", 3.20, 500.0),
            (1, 2, "充電", 3.70, 1100.0),  # tail exceeds sub-run 1's max
        ]
    )
    out = get_chdis_df(df)
    # Cycle 1 must contain ONLY sub-run 1 (no tail of sub-run 2 leaking in).
    cycle1_cap = out[(1, "ch", "電気量")].dropna().tolist()
    cycle1_v = out[(1, "ch", "電圧")].dropna().tolist()
    assert cycle1_cap == [0.0, 500.0, 1000.0]
    assert cycle1_v == [3.00, 3.50, 4.00]
    # Cycle 2 must contain sub-run 2's full data, untouched.
    cycle2_cap = out[(2, "ch", "電気量")].dropna().tolist()
    cycle2_v = out[(2, "ch", "電圧")].dropna().tolist()
    assert cycle2_cap == [0.0, 500.0, 1100.0]
    assert cycle2_v == [2.50, 3.20, 3.70]


def test_total_cycle_column_supersedes_cycle_en_mode() -> None:
    """EN-mode equivalent of test_total_cycle_column_supersedes_cycle."""
    df = _raw_with_total_cycle(
        [
            (1, 1, "charge", 3.50, 0.0),
            (1, 1, "charge", 3.60, 500.0),
            (1, 2, "charge", 3.55, 0.0),
            (1, 2, "charge", 3.65, 600.0),
        ],
        lang="en",
    )
    out = get_chdis_df(df, column_lang="en")
    assert sorted(out.columns.get_level_values("cycle").unique().tolist()) == [1, 2]
    assert out[(1, "ch", "capacity")].dropna().tolist() == [0.0, 500.0]
    assert out[(2, "ch", "capacity")].dropna().tolist() == [0.0, 600.0]
