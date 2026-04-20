"""Tests for :mod:`toyo_battery.core.chdis`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from toyo_battery.core.cell import Cell
from toyo_battery.core.chdis import get_chdis_df


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
    df = _raw(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 500.0),
            (1, "放電", 3.60, 0.0),
            (1, "放電", 3.40, 500.0),
        ],
        lang="en",
    )
    out = get_chdis_df(df, column_lang="en")
    assert set(out.columns.get_level_values("quantity")) == {"capacity", "voltage"}
    assert (1, "ch", "capacity") in out.columns
    assert out[(1, "dis", "voltage")].dropna().tolist() == [3.60, 3.40]


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
    """Cell.chdis_df is a cached_property — second access returns the same object."""
    cell_dir = make_cell_dir("renzoku", include_capacity_col=True)
    cell = Cell.from_dir(cell_dir)

    first = cell.chdis_df
    second = cell.chdis_df
    assert first is second
    assert (1, "ch", "電気量") in first.columns
    assert (1, "dis", "電気量") in first.columns


def test_cell_chdis_df_respects_column_lang(make_cell_dir: Callable[..., Path]) -> None:
    cell_dir = make_cell_dir("renzoku", include_capacity_col=True)
    cell = Cell.from_dir(cell_dir, column_lang="en")
    out = cell.chdis_df
    assert set(out.columns.get_level_values("quantity")) == {"capacity", "voltage"}


@pytest.mark.parametrize("layout", ["renzoku", "renzoku_py", "raw_6digit"])
def test_cell_chdis_df_from_all_layouts(make_cell_dir: Callable[..., Path], layout: str) -> None:
    """All three reader layouts produce a 2+2 row ch/dis shape for the shared fixture.

    Each fixture writes exactly 2 charge rows + 1 rest + 2 discharge rows (plus 1
    extra leading/trailing row in some variants). The reversal filter runs on
    ``|電気量|`` so the signed-capacity output of the raw-6-digit and
    renzoku-without-precomputed-capacity paths is preserved, not silently dropped.
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
