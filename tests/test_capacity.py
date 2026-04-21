"""Tests for :mod:`toyo_battery.core.capacity`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from toyo_battery.core.capacity import get_cap_df
from toyo_battery.core.cell import Cell


def _chdis(
    segments: dict[tuple[int, str], list[tuple[float, float]]],
    *,
    quantity_cap: str = "電気量",
    quantity_v: str = "電圧",
) -> pd.DataFrame:
    """Build a minimal chdis_df from ``(cycle, side) -> [(voltage, capacity)]``.

    Output mirrors the shape produced by :func:`toyo_battery.core.chdis.get_chdis_df`:
    a 3-level column MultiIndex ``(cycle, side, quantity)`` with NaN-padded
    rows so all segments share a row axis.
    """
    pieces: dict[tuple[int, str], pd.DataFrame] = {}
    for (cycle, side), rows in segments.items():
        if rows:
            df = pd.DataFrame(rows, columns=[quantity_v, quantity_cap])
        else:
            df = pd.DataFrame(columns=[quantity_v, quantity_cap])
        pieces[(cycle, side)] = df
    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
    out.columns = out.columns.set_names(["cycle", "side", "quantity"])
    return out


def _empty_chdis(*, lang: str = "ja") -> pd.DataFrame:
    """Return an empty chdis_df with the expected 3-level MultiIndex."""
    # Match the shape produced by chdis._empty_result.
    _ = lang  # Empty frames carry no quantity labels so lang is not material.
    idx = pd.MultiIndex.from_tuples([], names=["cycle", "side", "quantity"])
    return pd.DataFrame(columns=idx)


def test_three_cycle_monotone_pinned() -> None:
    """3-cycle fixture with distinct q_ch / q_dis per cycle.

    Pinned values: q_ch=[500, 480, 460], q_dis=[495, 475, 455],
    ce = 100 * q_dis / q_ch per cycle.
    """
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 500.0)],
        (1, "dis"): [(3.60, 0.0), (3.40, 495.0)],
        (2, "ch"): [(3.50, 0.0), (3.60, 480.0)],
        (2, "dis"): [(3.60, 0.0), (3.40, 475.0)],
        (3, "ch"): [(3.50, 0.0), (3.60, 460.0)],
        (3, "dis"): [(3.60, 0.0), (3.40, 455.0)],
    }
    cap = get_cap_df(_chdis(segments))

    assert list(cap.columns) == ["q_ch", "q_dis", "ce"]
    assert cap.index.name == "cycle"
    assert cap.index.tolist() == [1, 2, 3]
    assert cap["q_ch"].tolist() == [500.0, 480.0, 460.0]
    assert cap["q_dis"].tolist() == [495.0, 475.0, 455.0]
    np.testing.assert_allclose(
        cap["ce"].to_numpy(),
        np.array([100.0 * 495.0 / 500.0, 100.0 * 475.0 / 480.0, 100.0 * 455.0 / 460.0]),
    )
    # All columns float64
    for col in ("q_ch", "q_dis", "ce"):
        assert cap[col].dtype == np.float64


def test_cycle_with_only_ch_side_has_nan_dis_and_ce() -> None:
    """A cycle without a dis segment has q_dis=NaN, ce=NaN, row preserved."""
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 500.0)],
        (1, "dis"): [(3.60, 0.0), (3.40, 495.0)],
        (2, "ch"): [(3.50, 0.0), (3.60, 480.0)],
        # Note: no (2, "dis")
    }
    cap = get_cap_df(_chdis(segments))

    assert cap.index.tolist() == [1, 2]
    # Cycle 2 dis missing → NaN
    assert cap.loc[2, "q_ch"] == 480.0
    assert pd.isna(cap.loc[2, "q_dis"])
    assert pd.isna(cap.loc[2, "ce"])
    # Cycle 1 intact
    assert cap.loc[1, "q_ch"] == 500.0
    assert cap.loc[1, "q_dis"] == 495.0


def test_cycle_with_only_dis_side_has_nan_ch_and_ce() -> None:
    """Symmetric: a cycle without a ch segment has q_ch=NaN, ce=NaN."""
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 500.0)],
        (1, "dis"): [(3.60, 0.0), (3.40, 495.0)],
        (2, "dis"): [(3.60, 0.0), (3.40, 475.0)],
    }
    cap = get_cap_df(_chdis(segments))

    assert cap.index.tolist() == [1, 2]
    assert pd.isna(cap.loc[2, "q_ch"])
    assert cap.loc[2, "q_dis"] == 475.0
    assert pd.isna(cap.loc[2, "ce"])


def test_q_ch_zero_yields_nan_ce_not_inf() -> None:
    """q_ch == 0 → ce = NaN (neither inf nor raise)."""
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 0.0)],  # zero-capacity charge
        (1, "dis"): [(3.60, 0.0), (3.40, 10.0)],
    }
    cap = get_cap_df(_chdis(segments))

    assert cap.loc[1, "q_ch"] == 0.0
    assert cap.loc[1, "q_dis"] == 10.0
    assert pd.isna(cap.loc[1, "ce"])
    assert not np.isinf(cap.loc[1, "ce"])


def test_empty_chdis_returns_empty_frame_with_correct_columns_and_dtypes() -> None:
    """Empty chdis_df → empty cap_df with correct schema."""
    cap = get_cap_df(_empty_chdis())
    assert cap.empty
    assert list(cap.columns) == ["q_ch", "q_dis", "ce"]
    assert cap.index.name == "cycle"
    # Index dtype is int for consistency with the non-empty case.
    assert cap.index.dtype == np.int64
    for col in ("q_ch", "q_dis", "ce"):
        assert cap[col].dtype == np.float64


def test_column_lang_en_input_produces_same_english_output() -> None:
    """EN-labeled input quantity → same output shape & column names."""
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 500.0)],
        (1, "dis"): [(3.60, 0.0), (3.40, 495.0)],
        (2, "ch"): [(3.50, 0.0), (3.60, 480.0)],
        (2, "dis"): [(3.60, 0.0), (3.40, 475.0)],
    }
    chdis_en = _chdis(segments, quantity_cap="capacity", quantity_v="voltage")
    cap = get_cap_df(chdis_en, column_lang="en")

    assert list(cap.columns) == ["q_ch", "q_dis", "ce"]
    assert cap.index.name == "cycle"
    assert cap["q_ch"].tolist() == [500.0, 480.0]
    assert cap["q_dis"].tolist() == [495.0, 475.0]


def test_missing_required_quantity_raises_helpful_error() -> None:
    """Missing capacity label in the quantity level raises KeyError."""
    # Frame with 'voltage' only, no 電気量 / capacity.
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 500.0)],
    }
    chdis = _chdis(segments)
    # Rename the quantity level to something else so capacity is absent.
    chdis.columns = pd.MultiIndex.from_tuples(
        [(c, s, "まちがい") for (c, s, _q) in chdis.columns],
        names=["cycle", "side", "quantity"],
    )
    with pytest.raises(KeyError, match="電気量"):
        get_cap_df(chdis)


def test_wrong_column_lang_raises_with_context() -> None:
    """Passing JA-labeled chdis with column_lang='en' surfaces a helpful error."""
    segments: dict[tuple[int, str], list[tuple[float, float]]] = {
        (1, "ch"): [(3.50, 0.0), (3.60, 500.0)],
        (1, "dis"): [(3.60, 0.0), (3.40, 495.0)],
    }
    chdis = _chdis(segments)  # default JA quantity labels
    with pytest.raises(KeyError, match="column_lang='en'"):
        get_cap_df(chdis, column_lang="en")


def test_cell_cap_df_cached_and_columns(make_cell_dir: Callable[..., Path]) -> None:
    """Cell.cap_df integrates the capacity function and is cached."""
    cell = Cell.from_dir(make_cell_dir("renzoku"))
    first = cell.cap_df
    second = cell.cap_df
    assert first is second
    assert (cell.cap_df.columns == ["q_ch", "q_dis", "ce"]).all()
    assert cell.cap_df.index.name == "cycle"


def test_cell_cap_df_values_from_shared_fixture(make_cell_dir: Callable[..., Path]) -> None:
    """Shared fixture: one cycle, q_ch=q_dis=1000, ce=100%.

    The fixture's single cycle has elapsed=3600s, current=1mA, mass=1mg
    → capacity = 1·3600/(3600·0.001·1000) · 1000 = 1000 mAh/g at each
    segment end; the 電気量 column is pre-filled with exactly these values
    (renzoku) or derived from elapsed·current/mass (raw_6digit), so both
    paths produce q_ch = q_dis = 1000 and ce = 100.
    """
    cell = Cell.from_dir(make_cell_dir("renzoku"))
    cap = cell.cap_df
    assert cap.index.tolist() == [1]
    np.testing.assert_allclose(cap.loc[1, "q_ch"], 1000.0)
    np.testing.assert_allclose(cap.loc[1, "q_dis"], 1000.0)
    np.testing.assert_allclose(cap.loc[1, "ce"], 100.0)


def test_cell_cap_df_respects_column_lang(make_cell_dir: Callable[..., Path]) -> None:
    """Output column names are EN-fixed even when input column_lang='en'."""
    cell = Cell.from_dir(make_cell_dir("renzoku"), column_lang="en")
    cap = cell.cap_df
    assert list(cap.columns) == ["q_ch", "q_dis", "ce"]
