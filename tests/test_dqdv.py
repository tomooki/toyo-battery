"""Tests for :mod:`toyo_battery.core.dqdv`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from toyo_battery.core.cell import Cell
from toyo_battery.core.chdis import get_chdis_df
from toyo_battery.core.dqdv import get_dqdv_df


def _linear_chdis_ch_only(
    *,
    slope: float = 500.0,
    v_lo: float = 3.0,
    v_hi: float = 4.2,
    n_points: int = 200,
    lang: str = "ja",
) -> pd.DataFrame:
    """Build a ``chdis_df`` with a single cycle of linear Q = slope*(V - v_lo).

    The 2-point reversal filter in chdis is a no-op here because Q is
    monotone non-decreasing along the synthetic ramp.
    """
    v = np.linspace(v_lo, v_hi, n_points)
    q = slope * (v - v_lo)
    rows = [(1, "充電", float(vi), float(qi)) for vi, qi in zip(v, q)]
    if lang == "ja":
        columns = ["サイクル", "状態", "電圧", "電気量"]
    else:
        columns = ["cycle", "state", "voltage", "capacity"]
        # Even with EN column frame, chdis expects JA state strings.
    return pd.DataFrame(rows, columns=columns)


def test_linear_ramp_yields_constant_derivative() -> None:
    """Q = slope * (V - V_lo) → dQ/dV ≈ slope everywhere (within savgol bounds)."""
    slope = 500.0
    raw = _linear_chdis_ch_only(slope=slope, v_lo=3.0, v_hi=4.2, n_points=200)
    chdis = get_chdis_df(raw)

    out = get_dqdv_df(chdis)
    dqdv_series = out[(1, "ch", "dQ/dV")].dropna()
    # Interior samples (trim edges where savgol_filter mode="nearest"
    # mildly distorts) should recover the analytical slope.
    interior = dqdv_series.iloc[5:-5]
    np.testing.assert_allclose(np.median(interior), slope, rtol=0.05)


def test_segment_with_single_distinct_voltage_emits_nan_column() -> None:
    """A segment with <2 distinct V values must yield NaN columns (no exception)."""
    # Cycle 1 ch has 1 row only → <2 distinct V. Cycle 1 dis is a well-formed
    # ramp so the function has something to produce on the other side.
    v = np.linspace(3.0, 4.0, 200)
    q = 400.0 * (v - 3.0)
    rows: list[tuple[int, str, float, float]] = [(1, "充電", 3.5, 0.0)]
    rows.extend((1, "放電", float(vi), float(qi)) for vi, qi in zip(v, q))
    raw = pd.DataFrame(rows, columns=["サイクル", "状態", "電圧", "電気量"])
    chdis = get_chdis_df(raw)

    out = get_dqdv_df(chdis)
    assert (1, "ch", "電圧") in out.columns
    assert (1, "ch", "dQ/dV") in out.columns
    # ch has a single sample → chdis keeps it (no reversal to filter), but
    # after dedup there is only 1 distinct V → the whole column is NaN.
    assert out[(1, "ch", "電圧")].isna().all()
    assert out[(1, "ch", "dQ/dV")].isna().all()
    # dis retains real values.
    assert out[(1, "dis", "dQ/dV")].notna().any()


def test_column_multiindex_level_names() -> None:
    raw = _linear_chdis_ch_only()
    chdis = get_chdis_df(raw)
    out = get_dqdv_df(chdis)
    assert list(out.columns.names) == ["cycle", "side", "quantity"]


def test_inter_num_controls_row_axis_length() -> None:
    """With ``inter_num=50`` and a 1.2 V range, ipnum ~= 60 per segment; pinning
    the observable sanity bounds (>= 2, <= 50 when V-range < 1.0)."""
    # V range 3.8-4.2 → 0.4 V window. inter_num * range = 50 * 0.4 = 20.
    raw = _linear_chdis_ch_only(v_lo=3.8, v_hi=4.2, n_points=100)
    chdis = get_chdis_df(raw)
    out = get_dqdv_df(chdis, inter_num=50, window_length=11, polyorder=2)

    # Row count is the global max across segments. For a single ch segment
    # with V range 0.4 and inter_num=50, ipnum = 20 so row axis has exactly 20.
    assert 2 <= len(out) <= 50
    non_nan = out[(1, "ch", "電圧")].dropna()
    assert 2 <= len(non_nan) <= 50


def test_column_lang_en_uses_english_quantity_names() -> None:
    raw = _linear_chdis_ch_only(lang="en")
    chdis = get_chdis_df(raw, column_lang="en")
    out = get_dqdv_df(chdis, column_lang="en")
    qset = set(out.columns.get_level_values("quantity"))
    assert qset == {"voltage", "dq_dv"}
    # And the JA names are absent.
    assert "電圧" not in qset
    assert "dQ/dV" not in qset


def test_cell_dqdv_df_is_cached(make_cell_dir: Callable[..., Path]) -> None:
    """Cell.dqdv_df is cached_property — second access returns the same object."""
    cell_dir = make_cell_dir("renzoku")
    cell = Cell.from_dir(cell_dir)

    first = cell.dqdv_df
    second = cell.dqdv_df
    assert first is second
    # The synthetic renzoku fixture has a very small voltage span (0.1 V for
    # ch, 0.2 V for dis) so ipnum=int(100*0.1)=10 < window_length=11 → both
    # segments produce NaN columns by design. The shape + column layout must
    # still be correct.
    assert list(first.columns.names) == ["cycle", "side", "quantity"]
    assert (1, "ch", "電圧") in first.columns
    assert (1, "ch", "dQ/dV") in first.columns
    assert (1, "dis", "電圧") in first.columns
    assert (1, "dis", "dQ/dV") in first.columns
