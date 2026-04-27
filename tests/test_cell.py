"""Tests for :class:`echemplot.core.cell.Cell` immutability semantics.

The ``Cell`` contract:

- The constructor deep-copies its input ``raw_df`` so the caller can keep
  using their original frame without affecting the ``Cell`` (and vice
  versa).
- Every public DataFrame property (``raw_df``, ``chdis_df``, ``cap_df``,
  ``dqdv_df``) returns a fresh defensive copy on each access; mutating the
  returned frame must not corrupt the ``Cell``'s internal state.
- The underlying computation backing each derived property
  (``cached_property``) runs exactly once per ``Cell`` instance — the
  per-read copy is cheap, the compute is not.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from echemplot.core.cell import Cell


def _minimal_raw() -> pd.DataFrame:
    """Single-cycle charge → discharge frame in the canonical JA layout.

    Sized minimally — these tests only care about the immutability
    contract, not segmentation correctness.
    """
    return pd.DataFrame(
        [
            (1, "充電", 3.50, 0.0),
            (1, "充電", 3.60, 1000.0),
            (1, "放電", 3.40, 0.0),
            (1, "放電", 3.20, 1000.0),
        ],
        columns=["サイクル", "状態", "電圧", "電気量"],
    )


def test_cell_constructor_does_not_share_input_frame() -> None:
    """Mutating the original frame after ``Cell(...)`` must not affect the cell."""
    raw = _minimal_raw()
    original_voltage = raw.loc[0, "電圧"]
    cell = Cell(name="c1", mass_g=0.001, raw_df=raw)

    # Mutate the caller's frame — the Cell must be isolated.
    raw.loc[0, "電圧"] = 99.99
    raw.iloc[1, raw.columns.get_loc("電気量")] = -12345.0

    cell_raw = cell.raw_df
    assert cell_raw.loc[0, "電圧"] == original_voltage
    assert cell_raw.loc[1, "電気量"] == 1000.0


def test_cell_raw_df_returns_a_copy() -> None:
    """Mutating a returned ``raw_df`` must not affect subsequent reads."""
    cell = Cell(name="c1", mass_g=0.001, raw_df=_minimal_raw())

    first = cell.raw_df
    first.loc[0, "電圧"] = 42.0
    first.iloc[2, first.columns.get_loc("電気量")] = -1.0

    second = cell.raw_df
    assert second.loc[0, "電圧"] == 3.50
    assert second.loc[2, "電気量"] == 0.0
    # Defensive-copy contract: each access yields a distinct object.
    assert first is not second


def test_cell_chdis_df_returns_a_copy() -> None:
    """Mutating a returned ``chdis_df`` must not affect subsequent reads."""
    cell = Cell(name="c1", mass_g=0.001, raw_df=_minimal_raw())

    first = cell.chdis_df
    assert not first.empty, "fixture must produce a non-empty chdis_df"
    # Stomp every value in the returned copy.
    first.iloc[:, :] = 0.0

    second = cell.chdis_df
    assert first is not second
    # The fresh copy must reflect the original cached computation, not
    # the caller's mutation.
    pd.testing.assert_frame_equal(second, cell.chdis_df)
    # And it must not be all zeros (the original fixture has non-zero
    # capacity + voltage entries).
    assert (second.fillna(0.0).abs().to_numpy() > 0).any()


def test_cell_cached_property_only_computed_once() -> None:
    """Multiple ``cell.chdis_df`` reads must invoke ``get_chdis_df`` exactly once.

    The defensive copy on each public read is fine; the expensive
    computation behind it must still be cached.
    """
    cell = Cell(name="c1", mass_g=0.001, raw_df=_minimal_raw())

    with patch(
        "echemplot.core.cell.get_chdis_df",
        wraps=__import__("echemplot.core.chdis", fromlist=["get_chdis_df"]).get_chdis_df,
    ) as spy:
        _ = cell.chdis_df
        _ = cell.chdis_df
        _ = cell.chdis_df

    assert spy.call_count == 1, (
        f"get_chdis_df was invoked {spy.call_count} times - cached_property "
        "must memoize the computation across reads."
    )
