"""Tests for :mod:`echemplot.plotting.plotly_backend`.

Mirror of :mod:`test_matplotlib_backend` using in-memory :class:`Cell`
instances. Plotly is imported via :func:`pytest.importorskip` so the
suite is silently skipped when the ``[plotly]`` extra is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go

from echemplot.core.cell import Cell
from echemplot.io.schema import ColumnLang
from echemplot.plotting.plotly_backend import (
    plot_chdis,
    plot_cycle,
    plot_dqdv,
)

_RED = "red"
_BLACK = "black"


def _linear_cell(
    *,
    name: str = "synthetic",
    n_cycles: int = 2,
    column_lang: ColumnLang = "ja",
    n_points: int = 200,
    v_lo: float = 3.0,
    v_hi: float = 4.2,
) -> Cell:
    """Build a :class:`Cell` with ``n_cycles`` linear charge+discharge ramps.

    Wide voltage span (1.2 V default) so dQ/dV interpolation
    (``ipnum = int(100 * 1.2) = 120``) comfortably exceeds the default
    Savitzky-Golay window length (11) and produces populated columns.
    """
    rows: list[tuple[int, str, str, float, float]] = []
    v_ch = np.linspace(v_lo, v_hi, n_points)
    q_ch = 400.0 * (v_ch - v_lo)
    v_dis = np.linspace(v_hi, v_lo, n_points)
    q_dis = 400.0 * (v_hi - v_dis)
    if column_lang == "ja":
        charge_lbl, discharge_lbl = "充電", "放電"
        columns = ["サイクル", "モード", "状態", "電圧", "電気量"]
    else:
        # EN-mode chdis filters on EN state literals (issue #94).
        charge_lbl, discharge_lbl = "charge", "discharge"
        columns = ["cycle", "mode", "state", "voltage", "capacity"]
    for cycle in range(1, n_cycles + 1):
        rows += [(cycle, "1", charge_lbl, float(vi), float(qi)) for vi, qi in zip(v_ch, q_ch)]
        rows += [(cycle, "1", discharge_lbl, float(vi), float(qi)) for vi, qi in zip(v_dis, q_dis)]
    raw = pd.DataFrame(rows, columns=columns)
    return Cell(name=name, mass_g=0.001, raw_df=raw, column_lang=column_lang)


def _expected_grid(n: int) -> tuple[int, int]:
    """Mirror of the backend's grid formula (kept separate as a test guard)."""
    if n <= 3:
        return (1, max(n, 1))
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)


def _subplot_shape(fig: go.Figure) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` from the figure's grid_ref."""
    grid_ref = fig._grid_ref  # plotly internal; stable across 5.x
    nrows = len(grid_ref)
    ncols = len(grid_ref[0])
    return nrows, ncols


def test_plot_chdis_returns_plotly_figure() -> None:
    cell = _linear_cell(n_cycles=2)
    fig = plot_chdis([cell])
    assert isinstance(fig, go.Figure)


def test_plot_cycle_returns_plotly_figure() -> None:
    cell = _linear_cell(n_cycles=3)
    fig = plot_cycle([cell])
    assert isinstance(fig, go.Figure)


def test_plot_dqdv_returns_plotly_figure() -> None:
    cell = _linear_cell(n_cycles=2)
    fig = plot_dqdv([cell])
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_grid_shape_matches_formula(n: int) -> None:
    """Grid layout mirrors matplotlib: 1xN for N<=3, else near-square."""
    cells = [_linear_cell(name=f"cell_{i}", n_cycles=2) for i in range(n)]
    fig = plot_chdis(cells)
    assert _subplot_shape(fig) == _expected_grid(n)


def test_plot_chdis_cycle_1_red_others_black() -> None:
    """Cycle 1 traces are red; cycle 2+ are black. Pins the TOYO convention."""
    cell = _linear_cell(n_cycles=2)
    fig = plot_chdis([cell])

    # 2 cycles * 2 sides (ch + dis) = 4 traces
    assert len(fig.data) == 4
    colors = [trace.line.color for trace in fig.data]
    assert colors.count(_RED) == 2
    assert colors.count(_BLACK) == 2


def test_plot_chdis_respects_cycles_filter() -> None:
    """Requesting ``cycles=[1, 10]`` on a 3-cycle cell only plots cycle 1."""
    cell = _linear_cell(n_cycles=3)

    fig = plot_chdis([cell], cycles=[1, 10])

    # Only cycle 1 is present (cycle 10 doesn't exist) -> 1 * 2 sides = 2 traces.
    assert len(fig.data) == 2
    assert all(trace.line.color == _RED for trace in fig.data)


def test_plot_chdis_cycles_filter_exact_subset() -> None:
    """cycles=[1, 3] on a 5-cycle cell plots exactly those two."""
    cell = _linear_cell(n_cycles=5)

    fig = plot_chdis([cell], cycles=[1, 3])

    # 2 cycles * 2 sides = 4 traces; cycle 1 red (2), cycle 3 black (2).
    assert len(fig.data) == 4
    colors = [trace.line.color for trace in fig.data]
    assert colors.count(_RED) == 2
    assert colors.count(_BLACK) == 2


def test_plot_cycle_has_two_y_axes_per_subplot() -> None:
    """``plot_cycle`` uses make_subplots(secondary_y=True); verify both axes
    exist for each cell subplot.
    """
    cell_a = _linear_cell(name="cell_A", n_cycles=3)
    cell_b = _linear_cell(name="cell_B", n_cycles=3)
    fig = plot_cycle([cell_a, cell_b])

    # Each cell should have a primary and secondary y axis. Plotly names
    # secondary axes with ``overlaying`` set to the primary. Count axes
    # whose title matches each side.
    primaries = [
        ax
        for ax in fig.layout
        if ax.startswith("yaxis")
        and getattr(fig.layout[ax], "title", None) is not None
        and fig.layout[ax].title.text == "Discharge capacity [mAh/g]"
    ]
    secondaries = [
        ax
        for ax in fig.layout
        if ax.startswith("yaxis")
        and getattr(fig.layout[ax], "title", None) is not None
        and fig.layout[ax].title.text == "Coulombic efficiency [%]"
    ]
    assert len(primaries) == 2
    assert len(secondaries) == 2


def test_plot_cycle_wires_q_dis_primary_and_ce_secondary() -> None:
    """Regression guard: q_dis on primary Y, ce on secondary Y."""
    cell = _linear_cell(name="cell_A", n_cycles=3)
    fig = plot_cycle([cell])

    # Primary-y trace: q_dis. Secondary-y trace: ce.
    # Plotly encodes secondary-y as a trace on the secondary axis
    # (``yaxis='y2'`` for the first subplot).
    primary_traces = [t for t in fig.data if t.yaxis in (None, "y")]
    secondary_traces = [t for t in fig.data if t.yaxis == "y2"]
    assert len(primary_traces) == 1
    assert len(secondary_traces) == 1
    np.testing.assert_allclose(primary_traces[0].y, cell.cap_df["q_dis"].to_numpy())
    np.testing.assert_allclose(secondary_traces[0].y, cell.cap_df["ce"].to_numpy())


def test_plot_dqdv_discharge_traces_have_negative_values() -> None:
    """Discharge dQ/dV is negative by construction; verify the sign
    convention round-trips through the Plotly backend (raw signed
    values, no flipping).
    """
    cell = _linear_cell(n_cycles=1)
    fig = plot_dqdv([cell])

    # 1 cycle * 2 sides = 2 traces.
    assert len(fig.data) == 2
    medians = [float(np.median(trace.y)) for trace in fig.data]
    assert any(m > 0 for m in medians), "expected a positive-median charge branch"
    assert any(m < 0 for m in medians), "expected a negative-median discharge branch"


def test_plot_dqdv_english_labels_for_ja_cell() -> None:
    """Labels are fixed English regardless of ``cell.column_lang``."""
    cell = _linear_cell(column_lang="ja")
    fig = plot_dqdv([cell])

    # For a single-cell figure the axes are ``xaxis`` / ``yaxis``.
    assert fig.layout.xaxis.title.text == "Voltage [V]"
    assert fig.layout.yaxis.title.text == "dQ/dV [mAh/g/V]"


def test_plot_chdis_axis_labels() -> None:
    """chdis labels: x=Capacity, y=Voltage."""
    cell = _linear_cell(n_cycles=2)
    fig = plot_chdis([cell])

    assert fig.layout.xaxis.title.text == "Capacity [mAh/g]"
    assert fig.layout.yaxis.title.text == "Voltage [V]"


def test_empty_cells_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_chdis([])
    with pytest.raises(ValueError, match="non-empty"):
        plot_cycle([])
    with pytest.raises(ValueError, match="non-empty"):
        plot_dqdv([])


def test_column_lang_en_cell_plots_successfully() -> None:
    """A Cell built with ``column_lang='en'`` still plots."""
    cell = _linear_cell(column_lang="en", n_cycles=2)

    fig_chdis = plot_chdis([cell])
    fig_dqdv = plot_dqdv([cell])

    # 2 cycles * 2 sides = 4 traces per figure.
    assert len(fig_chdis.data) == 4
    assert len(fig_dqdv.data) == 4


def test_write_image_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: fig.write_image succeeds via kaleido."""
    pytest.importorskip("kaleido")
    cell = _linear_cell(n_cycles=2)
    fig = plot_chdis([cell])

    out = tmp_path / "out.png"
    fig.write_image(str(out))
    assert out.exists()
    assert out.stat().st_size > 0
