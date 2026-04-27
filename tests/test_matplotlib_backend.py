"""Tests for :mod:`echemplot.plotting.matplotlib_backend`.

The backend is exercised with small in-memory :class:`Cell` instances
(mirroring :mod:`test_dqdv`) so the tests run without reading real TOYO
data. Matplotlib is imported via :func:`pytest.importorskip` so the
suite is silently skipped when the ``[plot]`` extra is not installed.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from echemplot.core.cell import Cell
from echemplot.io.schema import ColumnLang
from echemplot.plotting.matplotlib_backend import (
    plot_chdis,
    plot_cycle,
    plot_dqdv,
)

_RED = to_rgba("red")
_BLACK = to_rgba("black")


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """Close all matplotlib figures after each test.

    Without this, the ~dozen tests in this module each leak a Figure and
    the suite eventually trips matplotlib's ``max_open_warning``.
    """
    yield
    plt.close("all")


def _visible(fig: Figure) -> list[Axes]:
    """Return the visible Axes on ``fig`` (hides the grid fillers)."""
    return [ax for ax in fig.axes if ax.get_visible()]


def _first_visible(fig: Figure) -> Axes:
    return _visible(fig)[0]


def _line_rgba(line: Line2D) -> tuple[float, float, float, float]:
    """Normalize a Line2D color to an RGBA tuple so comparisons don't depend
    on whether the backend stored it as a name, hex string, or tuple.
    """
    return to_rgba(line.get_color())


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


def test_plot_chdis_returns_figure_with_one_axes_per_cell() -> None:
    cell_a = _linear_cell(name="cell_A", n_cycles=2)
    cell_b = _linear_cell(name="cell_B", n_cycles=2)

    fig = plot_chdis([cell_a, cell_b])

    visible_axes = _visible(fig)
    assert len(visible_axes) == 2
    titles = {ax.get_title() for ax in visible_axes}
    assert titles == {"cell_A", "cell_B"}
    for ax in visible_axes:
        assert ax.get_xlabel() == "Capacity [mAh/g]"
        assert ax.get_ylabel() == "Voltage [V]"


def test_plot_chdis_cycle_1_red_others_black() -> None:
    """Cycle 1 lines are red; cycle 2+ lines are black. Pins the legacy
    TOYO coloring convention called out in issue #8.
    """
    cell = _linear_cell(n_cycles=2)

    fig = plot_chdis([cell])

    ax = _first_visible(fig)
    # 2 cycles * 2 sides (ch + dis) = 4 lines
    assert len(ax.lines) == 4
    colors = [_line_rgba(line) for line in ax.lines]
    assert colors.count(_RED) == 2
    assert colors.count(_BLACK) == 2


def test_plot_chdis_respects_cycles_filter() -> None:
    cell = _linear_cell(n_cycles=3)

    fig = plot_chdis([cell], cycles=[2])

    ax = _first_visible(fig)
    # Only cycle 2 -> 1 cycle * 2 sides = 2 lines, all black.
    assert len(ax.lines) == 2
    assert all(_line_rgba(line) == _BLACK for line in ax.lines)


def test_plot_chdis_default_cycles_covers_all_available() -> None:
    cell = _linear_cell(n_cycles=3)

    fig = plot_chdis([cell])

    ax = _first_visible(fig)
    # 3 cycles * 2 sides = 6 lines; cycle 1 red (2), others black (4).
    assert len(ax.lines) == 6
    colors = [_line_rgba(line) for line in ax.lines]
    assert colors.count(_RED) == 2
    assert colors.count(_BLACK) == 4


def test_plot_cycle_dual_y_has_two_y_axes() -> None:
    cell_a = _linear_cell(name="cell_A", n_cycles=3)
    cell_b = _linear_cell(name="cell_B", n_cycles=3)

    fig = plot_cycle([cell_a, cell_b])

    # Assertions pin axis *identity* via their ylabels rather than
    # counting total axes, which is more robust if matplotlib changes
    # how twins are enumerated.
    primaries = [a for a in fig.axes if a.get_ylabel() == "Discharge capacity [mAh/g]"]
    twins = [a for a in fig.axes if a.get_ylabel() == "Coulombic efficiency [%]"]
    assert len(primaries) == 2
    assert len(twins) == 2
    for ax in primaries:
        assert ax.get_xlabel() == "Cycle"


def test_plot_cycle_wires_q_dis_to_primary_and_ce_to_twin() -> None:
    """Data-to-axis regression guard: ``q_dis`` must be on the primary
    axis and ``ce`` on the twin. A silent swap would be visually
    plausible but numerically catastrophic.
    """
    cell = _linear_cell(name="cell_A", n_cycles=3)

    fig = plot_cycle([cell])

    primary = next(a for a in fig.axes if a.get_ylabel() == "Discharge capacity [mAh/g]")
    twin = next(a for a in fig.axes if a.get_ylabel() == "Coulombic efficiency [%]")
    # Single line per axis (q_dis or ce against cycle).
    np.testing.assert_allclose(primary.lines[0].get_ydata(), cell.cap_df["q_dis"].to_numpy())
    np.testing.assert_allclose(twin.lines[0].get_ydata(), cell.cap_df["ce"].to_numpy())


def test_plot_dqdv_axis_labels_english_even_for_ja_cell() -> None:
    """Labels are fixed English regardless of ``cell.column_lang``."""
    cell = _linear_cell(column_lang="ja")

    fig = plot_dqdv([cell])

    ax = _first_visible(fig)
    assert ax.get_xlabel() == "Voltage [V]"
    assert ax.get_ylabel() == "dQ/dV [mAh/g/V]"


def test_plot_dqdv_plots_both_ch_and_dis_sides() -> None:
    """Both charge (positive dQ/dV) and discharge (negative) branches render."""
    cell = _linear_cell(n_cycles=1)

    fig = plot_dqdv([cell])

    ax = _first_visible(fig)
    # 1 cycle * 2 sides = 2 lines (both colored red because cycle 1).
    assert len(ax.lines) == 2
    y_medians = [float(np.median(line.get_ydata())) for line in ax.lines]
    assert any(m > 0 for m in y_medians), "expected a positive-median charge branch"
    assert any(m < 0 for m in y_medians), "expected a negative-median discharge branch"


def test_empty_cells_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_chdis([])
    with pytest.raises(ValueError, match="non-empty"):
        plot_cycle([])
    with pytest.raises(ValueError, match="non-empty"):
        plot_dqdv([])


def test_plot_chdis_missing_cycle_in_filter_is_skipped_silently() -> None:
    """Requesting a cycle not present in a cell yields no lines — no raise."""
    cell = _linear_cell(n_cycles=1)

    fig = plot_chdis([cell], cycles=[99])

    ax = _first_visible(fig)
    assert len(ax.lines) == 0


def test_plot_chdis_cycles_filter_spans_cells_with_different_cycle_counts() -> None:
    """A shared ``cycles=[1, 2, 5]`` call on cells with different cycle
    inventories: each cell keeps only the cycles it actually has, so per-cell
    line counts reflect per-cell availability (2 sides per cycle).
    """
    small = _linear_cell(name="small", n_cycles=2)  # has cycles {1, 2}
    large = _linear_cell(name="large", n_cycles=6)  # has cycles {1..6}

    fig = plot_chdis([small, large], cycles=[1, 2, 5])

    axes_by_title = {ax.get_title(): ax for ax in _visible(fig)}
    # small only has {1, 2} of the requested {1, 2, 5} -> 2 cycles * 2 sides = 4.
    assert len(axes_by_title["small"].lines) == 4
    # large has {1, 2, 5} -> 3 cycles * 2 sides = 6.
    assert len(axes_by_title["large"].lines) == 6


def test_plot_dqdv_sg_override_changes_curve() -> None:
    """``plot_dqdv`` with a non-default ``sg_window_length`` must produce
    different y-values than the default.

    Guards the new SG-override plumbing on the matplotlib backend — at the
    defaults the cached :attr:`Cell.dqdv_df` is reused, but any override
    must force a recompute via :func:`get_dqdv_df`.
    """
    cell = _linear_cell(n_cycles=1)

    fig_default = plot_dqdv([cell])
    fig_wide = plot_dqdv([cell], sg_window_length=21)

    y_default = np.asarray(_first_visible(fig_default).lines[0].get_ydata(), dtype=float)
    y_wide = np.asarray(_first_visible(fig_wide).lines[0].get_ydata(), dtype=float)
    # Same sample count (interpolation grid is independent of the SG window)
    # but the smoothed derivatives should disagree.
    assert y_default.shape == y_wide.shape
    assert not np.allclose(y_default, y_wide, equal_nan=True)


def test_column_lang_en_cell_plots_successfully() -> None:
    """A Cell built with ``column_lang='en'`` still plots (regression guard
    for the internal ``_quantity`` lookup).
    """
    cell = _linear_cell(column_lang="en", n_cycles=2)

    fig_chdis = plot_chdis([cell])
    fig_dqdv = plot_dqdv([cell])

    ax_chdis = _first_visible(fig_chdis)
    ax_dqdv = _first_visible(fig_dqdv)
    # 2 cycles * 2 sides = 4 lines per figure.
    assert len(ax_chdis.lines) == 4
    assert len(ax_dqdv.lines) == 4
