"""Matplotlib plotting backend. Requires the ``[plot]`` extra.

Three user-facing functions operate on one or more :class:`echemplot.core.cell.Cell`
instances and return a :class:`matplotlib.figure.Figure` (the caller handles
``savefig``):

- :func:`plot_chdis` — charge/discharge V-vs-Q curves; cycle 1 is drawn in
  red, all other cycles in black (TOYO legacy convention).
- :func:`plot_cycle` — per-cycle dual-Y plot: discharge capacity on the
  left Y axis, Coulombic efficiency on the right Y axis.
- :func:`plot_dqdv` — dQ/dV-vs-V curves, same cycle-1-red / others-black
  coloring as :func:`plot_chdis`. Discharge dQ/dV is **negative** by
  construction (see :mod:`echemplot.core.dqdv` for the sign
  convention); raw signed values are plotted so charge and discharge
  branches live in different half-planes.

Layout: when multiple cells are passed, the returned figure contains one
Axes per cell arranged in a near-square grid (1xN for ``N<=3``, otherwise
``ncols = ceil(sqrt(N))``, ``nrows = ceil(N/ncols)``). Unused grid cells
are hidden.

Labels and legend text are **always English**, independent of each cell's
``column_lang``; this avoids matplotlib JP-font configuration issues.
Cell data access (``cell.chdis_df`` / ``cell.cap_df`` / ``cell.dqdv_df``)
uses the correct quantity-level label for the cell's own ``column_lang``.

Matplotlib is imported unconditionally at module level: this module is
only imported when plotting is actually needed, and a missing extra
surfaces as a standard ``ImportError`` from the import line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from echemplot.core.cell import Cell
from echemplot.core.dqdv import get_dqdv_df
from echemplot.io.schema import JA_COLS, JA_TO_EN, ColumnLang

_CYCLE_COLOR_FIRST = "red"
_CYCLE_COLOR_OTHER = "black"

# Per-subplot size (inches) used when building the grid. Pinned here so a
# tweak to the default layout touches one line rather than three.
_SUBPLOT_W_IN = 5.0
_SUBPLOT_H_IN = 4.0

# ``dQ/dV`` is a derived-quantity label, not a TOYO source column, so it is
# not part of the central ``JA_COLS`` and is kept local here.
_DQDV_LABEL_JA = "dQ/dV"
_DQDV_LABEL_EN = "dq_dv"


def _quantity(column_lang: ColumnLang, key: str) -> str:
    """Map a logical key (``"voltage"`` / ``"capacity"`` / ``"dqdv"``) to the
    ``quantity``-level MultiIndex label used by a cell with the given
    ``column_lang``.
    """
    if key == "dqdv":
        return _DQDV_LABEL_JA if column_lang == "ja" else _DQDV_LABEL_EN
    ja_col = JA_COLS[key]
    return ja_col if column_lang == "ja" else JA_TO_EN[ja_col]


def _subplot_grid(n: int) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` for ``n`` subplots.

    ``n <= 3`` → a single row. Larger counts fall back to a near-square
    grid: ``ncols = ceil(sqrt(n))``, ``nrows = ceil(n / ncols)``.
    """
    if n <= 3:
        return (1, max(n, 1))
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)


def _resolve_cycles(cell: Cell, cycles: Sequence[int] | None) -> list[int]:
    """Return the cycles to plot for ``cell``.

    ``cycles=None`` yields every cycle present in ``cell.cap_df.index``.
    An explicit sequence is intersected with the available cycles while
    preserving caller order; cycles missing from the cell are dropped
    silently so a shared ``cycles=[1, 5, 10]`` call can span cells with
    different cycle counts without raising.
    """
    available = {int(c) for c in cell.cap_df.index}
    if cycles is None:
        return sorted(available)
    return [int(c) for c in cycles if int(c) in available]


def _check_nonempty(cells: Sequence[Cell]) -> None:
    """Raise ``ValueError`` if ``cells`` is empty.

    Fails loudly rather than returning a blank figure — a zero-cell call
    is almost always a caller bug.
    """
    if len(cells) == 0:
        raise ValueError("cells must be a non-empty sequence of Cell instances")


def _build_grid(n: int) -> tuple[Figure, list[Axes]]:
    """Create a figure with one Axes per cell, hiding unused grid cells.

    Returns the figure and a flat list of ``n`` visible Axes (in cell
    order). Any extra Axes created by the grid layout are hidden in
    place.
    """
    nrows, ncols = _subplot_grid(n)
    fig, axes_array = plt.subplots(
        nrows,
        ncols,
        figsize=(_SUBPLOT_W_IN * ncols, _SUBPLOT_H_IN * nrows),
        squeeze=False,
    )
    flat: list[Axes] = [ax for row in axes_array for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _cycle_color(cycle: int) -> str:
    return _CYCLE_COLOR_FIRST if cycle == 1 else _CYCLE_COLOR_OTHER


def _add_cycle_legend(ax: Axes, cycles_plotted: list[int]) -> None:
    """Attach a 2-entry cycle-color legend when more than one cycle was drawn."""
    if len(cycles_plotted) <= 1:
        return
    handles = [
        Line2D([], [], color=_CYCLE_COLOR_FIRST, label="cycle 1"),
        Line2D([], [], color=_CYCLE_COLOR_OTHER, label="other cycles"),
    ]
    ax.legend(handles=handles, loc="best")


def plot_chdis(
    cells: Sequence[Cell],
    cycles: Sequence[int] | None = None,
) -> Figure:
    """Charge/discharge curves (capacity on X, voltage on Y).

    Parameters
    ----------
    cells
        One or more :class:`Cell` instances. One subplot is drawn per
        cell.
    cycles
        Cycles to include. ``None`` (default) plots every cycle present
        in each cell's ``cap_df``. When a sequence is given, cycles not
        present in a given cell are skipped silently for that cell only.

    Returns
    -------
    Figure
        A new matplotlib figure. The caller is responsible for
        ``savefig`` and ``close``.

    Raises
    ------
    ValueError
        If ``cells`` is empty.
    """
    _check_nonempty(cells)
    fig, axes = _build_grid(len(cells))

    for ax, cell in zip(axes, cells):
        v_name = _quantity(cell.column_lang, "voltage")
        q_name = _quantity(cell.column_lang, "capacity")
        chdis = cell.chdis_df

        plotted: list[int] = []
        for cycle in _resolve_cycles(cell, cycles):
            color = _cycle_color(cycle)
            drew_any = False
            for side in ("ch", "dis"):
                v_key = (cycle, side, v_name)
                q_key = (cycle, side, q_name)
                if v_key not in chdis.columns or q_key not in chdis.columns:
                    continue
                # Drop NaNs jointly so x/y stay positionally aligned; per-column
                # dropna would silently misplot if the two columns ever have
                # asymmetric NaN patterns.
                pair = chdis[[q_key, v_key]].dropna()
                if pair.empty:
                    continue
                ax.plot(pair[q_key], pair[v_key], color=color, linewidth=0.9)
                drew_any = True
            if drew_any:
                plotted.append(cycle)

        ax.set_xlabel("Capacity [mAh/g]")
        ax.set_ylabel("Voltage [V]")
        ax.set_title(cell.name)
        _add_cycle_legend(ax, plotted)

    fig.tight_layout()
    return fig


def plot_cycle(cells: Sequence[Cell]) -> Figure:
    """Per-cycle discharge capacity (left Y) and Coulombic efficiency (right Y).

    Unlike :func:`plot_chdis` / :func:`plot_dqdv`, this function has no
    ``cycles=`` parameter: the whole point of a cycle plot is the
    capacity-vs-cycle trajectory, and slicing out cycles would break the
    trend lines rather than filter them.

    Parameters
    ----------
    cells
        One or more :class:`Cell` instances. One subplot is drawn per
        cell, each with a twin-Y (``ax.twinx()``) for efficiency.

    Returns
    -------
    Figure
        A new matplotlib figure.

    Raises
    ------
    ValueError
        If ``cells`` is empty.
    """
    _check_nonempty(cells)
    fig, axes = _build_grid(len(cells))

    for ax, cell in zip(axes, cells):
        cap = cell.cap_df
        (line_q,) = ax.plot(
            cap.index,
            cap["q_dis"],
            "o-",
            color="C0",
            label="Discharge capacity",
        )
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Discharge capacity [mAh/g]")

        ax2 = ax.twinx()
        (line_ce,) = ax2.plot(
            cap.index,
            cap["ce"],
            "s--",
            color="C3",
            label="Coulombic efficiency",
        )
        ax2.set_ylabel("Coulombic efficiency [%]")

        ax.set_title(cell.name)
        ax.legend(handles=[line_q, line_ce], loc="lower right")

    fig.tight_layout()
    return fig


def plot_dqdv(
    cells: Sequence[Cell],
    cycles: Sequence[int] | None = None,
    *,
    sg_window_length: int = 11,
    sg_polyorder: int = 2,
) -> Figure:
    """dQ/dV vs voltage, cycle 1 red / other cycles black.

    Discharge dQ/dV is negative by construction (see
    :mod:`echemplot.core.dqdv`); the raw signed values are plotted so
    charge and discharge branches live in different half-planes.

    Parameters
    ----------
    cells
        One or more :class:`Cell` instances. One subplot per cell.
    cycles
        Cycles to include. ``None`` plots every cycle present in each
        cell's ``cap_df.index`` (the authoritative cycle inventory);
        cycles whose ``dqdv_df`` columns collapsed to all-NaN (e.g.
        ``ipnum < window_length`` for a narrow-voltage segment) are
        skipped silently for that cell.
    sg_window_length
        Savitzky-Golay ``window_length`` forwarded to
        :func:`echemplot.core.dqdv.get_dqdv_df`. When this and
        ``sg_polyorder`` are both at their defaults (``11`` / ``2``), the
        cached :attr:`Cell.dqdv_df` is reused; otherwise a fresh
        ``dqdv_df`` is computed per cell with the requested parameters.
        Must be a positive odd integer strictly greater than
        ``sg_polyorder``.
    sg_polyorder
        Savitzky-Golay polynomial order forwarded to
        :func:`echemplot.core.dqdv.get_dqdv_df`. See ``sg_window_length``
        for the cache-or-recompute behavior.

    Returns
    -------
    Figure
        A new matplotlib figure.

    Raises
    ------
    ValueError
        If ``cells`` is empty, or if Savitzky-Golay parameters violate
        :func:`echemplot.core.dqdv.get_dqdv_df`'s preconditions.
    """
    _check_nonempty(cells)
    fig, axes = _build_grid(len(cells))

    # Reuse the cached property only at the defaults; any override forces a
    # recompute via ``get_dqdv_df`` so the requested SG parameters actually
    # land in the output.
    use_default = sg_window_length == 11 and sg_polyorder == 2

    for ax, cell in zip(axes, cells):
        v_name = _quantity(cell.column_lang, "voltage")
        dqdv_name = _quantity(cell.column_lang, "dqdv")
        if use_default:
            df = cell.dqdv_df
        else:
            df = get_dqdv_df(
                cell.chdis_df,
                window_length=sg_window_length,
                polyorder=sg_polyorder,
                column_lang=cell.column_lang,
            )

        plotted: list[int] = []
        for cycle in _resolve_cycles(cell, cycles):
            color = _cycle_color(cycle)
            drew_any = False
            for side in ("ch", "dis"):
                v_key = (cycle, side, v_name)
                y_key = (cycle, side, dqdv_name)
                if v_key not in df.columns or y_key not in df.columns:
                    continue
                # Joint dropna — see plot_chdis for the rationale.
                pair = df[[v_key, y_key]].dropna()
                if pair.empty:
                    continue
                ax.plot(pair[v_key], pair[y_key], color=color, linewidth=0.9)
                drew_any = True
            if drew_any:
                plotted.append(cycle)

        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("dQ/dV [mAh/g/V]")
        ax.set_title(cell.name)
        _add_cycle_legend(ax, plotted)

    fig.tight_layout()
    return fig
