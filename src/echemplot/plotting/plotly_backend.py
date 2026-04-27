"""Plotly plotting backend. Requires the ``[plotly]`` extra.

Mirrors the :mod:`echemplot.plotting.matplotlib_backend` API — three
user-facing functions that accept one or more :class:`echemplot.core.cell.Cell`
instances and return a :class:`plotly.graph_objects.Figure` (the caller
handles ``fig.write_image(...)`` / ``fig.write_html(...)``):

- :func:`plot_chdis` — charge/discharge V-vs-Q curves; cycle 1 drawn in
  red, all others in black (TOYO legacy convention).
- :func:`plot_cycle` — per-cycle dual-Y plot: discharge capacity on the
  left Y axis, Coulombic efficiency on the right Y axis.
- :func:`plot_dqdv` — dQ/dV-vs-V curves, same cycle-1-red / others-black
  coloring as :func:`plot_chdis`. Discharge dQ/dV is **negative** by
  construction; raw signed values are plotted.

Grid layout matches the matplotlib backend verbatim: 1xN for ``N<=3``;
otherwise ``ncols = ceil(sqrt(N))``, ``nrows = ceil(N/ncols)``. Labels
and legend text are always English, independent of each cell's
``column_lang``.

Plotly is imported unconditionally at module level: this module is only
imported when plotly plotting is actually needed, and a missing extra
surfaces as a standard ``ImportError`` from the import line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from echemplot.core.cell import Cell
from echemplot.io.schema import JA_COLS, JA_TO_EN, ColumnLang

_CYCLE_COLOR_FIRST = "red"
_CYCLE_COLOR_OTHER = "black"

# Per-subplot size (pixels) used when sizing the figure. Pinned here so a
# tweak to the default layout touches one line rather than three.
_SUBPLOT_W_PX = 500
_SUBPLOT_H_PX = 400

# Colors mirroring the matplotlib ``C0`` / ``C3`` default-cycle entries
# used in ``plot_cycle``. Pinning explicit hex values keeps the Plotly
# output visually close to the matplotlib output even though the two
# backends have different default color palettes.
_C0_BLUE = "#1f77b4"
_C3_RED = "#d62728"

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


def _cycle_color(cycle: int) -> str:
    return _CYCLE_COLOR_FIRST if cycle == 1 else _CYCLE_COLOR_OTHER


def _cell_rowcol(idx: int, ncols: int) -> tuple[int, int]:
    """Return ``(row, col)`` (1-indexed) for the ``idx``-th cell in a grid
    with ``ncols`` columns.
    """
    return (idx // ncols + 1, idx % ncols + 1)


def _build_figure(n: int, *, secondary_y: bool = False) -> tuple[go.Figure, int, int]:
    """Create a subplot figure sized for ``n`` cells.

    Returns the figure and the ``(nrows, ncols)`` of the grid so callers
    can map cell index → (row, col).
    """
    nrows, ncols = _subplot_grid(n)
    specs: list[list[dict[str, bool]]] | None
    if secondary_y:
        specs = [[{"secondary_y": True} for _ in range(ncols)] for _ in range(nrows)]
    else:
        specs = None
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs)
    fig.update_layout(
        width=_SUBPLOT_W_PX * ncols,
        height=_SUBPLOT_H_PX * nrows,
        showlegend=False,
    )
    return fig, nrows, ncols


def plot_chdis(
    cells: Sequence[Cell],
    cycles: Sequence[int] | None = None,
) -> go.Figure:
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
    plotly.graph_objects.Figure
        A new Plotly figure. The caller is responsible for
        ``write_image`` / ``write_html``.

    Raises
    ------
    ValueError
        If ``cells`` is empty.
    """
    _check_nonempty(cells)
    fig, _nrows, ncols = _build_figure(len(cells))

    for idx, cell in enumerate(cells):
        row, col = _cell_rowcol(idx, ncols)
        v_name = _quantity(cell.column_lang, "voltage")
        q_name = _quantity(cell.column_lang, "capacity")
        chdis = cell.chdis_df

        for cycle in _resolve_cycles(cell, cycles):
            color = _cycle_color(cycle)
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
                fig.add_trace(
                    go.Scatter(
                        x=pair[q_key],
                        y=pair[v_key],
                        mode="lines",
                        line={"color": color, "width": 1.2},
                        name=f"cycle {cycle}",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(title_text="Capacity [mAh/g]", row=row, col=col)
        fig.update_yaxes(title_text="Voltage [V]", row=row, col=col)
        _annotate_title(fig, cell.name, row, col, ncols)

    return fig


def plot_cycle(cells: Sequence[Cell]) -> go.Figure:
    """Per-cycle discharge capacity (left Y) and Coulombic efficiency (right Y).

    Unlike :func:`plot_chdis` / :func:`plot_dqdv`, this function has no
    ``cycles=`` parameter: the whole point of a cycle plot is the
    capacity-vs-cycle trajectory, and slicing out cycles would break the
    trend lines rather than filter them.

    Parameters
    ----------
    cells
        One or more :class:`Cell` instances. One subplot is drawn per
        cell, each with a secondary Y axis for efficiency.

    Returns
    -------
    plotly.graph_objects.Figure
        A new Plotly figure.

    Raises
    ------
    ValueError
        If ``cells`` is empty.
    """
    _check_nonempty(cells)
    fig, _nrows, ncols = _build_figure(len(cells), secondary_y=True)

    for idx, cell in enumerate(cells):
        row, col = _cell_rowcol(idx, ncols)
        cap = cell.cap_df

        fig.add_trace(
            go.Scatter(
                x=cap.index,
                y=cap["q_dis"],
                mode="lines+markers",
                marker={"symbol": "circle", "color": _C0_BLUE},
                line={"color": _C0_BLUE},
                name="Discharge capacity",
                showlegend=False,
            ),
            row=row,
            col=col,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=cap.index,
                y=cap["ce"],
                mode="lines+markers",
                marker={"symbol": "square", "color": _C3_RED},
                line={"color": _C3_RED, "dash": "dash"},
                name="Coulombic efficiency",
                showlegend=False,
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Cycle", row=row, col=col)
        fig.update_yaxes(
            title_text="Discharge capacity [mAh/g]", row=row, col=col, secondary_y=False
        )
        fig.update_yaxes(title_text="Coulombic efficiency [%]", row=row, col=col, secondary_y=True)
        _annotate_title(fig, cell.name, row, col, ncols)

    return fig


def plot_dqdv(
    cells: Sequence[Cell],
    cycles: Sequence[int] | None = None,
) -> go.Figure:
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

    Returns
    -------
    plotly.graph_objects.Figure
        A new Plotly figure.

    Raises
    ------
    ValueError
        If ``cells`` is empty.
    """
    _check_nonempty(cells)
    fig, _nrows, ncols = _build_figure(len(cells))

    for idx, cell in enumerate(cells):
        row, col = _cell_rowcol(idx, ncols)
        v_name = _quantity(cell.column_lang, "voltage")
        dqdv_name = _quantity(cell.column_lang, "dqdv")
        dqdv = cell.dqdv_df

        for cycle in _resolve_cycles(cell, cycles):
            color = _cycle_color(cycle)
            for side in ("ch", "dis"):
                v_key = (cycle, side, v_name)
                y_key = (cycle, side, dqdv_name)
                if v_key not in dqdv.columns or y_key not in dqdv.columns:
                    continue
                # Joint dropna — see plot_chdis for the rationale.
                pair = dqdv[[v_key, y_key]].dropna()
                if pair.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=pair[v_key],
                        y=pair[y_key],
                        mode="lines",
                        line={"color": color, "width": 1.2},
                        name=f"cycle {cycle} {side}",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(title_text="Voltage [V]", row=row, col=col)
        fig.update_yaxes(title_text="dQ/dV [mAh/g/V]", row=row, col=col)
        _annotate_title(fig, cell.name, row, col, ncols)

    return fig


def _annotate_title(fig: go.Figure, title: str, row: int, col: int, ncols: int) -> None:
    """Attach a per-subplot title by updating the Plotly-generated annotation.

    ``make_subplots(subplot_titles=...)`` is the standard path, but the
    title list must be set at figure construction; since we build the
    figure generically in :func:`_build_figure` we set the per-cell title
    after the fact by targeting the subplot-title annotation that Plotly
    creates for each ``(row, col)``. If no annotation exists yet (the
    default when ``subplot_titles`` is unset), we append one positioned
    above the appropriate subplot.
    """
    # Compute the subplot index in row-major order (1-indexed) — this is
    # how Plotly names axes (``xaxis``, ``xaxis2``, ...).
    subplot_idx = (row - 1) * ncols + col
    xref = "x domain" if subplot_idx == 1 else f"x{subplot_idx} domain"
    yref = "y domain" if subplot_idx == 1 else f"y{subplot_idx} domain"
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref=xref,
        yref=yref,
        text=title,
        showarrow=False,
        font={"size": 14},
        xanchor="center",
        yanchor="bottom",
    )
