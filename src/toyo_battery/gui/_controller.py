"""Headless-testable adapter between the Tk view and the core pipeline.

The controller owns every piece of logic that needs to be verified in CI:
validation, cell loading, plot dispatch, axis-limit application. It
imports no ``tkinter`` symbols so it can run on a headless runner and be
unit-tested directly — the Tk view (``tk_app.py``) is a thin translator
from widget state into a :class:`GuiRequest`.

Design notes:

- The view builds a :class:`GuiRequest` and hands it to :func:`run`. The
  dataclass is ``frozen`` to make it cheap to log, compare, and round-trip
  through tests.
- Axis ranges are optional; when present they are applied by iterating
  ``fig.axes`` after the matplotlib backend has drawn. For :func:`plot_cycle`
  the dual-Y twin is matched by y-label rather than position so the
  capacity range never accidentally lands on the efficiency axis.
- The Savitzky-Golay ``window_length`` is not reachable through the cached
  :attr:`toyo_battery.core.cell.Cell.dqdv_df` property, which hard-codes
  the defaults. When the caller requests a non-default window, we compute
  a fresh ``dqdv_df`` via :func:`toyo_battery.core.dqdv.get_dqdv_df` and
  set it on the cell's ``__dict__`` to shadow the ``cached_property``
  lookup before handing the cell to the plotting backend. The override
  is local to freshly constructed cells and does not mutate any shared
  state outside this call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from toyo_battery.core.cell import Cell
from toyo_battery.core.dqdv import get_dqdv_df

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_VALID_KINDS: frozenset[str] = frozenset({"chdis", "cycle", "dqdv"})

# Labels set by the matplotlib backend; we key off these strings rather
# than axis ordering so twin-Y axes aren't confused with primary ones.
_YLABEL_VOLTAGE = "Voltage [V]"
_YLABEL_CAPACITY = "Discharge capacity [mAh/g]"
_YLABEL_DQDV = "dQ/dV [mAh/g/V]"


@dataclass(frozen=True)
class GuiRequest:
    """Inputs to :func:`run`, produced by the Tk view after parsing widgets.

    Attributes
    ----------
    dirs
        Cell directories to load. Each must be a directory acceptable to
        :meth:`Cell.from_dir`. Empty ``dirs`` raises ``ValueError``.
    kinds
        Subset of ``{"chdis", "cycle", "dqdv"}``. Unknown values raise
        ``ValueError`` in :func:`run`.
    cycles
        Cycles to highlight on charge/discharge and dQ/dV plots. Ignored
        by the cycle-vs-capacity plot. Empty sequence = plot every cycle.
    sg_window
        Savitzky-Golay ``window_length`` passed to
        :func:`toyo_battery.core.dqdv.get_dqdv_df`. Must be an odd integer
        ``>= 1``; even or non-positive values raise ``ValueError``.
    voltage_range, capacity_range, dqdv_range
        Optional ``(lo, hi)`` tuples applied to the corresponding plot's
        y-axis via :meth:`Axes.set_ylim`. ``None`` leaves the backend's
        auto-scaled limits in place.
    """

    dirs: Sequence[Path]
    kinds: frozenset[str]
    cycles: Sequence[int] = field(default_factory=tuple)
    sg_window: int = 11
    voltage_range: tuple[float, float] | None = None
    capacity_range: tuple[float, float] | None = None
    dqdv_range: tuple[float, float] | None = None


def _validate(request: GuiRequest) -> None:
    if not request.dirs:
        raise ValueError("dirs must contain at least one directory")
    unknown = set(request.kinds) - _VALID_KINDS
    if unknown:
        raise ValueError(
            f"unknown plot kinds {sorted(unknown)}; valid kinds are {sorted(_VALID_KINDS)}"
        )
    if not request.kinds:
        raise ValueError("kinds must contain at least one of chdis/cycle/dqdv")
    if request.sg_window < 1:
        raise ValueError(f"sg_window must be >= 1, got {request.sg_window}")
    if request.sg_window % 2 == 0:
        raise ValueError(f"sg_window must be odd, got {request.sg_window}")


def _load_cells(dirs: Sequence[Path], sg_window: int) -> list[Cell]:
    """Load one :class:`Cell` per directory.

    When ``sg_window`` differs from :func:`get_dqdv_df`'s default (11),
    we precompute ``dqdv_df`` with the requested window and install it on
    the cell's instance ``__dict__`` so the ``cached_property`` lookup on
    :attr:`Cell.dqdv_df` returns the overridden frame. This shadows the
    default without mutating the ``Cell`` class or any shared cache.
    """
    cells: list[Cell] = []
    for d in dirs:
        cell = Cell.from_dir(d)
        if sg_window != 11:
            cell.__dict__["dqdv_df"] = get_dqdv_df(
                cell.chdis_df,
                window_length=sg_window,
                column_lang=cell.column_lang,
            )
        cells.append(cell)
    return cells


def _apply_ylim(fig: Figure, ylabel: str, ylim: tuple[float, float]) -> None:
    """Set ``set_ylim(*ylim)`` on every Axes of ``fig`` whose y-label matches.

    Iterating all axes (rather than the first one) is deliberate: the
    grid layout emits one Axes per cell, and all of them share the same
    y-label, so the caller-supplied range applies to every subplot.
    """
    for ax in fig.axes:
        if ax.get_ylabel() == ylabel:
            ax.set_ylim(*ylim)


def run(request: GuiRequest) -> list[Figure]:
    """Load cells, dispatch to the matplotlib backend, return figures.

    Figures are returned in a stable order (``chdis``, ``cycle``,
    ``dqdv``) — independent of the iteration order of the input frozenset
    — so the view can present them predictably.

    Parameters
    ----------
    request
        The form state translated into a :class:`GuiRequest` by the Tk
        view. See that class for per-field constraints.

    Returns
    -------
    list of Figure
        One figure per requested plot kind. The caller owns the figures;
        they are not closed on return.

    Raises
    ------
    ValueError
        If ``request`` fails validation (empty dirs, unknown kinds, even
        or non-positive ``sg_window``).
    """
    # Import the backend lazily so a missing ``matplotlib`` extra surfaces
    # only when ``run`` is actually called — importing the controller by
    # itself (for unit tests, or during Tk view construction before the
    # Run button is clicked) stays cheap.
    from toyo_battery.plotting.matplotlib_backend import (
        plot_chdis,
        plot_cycle,
        plot_dqdv,
    )

    _validate(request)

    cells = _load_cells(request.dirs, request.sg_window)
    # ``None`` / empty cycles → let the backend default to all available.
    cycles_arg: Sequence[int] | None = list(request.cycles) if request.cycles else None

    figures: list[Figure] = []
    if "chdis" in request.kinds:
        fig = plot_chdis(cells, cycles=cycles_arg)
        if request.voltage_range is not None:
            _apply_ylim(fig, _YLABEL_VOLTAGE, request.voltage_range)
        figures.append(fig)
    if "cycle" in request.kinds:
        fig = plot_cycle(cells)
        if request.capacity_range is not None:
            _apply_ylim(fig, _YLABEL_CAPACITY, request.capacity_range)
        figures.append(fig)
    if "dqdv" in request.kinds:
        fig = plot_dqdv(cells, cycles=cycles_arg)
        if request.dqdv_range is not None:
            _apply_ylim(fig, _YLABEL_DQDV, request.dqdv_range)
        figures.append(fig)
    return figures
