"""Graph-creation helpers for the Origin adapter.

Graphs are instantiated from ``.otpu`` templates ported from OriginLab's
TOYO v2.01 pipeline. The three templates ship with the wheel under
``src/echemplot/origin/templates/`` and are picked up automatically;
``ECHEMPLOT_ORIGIN_TEMPLATE_DIR`` can override the lookup directory if a
user wants to substitute their own.

``originpro`` API assumptions (documented so real-Origin verification can
confirm them — see issue #15):

* ``op.new_graph(template=<otpu_path>, lname=<graph_name>)`` creates a
  graph from the template and returns a graph-like object with at least
  ``.name`` and ``.lname`` attributes.
* The returned graph is indexable by layer (``graph[0]``, ``graph[1]``).
  Each layer exposes ``add_plot(sheet, colx=<int>, coly=<int>)`` where
  ``colx`` / ``coly`` are 0-based column indices on the bound sheet.
  Passing only the sheet (``add_plot(sheet)``) leaves the plot with no
  column designation and Origin renders an empty graph window — the
  column indices are mandatory for data to appear, even when the
  template already provides plot types and axis scaling.
* Each layer exposes ``layer.axis(<"x"|"y">)`` returning an axis object
  with mutable ``begin`` and ``end`` float attributes. Setting these
  overrides the template's default scaling so per-cell and comparison
  graphs can share a common axis range (see issue #61). When the
  attribute-style access raises, :func:`_set_axis_limits` falls back to
  LabTalk via ``layer.lt_exec("x.from=<lo>;x.to=<hi>;")``.

The three templates expect distinct column layouts; the bind helpers in
this module encode the layout per category:

* ``chdis`` — flatten columns come in ``(電気量, 電圧)`` pairs, one pair
  per ``(cycle, side)``. One ``add_plot`` call per pair.
* ``cycle`` — flat columns ``[cycle, q_ch, q_dis, ce]``. The template
  is a dual-Y layout: left-Y layer plots ``q_dis`` vs ``cycle``, right-Y
  layer plots ``ce`` vs ``cycle``.
* ``dqdv`` — flatten columns come in ``(電圧, dqdv)`` pairs, one pair
  per ``(cycle, side)``. One ``add_plot`` call per pair.

Because the template handling depends on a runtime file that may not be
present, the helpers here centralize the ``FileNotFoundError`` with a
remediation message rather than spreading that concern across callers.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TEMPLATE_CHDIS = "charge_discharge.otpu"
_TEMPLATE_CYCLE = "cycle_efficiency.otpu"
_TEMPLATE_DQDV = "dqdv.otpu"

# Column indices for cap_df after reset_index(): [cycle, q_ch, q_dis, ce].
# Pinned here so the cycle_efficiency bind is resilient to cap_df column
# reorderings — a future refactor that changes the order must also touch
# these constants (and would fail a focused test).
_CYCLE_COL_CYCLE = 0
_CYCLE_COL_QDIS = 2
_CYCLE_COL_CE = 3

_TEMPLATE_ENV_VAR = "ECHEMPLOT_ORIGIN_TEMPLATE_DIR"

# Bundled template directory, resolved relative to this module's source
# location. We deliberately avoid ``importlib.resources.files`` here:
# ``templates/`` ships without an ``__init__.py`` (it's a data directory,
# not a Python module), which makes ``files()`` return a
# ``MultiplexedPath`` whose ``str()`` repr is wrapper text — not a usable
# filesystem path. ``__file__`` is well-defined for both editable and
# regular wheel installs (hatchling unpacks the wheel into site-packages
# as plain files), so this resolves cleanly in every supported deployment.
_BUNDLED_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_NOT_FOUND_MSG = (
    "Origin template {name!r} not found at {path!s}. "
    "The bundled templates ship with the wheel; this error usually means "
    "ECHEMPLOT_ORIGIN_TEMPLATE_DIR is set to an override directory that is "
    "missing one of charge_discharge.otpu, cycle_efficiency.otpu, or "
    "dqdv.otpu. Unset the env var to fall back to the bundled templates."
)


@dataclass(frozen=True)
class _GraphRanges:
    """Global (min, max) tuples across a Sequence of cells, one per axis.

    Used by :func:`compute_global_ranges` to share a single axis scale
    across every template-backed graph ``push_to_origin`` produces so
    per-cell and comparison plots are directly comparable (issue #61).
    ``None`` on a field means ``_safe_range`` found no finite values for
    that axis — the :func:`_set_axis_limits` helper treats ``None`` as
    "leave the template default in place".
    """

    chdis_x: tuple[float, float] | None
    chdis_y: tuple[float, float] | None
    cycle_x: tuple[float, float] | None
    cycle_left_y: tuple[float, float] | None
    cycle_right_y: tuple[float, float] | None
    dqdv_x: tuple[float, float] | None
    dqdv_y: tuple[float, float] | None


def _safe_range(values: pd.DataFrame | pd.Series | np.ndarray) -> tuple[float, float] | None:
    """Return ``(min, max)`` of ``values`` ignoring NaN; ``None`` if no finite entries.

    Empty inputs and all-NaN / all-inf inputs both yield ``None`` so
    callers can pass the result straight to :func:`_set_axis_limits`,
    which treats ``None`` as a no-op and falls back to the template's
    default axis range.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def compute_global_ranges(cells: Sequence[Any]) -> _GraphRanges:
    """Compute axis ranges across all cells' dataframes for template-backed graphs.

    The function walks every cell once and collects per-axis values from
    ``cell.chdis_df`` (even columns = capacity X, odd = voltage Y),
    ``cell.cap_df`` (``[cycle, q_ch, q_dis, ce]`` with indices pinned by
    ``_CYCLE_COL_*`` constants), and ``cell.dqdv_df`` (even columns =
    voltage X, odd = dQ/dV Y). Concatenated arrays are reduced by
    :func:`_safe_range` so empty / all-NaN inputs surface as ``None`` on
    the returned :class:`_GraphRanges` rather than raising.

    When the input is a single cell, "global" and "that cell's range"
    coincide — the autoscale for a one-cell push is simply the cell's
    own data range.
    """
    chdis_x_vals: list[np.ndarray] = []
    chdis_y_vals: list[np.ndarray] = []
    dqdv_x_vals: list[np.ndarray] = []
    dqdv_y_vals: list[np.ndarray] = []
    cycle_x_vals: list[np.ndarray] = []
    cycle_left_vals: list[np.ndarray] = []
    cycle_right_vals: list[np.ndarray] = []
    for cell in cells:
        chdis = cell.chdis_df
        if chdis.shape[1] >= 2:
            # even = capacity (x), odd = voltage (y)
            chdis_x_vals.append(chdis.iloc[:, 0::2].to_numpy(dtype=float))
            chdis_y_vals.append(chdis.iloc[:, 1::2].to_numpy(dtype=float))
        dqdv = cell.dqdv_df
        if dqdv.shape[1] >= 2:
            dqdv_x_vals.append(dqdv.iloc[:, 0::2].to_numpy(dtype=float))
            dqdv_y_vals.append(dqdv.iloc[:, 1::2].to_numpy(dtype=float))
        # cap_df carries the cycle number in its index; surface it as a
        # column to match the post-``reset_index`` layout used when
        # writing the sheet. See :func:`._worksheets.write_cell_sheets`.
        cap = cell.cap_df.reset_index()
        if cap.shape[1] > _CYCLE_COL_CE:
            cycle_x_vals.append(cap.iloc[:, _CYCLE_COL_CYCLE].to_numpy(dtype=float))
            cycle_left_vals.append(cap.iloc[:, _CYCLE_COL_QDIS].to_numpy(dtype=float))
            cycle_right_vals.append(cap.iloc[:, _CYCLE_COL_CE].to_numpy(dtype=float))
    return _GraphRanges(
        chdis_x=_safe_range(np.concatenate(chdis_x_vals)) if chdis_x_vals else None,
        chdis_y=_safe_range(np.concatenate(chdis_y_vals)) if chdis_y_vals else None,
        cycle_x=_safe_range(np.concatenate(cycle_x_vals)) if cycle_x_vals else None,
        cycle_left_y=_safe_range(np.concatenate(cycle_left_vals)) if cycle_left_vals else None,
        cycle_right_y=_safe_range(np.concatenate(cycle_right_vals)) if cycle_right_vals else None,
        dqdv_x=_safe_range(np.concatenate(dqdv_x_vals)) if dqdv_x_vals else None,
        dqdv_y=_safe_range(np.concatenate(dqdv_y_vals)) if dqdv_y_vals else None,
    )


def _set_axis_limits(layer: Any, axis: str, limits: tuple[float, float] | None) -> None:
    """Apply ``(lo, hi)`` to ``layer``'s named axis. No-op when ``limits`` is ``None``.

    Tries attribute-style originpro API first (``layer.axis(axis).begin``
    and ``.end``) and falls back to LabTalk via ``layer.lt_exec`` when
    that raises. The fallback path is exercised only in real Origin —
    our mocks satisfy the attribute-style call so it never trips in
    CI. Both paths need to be confirmed against real Origin; see issue
    #15 for the tracking checklist.

    Degenerate ``lo == hi`` ranges are left alone so the template's own
    scaling picks a sensible unit-wide window rather than collapsing the
    axis to a zero-width line.
    """
    if limits is None:
        return
    lo, hi = limits
    if lo == hi:
        return  # degenerate range; leave template scaling in place
    try:
        ax = layer.axis(axis)
        ax.begin = lo
        ax.end = hi
    except Exception:  # pragma: no cover - exercised only in real Origin
        # LabTalk fallback. ``layer.lt_exec`` selects ``layer`` implicitly.
        cmd = "x" if axis == "x" else "y"
        layer.lt_exec(f"{cmd}.from={lo};{cmd}.to={hi};")


def _template_path(name: str) -> Path:
    """Resolve ``name`` under the template directory, env-var override first.

    The resolution does **not** check for existence — callers use
    :func:`_require_template` below when the on-disk presence matters.
    Keeping the lookup and the existence-check separate lets tests stub
    out the path without patching ``Path.exists``.
    """
    env = os.environ.get(_TEMPLATE_ENV_VAR)
    if env:
        return Path(env) / name
    return _BUNDLED_TEMPLATE_DIR / name


def _require_template(name: str) -> Path:
    """Return the on-disk template path, raising ``FileNotFoundError`` if absent.

    The error message explicitly enumerates both remediation paths (ship
    the file into the package, or point the env var at it) so users hit
    with a missing template inside Origin don't need to cross-reference
    the docs to unblock themselves.
    """
    path = _template_path(name)
    if not path.exists():
        raise FileNotFoundError(_NOT_FOUND_MSG.format(name=name, path=path))
    return path


def _new_graph_from_template(op: Any, template_name: str, graph_name: str) -> Any:
    """Return a graph instantiated from a template, with no data bound yet."""
    template_path = _require_template(template_name)
    return op.new_graph(template=str(template_path), lname=graph_name)


def _bind_xy_pairs(layer: Any, sheet: Any, ncols: int) -> None:
    """Bind every ``(colx, coly)`` pair from ``sheet`` onto ``layer``.

    Used for ``chdis`` and ``dqdv`` sheets, whose flattened columns line
    up in contiguous pairs: ``(電気量, 電圧)`` for chdis,
    ``(電圧, dqdv)`` for dqdv. Trailing odd columns — which should never
    occur given the upstream pair-producing pipeline — are skipped
    rather than bound as an orphan plot.
    """
    for i in range(0, ncols - 1, 2):
        layer.add_plot(sheet, colx=i, coly=i + 1)


def _bind_cycle(graph: Any, sheet: Any) -> None:
    """Bind the cycle_efficiency dual-Y layout.

    The template has two layers: ``graph[0]`` is the left Y (discharge
    capacity), ``graph[1]`` is the right Y (Coulombic efficiency). Both
    share ``cycle`` as X.
    """
    graph[0].add_plot(sheet, colx=_CYCLE_COL_CYCLE, coly=_CYCLE_COL_QDIS)
    graph[1].add_plot(sheet, colx=_CYCLE_COL_CYCLE, coly=_CYCLE_COL_CE)


def create_cell_plots(
    op: Any,
    cell: Any,
    sheets: dict[str, Any],
    *,
    ranges: _GraphRanges | None = None,
) -> list[Any]:
    """Create the three per-cell graphs from templates.

    ``sheets`` is the mapping returned by
    :func:`echemplot.origin._worksheets.write_cell_sheets`. ``cell``
    is used to read the per-category column counts so the bind helpers
    know how many plot pairs to emit.

    ``ranges`` — when given, the ``(lo, hi)`` tuples on this
    :class:`_GraphRanges` are applied to the corresponding axes on each
    graph via :func:`_set_axis_limits`. Pass ``None`` (the default) to
    keep the template's built-in axis scaling, matching the legacy
    pre-#61 behaviour.
    """
    chdis_graph = _new_graph_from_template(op, _TEMPLATE_CHDIS, f"{cell.name}_chdis_plot")
    _bind_xy_pairs(chdis_graph[0], sheets["chdis"], cell.chdis_df.shape[1])
    if ranges is not None:
        _set_axis_limits(chdis_graph[0], "x", ranges.chdis_x)
        _set_axis_limits(chdis_graph[0], "y", ranges.chdis_y)

    cycle_graph = _new_graph_from_template(op, _TEMPLATE_CYCLE, f"{cell.name}_cycle_plot")
    _bind_cycle(cycle_graph, sheets["cycle"])
    if ranges is not None:
        _set_axis_limits(cycle_graph[0], "x", ranges.cycle_x)
        _set_axis_limits(cycle_graph[0], "y", ranges.cycle_left_y)
        _set_axis_limits(cycle_graph[1], "x", ranges.cycle_x)
        _set_axis_limits(cycle_graph[1], "y", ranges.cycle_right_y)

    dqdv_graph = _new_graph_from_template(op, _TEMPLATE_DQDV, f"{cell.name}_dqdv_plot")
    _bind_xy_pairs(dqdv_graph[0], sheets["dqdv"], cell.dqdv_df.shape[1])
    if ranges is not None:
        _set_axis_limits(dqdv_graph[0], "x", ranges.dqdv_x)
        _set_axis_limits(dqdv_graph[0], "y", ranges.dqdv_y)

    return [chdis_graph, cycle_graph, dqdv_graph]


def create_comparison_plots(
    op: Any,
    cells: Sequence[Any],
    per_cell_sheets: list[dict[str, Any]],
    *,
    ranges: _GraphRanges | None = None,
) -> list[Any]:
    """Create three overlay graphs that combine every cell's sheets.

    One graph per category (``chdis``, ``cycle``, ``dqdv``); each cell's
    sheet contributes its own set of ``add_plot`` calls. Graph names are
    fixed (``comparison_chdis_plot`` etc.) since there is no per-cell
    disambiguation to perform.

    Per-category bind shape mirrors :func:`create_cell_plots`:
    ``chdis`` / ``dqdv`` emit one ``add_plot`` per ``(cycle, side)``
    column pair on ``graph[0]``; ``cycle`` emits two calls per cell, one
    per dual-Y layer (``graph[0]`` = discharge capacity,
    ``graph[1]`` = Coulombic efficiency).

    ``ranges`` applies the same per-axis ``(lo, hi)`` overrides as in
    :func:`create_cell_plots`, so per-cell and comparison graphs share a
    single axis scale — the core of the issue #61 fix.
    """
    chdis_graph = _new_graph_from_template(op, _TEMPLATE_CHDIS, "comparison_chdis_plot")
    for cell, sheets in zip(cells, per_cell_sheets):
        _bind_xy_pairs(chdis_graph[0], sheets["chdis"], cell.chdis_df.shape[1])
    if ranges is not None:
        _set_axis_limits(chdis_graph[0], "x", ranges.chdis_x)
        _set_axis_limits(chdis_graph[0], "y", ranges.chdis_y)

    cycle_graph = _new_graph_from_template(op, _TEMPLATE_CYCLE, "comparison_cycle_plot")
    for _cell, sheets in zip(cells, per_cell_sheets):
        _bind_cycle(cycle_graph, sheets["cycle"])
    if ranges is not None:
        _set_axis_limits(cycle_graph[0], "x", ranges.cycle_x)
        _set_axis_limits(cycle_graph[0], "y", ranges.cycle_left_y)
        _set_axis_limits(cycle_graph[1], "x", ranges.cycle_x)
        _set_axis_limits(cycle_graph[1], "y", ranges.cycle_right_y)

    dqdv_graph = _new_graph_from_template(op, _TEMPLATE_DQDV, "comparison_dqdv_plot")
    for cell, sheets in zip(cells, per_cell_sheets):
        _bind_xy_pairs(dqdv_graph[0], sheets["dqdv"], cell.dqdv_df.shape[1])
    if ranges is not None:
        _set_axis_limits(dqdv_graph[0], "x", ranges.dqdv_x)
        _set_axis_limits(dqdv_graph[0], "y", ranges.dqdv_y)

    return [chdis_graph, cycle_graph, dqdv_graph]
