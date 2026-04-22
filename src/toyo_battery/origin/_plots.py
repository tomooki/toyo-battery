"""Graph-creation helpers for the Origin adapter.

Graphs are instantiated from ``.otpu`` templates ported from OriginLab's
TOYO v2.01 pipeline. The three templates ship with the wheel under
``src/toyo_battery/origin/templates/`` and are picked up automatically;
``TOYO_ORIGIN_TEMPLATE_DIR`` can override the lookup directory if a user
wants to substitute their own.

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
from pathlib import Path
from typing import Any

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

_TEMPLATE_ENV_VAR = "TOYO_ORIGIN_TEMPLATE_DIR"

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
    "TOYO_ORIGIN_TEMPLATE_DIR is set to an override directory that is "
    "missing one of charge_discharge.otpu, cycle_efficiency.otpu, or "
    "dqdv.otpu. Unset the env var to fall back to the bundled templates."
)


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


def create_cell_plots(op: Any, cell: Any, sheets: dict[str, Any]) -> list[Any]:
    """Create the three per-cell graphs from templates.

    ``sheets`` is the mapping returned by
    :func:`toyo_battery.origin._worksheets.write_cell_sheets`. ``cell``
    is used to read the per-category column counts so the bind helpers
    know how many plot pairs to emit.
    """
    chdis_graph = _new_graph_from_template(op, _TEMPLATE_CHDIS, f"{cell.name}_chdis_plot")
    _bind_xy_pairs(chdis_graph[0], sheets["chdis"], cell.chdis_df.shape[1])

    cycle_graph = _new_graph_from_template(op, _TEMPLATE_CYCLE, f"{cell.name}_cycle_plot")
    _bind_cycle(cycle_graph, sheets["cycle"])

    dqdv_graph = _new_graph_from_template(op, _TEMPLATE_DQDV, f"{cell.name}_dqdv_plot")
    _bind_xy_pairs(dqdv_graph[0], sheets["dqdv"], cell.dqdv_df.shape[1])

    return [chdis_graph, cycle_graph, dqdv_graph]


def create_comparison_plots(
    op: Any,
    cells: Sequence[Any],
    per_cell_sheets: list[dict[str, Any]],
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
    """
    chdis_graph = _new_graph_from_template(op, _TEMPLATE_CHDIS, "comparison_chdis_plot")
    for cell, sheets in zip(cells, per_cell_sheets):
        _bind_xy_pairs(chdis_graph[0], sheets["chdis"], cell.chdis_df.shape[1])

    cycle_graph = _new_graph_from_template(op, _TEMPLATE_CYCLE, "comparison_cycle_plot")
    for _cell, sheets in zip(cells, per_cell_sheets):
        _bind_cycle(cycle_graph, sheets["cycle"])

    dqdv_graph = _new_graph_from_template(op, _TEMPLATE_DQDV, "comparison_dqdv_plot")
    for cell, sheets in zip(cells, per_cell_sheets):
        _bind_xy_pairs(dqdv_graph[0], sheets["dqdv"], cell.dqdv_df.shape[1])

    return [chdis_graph, cycle_graph, dqdv_graph]
