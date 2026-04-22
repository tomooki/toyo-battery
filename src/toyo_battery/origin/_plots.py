"""Graph-creation helpers for the Origin adapter.

Graphs are instantiated from ``.otpu`` templates shipped by OriginLab's
TOYO v2.01 pipeline. The templates themselves are not bundled with this
package (license review pending — see issue #15); users either drop them
into ``src/toyo_battery/origin/templates/`` at install time or point
``TOYO_ORIGIN_TEMPLATE_DIR`` at an external directory.

``originpro`` API assumptions (documented so real-Origin verification can
confirm them — see issue #15):

* ``op.new_graph(template=<otpu_path>, lname=<graph_name>)`` creates a
  graph from the template and returns a graph-like object with at least
  ``.name`` and ``.lname`` attributes.
* The returned graph exposes a ``set_xy`` / ``add_plot`` mechanism. We
  use the lowest-common-denominator path here: resolve the first layer
  via ``graph[0]`` (real ``originpro`` graphs are indexable by layer
  index) and call ``add_plot(sheet, coly=<col>, colx=<col>)``. Real
  ``.otpu`` templates supply their own plot types + axis scaling, so the
  data bind is the only thing we do from Python.

Because the template handling depends on a runtime file that may not be
present, the helpers here centralize the ``FileNotFoundError`` with a
remediation message rather than spreading that concern across callers.
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path
from typing import Any

_TEMPLATE_CHDIS = "charge_discharge.otpu"
_TEMPLATE_CYCLE = "cycle_efficiency.otpu"
_TEMPLATE_DQDV = "dqdv.otpu"

_TEMPLATE_ENV_VAR = "TOYO_ORIGIN_TEMPLATE_DIR"

_NOT_FOUND_MSG = (
    "Origin template {name!r} not found at {path!s}. "
    "Copy the v2.01 .otpu templates into the package directory "
    "(src/toyo_battery/origin/templates/), or set the environment variable "
    "TOYO_ORIGIN_TEMPLATE_DIR to a directory containing "
    "charge_discharge.otpu, cycle_efficiency.otpu, and dqdv.otpu."
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
    # ``importlib.resources.files`` returns a ``Traversable`` which on
    # filesystem packages is a ``PosixPath``/``WindowsPath`` subclass —
    # wrapping in ``Path`` normalizes the return type for downstream
    # consumers and avoids a mypy complaint about the union.
    return Path(str(importlib.resources.files("toyo_battery.origin.templates"))) / name


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


def _make_graph(op: Any, template_name: str, graph_name: str, sheet: Any) -> Any:
    """Instantiate a graph from a template and bind it to a sheet.

    The sheet-binding step is a single ``add_plot`` call on the graph's
    first layer. We intentionally do not specify a plot type — the
    ``.otpu`` template carries the type, axis limits, legend, etc.
    """
    template_path = _require_template(template_name)
    graph = op.new_graph(template=str(template_path), lname=graph_name)
    # ``graph[0]`` is the first layer. Real ``originpro`` graphs are
    # indexable; the mocked test uses a MagicMock where indexing returns
    # another MagicMock, so this line is exercised in tests.
    layer = graph[0]
    layer.add_plot(sheet)
    return graph


def create_cell_plots(op: Any, cell: Any, sheets: dict[str, Any]) -> list[Any]:
    """Create the three per-cell graphs from templates.

    ``sheets`` is the mapping returned by
    :func:`toyo_battery.origin._worksheets.write_cell_sheets`.
    """
    graphs: list[Any] = []
    graphs.append(
        _make_graph(op, _TEMPLATE_CHDIS, f"{cell.name}_chdis_plot", sheets["chdis"]),
    )
    graphs.append(
        _make_graph(op, _TEMPLATE_CYCLE, f"{cell.name}_cycle_plot", sheets["cycle"]),
    )
    graphs.append(
        _make_graph(op, _TEMPLATE_DQDV, f"{cell.name}_dqdv_plot", sheets["dqdv"]),
    )
    return graphs


def create_comparison_plots(op: Any, per_cell_sheets: list[dict[str, Any]]) -> list[Any]:
    """Create three overlay graphs that combine every cell's sheets.

    One graph per category (``chdis``, ``cycle``, ``dqdv``); each graph's
    first layer has one ``add_plot`` call per cell. Graph names are
    fixed (``comparison_chdis_plot`` etc.) since there is no per-cell
    disambiguation to perform.
    """
    graphs: list[Any] = []
    for category, template_name, graph_name in (
        ("chdis", _TEMPLATE_CHDIS, "comparison_chdis_plot"),
        ("cycle", _TEMPLATE_CYCLE, "comparison_cycle_plot"),
        ("dqdv", _TEMPLATE_DQDV, "comparison_dqdv_plot"),
    ):
        template_path = _require_template(template_name)
        graph = op.new_graph(template=str(template_path), lname=graph_name)
        layer = graph[0]
        for sheets in per_cell_sheets:
            layer.add_plot(sheets[category])
        graphs.append(graph)
    return graphs
