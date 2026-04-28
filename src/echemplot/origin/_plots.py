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
  graphs can share a common axis range (see issue #61). The attribute
  path is treated as "tentative": :func:`_set_axis_limits` writes the
  values, then reads them back and — if the round-trip disagrees or
  the attribute access raised — falls back to LabTalk via
  ``op.lt_exec("x.from=<lo>;x.to=<hi>;")``. The module-level
  :func:`op.lt_exec` targets the currently-active graph, which is the
  graph we just created via :func:`op.new_graph`; it is the documented
  scripting entry point and does not assume any per-layer attribute
  contract.
* Each layer exposes ``layer.rescale()``, which fits the axes to the
  currently-bound data. The template defaults do not know the data
  range, so without this call the graph renders with the template's
  original scale and data outside that window is clipped. The bind
  helpers below call ``rescale`` after their ``add_plot`` loop;
  :func:`_set_axis_limits` (called later from the orchestrators when
  explicit ``ranges`` are provided) subsequently overrides those
  autoscaled values, so the issue #61 shared-axis path is unaffected.

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

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from echemplot.origin._worksheets import _CYCLE_SHEET_COLUMNS

_logger = logging.getLogger("echemplot.origin")

# Exception types that the originpro attribute-API path is observed to raise
# when the layer / axis bridge is unavailable or misbehaves. ``originpro``
# does not document a public exception hierarchy, so we whitelist the
# narrow set of types we have actually seen in the wild and in the test
# mocks: ``RuntimeError`` (the generic raise pattern from the C bridge
# wrapper), ``AttributeError`` (older ``originpro`` builds where ``layer``
# does not expose ``axis()`` at all), and ``TypeError`` (defensive — covers
# ``axis()`` returning ``None`` or some other non-axis sentinel that fails
# attribute writes). Bare ``Exception`` is deliberately avoided so genuine
# bugs (e.g. a typo in this module) surface rather than silently degrading
# to template-default scaling.
_AXIS_API_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    AttributeError,
    TypeError,
)

_TEMPLATE_CHDIS = "charge_discharge.otpu"
_TEMPLATE_CYCLE = "cycle_efficiency.otpu"
_TEMPLATE_DQDV = "dqdv.otpu"

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
    ``cell.cap_df`` (resolved to ``[cycle, q_ch, q_dis, ce]`` after
    ``reset_index()`` and looked up by name — see
    :data:`_CYCLE_SHEET_COLUMNS`), and ``cell.dqdv_df`` (even columns =
    voltage X, odd = dQ/dV Y). Concatenated arrays are reduced by
    :func:`_safe_range` so empty / all-NaN inputs surface as ``None`` on
    the returned :class:`_GraphRanges` rather than raising.

    When the input is a single cell, "global" and "that cell's range"
    coincide — the autoscale for a one-cell push is simply the cell's
    own data range.

    Cells whose ``cap_df.reset_index()`` does not satisfy the
    :data:`_CYCLE_SHEET_COLUMNS` contract are silently skipped for the
    cycle_x / cycle_left_y / cycle_right_y aggregation here — the hard
    failure is deferred to
    :func:`echemplot.origin._worksheets.write_cell_sheets`, which raises
    :class:`echemplot.origin._worksheets.OriginContractError` at
    worksheet-write time. Surfacing the contract violation from the
    range-aggregation path would tie compute_global_ranges to the same
    error semantics for no benefit; the write path is the single
    enforcement point.
    """
    chdis_x_vals: list[np.ndarray] = []
    chdis_y_vals: list[np.ndarray] = []
    dqdv_x_vals: list[np.ndarray] = []
    dqdv_y_vals: list[np.ndarray] = []
    cycle_x_vals: list[np.ndarray] = []
    cycle_left_vals: list[np.ndarray] = []
    cycle_right_vals: list[np.ndarray] = []
    # Each cell's chdis_df / dqdv_df can have a different number of
    # (cycle, side) column pairs and a different row count, so the 2D
    # arrays from different cells do not share a common axis-1 size.
    # ``np.concatenate`` (default axis=0) requires every non-concat axis
    # to match exactly, which would raise "along dimension 1, size N vs.
    # size M" when cells have differing pair counts. ``_safe_range``
    # ravels its input before computing min/max anyway, so we ravel here
    # at append time and concatenate flat 1D arrays — equivalent result,
    # no shape constraint.
    for cell in cells:
        chdis = cell.chdis_df
        if chdis.shape[1] >= 2:
            # even = capacity (x), odd = voltage (y)
            chdis_x_vals.append(chdis.iloc[:, 0::2].to_numpy(dtype=float).ravel())
            chdis_y_vals.append(chdis.iloc[:, 1::2].to_numpy(dtype=float).ravel())
        dqdv = cell.dqdv_df
        if dqdv.shape[1] >= 2:
            dqdv_x_vals.append(dqdv.iloc[:, 0::2].to_numpy(dtype=float).ravel())
            dqdv_y_vals.append(dqdv.iloc[:, 1::2].to_numpy(dtype=float).ravel())
        # cap_df carries the cycle number in its index; surface it as a
        # column to match the post-``reset_index`` layout used when
        # writing the sheet. See :func:`._worksheets.write_cell_sheets`.
        cap = cell.cap_df.reset_index()
        cap_cols = list(cap.columns)
        # Only contribute to the aggregation when every contracted
        # column is present. The write path is the contract gate; here
        # we just need the lookups to be safe so a malformed cell does
        # not raise mid-aggregation before write_cell_sheets can produce
        # the proper OriginContractError.
        if all(name in cap_cols for name in _CYCLE_SHEET_COLUMNS):
            cycle_x_vals.append(cap["cycle"].to_numpy(dtype=float))
            cycle_left_vals.append(cap["q_dis"].to_numpy(dtype=float))
            cycle_right_vals.append(cap["ce"].to_numpy(dtype=float))
    return _GraphRanges(
        chdis_x=_safe_range(np.concatenate(chdis_x_vals)) if chdis_x_vals else None,
        chdis_y=_safe_range(np.concatenate(chdis_y_vals)) if chdis_y_vals else None,
        cycle_x=_safe_range(np.concatenate(cycle_x_vals)) if cycle_x_vals else None,
        cycle_left_y=_safe_range(np.concatenate(cycle_left_vals)) if cycle_left_vals else None,
        cycle_right_y=_safe_range(np.concatenate(cycle_right_vals)) if cycle_right_vals else None,
        dqdv_x=_safe_range(np.concatenate(dqdv_x_vals)) if dqdv_x_vals else None,
        dqdv_y=_safe_range(np.concatenate(dqdv_y_vals)) if dqdv_y_vals else None,
    )


def _set_axis_limits(
    layer: Any,
    axis: str,
    limits: tuple[float, float] | None,
    *,
    op: Any | None = None,
    strict: bool = False,
) -> None:
    """Apply ``(lo, hi)`` to ``layer``'s named axis. No-op when ``limits`` is ``None``.

    Strategy: try the attribute-style originpro API
    (``layer.axis(axis).begin`` / ``.end``), then **verify** by reading
    the values back — a plain Python object silently accepts unknown
    attribute writes, which would let a missing real-Origin contract
    slip through as a degraded "template default scaling" graph instead
    of a raised error. If either the set or the readback disagrees with
    ``(lo, hi)``, fall back to LabTalk via ``op.lt_exec`` (documented
    module-level function) against the currently-active graph, which
    is the one the caller just created.

    ``op`` is optional for backward compatibility with call sites that
    don't thread the module through; when omitted the fallback is
    suppressed. Callers that care about the degraded case
    (:func:`create_cell_plots`, :func:`create_comparison_plots`) always
    pass ``op`` explicitly.

    Degenerate ``lo == hi`` ranges are left alone so the template's own
    scaling picks a sensible unit-wide window rather than collapsing the
    axis to a zero-width line. NaN / inf are filtered upstream by
    :func:`_safe_range`, so they never reach this helper.

    Exception handling: the attribute-API path catches a narrow tuple
    of exception types observed when calling ``originpro`` from real
    Origin (``RuntimeError`` from the C bridge, ``AttributeError`` from
    older builds without ``layer.axis()``, ``TypeError`` for
    ``axis()`` returning a sentinel that fails attribute writes). Other
    exception types propagate unchanged so genuine bugs surface. When
    a known exception is caught, a ``WARNING`` is logged on the
    ``echemplot.origin`` logger naming the axis, the attempted range,
    and the exception type+message; the call then falls through to the
    LabTalk path. Set ``strict=True`` to re-raise the caught exception
    instead of warn-and-continue — useful for callers that prefer a
    hard failure over a silently degraded graph.

    Both the attribute and LabTalk paths ultimately need real-Origin
    verification (see issue #15). The round-trip check is the best-effort
    safety net we can run from the Python side.
    """
    if limits is None:
        return
    lo, hi = limits
    if lo == hi:
        return  # degenerate range; leave template scaling in place

    attr_ok = False
    try:
        ax = layer.axis(axis)
        ax.begin = lo
        ax.end = hi
        # Round-trip verification: if ``ax`` is a plain Python object it
        # accepted the writes silently but the real Origin axis is
        # untouched — the graph would render with template defaults and
        # no error would surface. ``math.isclose`` guards against
        # float-roundtrip noise from the originpro C bridge; the abs_tol
        # covers near-zero ranges (e.g. dQ/dV around the baseline).
        attr_ok = math.isclose(float(ax.begin), lo, rel_tol=1e-9, abs_tol=1e-12) and math.isclose(
            float(ax.end), hi, rel_tol=1e-9, abs_tol=1e-12
        )
    except _AXIS_API_EXCEPTIONS as exc:
        if strict:
            raise
        _logger.warning(
            "originpro axis attribute API failed for axis=%r range=(%r, %r): %s: %s; "
            "falling back to LabTalk",
            axis,
            lo,
            hi,
            type(exc).__name__,
            exc,
        )
        attr_ok = False

    if attr_ok:
        return

    # Fallback: LabTalk against the active graph. ``op.lt_exec`` is the
    # documented module-level entry point; individual layer objects do
    # not reliably expose ``lt_exec``.
    if op is None:  # pragma: no cover - defensive; call sites always pass op
        return
    cmd = "x" if axis == "x" else "y"
    op.lt_exec(f"{cmd}.from={lo};{cmd}.to={hi};")


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

    ``layer.rescale()`` fits the axes to the just-bound data so the
    template's default scaling does not clip the plot. When called
    repeatedly on the same layer from a comparison loop the call is
    idempotent — each invocation expands the axes to cover everything
    currently bound — so the final state is correct regardless of the
    number of cells.
    """
    for i in range(0, ncols - 1, 2):
        layer.add_plot(sheet, colx=i, coly=i + 1)
    layer.rescale()


def _bind_cycle(graph: Any, sheet: Any) -> None:
    """Bind the cycle_efficiency dual-Y layout.

    The template has two layers: ``graph[0]`` is the left Y (discharge
    capacity), ``graph[1]`` is the right Y (Coulombic efficiency). Both
    share ``cycle`` as X. Each layer is rescaled after binding so the
    autoscale survives the template's default axis range.

    Column indices are resolved by **name** against
    :data:`_CYCLE_SHEET_COLUMNS` rather than baked-in positionals, so a
    future ``get_cap_df`` schema change cannot silently shift which
    column gets bound to which axis. The contract is enforced at
    worksheet-write time by
    :func:`echemplot.origin._worksheets.write_cell_sheets`, so by the
    time we reach this helper the sheet is guaranteed to expose the
    four contracted columns in their canonical order — the ``index()``
    lookups below are therefore O(1) on a fixed-size tuple and cannot
    miss.
    """
    cycle_idx = _CYCLE_SHEET_COLUMNS.index("cycle")
    qdis_idx = _CYCLE_SHEET_COLUMNS.index("q_dis")
    ce_idx = _CYCLE_SHEET_COLUMNS.index("ce")
    graph[0].add_plot(sheet, colx=cycle_idx, coly=qdis_idx)
    graph[1].add_plot(sheet, colx=cycle_idx, coly=ce_idx)
    graph[0].rescale()
    graph[1].rescale()


def create_cell_plots(
    op: Any,
    cell: Any,
    sheets: dict[str, Any],
    *,
    ranges: _GraphRanges | None = None,
    strict_axis: bool = False,
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

    ``strict_axis`` is forwarded to :func:`_set_axis_limits`. With the
    default ``False`` an originpro attribute-API failure is logged at
    WARNING and the LabTalk fallback runs; with ``True`` the caught
    exception is re-raised so the caller can fail-fast on a degraded
    axis range instead of silently shipping a template-default graph.
    """
    chdis_graph = _new_graph_from_template(op, _TEMPLATE_CHDIS, f"{cell.name}_chdis_plot")
    _bind_xy_pairs(chdis_graph[0], sheets["chdis"], cell.chdis_df.shape[1])
    if ranges is not None:
        _set_axis_limits(chdis_graph[0], "x", ranges.chdis_x, op=op, strict=strict_axis)
        _set_axis_limits(chdis_graph[0], "y", ranges.chdis_y, op=op, strict=strict_axis)

    cycle_graph = _new_graph_from_template(op, _TEMPLATE_CYCLE, f"{cell.name}_cycle_plot")
    _bind_cycle(cycle_graph, sheets["cycle"])
    if ranges is not None:
        _set_axis_limits(cycle_graph[0], "x", ranges.cycle_x, op=op, strict=strict_axis)
        _set_axis_limits(cycle_graph[0], "y", ranges.cycle_left_y, op=op, strict=strict_axis)
        _set_axis_limits(cycle_graph[1], "x", ranges.cycle_x, op=op, strict=strict_axis)
        _set_axis_limits(cycle_graph[1], "y", ranges.cycle_right_y, op=op, strict=strict_axis)

    dqdv_graph = _new_graph_from_template(op, _TEMPLATE_DQDV, f"{cell.name}_dqdv_plot")
    _bind_xy_pairs(dqdv_graph[0], sheets["dqdv"], cell.dqdv_df.shape[1])
    if ranges is not None:
        _set_axis_limits(dqdv_graph[0], "x", ranges.dqdv_x, op=op, strict=strict_axis)
        _set_axis_limits(dqdv_graph[0], "y", ranges.dqdv_y, op=op, strict=strict_axis)

    return [chdis_graph, cycle_graph, dqdv_graph]


def create_comparison_plots(
    op: Any,
    cells: Sequence[Any],
    per_cell_sheets: list[dict[str, Any]],
    *,
    ranges: _GraphRanges | None = None,
    strict_axis: bool = False,
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

    ``strict_axis`` is forwarded to :func:`_set_axis_limits` with the
    same semantics as :func:`create_cell_plots`.
    """
    chdis_graph = _new_graph_from_template(op, _TEMPLATE_CHDIS, "comparison_chdis_plot")
    for cell, sheets in zip(cells, per_cell_sheets):
        _bind_xy_pairs(chdis_graph[0], sheets["chdis"], cell.chdis_df.shape[1])
    if ranges is not None:
        _set_axis_limits(chdis_graph[0], "x", ranges.chdis_x, op=op, strict=strict_axis)
        _set_axis_limits(chdis_graph[0], "y", ranges.chdis_y, op=op, strict=strict_axis)

    cycle_graph = _new_graph_from_template(op, _TEMPLATE_CYCLE, "comparison_cycle_plot")
    for _cell, sheets in zip(cells, per_cell_sheets):
        _bind_cycle(cycle_graph, sheets["cycle"])
    if ranges is not None:
        _set_axis_limits(cycle_graph[0], "x", ranges.cycle_x, op=op, strict=strict_axis)
        _set_axis_limits(cycle_graph[0], "y", ranges.cycle_left_y, op=op, strict=strict_axis)
        _set_axis_limits(cycle_graph[1], "x", ranges.cycle_x, op=op, strict=strict_axis)
        _set_axis_limits(cycle_graph[1], "y", ranges.cycle_right_y, op=op, strict=strict_axis)

    dqdv_graph = _new_graph_from_template(op, _TEMPLATE_DQDV, "comparison_dqdv_plot")
    for cell, sheets in zip(cells, per_cell_sheets):
        _bind_xy_pairs(dqdv_graph[0], sheets["dqdv"], cell.dqdv_df.shape[1])
    if ranges is not None:
        _set_axis_limits(dqdv_graph[0], "x", ranges.dqdv_x, op=op, strict=strict_axis)
        _set_axis_limits(dqdv_graph[0], "y", ranges.dqdv_y, op=op, strict=strict_axis)

    return [chdis_graph, cycle_graph, dqdv_graph]
