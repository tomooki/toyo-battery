"""Worksheet-materialization helpers for the Origin adapter.

These helpers translate pandas DataFrames into Origin worksheets via the
``originpro`` API. All functions in this module assume that
:func:`echemplot.origin._require_originpro` has already succeeded and
accept the resulting ``op`` module as a parameter rather than re-importing
it — this keeps the helpers unit-testable by passing in a mock.

``originpro`` API assumptions (documented so real-Origin verification can
confirm them — see issue #15):

* ``op.new_sheet(type="w", lname=<name>)`` creates a new worksheet and
  returns a worksheet-like object. The ``lname`` kwarg sets the sheet's
  long name (the human-readable label used to identify it from plots and
  scripts). The ``type="w"`` argument is the documented spelling for a
  standard worksheet (``"m"`` = matrix).
* The returned object exposes ``from_df(df)``, which writes the DataFrame's
  values + column headers into the sheet. Real ``originpro`` worksheets
  also expose a ``name`` attribute; helpers here return the sheet object
  so the caller can read ``.name`` / ``.lname`` if it wants to wire the
  sheet into a plot template.
* The returned object also exposes ``cols_axis(types)``, where ``types``
  is a string like ``"XYY"`` assigning one plot role per column. We call
  this after ``from_df`` because ``from_df`` leaves every column as
  default-``"Y"``; template-backed plots need at least one ``"X"`` column
  designated or they render with an empty data binding.
* Origin's sheet-name length limit is 32 characters. This module enforces
  that cap in ``_sanitize_sheet_name`` before calling ``new_sheet``.

The flattening convention for pandas MultiIndex columns is
``"{level0}_{level1}_..."`` with ``"_"`` as the join separator. This is
the shape Origin users expect from the legacy v2.01 pipeline and is
stable across column orders (pandas preserves tuple order in
``df.columns``).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("echemplot.origin")

# Origin's worksheet-name length limit. Hard-coded here rather than
# re-read from ``originpro`` because the constant is a platform invariant
# (not a per-version setting) and we want the sanitizer to be callable
# without an ``op`` handle in tests.
_SHEET_NAME_MAX = 32

# Contracted column shape for ``cap_df.reset_index()``: the cycle index
# surfaced as the first column followed by the three derived metrics. The
# metric columns are always English (``q_ch`` / ``q_dis`` / ``ce``) per
# :func:`echemplot.core.capacity.get_cap_df`'s "fixed English" contract,
# regardless of the cell's ``column_lang`` (which only selects the *input*
# quantity label read out of ``chdis_df``). The cycle index is always
# named ``cycle`` (set in ``get_cap_df``'s ``out.index.name = "cycle"``).
#
# This tuple is the single source of truth for both:
# * the cycle_efficiency template's add_plot column indices
#   (:func:`echemplot.origin._plots._bind_cycle`), and
# * the worksheet-write-time contract assertion enforced in
#   :func:`write_cell_sheets`.
#
# A future refactor that changes ``get_cap_df`` 's column names or order
# must update this tuple, otherwise :class:`OriginContractError` fires
# loudly during the next push instead of letting Origin silently bind the
# wrong column to the cycle_efficiency plot.
_CYCLE_SHEET_COLUMNS: tuple[str, ...] = ("cycle", "q_ch", "q_dis", "ce")


class OriginContractError(ValueError):
    """Raised when a DataFrame handed to the Origin push path violates a contract.

    The Origin adapter binds plots to worksheet columns by *index*
    (``add_plot(sheet, colx=..., coly=...)``) since ``originpro``'s
    template-backed graph API takes 0-based positionals, not column
    names. Inside the package we look up those positionals by name
    against a contracted shape (e.g. :data:`_CYCLE_SHEET_COLUMNS` for
    ``cap_df.reset_index()``); when the shape doesn't match, binding
    silently to the wrong columns would produce a graph with the right
    template but wrong-axis data — the worst kind of failure mode for a
    visual diagnostic tool.

    Subclasses :class:`ValueError` so callers that already catch
    ``ValueError`` (e.g. ``push_to_origin``'s ``sg_window`` validation)
    keep working without an extra ``except`` clause.

    The package-internal nature of this exception is intentional: callers
    of :func:`echemplot.origin.push_to_origin` should treat it as a
    "your input frame is malformed" :class:`ValueError`, not a special
    type to catch by name. Re-export from
    :mod:`echemplot.origin` is therefore deliberately omitted.
    """


# Length of the hash suffix appended when a name must be truncated.
# 8 hex chars from a SHA-1 gives ~4e9 uniqueness — ample for per-project
# disambiguation and short enough to leave room for a recognizable prefix.
_HASH_SUFFIX_LEN = 8


def _sanitize_sheet_name(name: str) -> str:
    """Return a sheet name guaranteed to fit Origin's 32-char limit.

    Names at or below the limit pass through unchanged. Longer names are
    truncated and suffixed with an 8-char SHA-1 hash of the full original
    so two long names that share a prefix still get distinct sheets.
    """
    if len(name) <= _SHEET_NAME_MAX:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_LEN]
    # Leave a single "_" separator between prefix and hash.
    prefix_len = _SHEET_NAME_MAX - _HASH_SUFFIX_LEN - 1
    truncated = f"{name[:prefix_len]}_{digest}"
    logger.info("sheet name truncated: %r -> %r (original len %d)", name, truncated, len(name))
    return truncated


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column frame to single-level string columns.

    Non-MultiIndex frames are returned unchanged (a shallow copy is *not*
    made; callers that mutate should copy themselves). MultiIndex columns
    are joined with ``"_"`` after stringifying each level entry. Empty
    level values — common at the outer levels of partial MultiIndex
    frames — are dropped from the join so ``("1", "ch", "")`` becomes
    ``"1_ch"`` rather than ``"1_ch_"``.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat_names = [
        "_".join(str(lvl) for lvl in tup if str(lvl) != "") for tup in df.columns.to_flat_index()
    ]
    out = df.copy()
    # ``dtype=object`` pins the Index to numpy-compatible object dtype
    # instead of pandas' ``StringDtype``. See :func:`_coerce_for_originpro`
    # for why ``StringDtype`` breaks ``originpro.from_df``.
    out.columns = pd.Index(flat_names, dtype=object)
    # Explicit DataFrame wrap pins the return type against pandas-stubs
    # occasionally inferring ``.copy`` as ``Any``. Matches the idiom used
    # in :mod:`echemplot.core.stats`.
    return pd.DataFrame(out)


def _coerce_for_originpro(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce pandas extension dtypes to numpy-compatible dtypes for ``from_df``.

    ``originpro.worksheet.from_df`` dispatches per-column storage via
    ``series.dtype.char`` (and also touches ``df.columns.dtype``). Pandas
    extension dtypes — most commonly ``StringDtype`` when a user has
    ``pd.options.future.infer_string = True`` set, or when Origin's
    embedded Python ships a pandas build that defaults it on — do not
    expose ``.char`` and raise ``AttributeError: 'StringDtype' object has
    no attribute 'char'`` (issue #75).

    The helper:

    * Rebuilds the column ``Index`` with ``dtype=object`` if it isn't
      already a numpy dtype.
    * Converts any column whose dtype is not a numpy dtype to ``object``
      (string-like extension arrays) — preserving values while giving
      originpro a ``.char`` to read.

    No-op when every dtype is already numpy-native, so the helper is
    cheap on the numeric-only cell sheets that dominate the push path.
    """
    new_columns = None
    if not isinstance(df.columns.dtype, np.dtype):
        new_columns = pd.Index(list(df.columns), dtype=object)

    conversions: dict[Any, str] = {
        name: "object" for name, dt in zip(df.columns, df.dtypes) if not isinstance(dt, np.dtype)
    }
    if not conversions and new_columns is None:
        return df

    out = df.astype(conversions) if conversions else df.copy()
    if new_columns is not None:
        out.columns = new_columns
    return pd.DataFrame(out)


def _write_sheet(
    op: Any,
    sheet_name: str,
    df: pd.DataFrame,
    *,
    axis_types: str | None = None,
) -> Any:
    """Create a worksheet and populate it from a DataFrame.

    Assumes ``op.new_sheet(type="w", lname=sheet_name)`` returns a
    worksheet-like object with a ``from_df`` method. Returns that object
    so callers (e.g. :mod:`._plots`) can reference the sheet when
    instantiating a template-backed graph.

    ``axis_types`` — when given, is forwarded to ``wks.cols_axis`` after
    the DataFrame is written. Each character designates one column's
    plot role (``"X"``, ``"Y"``, ``"Z"``, ``"E"`` etc.). Without this
    call, ``from_df`` leaves every column as plain ``"Y"``, which means
    template-backed plots have no X source to bind against and render
    empty. Pass ``None`` to keep the default (e.g. for ``stat_table``,
    which is never plotted).
    """
    safe_name = _sanitize_sheet_name(sheet_name)
    flat = _coerce_for_originpro(_flatten_columns(df))
    wks = op.new_sheet(type="w", lname=safe_name)
    wks.from_df(flat)
    # ``None`` = caller opted out (e.g. ``stat_table``). Empty string =
    # zero-column sheet (``_xy_pairs_axis(0)`` or ``_single_x_axis(0)``);
    # both skip the call since there is nothing to designate.
    if axis_types is not None and axis_types:
        wks.cols_axis(axis_types)
    return wks


def _xy_pairs_axis(ncols: int) -> str:
    """Return ``"XY"`` repeated for ``ncols // 2`` pairs.

    Used for ``chdis`` / ``dqdv`` sheets whose flattened columns come in
    ``(X, Y)`` pairs — ``(電気量, 電圧)`` for chdis and ``(電圧, dqdv)``
    for dqdv. Odd column counts fall back to ``"Y"`` for the trailing
    column, matching Origin's per-column default.
    """
    pairs, tail = divmod(ncols, 2)
    return "XY" * pairs + ("Y" if tail else "")


def _single_x_axis(ncols: int) -> str:
    """Return ``"X"`` followed by ``"Y"`` repeated ``ncols - 1`` times.

    Used for the ``cycle`` sheet whose flattened layout is
    ``[cycle, q_ch, q_dis, ce]`` — one leading X column and the rest Y.
    Empty input yields ``""`` so the ``_write_sheet`` no-op guard
    handles the zero-cycle degenerate case without a ``cols_axis`` call.
    """
    if ncols <= 0:
        return ""
    return "X" + "Y" * (ncols - 1)


def write_cell_sheets(
    op: Any,
    cell: Any,
    *,
    dqdv_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Materialize the three per-cell worksheets.

    Parameters
    ----------
    op
        The ``originpro`` module handle returned by
        :func:`echemplot.origin._require_originpro`.
    cell
        A :class:`echemplot.core.cell.Cell` (typed as ``Any`` here to
        keep the mocked-originpro test path import-light).
    dqdv_df
        Optional pre-computed dQ/dV DataFrame. When ``None`` (the
        default) the cached :attr:`Cell.dqdv_df` property is used, which
        hard-codes the Savitzky-Golay defaults. Callers that need the
        dQ/dV worksheet to reflect a non-default SG ``window_length`` /
        ``polyorder`` must compute the frame via
        :func:`echemplot.core.dqdv.get_dqdv_df` and pass it here —
        :func:`push_to_origin` does so on behalf of GUI callers when
        the user edits the SG window.

    Returns
    -------
    dict
        Mapping ``{"chdis": wks, "cycle": wks, "dqdv": wks}`` of the
        created worksheet objects, keyed by the logical role. Keys are
        the sheet-name suffix (without the ``{cell.name}_`` prefix) so
        callers can look up each sheet without re-computing the
        sanitized full name.
    """
    chdis = cell.chdis_df
    # cap_df carries the cycle number in its index; surface it as a
    # regular column so Origin has an X source for the cycle_efficiency
    # plot. Matches the pattern :func:`write_stat_table` already uses.
    cap = cell.cap_df.reset_index()
    # Contract gate: the cycle_efficiency plot binds columns by 0-based
    # positional index, and those indices are derived in :mod:`._plots`
    # from the canonical name order in :data:`_CYCLE_SHEET_COLUMNS`. If
    # ``get_cap_df``'s output shape ever drifts (column rename, new
    # column inserted in the middle, etc.) we want the push to fail
    # loudly here — a wrong-axis cycle plot is a silent-correctness bug
    # that would slip past every existing test. Verifying both names AND
    # order guards against partial drift (e.g. q_ch and q_dis swapping
    # places: the column set is still correct, but the bind would now
    # plot charge capacity on the discharge-capacity axis).
    actual_cap_cols = tuple(str(c) for c in cap.columns)
    if actual_cap_cols != _CYCLE_SHEET_COLUMNS:
        raise OriginContractError(
            "cap_df.reset_index() column shape violates the cycle_efficiency "
            "plot contract. Expected columns "
            f"{list(_CYCLE_SHEET_COLUMNS)} (in this order); "
            f"got {list(actual_cap_cols)}. "
            "If echemplot.core.capacity.get_cap_df's output schema "
            "changed intentionally, update _CYCLE_SHEET_COLUMNS in "
            "echemplot.origin._worksheets and the corresponding bind "
            "logic in echemplot.origin._plots."
        )
    dqdv = cell.dqdv_df if dqdv_df is None else dqdv_df

    sheets: dict[str, Any] = {}
    sheets["chdis"] = _write_sheet(
        op,
        f"{cell.name}_chdis",
        chdis,
        axis_types=_xy_pairs_axis(chdis.shape[1]),
    )
    # cap_df after reset_index: [cycle, q_ch, q_dis, ce] → "XYYY".
    # q_ch is kept as Y rather than dropped so the user can easily
    # re-plot it without re-running the pipeline. Column order is
    # contract-checked above.
    sheets["cycle"] = _write_sheet(
        op,
        f"{cell.name}_cycle",
        cap,
        axis_types=_single_x_axis(cap.shape[1]),
    )
    sheets["dqdv"] = _write_sheet(
        op,
        f"{cell.name}_dqdv",
        dqdv,
        axis_types=_xy_pairs_axis(dqdv.shape[1]),
    )
    return sheets


def write_stat_table(op: Any, stat_df: pd.DataFrame) -> Any:
    """Create the ``stat_table`` worksheet from a pre-computed DataFrame.

    ``stat_df`` is the output of
    :func:`echemplot.core.stats.stat_table`. Its index (``cell``) is
    reset into a regular column before writing so Origin users see the
    cell name as data rather than as a hidden row label.
    """
    # ``reset_index`` surfaces ``cell`` as the first column, matching the
    # v2.01 layout that downstream Origin scripts expect.
    populated = stat_df.reset_index()
    return _write_sheet(op, "stat_table", populated)
