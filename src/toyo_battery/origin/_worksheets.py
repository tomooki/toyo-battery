"""Worksheet-materialization helpers for the Origin adapter.

These helpers translate pandas DataFrames into Origin worksheets via the
``originpro`` API. All functions in this module assume that
:func:`toyo_battery.origin._require_originpro` has already succeeded and
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
from typing import Any

import pandas as pd

# Origin's worksheet-name length limit. Hard-coded here rather than
# re-read from ``originpro`` because the constant is a platform invariant
# (not a per-version setting) and we want the sanitizer to be callable
# without an ``op`` handle in tests.
_SHEET_NAME_MAX = 32

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
    return f"{name[:prefix_len]}_{digest}"


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
    out.columns = pd.Index(flat_names)
    # Explicit DataFrame wrap pins the return type against pandas-stubs
    # occasionally inferring ``.copy`` as ``Any``. Matches the idiom used
    # in :mod:`toyo_battery.core.stats`.
    return pd.DataFrame(out)


def _write_sheet(op: Any, sheet_name: str, df: pd.DataFrame) -> Any:
    """Create a worksheet and populate it from a DataFrame.

    Assumes ``op.new_sheet(type="w", lname=sheet_name)`` returns a
    worksheet-like object with a ``from_df`` method. Returns that object
    so callers (e.g. :mod:`._plots`) can reference the sheet when
    instantiating a template-backed graph.
    """
    safe_name = _sanitize_sheet_name(sheet_name)
    flat = _flatten_columns(df)
    wks = op.new_sheet(type="w", lname=safe_name)
    wks.from_df(flat)
    return wks


def write_cell_sheets(op: Any, cell: Any) -> dict[str, Any]:
    """Materialize the three per-cell worksheets.

    Parameters
    ----------
    op
        The ``originpro`` module handle returned by
        :func:`toyo_battery.origin._require_originpro`.
    cell
        A :class:`toyo_battery.core.cell.Cell` (typed as ``Any`` here to
        keep the mocked-originpro test path import-light).

    Returns
    -------
    dict
        Mapping ``{"chdis": wks, "cycle": wks, "dqdv": wks}`` of the
        created worksheet objects, keyed by the logical role. Keys are
        the sheet-name suffix (without the ``{cell.name}_`` prefix) so
        callers can look up each sheet without re-computing the
        sanitized full name.
    """
    sheets: dict[str, Any] = {}
    sheets["chdis"] = _write_sheet(op, f"{cell.name}_chdis", cell.chdis_df)
    sheets["cycle"] = _write_sheet(op, f"{cell.name}_cycle", cell.cap_df)
    sheets["dqdv"] = _write_sheet(op, f"{cell.name}_dqdv", cell.dqdv_df)
    return sheets


def write_stat_table(op: Any, stat_df: pd.DataFrame) -> Any:
    """Create the ``stat_table`` worksheet from a pre-computed DataFrame.

    ``stat_df`` is the output of
    :func:`toyo_battery.core.stats.stat_table`. Its index (``cell``) is
    reset into a regular column before writing so Origin users see the
    cell name as data rather than as a hidden row label.
    """
    # ``reset_index`` surfaces ``cell`` as the first column, matching the
    # v2.01 layout that downstream Origin scripts expect.
    populated = stat_df.reset_index()
    return _write_sheet(op, "stat_table", populated)
