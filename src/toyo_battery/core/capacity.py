"""Per-cycle charge/discharge capacity and Coulombic efficiency.

Computes, from a :mod:`toyo_battery.core.chdis` output, one row per cycle
with the maximum charge capacity (``q_ch``), maximum discharge capacity
(``q_dis``), and Coulombic efficiency ``ce = 100 * q_dis / q_ch`` (in %).

The derived column names are **fixed English** (``q_ch``, ``q_dis``,
``ce``) regardless of the ``column_lang`` used for the input frame —
``column_lang`` selects which *input* quantity label (JA ``電気量`` or EN
``capacity``) to read from, not the output schema.

Rewrite note (vs. legacy TOYO_Origin_2.01): v2.01's ``get_cap_df`` pivoted
through ``df.query('状態 in [...]')`` on the raw frame and used
``df["状態"][1]`` to decide whether the first cycle was charge- or
discharge-first. That check is positionally brittle (row index 1 can be a
rest row) and ties capacity computation to raw-state string semantics.
Here we operate on the already-segmented ``chdis_df``, whose
charge-first normalization is resolved once in :mod:`chdis`. This frees
``get_cap_df`` from ever touching raw state labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from toyo_battery.io.schema import JA_TO_EN, ColumnLang

_JA_COLS: dict[str, str] = {
    "capacity": "電気量",
}


def _resolve_cols(column_lang: ColumnLang) -> dict[str, str]:
    if column_lang == "ja":
        return _JA_COLS
    return {k: JA_TO_EN[v] for k, v in _JA_COLS.items()}


def _empty_result() -> pd.DataFrame:
    idx = pd.Index([], dtype="int64", name="cycle")
    return pd.DataFrame(
        {
            "q_ch": pd.Series([], dtype="float64"),
            "q_dis": pd.Series([], dtype="float64"),
            "ce": pd.Series([], dtype="float64"),
        },
        index=idx,
    )


def get_cap_df(chdis_df: pd.DataFrame, *, column_lang: ColumnLang = "ja") -> pd.DataFrame:
    """Per-cycle charge/discharge capacity and Coulombic efficiency.

    Parameters
    ----------
    chdis_df
        Output of :func:`toyo_battery.core.chdis.get_chdis_df`. Must have a
        3-level column MultiIndex ``(cycle, side, quantity)`` where
        ``side`` is in ``{"ch", "dis"}`` and ``quantity`` contains the
        capacity label (``電気量`` or ``capacity``).
    column_lang
        Language of the input's ``quantity`` level. Does *not* affect the
        output column names — those are always English.

    Returns
    -------
    DataFrame
        Index is ``cycle`` (int). Columns are ``["q_ch", "q_dis", "ce"]``
        (all float). ``ce`` is in percent. ``ce`` is ``NaN`` when
        ``q_ch == 0`` (not ``inf``). Missing ch or dis for a cycle yields
        ``NaN`` for that column; the cycle row is preserved.

    Raises
    ------
    KeyError
        If ``chdis_df``'s ``quantity`` level does not contain the
        capacity label for the requested ``column_lang``.
    """
    cols = _resolve_cols(column_lang)
    cap_col = cols["capacity"]

    # Empty / no-column input → empty result with correct schema.
    if chdis_df.empty and chdis_df.columns.empty:
        return _empty_result()

    # Validate that the chdis_df has the expected 3-level MultiIndex and
    # carries the requested capacity quantity label. We check against the
    # quantity level rather than attempting a cross-section so the error
    # message is explicit about the column_lang mismatch.
    if not isinstance(chdis_df.columns, pd.MultiIndex) or chdis_df.columns.nlevels != 3:
        raise KeyError(
            "chdis_df.columns must be a 3-level MultiIndex "
            "(cycle, side, quantity); "
            f"got {chdis_df.columns!r}"
        )

    quantity_values = set(chdis_df.columns.get_level_values("quantity"))
    if cap_col not in quantity_values:
        raise KeyError(
            f"chdis_df missing required quantity {cap_col!r} "
            f"for column_lang={column_lang!r}; got quantities={sorted(quantity_values)}"
        )

    # Cross-section the capacity quantity → columns become (cycle, side).
    cap_only = chdis_df.xs(cap_col, axis=1, level="quantity")

    # Max per segment, skipping NaN. Result indexed by (cycle, side).
    per_segment_max = cap_only.max(axis=0, skipna=True)

    # Unstack side → rows=cycle, columns=side. Reindex to ensure both
    # "ch" and "dis" exist even if only one side has any data globally.
    wide = per_segment_max.unstack(level="side")
    wide = wide.reindex(columns=["ch", "dis"])

    # Preserve every cycle from the input, including ones with only one side.
    cycles = sorted(set(chdis_df.columns.get_level_values("cycle")))
    wide = wide.reindex(index=cycles)

    q_ch = wide["ch"].astype("float64")
    q_dis = wide["dis"].astype("float64")

    # Coulombic efficiency — guard q_ch == 0 as NaN (not inf / not error).
    # `where(cond, other=NaN)` keeps the value where cond is True and
    # substitutes NaN elsewhere; cast q_ch to a safe divisor first.
    with np.errstate(divide="ignore", invalid="ignore"):
        ce = 100.0 * q_dis / q_ch
    ce = ce.where(q_ch != 0, other=np.nan)

    out = pd.DataFrame(
        {
            "q_ch": q_ch.to_numpy(dtype="float64"),
            "q_dis": q_dis.to_numpy(dtype="float64"),
            "ce": ce.to_numpy(dtype="float64"),
        },
        index=pd.Index([int(c) for c in cycles], name="cycle", dtype="int64"),
    )
    return out
