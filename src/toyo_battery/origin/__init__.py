"""Origin adapter subpackage.

`originpro` is shipped with OriginLab's embedded Python; it is NOT listed as a
PyPI dependency. Importing this subpackage outside Origin is fine — calling
:func:`push_to_origin` (which routes through :func:`_require_originpro`) is
what raises ``ImportError`` at runtime.

The package is split into three small modules so the helpers can be unit
tested against a mocked ``originpro`` module without importing the top-level
:func:`push_to_origin` orchestrator:

* :mod:`._worksheets` — DataFrame → Origin worksheet helpers.
* :mod:`._plots` — template-backed graph creation.
* :mod:`.templates` — the ``.otpu`` asset directory (README only in this
  PR; users provide the proprietary templates — see
  ``templates/README.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from toyo_battery.origin._plots import create_cell_plots, create_comparison_plots
from toyo_battery.origin._worksheets import write_cell_sheets, write_stat_table

if TYPE_CHECKING:
    from toyo_battery.core.cell import Cell


def _require_originpro() -> object:
    try:
        import originpro as op
    except ImportError as e:
        raise ImportError(
            "toyo_battery.origin requires `originpro`, which is provided by "
            "OriginLab's embedded Python. Run this module from Origin's Python."
        ) from e
    return op


def push_to_origin(
    cells: Sequence[Cell],
    *,
    project_path: str | None = None,
    stat_cycles: Sequence[int] = (10, 50),
) -> None:
    """Populate the current Origin project with per-cell sheets + plots + stats.

    For every cell in ``cells`` this creates three worksheets
    (``{cell.name}_chdis``, ``{cell.name}_cycle``, ``{cell.name}_dqdv``)
    populated from ``cell.chdis_df``, ``cell.cap_df``, and ``cell.dqdv_df``
    respectively, then three graphs instantiated from the v2.01 ``.otpu``
    templates (``charge_discharge.otpu``, ``cycle_efficiency.otpu``,
    ``dqdv.otpu``). When ``len(cells) > 1`` a further three overlay-style
    comparison graphs are produced. Finally a ``stat_table`` worksheet is
    written from :func:`toyo_battery.core.stats.stat_table`.

    Parameters
    ----------
    cells
        One or more :class:`toyo_battery.core.cell.Cell` instances.
    project_path
        If provided, ``op.open(project_path)`` is called at the start and
        ``op.save(project_path)`` at the end. When ``None`` the function
        operates on the current in-memory project without touching disk.
    stat_cycles
        Cycle numbers forwarded to
        :func:`toyo_battery.core.stats.stat_table` as ``target_cycles``.

    Raises
    ------
    ImportError
        From :func:`_require_originpro` when ``originpro`` is not available
        (i.e. when the caller is not running inside Origin).
    FileNotFoundError
        When a required ``.otpu`` template cannot be located under the
        package ``templates/`` directory or
        ``$TOYO_ORIGIN_TEMPLATE_DIR``. The message names both remediation
        paths.
    """
    op = _require_originpro()

    # Import inside the function so users who only ever call the helpers
    # directly don't pay the stats-import cost. ``stat_table`` pulls in
    # scipy.integrate at import time.
    from toyo_battery.core.stats import stat_table

    if project_path is not None:
        op.open(project_path)  # type: ignore[attr-defined]

    per_cell_sheets: list[dict[str, object]] = []
    for cell in cells:
        sheets = write_cell_sheets(op, cell)
        create_cell_plots(op, cell, sheets)
        per_cell_sheets.append(sheets)

    if len(cells) > 1:
        create_comparison_plots(op, per_cell_sheets)

    stat_df = stat_table(cells, target_cycles=list(stat_cycles))
    write_stat_table(op, stat_df)

    if project_path is not None:
        op.save(project_path)  # type: ignore[attr-defined]


__all__ = [
    "_require_originpro",
    "push_to_origin",
]
