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
* :mod:`.templates` — the bundled ``.otpu`` asset directory.

For the end-user "default install only" path inside Origin, this module
also exposes :func:`launch_gui`, a thin wrapper around
:func:`toyo_battery.gui.launch_gui` that injects an ``on_complete``
callback routing the loaded cells through :func:`push_to_origin`. That
keeps the Tk view (in :mod:`toyo_battery.gui.tk_app`) free of any
``originpro`` coupling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from toyo_battery.origin._plots import create_cell_plots, create_comparison_plots
from toyo_battery.origin._worksheets import write_cell_sheets, write_stat_table

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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
        create_comparison_plots(op, list(cells), per_cell_sheets)

    stat_df = stat_table(cells, target_cycles=list(stat_cycles))
    write_stat_table(op, stat_df)

    if project_path is not None:
        op.save(project_path)  # type: ignore[attr-defined]


def launch_gui(
    *,
    project_path: str | None = None,
    stat_cycles: Sequence[int] = (10, 50),
) -> None:
    """Launch the Tk GUI from inside Origin and push results to the project.

    This is the one-liner entry point for the "Origin default install only"
    workflow: after ``pip install toyo-battery[origin]`` inside Origin's
    embedded Python, ``from toyo_battery.origin import launch_gui;
    launch_gui()`` brings up the existing Tk directory picker, and Run
    forwards the loaded :class:`Cell` list to :func:`push_to_origin` —
    populating the active Origin project with worksheets, template-backed
    graphs, and a ``stat_table`` sheet without writing any PNG/CSV
    intermediates.

    The originpro check runs **before** Tk is constructed so a missing
    embedded-Python build fails fast with the canonical
    :func:`_require_originpro` message rather than after the user has
    spent time selecting directories.

    Parameters
    ----------
    project_path
        Forwarded to :func:`push_to_origin` for each Run. ``None`` (the
        default) operates on the in-memory project.
    stat_cycles
        Forwarded to :func:`push_to_origin` as ``stat_cycles``.

    Raises
    ------
    ImportError
        From :func:`_require_originpro` when ``originpro`` is not
        importable (i.e. not running inside Origin), or from
        ``import tkinter`` when the host Python lacks ``_tkinter`` (some
        older OriginLab releases — see issue #16). Both errors propagate
        unmodified so the failure mode is obvious from the traceback.
    """
    _require_originpro()
    # Imported lazily so this module stays importable on hosts without
    # Tk / matplotlib (the standalone ``push_to_origin`` path needs
    # neither). ``Figure`` is annotation-only; the runtime ``plt.close``
    # call below resolves it via ``matplotlib.pyplot``.
    from toyo_battery.gui import launch_gui as _launch_tk_gui

    def _push(cells: Sequence[Cell], figures: Sequence[Figure]) -> None:
        push_to_origin(cells, project_path=project_path, stat_cycles=stat_cycles)
        # Close the figures the controller built. The Origin path
        # bypasses ``_show_figure`` (which would otherwise own the close
        # via ``WM_DELETE_WINDOW``), so without an explicit close a long
        # session of repeated Run clicks accumulates figures in pyplot's
        # global registry and eventually trips ``max_open_warning``.
        import matplotlib.pyplot as plt

        for fig in figures:
            plt.close(fig)

    _launch_tk_gui(on_complete=_push)


__all__ = [
    "_require_originpro",
    "launch_gui",
    "push_to_origin",
]
