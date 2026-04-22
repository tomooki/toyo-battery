"""Tests for :func:`echemplot.origin.launch_gui`.

The launcher is the one-liner Origin users invoke from the embedded
Python Console. It composes three pieces:

1. ``_require_originpro`` — must run before Tk so a missing ``originpro``
   fails fast.
2. ``echemplot.gui.launch_gui`` — the standalone Tk entry point;
   patched here so no real window opens.
3. The injected ``on_complete`` callback must invoke
   :func:`echemplot.origin.push_to_origin` with the cells produced by
   the GUI Run.

We mock both ``originpro`` and the Tk launcher so the test stays
headless.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _install_mock_originpro(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mirror the helper in ``test_origin.py`` so this file is self-contained."""
    mock_op = MagicMock(name="originpro")
    monkeypatch.setitem(sys.modules, "originpro", mock_op)
    return mock_op


def test_launch_gui_requires_originpro_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``originpro`` is not importable the launcher must raise *before*
    importing ``echemplot.gui`` (which would otherwise pull in
    matplotlib + tkinter)."""
    monkeypatch.setitem(sys.modules, "originpro", None)
    # If gui is imported it should not be reached — guard by a sentinel
    # exception that would mask the expected ``ImportError``.
    import echemplot.gui as gui_pkg

    monkeypatch.setattr(
        gui_pkg, "launch_gui", lambda **_kw: pytest.fail("Tk launched despite missing originpro")
    )

    from echemplot.origin import launch_gui

    with pytest.raises(ImportError, match="OriginLab's embedded Python"):
        launch_gui()


def test_launch_gui_passes_callback_that_calls_push_to_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The launcher must hand a callback to the Tk GUI which, when fired
    by the view's Run path, calls :func:`push_to_origin` with the cells
    the controller loaded.

    We capture the callback at the ``echemplot.gui.launch_gui`` boundary
    and invoke it manually with sentinel cells/figures, mirroring what
    the view's ``_on_run`` would do post-Run.
    """
    _install_mock_originpro(monkeypatch)

    captured: dict[str, object] = {}

    def fake_tk_launch(**kwargs: object) -> None:
        captured.update(kwargs)

    import echemplot.gui as gui_pkg

    monkeypatch.setattr(gui_pkg, "launch_gui", fake_tk_launch)

    push_calls: list[dict[str, object]] = []

    def fake_push(cells: object, **kw: object) -> None:
        push_calls.append({"cells": cells, **kw})

    import echemplot.origin as origin_mod

    monkeypatch.setattr(origin_mod, "push_to_origin", fake_push)

    origin_mod.launch_gui(project_path=str(tmp_path / "p.opju"), stat_cycles=(7, 13))

    # Issue #60: the Origin launcher must put the Tk GUI into its
    # constrained ``origin_mode`` so ineffective options are greyed out.
    assert captured.get("origin_mode") is True

    on_complete = captured.get("on_complete")
    assert callable(on_complete), "launcher must inject an on_complete callback"

    sentinel_cells = ["cell-A", "cell-B"]  # opaque to push_to_origin in this fake
    sentinel_figs = ["fig0", "fig1", "fig2"]
    # Third positional arg is ``sg_window`` — the Tk view surfaces it so
    # the Origin launcher can propagate a non-default value through to
    # the dQ/dV worksheet (issue #60 follow-up).
    on_complete(sentinel_cells, sentinel_figs, 11)  # type: ignore[operator]

    assert len(push_calls) == 1
    call = push_calls[0]
    assert call["cells"] is sentinel_cells
    assert call["project_path"] == str(tmp_path / "p.opju")
    assert call["stat_cycles"] == (7, 13)
    assert call["sg_window"] == 11


def test_launch_gui_defaults_match_push_to_origin_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user calls ``launch_gui()`` with no kwargs, the injected
    callback must forward the documented push_to_origin defaults
    (``project_path=None``, ``stat_cycles=(10, 50)``).
    """
    _install_mock_originpro(monkeypatch)

    captured: dict[str, object] = {}

    def fake_tk_launch(**kwargs: object) -> None:
        captured.update(kwargs)

    import echemplot.gui as gui_pkg

    monkeypatch.setattr(gui_pkg, "launch_gui", fake_tk_launch)

    push_calls: list[dict[str, object]] = []

    def fake_push(cells: object, **kw: object) -> None:
        push_calls.append({"cells": cells, **kw})

    import echemplot.origin as origin_mod

    monkeypatch.setattr(origin_mod, "push_to_origin", fake_push)

    origin_mod.launch_gui()

    on_complete = captured.get("on_complete")
    assert callable(on_complete)
    on_complete(["c"], ["f"], 11)  # type: ignore[operator]

    call = push_calls[0]
    assert call["project_path"] is None
    assert call["stat_cycles"] == (10, 50)
    assert call["sg_window"] == 11
