"""Tests for :mod:`echemplot.origin`.

Real ``originpro`` is not available outside Origin's embedded Python, so
these tests build a fake module on ``sys.modules`` before calling
:func:`push_to_origin`. All assertions are against call counts / call
shapes on that mock — the real-Origin end-to-end verification is tracked
separately in issue #15.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from echemplot.core.cell import Cell

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _linear_cell(name: str, *, q_ch: float = 1000.0, q_dis: float = 990.0) -> Cell:
    """Build a :class:`Cell` with a single linear charge+discharge cycle.

    Borrowed in spirit from ``tests/test_stats.py`` — enough points to
    exercise the chdis / cap / dqdv pipelines but not so many that the
    DataFrame churn slows down the suite.
    """
    n_points = 30
    rows: list[tuple[int, str, str, float, float]] = []
    v_ch = np.linspace(3.0, 4.2, n_points)
    q_ch_arr = np.linspace(0.0, q_ch, n_points)
    rows.extend((1, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch_arr))
    v_dis = np.linspace(4.2, 3.0, n_points)
    q_dis_arr = np.linspace(0.0, q_dis, n_points)
    rows.extend((1, "1", "放電", float(v), float(q)) for v, q in zip(v_dis, q_dis_arr))
    raw = pd.DataFrame(rows, columns=["サイクル", "モード", "状態", "電圧", "電気量"])
    return Cell(name=name, mass_g=0.001, raw_df=raw)


def _install_mock_originpro(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Register a ``MagicMock`` as ``sys.modules["originpro"]`` and return it."""
    mock_op = MagicMock(name="originpro")
    monkeypatch.setitem(sys.modules, "originpro", mock_op)
    return mock_op


def _stub_templates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``ECHEMPLOT_ORIGIN_TEMPLATE_DIR`` at a dir with zero-byte templates.

    Used by tests that exercise the env-var override path explicitly, and
    by the bulk of the suite to keep the assertions focused on call
    counts / shapes rather than on the bundled templates' real bytes.
    Tests that need to verify the bundled-default lookup should NOT call
    this and should instead unset the env var (see
    :func:`test_default_templates_resolve_from_bundled_directory`).
    """
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    for name in ("charge_discharge.otpu", "cycle_efficiency.otpu", "dqdv.otpu"):
        (tpl_dir / name).write_bytes(b"")
    monkeypatch.setenv("ECHEMPLOT_ORIGIN_TEMPLATE_DIR", str(tpl_dir))
    return tpl_dir


# ----------------------------------------------------------------------
# _require_originpro
# ----------------------------------------------------------------------


def test_require_originpro_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ``sys.modules["originpro"] = None`` blocks the import."""
    # ``monkeypatch.setitem`` with value ``None`` causes ``import originpro``
    # to raise ``ImportError`` — this is the canonical way to simulate an
    # unimportable module in the stdlib.
    monkeypatch.setitem(sys.modules, "originpro", None)
    from echemplot.origin import _require_originpro

    with pytest.raises(ImportError, match="OriginLab's embedded Python"):
        _require_originpro()


def test_require_originpro_returns_module_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    from echemplot.origin import _require_originpro

    assert _require_originpro() is mock_op


# ----------------------------------------------------------------------
# push_to_origin — single cell
# ----------------------------------------------------------------------


def test_push_to_origin_single_cell_creates_sheets_and_plots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # 3 per-cell sheets + 1 stat_table sheet = 4 new_sheet calls.
    assert mock_op.new_sheet.call_count == 4
    # 3 per-cell plots, no comparisons (only one cell).
    assert mock_op.new_graph.call_count == 3

    # Sheet long-names include the three per-cell names + stat_table.
    sheet_lnames = {call.kwargs["lname"] for call in mock_op.new_sheet.call_args_list}
    assert sheet_lnames == {"A_chdis", "A_cycle", "A_dqdv", "stat_table"}

    # Plot long-names match the documented pattern.
    graph_lnames = {call.kwargs["lname"] for call in mock_op.new_graph.call_args_list}
    assert graph_lnames == {"A_chdis_plot", "A_cycle_plot", "A_dqdv_plot"}


def test_push_to_origin_calls_require_originpro_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The orchestrator must resolve originpro exactly once per call."""
    _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell = _linear_cell("A")

    import echemplot.origin as origin_mod

    calls = {"n": 0}
    real = origin_mod._require_originpro

    def counting() -> object:
        calls["n"] += 1
        return real()

    monkeypatch.setattr(origin_mod, "_require_originpro", counting)
    origin_mod.push_to_origin([cell], stat_cycles=(1,))
    assert calls["n"] == 1


# ----------------------------------------------------------------------
# push_to_origin — multi-cell comparison
# ----------------------------------------------------------------------


def test_push_to_origin_multi_cell_includes_comparison_plots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell_a = _linear_cell("A")
    cell_b = _linear_cell("B", q_ch=800.0, q_dis=760.0)

    from echemplot.origin import push_to_origin

    push_to_origin([cell_a, cell_b], stat_cycles=(1,))

    # 3 sheets per cell x 2 cells + 1 stat_table = 7 new_sheet calls.
    assert mock_op.new_sheet.call_count == 7
    # 3 per-cell plots x 2 cells + 3 comparison plots = 9 new_graph calls.
    assert mock_op.new_graph.call_count == 9

    graph_lnames = [call.kwargs["lname"] for call in mock_op.new_graph.call_args_list]
    assert "comparison_chdis_plot" in graph_lnames
    assert "comparison_cycle_plot" in graph_lnames
    assert "comparison_dqdv_plot" in graph_lnames


# ----------------------------------------------------------------------
# Template resolution
# ----------------------------------------------------------------------


def test_push_to_origin_missing_template_raises_with_remediation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_mock_originpro(monkeypatch)
    # Point at an empty directory so no templates can be resolved.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("ECHEMPLOT_ORIGIN_TEMPLATE_DIR", str(empty))

    from echemplot.origin import push_to_origin

    cell = _linear_cell("A")
    with pytest.raises(FileNotFoundError) as excinfo:
        push_to_origin([cell], stat_cycles=(1,))

    msg = str(excinfo.value)
    assert "charge_discharge.otpu" in msg
    assert "ECHEMPLOT_ORIGIN_TEMPLATE_DIR" in msg


def test_default_templates_resolve_from_bundled_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``ECHEMPLOT_ORIGIN_TEMPLATE_DIR`` set, the three required ``.otpu``
    templates must resolve to non-empty files inside the package.

    Guards the "default install only" UX: a fresh ``pip install
    echemplot[origin]`` inside Origin must produce graphs without any
    template-copy step. Regressing this means ``push_to_origin`` would
    raise ``FileNotFoundError`` for every Origin user.
    """
    monkeypatch.delenv("ECHEMPLOT_ORIGIN_TEMPLATE_DIR", raising=False)
    from echemplot.origin._plots import _require_template

    for name in ("charge_discharge.otpu", "cycle_efficiency.otpu", "dqdv.otpu"):
        path = _require_template(name)
        assert path.exists(), name
        assert path.stat().st_size > 0, name


# ----------------------------------------------------------------------
# Sheet-name sanitization
# ----------------------------------------------------------------------


def test_long_cell_name_is_truncated_to_origin_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    # 50-char name; suffixes like ``_chdis`` push the full sheet name to
    # 56 chars which must be truncated.
    long_name = "a" * 50
    cell = _linear_cell(long_name)

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    for call in mock_op.new_sheet.call_args_list:
        assert len(call.kwargs["lname"]) <= 32, call.kwargs


def test_sanitize_sheet_name_preserves_short_names() -> None:
    from echemplot.origin._worksheets import _sanitize_sheet_name

    assert _sanitize_sheet_name("A_chdis") == "A_chdis"
    assert _sanitize_sheet_name("x" * 32) == "x" * 32


def test_sanitize_sheet_name_disambiguates_long_prefix_collisions() -> None:
    """Two overlong names sharing a 31-char prefix must yield distinct sheets."""
    from echemplot.origin._worksheets import _sanitize_sheet_name

    a = "x" * 40 + "A"
    b = "x" * 40 + "B"
    assert _sanitize_sheet_name(a) != _sanitize_sheet_name(b)
    assert len(_sanitize_sheet_name(a)) <= 32


# ----------------------------------------------------------------------
# MultiIndex flattening
# ----------------------------------------------------------------------


def test_flatten_columns_joins_multiindex_levels_with_underscore() -> None:
    from echemplot.origin._worksheets import _flatten_columns

    df = pd.DataFrame(
        [[1.0, 2.0]],
        columns=pd.MultiIndex.from_tuples(
            [(1, "ch", "電圧"), (1, "dis", "電圧")],
            names=["cycle", "side", "quantity"],
        ),
    )
    flat = _flatten_columns(df)
    assert list(flat.columns) == ["1_ch_電圧", "1_dis_電圧"]


def test_flatten_columns_passes_single_level_columns_through() -> None:
    from echemplot.origin._worksheets import _flatten_columns

    df = pd.DataFrame({"q_ch": [1.0], "q_dis": [2.0]})
    assert list(_flatten_columns(df).columns) == ["q_ch", "q_dis"]


def test_flatten_columns_yields_numpy_object_column_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flattened column Index must be numpy-object, not pandas ``StringDtype``.

    ``originpro.worksheet.from_df`` dispatches on ``.dtype.char``, which
    pandas extension dtypes don't expose. With
    ``future.infer_string=True``, ``pd.Index([...])`` on strings yields
    ``StringDtype`` and the push path fails with
    ``AttributeError: 'StringDtype' object has no attribute 'char'``
    (issue #75). Pinning ``dtype=object`` in ``_flatten_columns`` guards
    against that configuration regardless of the caller's pandas option.
    """
    from echemplot.origin._worksheets import _flatten_columns

    monkeypatch.setattr(pd.options.future, "infer_string", True)
    df = pd.DataFrame(
        [[1.0, 2.0]],
        columns=pd.MultiIndex.from_tuples(
            [(1, "ch", "q"), (1, "dis", "q")],
            names=["cycle", "side", "quantity"],
        ),
    )
    flat = _flatten_columns(df)
    assert isinstance(flat.columns.dtype, np.dtype)
    assert flat.columns.dtype == object


# ----------------------------------------------------------------------
# _coerce_for_originpro — extension-dtype defence for originpro.from_df
# ----------------------------------------------------------------------


def test_coerce_for_originpro_converts_stringdtype_column_to_object() -> None:
    """StringDtype columns must become object so ``from_df`` sees ``.dtype.char``."""
    from echemplot.origin._worksheets import _coerce_for_originpro

    df = pd.DataFrame({"cell": pd.array(["a", "b"], dtype="string"), "x": [1.0, 2.0]})
    out = _coerce_for_originpro(df)
    assert out["cell"].dtype == object
    assert out["cell"].dtype.char == "O"
    assert out["x"].dtype == np.float64
    assert list(out["cell"]) == ["a", "b"]


def test_coerce_for_originpro_rebuilds_string_column_index_as_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``StringDtype`` column Index must be rebuilt as numpy-object.

    ``from_df`` touches ``df.columns.dtype`` in addition to per-column
    dtypes; with ``future.infer_string=True`` the Index dtype alone can
    trip the same ``AttributeError``.
    """
    from echemplot.origin._worksheets import _coerce_for_originpro

    monkeypatch.setattr(pd.options.future, "infer_string", True)
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    # Re-constructing the Index under the opted-in option produces StringDtype.
    df.columns = pd.Index(["a", "b"])
    assert not isinstance(df.columns.dtype, np.dtype)  # precondition

    out = _coerce_for_originpro(df)
    assert isinstance(out.columns.dtype, np.dtype)
    assert out.columns.dtype == object
    assert list(out.columns) == ["a", "b"]


def test_coerce_for_originpro_is_identity_for_numpy_dtypes() -> None:
    """Numeric-only frames with an object-dtype column Index short-circuit.

    The column Index is pinned to ``dtype=object`` explicitly so the
    assertion stays stable across pandas versions — recent releases
    construct ``pd.DataFrame({...}).columns`` as ``StringDtype`` when
    ``future.infer_string`` is on (py3.11/3.12 CI resolves a pandas
    build where that is the case), which would legitimately send the
    frame through the rebuild path and defeat the identity check.
    """
    from echemplot.origin._worksheets import _coerce_for_originpro

    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    df.columns = pd.Index(["a", "b"], dtype=object)
    assert _coerce_for_originpro(df) is df


def test_push_to_origin_succeeds_under_future_infer_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for issue #75: completion hook failed with
    ``'StringDtype' object has no attribute 'char'`` when the host pandas
    had ``future.infer_string`` enabled, because the Index built in
    ``_flatten_columns`` and the ``cell`` column from
    ``stat_table.reset_index()`` both surfaced as ``StringDtype``. Every
    frame handed to ``from_df`` must expose numpy-native dtypes on both
    the values and the column Index.
    """
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    monkeypatch.setattr(pd.options.future, "infer_string", True)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # Every from_df call must receive numpy-native dtypes so originpro's
    # ``.dtype.char`` dispatch survives. Guards against any future dtype
    # inflation slipping past ``_coerce_for_originpro``.
    for sheet_call in mock_op.new_sheet.return_value.from_df.call_args_list:
        frame = sheet_call.args[0]
        assert isinstance(frame.columns.dtype, np.dtype), frame.columns.dtype
        for col in frame.columns:
            assert isinstance(frame[col].dtype, np.dtype), (col, frame[col].dtype)


# ----------------------------------------------------------------------
# project_path round-trip
# ----------------------------------------------------------------------


def test_push_to_origin_opens_and_saves_when_project_path_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell = _linear_cell("A")
    project = str(tmp_path / "proj.opju")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], project_path=project, stat_cycles=(1,))

    mock_op.open.assert_called_once_with(project)
    mock_op.save.assert_called_once_with(project)


def test_push_to_origin_skips_open_save_when_project_path_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))
    assert not mock_op.open.called
    assert not mock_op.save.called


# ----------------------------------------------------------------------
# Sheet binding (worksheet + graph wiring)
# ----------------------------------------------------------------------


def _install_tracking_originpro(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, list[MagicMock], list[MagicMock]]:
    """Install a mock originpro that returns a fresh MagicMock per sheet/graph.

    Returning distinct mocks (instead of the default single shared
    ``return_value``) lets tests correlate individual sheet objects with
    the ``add_plot`` calls bound to them. Worksheet mocks also get a
    ``cols_axis`` spec so assertions on column designation stay honest.
    """
    mock_op = _install_mock_originpro(monkeypatch)

    sheet_objs: list[MagicMock] = []

    def _fake_new_sheet(**_kwargs: Any) -> MagicMock:
        s = MagicMock(name=f"sheet_{len(sheet_objs)}")
        sheet_objs.append(s)
        return s

    graph_objs: list[MagicMock] = []

    def _fake_new_graph(**_kwargs: Any) -> MagicMock:
        g = MagicMock(name=f"graph_{len(graph_objs)}")
        graph_objs.append(g)
        return g

    mock_op.new_sheet.side_effect = _fake_new_sheet
    mock_op.new_graph.side_effect = _fake_new_graph
    return mock_op, sheet_objs, graph_objs


def test_each_per_cell_graph_binds_its_own_sheet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each per-cell graph's add_plot calls must reference only its own sheet."""
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, graph_objs = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # write_cell_sheets creates sheets in order chdis → cycle → dqdv (the
    # stat_table sheet comes last, see push_to_origin orchestration).
    # create_cell_plots emits graphs in the same chdis → cycle → dqdv order.
    assert len(graph_objs) >= 3, "expected at least 3 per-cell graphs"
    for graph, expected_sheet in zip(graph_objs[:3], sheet_objs[:3]):
        # MagicMock's __getitem__ returns the same inner mock regardless
        # of index, so both graph[0] and graph[1] share call_args_list.
        # chdis / dqdv emit multiple add_plot calls (one per (cycle,
        # side) pair); cycle emits two (one per layer). Every call must
        # point at the same sheet.
        calls = graph.__getitem__.return_value.add_plot.call_args_list
        assert calls, f"no add_plot calls on {graph!r}"
        for call in calls:
            assert call.args[0] is expected_sheet, (
                f"add_plot bound wrong sheet on {graph!r}: {call}"
            )


def test_add_plot_passes_column_indices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every ``add_plot`` call must carry ``colx`` and ``coly`` kwargs.

    Without these, Origin renders the template's axes/legend but binds
    no data — the v0.0.2 regression that this test guards against.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, _, graph_objs = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    for graph in graph_objs[:3]:
        calls = graph.__getitem__.return_value.add_plot.call_args_list
        assert calls, f"no add_plot calls on {graph!r}"
        for call in calls:
            assert "colx" in call.kwargs, f"missing colx on {graph!r}: {call}"
            assert "coly" in call.kwargs, f"missing coly on {graph!r}: {call}"


def test_cycle_plot_binds_both_layers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """cycle_efficiency's dual-Y template has two layers — bind both.

    ``graph[0]`` takes discharge capacity (col 2), ``graph[1]`` takes
    Coulombic efficiency (col 3). Both share ``cycle`` (col 0) as X.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, _, graph_objs = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # Per-cell graphs emit in order chdis → cycle → dqdv.
    cycle_graph = graph_objs[1]
    # MagicMock indexing history: __getitem__.call_args_list records each
    # access. We expect both 0 and 1 to have been accessed.
    indexed = {call.args[0] for call in cycle_graph.__getitem__.call_args_list}
    assert 0 in indexed and 1 in indexed, f"cycle graph accessed only {indexed}"

    # Two add_plot calls total on the shared inner mock: one per layer.
    call_list = cycle_graph.__getitem__.return_value.add_plot.call_args_list
    colys = sorted(call.kwargs["coly"] for call in call_list)
    assert colys == [2, 3], f"expected q_dis + ce bound, got coly={colys}"
    assert all(call.kwargs["colx"] == 0 for call in call_list), call_list


def test_every_per_cell_graph_layer_is_rescaled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every template-backed layer must be rescaled after ``add_plot``.

    Without an explicit ``layer.rescale()`` call, Origin keeps the
    template's default axis range and data outside that window renders
    clipped — the autoscale-not-working regression this test guards.
    ``MagicMock.__getitem__`` returns the same inner mock for every
    index, so the cycle graph's two layers share a single ``rescale``
    counter; we assert ``>= 1`` rather than an exact count.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, _, graph_objs = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    for graph in graph_objs[:3]:
        layer = graph.__getitem__.return_value
        assert layer.rescale.called, f"rescale not called on layer of {graph!r}"


def test_cols_axis_is_set_per_category(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Worksheets for plot-targeted categories must declare X/Y designations.

    ``from_df`` leaves every column as ``"Y"``; without a ``cols_axis``
    call the template-backed plot has no X to bind against and renders
    empty. ``stat_table`` is never plotted so it skips the call.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, _ = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # Sheet creation order: chdis → cycle → dqdv → stat_table.
    chdis_sheet, cycle_sheet, dqdv_sheet, stat_sheet = sheet_objs[:4]

    # chdis / dqdv: "XY" repeated — one pair per (cycle, side) column pair.
    for sheet in (chdis_sheet, dqdv_sheet):
        args = sheet.cols_axis.call_args
        assert args is not None, f"cols_axis not called on {sheet!r}"
        types = args.args[0]
        assert set(types) <= {"X", "Y"}, types
        assert types.startswith("XY"), types
        assert len(types) % 2 == 0, types

    # cycle: [cycle, q_ch, q_dis, ce] → "XYYY".
    cycle_sheet.cols_axis.assert_called_once_with("XYYY")

    # stat_table must NOT have a cols_axis call (never plotted).
    assert not stat_sheet.cols_axis.called


def test_cap_df_written_with_cycle_as_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cap_df's ``cycle`` index must be surfaced as a column before from_df.

    Without ``reset_index``, Origin receives only ``[q_ch, q_dis, ce]``
    and the cycle_efficiency plot has no X source — graph window opens
    but nothing is plotted.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, _ = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # Second sheet created is the cycle sheet (chdis=0, cycle=1, dqdv=2).
    cycle_sheet = sheet_objs[1]
    from_df_call = cycle_sheet.from_df.call_args
    assert from_df_call is not None, "from_df not called on cycle sheet"
    written = from_df_call.args[0]
    assert "cycle" in written.columns, (
        f"cycle column missing from cap_df write: {list(written.columns)}"
    )
    # Order must match the _CYCLE_SHEET_COLUMNS contract in
    # echemplot.origin._worksheets — the cycle_efficiency template's
    # add_plot indices are derived from that tuple.
    assert next(iter(written.columns)) == "cycle", list(written.columns)


# ----------------------------------------------------------------------
# Savitzky-Golay override propagation (issue #60 follow-up)
# ----------------------------------------------------------------------


def test_push_to_origin_non_default_sg_window_reaches_dqdv_sheet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-default ``sg_window`` must change the dQ/dV worksheet content.

    ``Cell.dqdv_df`` is cached at the SG defaults (window=11, polyorder=2),
    so if :func:`push_to_origin` blindly reused that cached frame the user's
    GUI ``SG window_length`` edit (issue #60) would silently be ignored —
    exactly the failure mode this test guards against. Here we compare the
    dQ/dV DataFrame handed to ``from_df`` against the cached default and
    assert it differs when ``sg_window`` is overridden.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, _ = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    # Cache the default-SG frame before push; push_to_origin must override
    # with a freshly-computed frame using sg_window=5 rather than reuse this.
    default_dqdv = cell.dqdv_df.copy()

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,), sg_window=5)

    # sheet_objs: chdis=0, cycle=1, dqdv=2 (stat_table created last).
    dqdv_sheet = sheet_objs[2]
    from_df_call = dqdv_sheet.from_df.call_args
    assert from_df_call is not None, "from_df not called on dqdv sheet"
    written = from_df_call.args[0]

    # Column structure must be preserved — only the values should differ.
    assert written.shape == default_dqdv.shape, (
        f"shape drift: default={default_dqdv.shape} overridden={written.shape}"
    )
    # Values must differ for at least one column (the SG smoothing window
    # changed, so the derivative estimates cannot coincide everywhere).
    import numpy as np

    assert not np.allclose(
        written.to_numpy(dtype=float),
        default_dqdv.to_numpy(dtype=float),
        equal_nan=True,
    ), "dqdv sheet content identical to default-SG cached frame; override not applied"


def test_push_to_origin_default_sg_window_writes_cached_dqdv_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """At the defaults, the dQ/dV sheet content matches the cached frame.

    ``_write_sheet`` flattens MultiIndex columns before handing the frame to
    ``from_df``, so object identity does not survive; we compare values
    instead. If a future refactor accidentally starts recomputing at the
    defaults, this test still guards against value drift.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, _ = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    default_dqdv = cell.dqdv_df.copy()

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    dqdv_sheet = sheet_objs[2]
    from_df_call = dqdv_sheet.from_df.call_args
    assert from_df_call is not None
    written = from_df_call.args[0]

    import numpy as np

    assert written.shape == default_dqdv.shape
    assert np.allclose(
        written.to_numpy(dtype=float),
        default_dqdv.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_push_to_origin_rejects_even_sg_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even ``sg_window`` values are invalid Savitzky-Golay inputs.

    Validation lives in ``push_to_origin`` itself so an even value fails
    before any Origin sheets are created, mirroring the controller-side
    ``GuiRequest`` validation.
    """
    _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    with pytest.raises(ValueError, match="sg_window"):
        push_to_origin([cell], stat_cycles=(1,), sg_window=10)


# ----------------------------------------------------------------------
# Shared-scale autoscaling (issue #61)
# ----------------------------------------------------------------------


def _fake_cell(
    *,
    chdis: pd.DataFrame | None = None,
    cap: pd.DataFrame | None = None,
    dqdv: pd.DataFrame | None = None,
    name: str = "fake",
) -> MagicMock:
    """Build a duck-typed Cell exposing only the three dataframes + name.

    ``compute_global_ranges`` touches nothing else, so the full
    :class:`Cell` construction (raw_df parsing, mass, etc.) is
    unnecessary for range-computation tests.
    """
    cell = MagicMock(spec=["name", "chdis_df", "cap_df", "dqdv_df"])
    cell.name = name
    cell.chdis_df = chdis if chdis is not None else pd.DataFrame()
    # ``compute_global_ranges`` calls ``cap_df.reset_index()``; a
    # DataFrame provides that out of the box so no extra plumbing is
    # needed on the mock.
    cell.cap_df = cap if cap is not None else pd.DataFrame()
    cell.dqdv_df = dqdv if dqdv is not None else pd.DataFrame()
    return cell


def test_safe_range_ignores_nan_and_returns_none_for_empty() -> None:
    from echemplot.origin._plots import _safe_range

    assert _safe_range(np.array([1.0, 2.0, np.nan, 3.0])) == (1.0, 3.0)
    assert _safe_range(np.array([np.nan, np.nan])) is None
    assert _safe_range(np.array([])) is None


def test_compute_global_ranges_extracts_per_axis_min_max() -> None:
    """Hand-crafted cells → exact min/max tuples across the sequence."""
    from echemplot.origin._plots import compute_global_ranges

    # chdis: even cols = capacity (x), odd = voltage (y).
    chdis_a = pd.DataFrame({"q_ch": [0.0, 500.0], "v_ch": [3.0, 4.0]})
    chdis_b = pd.DataFrame({"q_ch": [0.0, 800.0], "v_ch": [3.1, 4.2]})

    # cap after reset_index → [cycle, q_ch, q_dis, ce].
    cap_a = pd.DataFrame(
        {"cycle": [1, 2], "q_ch": [1000.0, 990.0], "q_dis": [990.0, 980.0], "ce": [99.0, 98.9]}
    ).set_index("cycle")
    cap_b = pd.DataFrame(
        {
            "cycle": [1, 2, 3],
            "q_ch": [800.0, 790.0, 780.0],
            "q_dis": [760.0, 755.0, 750.0],
            "ce": [95.0, 95.5, 96.1],
        }
    ).set_index("cycle")

    # dqdv: even cols = voltage (x), odd = dQ/dV (y).
    dqdv_a = pd.DataFrame({"v": [3.0, 3.5, 4.0], "dqdv": [10.0, 20.0, -5.0]})
    dqdv_b = pd.DataFrame({"v": [3.1, 3.6, 4.1], "dqdv": [12.0, 22.0, -8.0]})

    a = _fake_cell(chdis=chdis_a, cap=cap_a, dqdv=dqdv_a, name="A")
    b = _fake_cell(chdis=chdis_b, cap=cap_b, dqdv=dqdv_b, name="B")

    ranges = compute_global_ranges([a, b])

    assert ranges.chdis_x == (0.0, 800.0)
    assert ranges.chdis_y == (3.0, 4.2)
    assert ranges.cycle_x == (1.0, 3.0)
    assert ranges.cycle_left_y == (750.0, 990.0)
    assert ranges.cycle_right_y == (95.0, 99.0)
    assert ranges.dqdv_x == (3.0, 4.1)
    assert ranges.dqdv_y == (-8.0, 22.0)


def test_compute_global_ranges_handles_cells_with_different_column_counts() -> None:
    """Cells with mismatched (cycle, side) pair counts must not raise.

    Regression for the np.concatenate "along dimension 1" failure that
    surfaced when Run-ing multiple cells whose chdis_df / dqdv_df had
    different numbers of column pairs (e.g. 50 cycles x 2 sides vs.
    1 cycle x 1 side).
    """
    from echemplot.origin._plots import compute_global_ranges

    chdis_a = pd.DataFrame(
        {
            "q_ch": [0.0, 500.0],
            "v_ch": [3.0, 4.0],
            "q_dis": [0.0, 480.0],
            "v_dis": [4.0, 3.0],
        }
    )
    chdis_b = pd.DataFrame({"q_ch": [0.0, 800.0], "v_ch": [3.1, 4.2]})

    cap_a = pd.DataFrame(
        {
            "cycle": [1, 2],
            "q_ch": [1000.0, 990.0],
            "q_dis": [990.0, 980.0],
            "ce": [99.0, 98.9],
        }
    ).set_index("cycle")
    cap_b = pd.DataFrame(
        {"cycle": [1], "q_ch": [800.0], "q_dis": [760.0], "ce": [95.0]}
    ).set_index("cycle")

    dqdv_a = pd.DataFrame(
        {
            "v_ch": [3.0, 3.5, 4.0],
            "dqdv_ch": [10.0, 20.0, -5.0],
            "v_dis": [4.0, 3.5, 3.0],
            "dqdv_dis": [-3.0, -8.0, -1.0],
        }
    )
    dqdv_b = pd.DataFrame({"v": [3.1, 4.1], "dqdv": [12.0, -8.0]})

    a = _fake_cell(chdis=chdis_a, cap=cap_a, dqdv=dqdv_a, name="A")
    b = _fake_cell(chdis=chdis_b, cap=cap_b, dqdv=dqdv_b, name="B")

    ranges = compute_global_ranges([a, b])

    assert ranges.chdis_x == (0.0, 800.0)
    assert ranges.chdis_y == (3.0, 4.2)
    assert ranges.dqdv_x == (3.0, 4.1)
    assert ranges.dqdv_y == (-8.0, 20.0)


def test_compute_global_ranges_single_cell_equals_cell_range() -> None:
    """With a single cell, global == that cell's own data range."""
    from echemplot.origin._plots import compute_global_ranges

    chdis = pd.DataFrame({"q_ch": [0.0, 1000.0], "v_ch": [3.0, 4.2]})
    cap = pd.DataFrame(
        {"cycle": [1, 2], "q_ch": [1000.0, 990.0], "q_dis": [990.0, 980.0], "ce": [99.0, 98.9]}
    ).set_index("cycle")
    dqdv = pd.DataFrame({"v": [3.0, 4.0], "dqdv": [5.0, 15.0]})

    cell = _fake_cell(chdis=chdis, cap=cap, dqdv=dqdv)
    ranges = compute_global_ranges([cell])

    assert ranges.chdis_x == (0.0, 1000.0)
    assert ranges.chdis_y == (3.0, 4.2)
    assert ranges.cycle_x == (1.0, 2.0)
    assert ranges.cycle_left_y == (980.0, 990.0)
    assert ranges.cycle_right_y == (98.9, 99.0)
    assert ranges.dqdv_x == (3.0, 4.0)
    assert ranges.dqdv_y == (5.0, 15.0)


def test_compute_global_ranges_nan_only_yields_none() -> None:
    """All-NaN columns produce ``None`` per-axis instead of NaN tuples."""
    from echemplot.origin._plots import compute_global_ranges

    nan_df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    cap_nan = pd.DataFrame(
        {
            "cycle": [1, 2],
            "q_ch": [np.nan, np.nan],
            "q_dis": [np.nan, np.nan],
            "ce": [np.nan, np.nan],
        }
    ).set_index("cycle")

    cell = _fake_cell(chdis=nan_df, cap=cap_nan, dqdv=nan_df)
    ranges = compute_global_ranges([cell])

    assert ranges.chdis_x is None
    assert ranges.chdis_y is None
    assert ranges.cycle_x == (1.0, 2.0)  # cycle index is finite
    assert ranges.cycle_left_y is None
    assert ranges.cycle_right_y is None
    assert ranges.dqdv_x is None
    assert ranges.dqdv_y is None


def test_compute_global_ranges_empty_dataframes_are_safe() -> None:
    """Empty cell dataframes → ``_GraphRanges`` with all ``None`` fields, no raises."""
    from echemplot.origin._plots import compute_global_ranges

    cell = _fake_cell()
    ranges = compute_global_ranges([cell])

    for field_name in (
        "chdis_x",
        "chdis_y",
        "cycle_x",
        "cycle_left_y",
        "cycle_right_y",
        "dqdv_x",
        "dqdv_y",
    ):
        assert getattr(ranges, field_name) is None, field_name


def test_set_axis_limits_is_noop_when_limits_none() -> None:
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    _set_axis_limits(layer, "x", None)
    layer.axis.assert_not_called()
    layer.lt_exec.assert_not_called()


def test_set_axis_limits_skips_degenerate_range() -> None:
    """``lo == hi`` → leave template scaling alone (no axis API call)."""
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    _set_axis_limits(layer, "x", (3.0, 3.0))
    layer.axis.assert_not_called()
    layer.lt_exec.assert_not_called()


def test_set_axis_limits_sets_begin_and_end_via_attribute_api() -> None:
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    _set_axis_limits(layer, "x", (0.0, 10.0))
    layer.axis.assert_called_once_with("x")
    axis_obj = layer.axis.return_value
    assert axis_obj.begin == 0.0
    assert axis_obj.end == 10.0


def test_set_axis_limits_falls_back_to_lt_exec_when_attr_api_raises() -> None:
    """When ``layer.axis()`` itself raises, the fallback runs on ``op.lt_exec``.

    The fallback uses the module-level :func:`op.lt_exec` rather than
    ``layer.lt_exec`` because originpro's Python bindings reliably expose
    ``lt_exec`` at the module level but not consistently on per-layer
    proxies. :func:`op.lt_exec` targets the currently-active graph, which
    is the one :func:`_new_graph_from_template` just created.
    """
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    layer.axis.side_effect = RuntimeError("no axis() on this layer build")
    op = MagicMock()
    _set_axis_limits(layer, "y", (1.0, 5.0), op=op)
    op.lt_exec.assert_called_once()
    cmd = op.lt_exec.call_args.args[0]
    assert cmd.startswith("y.from=1.0")
    assert "y.to=5.0" in cmd
    # ``layer.lt_exec`` must NOT be called — that was the old contract.
    layer.lt_exec.assert_not_called()


def test_set_axis_limits_falls_back_when_readback_disagrees() -> None:
    """Silent no-op detection: if ``ax.begin`` readback != ``lo`` then fall back.

    Covers the case where ``layer.axis()`` returns an object that
    accepts attribute writes (plain Python objects always do) but
    doesn't actually propagate them to Origin — the graph would render
    with template defaults and no exception would surface. A round-trip
    read guards against that.
    """
    from echemplot.origin._plots import _set_axis_limits

    # Custom axis proxy: accepts writes but returns stale readbacks.
    class _StaleAxis:
        def __init__(self) -> None:
            self.begin = 999.0
            self.end = 999.0

        def __setattr__(self, name: str, value: object) -> None:
            # Accept the write but don't actually update (simulates a
            # proxy whose setter doesn't wire through to Origin).
            if not hasattr(self, name):
                # Allow the initial __init__ writes.
                object.__setattr__(self, name, value)

    stale_axis = _StaleAxis()
    layer = MagicMock()
    layer.axis.return_value = stale_axis
    op = MagicMock()
    _set_axis_limits(layer, "x", (0.0, 10.0), op=op)
    # Round-trip failed (begin/end still 999.0) → fallback fires.
    op.lt_exec.assert_called_once()
    cmd = op.lt_exec.call_args.args[0]
    assert cmd.startswith("x.from=0.0")
    assert "x.to=10.0" in cmd


def test_set_axis_limits_no_fallback_when_op_is_none() -> None:
    """Without an ``op`` handle the fallback is suppressed, not crashed.

    Preserves the pre-#67 call-site ergonomics: unit tests that call
    :func:`_set_axis_limits` directly without plumbing ``op`` through
    must not raise when the attribute path is unreliable.
    """
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    layer.axis.side_effect = RuntimeError("no axis() on this layer build")
    # Must not raise despite no fallback being available.
    _set_axis_limits(layer, "y", (1.0, 5.0))
    layer.lt_exec.assert_not_called()


def test_set_axis_limits_warns_on_originpro_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attribute-API failure must be observable via a WARNING log record.

    Issue #99: the previous bare ``except Exception`` swallowed
    originpro-side failures and let the graph silently degrade to the
    template's default axis range. A WARNING on the
    ``echemplot.origin`` logger now surfaces the axis name, the
    attempted range, and the exception type+message before the LabTalk
    fallback runs.
    """
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    layer.axis.side_effect = RuntimeError("no axis() on this layer build")
    op = MagicMock()
    with caplog.at_level(logging.WARNING, logger="echemplot.origin"):
        _set_axis_limits(layer, "y", (1.0, 5.0), op=op)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING record from the axis fallback"
    msg = warnings[0].getMessage()
    assert "'y'" in msg, msg  # axis name present
    assert "RuntimeError" in msg, msg  # exception type surfaced
    assert "no axis() on this layer build" in msg, msg  # exception message surfaced
    # LabTalk fallback still ran — this is warn-and-continue, not warn-and-skip.
    op.lt_exec.assert_called_once()


def test_set_axis_limits_raises_when_strict() -> None:
    """``strict=True`` re-raises the caught originpro exception.

    Issue #99: callers that prefer fail-fast over a silently degraded
    graph can opt in via ``strict_axis=True`` on
    :func:`push_to_origin`, which propagates here as ``strict``. The
    LabTalk fallback must NOT run when re-raising.
    """
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    layer.axis.side_effect = RuntimeError("no axis() on this layer build")
    op = MagicMock()
    with pytest.raises(RuntimeError, match="no axis"):
        _set_axis_limits(layer, "y", (1.0, 5.0), op=op, strict=True)
    op.lt_exec.assert_not_called()


def test_set_axis_limits_silent_on_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal attribute-API path must not emit any WARNING.

    Guards against the warn-on-success regression: the WARNING is
    reserved for the failure path so log noise stays meaningful.
    """
    from echemplot.origin._plots import _set_axis_limits

    layer = MagicMock()
    op = MagicMock()
    with caplog.at_level(logging.WARNING, logger="echemplot.origin"):
        _set_axis_limits(layer, "x", (0.0, 10.0), op=op)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, f"unexpected WARNING records on success path: {warnings}"
    # And the LabTalk fallback must not have been triggered either.
    op.lt_exec.assert_not_called()


def test_create_cell_plots_applies_ranges_to_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``ranges`` is passed, each layer receives begin/end writes per axis.

    Calls ``create_cell_plots`` directly with a non-degenerate
    ``_GraphRanges`` so every axis triggers an ``axis()`` setter — the
    single-cycle ``_linear_cell`` would otherwise collapse cycle ranges
    to ``lo == hi`` and short-circuit :func:`_set_axis_limits`.
    """
    import echemplot.origin._plots as plots_mod
    from echemplot.origin._plots import _GraphRanges, create_cell_plots

    # Make graph[0] and graph[1] share the same inner layer mock so
    # axis() call history is observable in one place.
    created_graphs: list[MagicMock] = []

    def _fake_new_graph(*_args: Any, **_kwargs: Any) -> MagicMock:
        g = MagicMock()
        created_graphs.append(g)
        return g

    monkeypatch.setattr(plots_mod, "_new_graph_from_template", _fake_new_graph)

    cell = _linear_cell("A")
    sheets = {"chdis": MagicMock(), "cycle": MagicMock(), "dqdv": MagicMock()}
    ranges = _GraphRanges(
        chdis_x=(0.0, 1000.0),
        chdis_y=(3.0, 4.2),
        cycle_x=(1.0, 5.0),
        cycle_left_y=(900.0, 1000.0),
        cycle_right_y=(95.0, 99.0),
        dqdv_x=(3.0, 4.2),
        dqdv_y=(-10.0, 10.0),
    )
    create_cell_plots(MagicMock(), cell, sheets, ranges=ranges)

    chdis_graph, cycle_graph, dqdv_graph = created_graphs

    # chdis: one layer, two axis() calls (x + y).
    chdis_axes = [c.args[0] for c in chdis_graph.__getitem__.return_value.axis.call_args_list]
    assert chdis_axes == ["x", "y"], chdis_axes

    # cycle: two layers but MagicMock collapses graph[0] / graph[1]
    # into the same return value, so we see 4 axis() calls in order.
    cycle_axes = [c.args[0] for c in cycle_graph.__getitem__.return_value.axis.call_args_list]
    assert cycle_axes == ["x", "y", "x", "y"], cycle_axes

    # dqdv: one layer, two axis() calls.
    dqdv_axes = [c.args[0] for c in dqdv_graph.__getitem__.return_value.axis.call_args_list]
    assert dqdv_axes == ["x", "y"], dqdv_axes


def test_comparison_plots_share_the_per_cell_ranges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Comparison graphs must receive the same ``(lo, hi)`` as per-cell graphs."""
    _stub_templates(monkeypatch, tmp_path)
    _, _, graph_objs = _install_tracking_originpro(monkeypatch)
    cell_a = _linear_cell("A")
    cell_b = _linear_cell("B", q_ch=800.0, q_dis=760.0)

    from echemplot.origin import push_to_origin
    from echemplot.origin._plots import compute_global_ranges

    push_to_origin([cell_a, cell_b], stat_cycles=(1,))

    # Order: per-cell A (3), per-cell B (3), comparison (3) = 9 graphs.
    assert len(graph_objs) == 9

    expected = compute_global_ranges([cell_a, cell_b])

    # Per-cell A chdis graph (index 0) and comparison chdis graph
    # (index 6) should both have begin/end set to the shared range.
    per_cell_chdis_axis = graph_objs[0].__getitem__.return_value.axis
    comparison_chdis_axis = graph_objs[6].__getitem__.return_value.axis

    # Both must have been called with "x" and "y" at least once.
    per_cell_axes = {c.args[0] for c in per_cell_chdis_axis.call_args_list}
    comparison_axes = {c.args[0] for c in comparison_chdis_axis.call_args_list}
    assert per_cell_axes == {"x", "y"}
    assert comparison_axes == {"x", "y"}

    # Same axis object is returned by MagicMock for every axis() call,
    # so the last begin/end written wins. Assert on that final value
    # which is the comparison graph's y-range write — matching the
    # global chdis_y range.
    assert expected.chdis_y is not None
    final_begin = comparison_chdis_axis.return_value.begin
    final_end = comparison_chdis_axis.return_value.end
    assert final_begin == expected.chdis_y[0]
    assert final_end == expected.chdis_y[1]


def test_ranges_not_applied_when_dataframes_yield_no_finite_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All-None ranges → no axis() or lt_exec() calls on any layer."""
    _stub_templates(monkeypatch, tmp_path)
    _install_mock_originpro(monkeypatch)

    from echemplot.origin._plots import _GraphRanges, create_cell_plots

    layer = MagicMock()
    # Make graph[0] and graph[1] both return the same layer mock.
    graph = MagicMock()
    graph.__getitem__.return_value = layer

    # Patch _new_graph_from_template to return our tracked graph every
    # call, avoiding any template-file resolution concerns here.
    import echemplot.origin._plots as plots_mod

    monkeypatch.setattr(plots_mod, "_new_graph_from_template", lambda *a, **kw: graph)

    empty_ranges = _GraphRanges(
        chdis_x=None,
        chdis_y=None,
        cycle_x=None,
        cycle_left_y=None,
        cycle_right_y=None,
        dqdv_x=None,
        dqdv_y=None,
    )

    cell = _linear_cell("A")
    sheets = {
        "chdis": MagicMock(),
        "cycle": MagicMock(),
        "dqdv": MagicMock(),
    }
    create_cell_plots(MagicMock(), cell, sheets, ranges=empty_ranges)

    layer.axis.assert_not_called()
    layer.lt_exec.assert_not_called()


# ----------------------------------------------------------------------
# cap_df cycle-sheet column contract (issue #100)
# ----------------------------------------------------------------------


def _linear_cell_en(name: str, *, q_ch: float = 1000.0, q_dis: float = 990.0) -> Cell:
    """Same shape as :func:`_linear_cell` but with EN raw columns + ``column_lang="en"``.

    Used to confirm that the cycle-sheet contract holds across the
    ``column_lang`` axis: ``get_cap_df`` always emits English derived
    column names regardless of input language, so the same
    :data:`_CYCLE_SHEET_COLUMNS` shape applies to both modes.
    ``状態`` cell values stay JA — that's the documented per-row state
    convention; only the *column headers* switch.
    """
    n_points = 30
    rows: list[tuple[int, str, str, float, float]] = []
    v_ch = np.linspace(3.0, 4.2, n_points)
    q_ch_arr = np.linspace(0.0, q_ch, n_points)
    rows.extend((1, "1", "充電", float(v), float(q)) for v, q in zip(v_ch, q_ch_arr))
    v_dis = np.linspace(4.2, 3.0, n_points)
    q_dis_arr = np.linspace(0.0, q_dis, n_points)
    rows.extend((1, "1", "放電", float(v), float(q)) for v, q in zip(v_dis, q_dis_arr))
    raw = pd.DataFrame(rows, columns=["cycle", "mode", "state", "voltage", "capacity"])
    return Cell(name=name, mass_g=0.001, raw_df=raw, column_lang="en")


def test_origin_cycle_sheet_contract_holds_for_default_capdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A normally-built ``Cell`` must satisfy the cycle-sheet column contract.

    The Origin cycle_efficiency plot binds columns by 0-based positional
    index; those indices are derived from
    :data:`_CYCLE_SHEET_COLUMNS`. As long as
    :func:`echemplot.core.capacity.get_cap_df` keeps emitting
    ``[cycle, q_ch, q_dis, ce]`` in that order, ``push_to_origin`` must
    complete without raising :class:`OriginContractError`.
    """
    _stub_templates(monkeypatch, tmp_path)
    _install_mock_originpro(monkeypatch)
    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    # Must complete without raising the contract error. We don't assert
    # on call counts here (separate tests already do); the goal is to
    # exercise the contract gate on the happy path.
    push_to_origin([cell], stat_cycles=(1,))


def test_origin_cycle_sheet_contract_raises_on_reordered_capdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reordering ``cap_df`` columns must raise :class:`OriginContractError`.

    Patches :func:`echemplot.core.capacity.get_cap_df` (looked up where
    :class:`Cell` imports it from, in :mod:`echemplot.core.cell`) to
    return a frame whose columns swap ``q_ch`` and ``q_dis`` —
    exercises the partial-drift case where the column *set* still matches
    but the *order* is wrong, which would silently bind charge capacity
    to the discharge-capacity axis if the contract weren't enforced.

    The raised error must name both the expected and actual column
    lists so a future maintainer can read the contract violation
    straight from the traceback.
    """
    _stub_templates(monkeypatch, tmp_path)
    _install_mock_originpro(monkeypatch)

    from echemplot.core import cell as cell_mod
    from echemplot.origin._worksheets import OriginContractError

    real_get_cap_df = cell_mod.get_cap_df

    def _reordered_get_cap_df(*args: Any, **kwargs: Any) -> pd.DataFrame:
        df = real_get_cap_df(*args, **kwargs)
        # Swap q_ch and q_dis: column set is unchanged, order is wrong.
        return df[["q_dis", "q_ch", "ce"]]

    monkeypatch.setattr(cell_mod, "get_cap_df", _reordered_get_cap_df)

    cell = _linear_cell("A")

    from echemplot.origin import push_to_origin

    with pytest.raises(OriginContractError) as excinfo:
        push_to_origin([cell], stat_cycles=(1,))

    msg = str(excinfo.value)
    # Both the expected and actual column lists must surface in the
    # message so a future maintainer can diagnose the violation
    # without grepping the source.
    assert "cycle" in msg and "q_ch" in msg and "q_dis" in msg and "ce" in msg, msg
    # The "got" list is the swapped order produced above
    # (``cycle`` comes first via ``reset_index`` then the swapped pair).
    assert "['cycle', 'q_dis', 'q_ch', 'ce']" in msg, msg
    # And the expected list with the canonical order.
    assert "['cycle', 'q_ch', 'q_dis', 'ce']" in msg, msg


def test_origin_contract_error_subclasses_value_error() -> None:
    """``OriginContractError`` must subclass ``ValueError``.

    Callers of :func:`push_to_origin` already handle ``ValueError`` (the
    ``sg_window`` validation raises plain ``ValueError``). Subclassing
    keeps the broad ``except ValueError`` clauses in their code working
    without an extra ``except OriginContractError`` arm.
    """
    from echemplot.origin._worksheets import OriginContractError

    assert issubclass(OriginContractError, ValueError)


@pytest.mark.parametrize("column_lang", ["ja", "en"])
def test_origin_cycle_sheet_contract_works_for_both_langs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, column_lang: str
) -> None:
    """The cycle-sheet contract is the same in JA and EN ``column_lang`` modes.

    :func:`echemplot.core.capacity.get_cap_df` documents its derived
    column names as "fixed English" — ``column_lang`` only selects the
    *input* quantity label read out of ``chdis_df``, not the output
    schema. This test makes that invariant explicit at the Origin push
    boundary so a future change to ``get_cap_df`` that started honoring
    ``column_lang`` for output names would fail loudly here AND the
    underlying ``OriginContractError`` raised from ``write_cell_sheets``.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, _ = _install_tracking_originpro(monkeypatch)

    cell = _linear_cell("A") if column_lang == "ja" else _linear_cell_en("A")

    from echemplot.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # Sheet creation order: chdis → cycle → dqdv → stat_table.
    cycle_sheet = sheet_objs[1]
    from_df_call = cycle_sheet.from_df.call_args
    assert from_df_call is not None, "from_df not called on cycle sheet"
    written = from_df_call.args[0]
    # The contract holds in both modes: same English column names in
    # the same order, regardless of which language the cell was built
    # with. If this ever flips the test catches it before Origin
    # silently mis-binds.
    assert list(written.columns) == ["cycle", "q_ch", "q_dis", "ce"], (
        f"column_lang={column_lang!r} produced {list(written.columns)}"
    )
