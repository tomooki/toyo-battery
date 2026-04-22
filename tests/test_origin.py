"""Tests for :mod:`toyo_battery.origin`.

Real ``originpro`` is not available outside Origin's embedded Python, so
these tests build a fake module on ``sys.modules`` before calling
:func:`push_to_origin`. All assertions are against call counts / call
shapes on that mock — the real-Origin end-to-end verification is tracked
separately in issue #15.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from toyo_battery.core.cell import Cell

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
    """Point ``TOYO_ORIGIN_TEMPLATE_DIR`` at a dir with zero-byte templates.

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
    monkeypatch.setenv("TOYO_ORIGIN_TEMPLATE_DIR", str(tpl_dir))
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
    from toyo_battery.origin import _require_originpro

    with pytest.raises(ImportError, match="OriginLab's embedded Python"):
        _require_originpro()


def test_require_originpro_returns_module_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    from toyo_battery.origin import _require_originpro

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

    from toyo_battery.origin import push_to_origin

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

    import toyo_battery.origin as origin_mod

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

    from toyo_battery.origin import push_to_origin

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
    monkeypatch.setenv("TOYO_ORIGIN_TEMPLATE_DIR", str(empty))

    from toyo_battery.origin import push_to_origin

    cell = _linear_cell("A")
    with pytest.raises(FileNotFoundError) as excinfo:
        push_to_origin([cell], stat_cycles=(1,))

    msg = str(excinfo.value)
    assert "charge_discharge.otpu" in msg
    assert "TOYO_ORIGIN_TEMPLATE_DIR" in msg


def test_default_templates_resolve_from_bundled_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``TOYO_ORIGIN_TEMPLATE_DIR`` set, the three required ``.otpu``
    templates must resolve to non-empty files inside the package.

    Guards the "default install only" UX: a fresh ``pip install
    toyo-battery[origin]`` inside Origin must produce graphs without any
    template-copy step. Regressing this means ``push_to_origin`` would
    raise ``FileNotFoundError`` for every Origin user.
    """
    monkeypatch.delenv("TOYO_ORIGIN_TEMPLATE_DIR", raising=False)
    from toyo_battery.origin._plots import _require_template

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

    from toyo_battery.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    for call in mock_op.new_sheet.call_args_list:
        assert len(call.kwargs["lname"]) <= 32, call.kwargs


def test_sanitize_sheet_name_preserves_short_names() -> None:
    from toyo_battery.origin._worksheets import _sanitize_sheet_name

    assert _sanitize_sheet_name("A_chdis") == "A_chdis"
    assert _sanitize_sheet_name("x" * 32) == "x" * 32


def test_sanitize_sheet_name_disambiguates_long_prefix_collisions() -> None:
    """Two overlong names sharing a 31-char prefix must yield distinct sheets."""
    from toyo_battery.origin._worksheets import _sanitize_sheet_name

    a = "x" * 40 + "A"
    b = "x" * 40 + "B"
    assert _sanitize_sheet_name(a) != _sanitize_sheet_name(b)
    assert len(_sanitize_sheet_name(a)) <= 32


# ----------------------------------------------------------------------
# MultiIndex flattening
# ----------------------------------------------------------------------


def test_flatten_columns_joins_multiindex_levels_with_underscore() -> None:
    from toyo_battery.origin._worksheets import _flatten_columns

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
    from toyo_battery.origin._worksheets import _flatten_columns

    df = pd.DataFrame({"q_ch": [1.0], "q_dis": [2.0]})
    assert list(_flatten_columns(df).columns) == ["q_ch", "q_dis"]


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

    from toyo_battery.origin import push_to_origin

    push_to_origin([cell], project_path=project, stat_cycles=(1,))

    mock_op.open.assert_called_once_with(project)
    mock_op.save.assert_called_once_with(project)


def test_push_to_origin_skips_open_save_when_project_path_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_op = _install_mock_originpro(monkeypatch)
    _stub_templates(monkeypatch, tmp_path)
    cell = _linear_cell("A")

    from toyo_battery.origin import push_to_origin

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

    from toyo_battery.origin import push_to_origin

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

    from toyo_battery.origin import push_to_origin

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

    from toyo_battery.origin import push_to_origin

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


def test_cols_axis_is_set_per_category(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Worksheets for plot-targeted categories must declare X/Y designations.

    ``from_df`` leaves every column as ``"Y"``; without a ``cols_axis``
    call the template-backed plot has no X to bind against and renders
    empty. ``stat_table`` is never plotted so it skips the call.
    """
    _stub_templates(monkeypatch, tmp_path)
    _, sheet_objs, _ = _install_tracking_originpro(monkeypatch)
    cell = _linear_cell("A")

    from toyo_battery.origin import push_to_origin

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

    from toyo_battery.origin import push_to_origin

    push_to_origin([cell], stat_cycles=(1,))

    # Second sheet created is the cycle sheet (chdis=0, cycle=1, dqdv=2).
    cycle_sheet = sheet_objs[1]
    from_df_call = cycle_sheet.from_df.call_args
    assert from_df_call is not None, "from_df not called on cycle sheet"
    written = from_df_call.args[0]
    assert "cycle" in written.columns, (
        f"cycle column missing from cap_df write: {list(written.columns)}"
    )
    # Order must match the _CYCLE_COL_* constants in _plots.py.
    assert next(iter(written.columns)) == "cycle", list(written.columns)
