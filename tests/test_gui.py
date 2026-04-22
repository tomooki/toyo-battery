"""Tests for :mod:`echemplot.gui`.

Only the controller is exercised — the Tk view is touched with
import-only smoke to avoid spinning up a display in CI. Matplotlib is
imported via :func:`pytest.importorskip` so this module is silently
skipped when the ``[plot]`` / ``[gui]`` extras aren't installed.

``_write_wide_renzoku`` below builds a cell directory with a 1.2 V-wide
linear charge/discharge over two cycles — wide enough that the dQ/dV
interpolation density comfortably exceeds the default Savitzky-Golay
window (``ipnum = 100 * 1.2 = 120 >> 11``) and populates dQ/dV columns.
The conftest ``make_cell_dir`` fixture uses 0.2 V, which collapses dQ/dV
to all-NaN and defeats the SG-window assertion below.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from echemplot.gui._controller import GuiRequest, run


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """Close figures after each test so matplotlib's warning threshold
    isn't tripped as the suite grows.
    """
    yield
    plt.close("all")


def _write_wide_renzoku(
    cell_dir: Path,
    *,
    mass_mg: float = 1.0,
    n_cycles: int = 2,
    n_points: int = 200,
    v_lo: float = 3.0,
    v_hi: float = 4.2,
) -> None:
    """Write a ``連続データ.csv`` with a wide-voltage linear ramp per cycle.

    Mirrors the layout expected by :func:`echemplot.io.reader.read_cell_dir`
    for the renzoku path (mass row, JP header, channel row, separator,
    units, then data).
    """
    v_ch = np.linspace(v_lo, v_hi, n_points)
    q_ch = 400.0 * (v_ch - v_lo)
    v_dis = np.linspace(v_hi, v_lo, n_points)
    q_dis = 400.0 * (v_hi - v_dis)

    lines = [
        ",試験名,C:¥synthetic¥test¥path,,,,開始日時,2026-01-01 00:00:00",
        ",測定備考,",
        f",重量[mg],{mass_mg:.3f}",
        "サイクル,モード,状態,電圧,電気量",
        "1ch,1ch,1ch,1ch,1ch",
        "-,-,-,-,-",
        "[],[],[],[V],[mAh/g]",
    ]
    for cycle in range(1, n_cycles + 1):
        for v, q in zip(v_ch, q_ch):
            lines.append(f"{cycle},1,充電,{v:.4f},{q:.6f}")
        # Mid-cycle rest so chdis still sees a clean boundary.
        lines.append(f"{cycle},1,充電休止,{v_hi:.4f},{q_ch[-1]:.6f}")
        for v, q in zip(v_dis, q_dis):
            lines.append(f"{cycle},1,放電,{v:.4f},{q:.6f}")
    (cell_dir / "連続データ.csv").write_text("\n".join(lines) + "\n", encoding="shift_jis")


@pytest.fixture
def wide_cell_dir(tmp_path: Path) -> Path:
    d = tmp_path / "wide_cell"
    d.mkdir()
    _write_wide_renzoku(d)
    return d


# ---- controller returns ----------------------------------------------------


def test_run_chdis_returns_single_figure_with_axis_labels(
    wide_cell_dir: Path,
) -> None:
    result = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"chdis"}),
            cycles=[1, 2],
        )
    )
    assert len(result.figures) == 1
    assert len(result.cells) == 1
    ax = result.figures[0].axes[0]
    assert ax.get_xlabel() == "Capacity [mAh/g]"
    assert ax.get_ylabel() == "Voltage [V]"


def test_run_all_kinds_returns_three_figures(wide_cell_dir: Path) -> None:
    result = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"chdis", "cycle", "dqdv"}),
            cycles=[1, 2],
        )
    )
    assert len(result.figures) == 3
    ylabels = {fig.axes[0].get_ylabel() for fig in result.figures}
    # One of the figures is the cycle dual-Y plot whose primary axis is
    # "Discharge capacity [mAh/g]"; the chdis and dQ/dV figures contribute
    # the other two labels.
    assert ylabels == {
        "Voltage [V]",
        "Discharge capacity [mAh/g]",
        "dQ/dV [mAh/g/V]",
    }


# ---- sg_window propagation -------------------------------------------------


def test_sg_window_override_changes_dqdv_values(wide_cell_dir: Path) -> None:
    """A non-default ``sg_window`` must actually flow into the dQ/dV
    computation, not silently fall back to the cached-property default.

    Compares the y-data of the first dQ/dV line between a default-window
    run (11) and an override (21); the smoother window produces numerically
    distinct values across the curve.
    """
    fig_default = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"dqdv"}),
            cycles=[1],
            sg_window=11,
        )
    ).figures[0]
    fig_wide = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"dqdv"}),
            cycles=[1],
            sg_window=21,
        )
    ).figures[0]
    y_default = np.asarray(fig_default.axes[0].lines[0].get_ydata(), dtype=float)
    y_wide = np.asarray(fig_wide.axes[0].lines[0].get_ydata(), dtype=float)
    # Same sample count (interpolation grid is independent of the SG window)
    # but the smoothed derivatives should disagree on a non-trivial fraction
    # of samples.
    assert y_default.shape == y_wide.shape
    assert not np.allclose(y_default, y_wide)


def test_controller_passes_sg_window_to_plot_dqdv(
    monkeypatch: pytest.MonkeyPatch, wide_cell_dir: Path
) -> None:
    """The controller must forward ``request.sg_window`` as ``plot_dqdv``'s
    ``sg_window_length`` kwarg.

    Implementation note: the controller imports ``plot_dqdv`` lazily inside
    :func:`run`, so monkeypatching the controller-module attribute would be
    a no-op for the first call. We patch the symbol on the source module
    (``echemplot.plotting.matplotlib_backend``) which is what the lazy
    import resolves to.
    """
    import echemplot.plotting.matplotlib_backend as backend

    calls: list[dict[str, object]] = []

    def fake_plot_dqdv(
        cells: object,
        cycles: object = None,
        *,
        sg_window_length: int = 11,
        sg_polyorder: int = 2,
    ) -> object:
        calls.append(
            {
                "sg_window_length": sg_window_length,
                "sg_polyorder": sg_polyorder,
                "cycles": cycles,
            }
        )
        return plt.figure()

    monkeypatch.setattr(backend, "plot_dqdv", fake_plot_dqdv)

    result = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"dqdv"}),
            cycles=[1],
            sg_window=21,
        )
    )

    assert len(result.figures) == 1
    assert len(calls) == 1
    assert calls[0]["sg_window_length"] == 21


# ---- axis ranges -----------------------------------------------------------


def test_voltage_range_applied_to_chdis(wide_cell_dir: Path) -> None:
    result = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"chdis"}),
            cycles=[1],
            voltage_range=(3.0, 4.2),
        )
    )
    fig = result.figures[0]
    # All visible axes in the chdis figure are "Voltage [V]" primaries.
    lo, hi = fig.axes[0].get_ylim()
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(4.2)


def test_capacity_range_applied_only_to_primary_not_twin(
    wide_cell_dir: Path,
) -> None:
    """The ``cycle`` plot has a twin-Y (Coulombic efficiency). The
    capacity range must land on the primary axis only — a naive
    ``fig.axes[0].set_ylim`` on all axes would clobber the efficiency
    scale too. This test guards that labeled-axis dispatch.
    """
    result = run(
        GuiRequest(
            dirs=[wide_cell_dir],
            kinds=frozenset({"cycle"}),
            capacity_range=(0.0, 9999.0),
        )
    )
    fig = result.figures[0]
    primaries = [ax for ax in fig.axes if ax.get_ylabel() == "Discharge capacity [mAh/g]"]
    twins = [ax for ax in fig.axes if ax.get_ylabel() == "Coulombic efficiency [%]"]
    assert primaries and twins
    for ax in primaries:
        assert ax.get_ylim() == pytest.approx((0.0, 9999.0))
    # Twin y-axis limits are whatever matplotlib auto-picked — crucially
    # NOT (0, 9999), which would indicate cross-contamination.
    for ax in twins:
        assert ax.get_ylim() != pytest.approx((0.0, 9999.0))


# ---- validation ------------------------------------------------------------


def test_invalid_sg_window_even_raises(wide_cell_dir: Path) -> None:
    with pytest.raises(ValueError, match="sg_window must be odd"):
        run(
            GuiRequest(
                dirs=[wide_cell_dir],
                kinds=frozenset({"dqdv"}),
                sg_window=10,
            )
        )


def test_invalid_sg_window_nonpositive_raises(wide_cell_dir: Path) -> None:
    with pytest.raises(ValueError, match="sg_window must be >= 1"):
        run(
            GuiRequest(
                dirs=[wide_cell_dir],
                kinds=frozenset({"dqdv"}),
                sg_window=0,
            )
        )


def test_unknown_kind_raises(wide_cell_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown plot kinds"):
        run(
            GuiRequest(
                dirs=[wide_cell_dir],
                kinds=frozenset({"chdis", "stats"}),  # "stats" isn't a GUI plot
            )
        )


def test_empty_dirs_raises() -> None:
    with pytest.raises(ValueError, match="dirs must contain"):
        run(GuiRequest(dirs=[], kinds=frozenset({"chdis"})))


def test_empty_kinds_raises(wide_cell_dir: Path) -> None:
    with pytest.raises(ValueError, match="kinds must contain"):
        run(GuiRequest(dirs=[wide_cell_dir], kinds=frozenset()))


# ---- smoke -----------------------------------------------------------------


def test_tk_app_module_importable_without_display() -> None:
    """Importing the view must not require a live display.

    If the import itself spun up a Tk root or called ``mainloop``, CI
    (no X server, no Windows session) would hang or raise. We accept a
    ``TclError`` from any deeper setup — but the import line itself must
    succeed.
    """
    import echemplot.gui.tk_app as tk_app

    assert callable(tk_app.launch_gui)


def test_gui_package_reexports_launch_gui() -> None:
    """``launch_gui`` is the public entry point — callable from CLI
    (``python -m echemplot.gui``) and from host Python processes such
    as Origin's embedded console (``from echemplot.gui import
    launch_gui``). Guarding the re-export keeps that import path stable.
    """
    import echemplot.gui as gui

    assert callable(gui.launch_gui)


def test_add_dir_deduplicates_paths(tmp_path: Path) -> None:
    """``_add_dir`` is the single path through which both Add-button and
    drag-and-drop reach the state list; adding the same directory twice
    must leave exactly one entry so Remove-selected / Run stays
    unambiguous.

    Exercised without a live Tk — we stub the Listbox ``insert`` call
    since we only care about the ``self._dirs`` invariant and asserting
    ``insert`` is not called on the duplicate attempt.
    """
    import echemplot.gui.tk_app as tk_app

    class _FakeListbox:
        def __init__(self) -> None:
            self.inserts: list[str] = []

        def insert(self, _where: object, value: str) -> None:
            self.inserts.append(value)

    app = tk_app._App.__new__(tk_app._App)
    app._dirs = []  # type: ignore[attr-defined]
    app._dirs_list = _FakeListbox()  # type: ignore[attr-defined]

    d = tmp_path / "cell_a"
    d.mkdir()

    app._add_dir(d)
    app._add_dir(d)  # duplicate — should be ignored
    app._add_dir(tmp_path / "cell_b")  # different path — counted even though missing

    assert app._dirs == [d, tmp_path / "cell_b"]
    assert app._dirs_list.inserts == [str(d), str(tmp_path / "cell_b")]  # type: ignore[attr-defined]
