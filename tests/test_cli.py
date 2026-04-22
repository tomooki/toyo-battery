"""Tests for :mod:`toyo_battery.cli` (the Typer batch CLI).

Invoked through :class:`typer.testing.CliRunner` so we exercise the full
parse + dispatch path rather than reaching into helpers. Synthetic TOYO
cell directories come from the shared :func:`make_cell_dir` fixture in
``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from toyo_battery.cli import app

runner = CliRunner()


def test_help_lists_all_three_subcommands() -> None:
    """``--help`` must mention each of ``process``, ``plot``, ``stats``."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("process", "plot", "stats"):
        assert name in result.output


def test_process_writes_three_csvs_per_cell(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    """``process <dir>`` writes ``{name}_chdis.csv`` / ``_cap.csv`` / ``_dqdv.csv``."""
    cell_dir = make_cell_dir("renzoku", name="cell_A")
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["process", str(cell_dir), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "cell_A_chdis.csv").is_file()
    assert (out_dir / "cell_A_cap.csv").is_file()
    assert (out_dir / "cell_A_dqdv.csv").is_file()


def test_plot_writes_png_for_requested_kind(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    """``plot <dir> --kinds chdis`` produces ``chdis.png`` in ``--out``."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    cell_dir = make_cell_dir("renzoku", name="cell_A")
    out_dir = tmp_path / "plots"

    result = runner.invoke(
        app,
        ["plot", str(cell_dir), "--out", str(out_dir), "--kinds", "chdis"],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "chdis.png").is_file()


def test_stats_writes_csv_with_expected_columns(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    """``stats <dir> --cycles 10 --out stats.csv`` writes a parseable CSV
    whose columns match the :func:`stat_table` contract.
    """
    cell_dir = make_cell_dir("renzoku", name="cell_A")
    out_path = tmp_path / "stats.csv"

    result = runner.invoke(
        app,
        ["stats", str(cell_dir), "--cycles", "10", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert out_path.is_file()
    df = pd.read_csv(out_path, index_col=0)
    # Pin the column schema for the N=10 single-cycle case. If stat_table
    # ever gains or renames a column, this test becomes the canary.
    assert list(df.columns) == [
        "Q_dis_max",
        "cycle_at_80pct",
        "Q_dis@10",
        "CE@10",
        "V_mean_dis@10",
        "EE@10",
        "retention@10",
        "CE_mean_1to10",
    ]


def test_invalid_cycles_exits_non_zero_with_bad_parameter(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    """Non-numeric ``--cycles`` values surface as a ``BadParameter`` error."""
    cell_dir = make_cell_dir("renzoku", name="cell_A")
    out_path = tmp_path / "stats.csv"

    result = runner.invoke(
        app,
        ["stats", str(cell_dir), "--cycles", "foo", "--out", str(out_path)],
    )

    assert result.exit_code != 0
    # Typer renders BadParameter as "Invalid value for '--cycles': ..." —
    # the substring match is loose enough to survive Typer cosmetic changes.
    assert "Invalid value" in result.output or "invalid" in result.output.lower()


def test_stats_multi_dir_produces_one_row_per_cell(
    make_cell_dir: Callable[..., Path], tmp_path: Path
) -> None:
    """Passing two cell dirs yields a CSV with two data rows (one per cell)."""
    dir_a = make_cell_dir("renzoku", name="cell_A")
    dir_b = make_cell_dir("renzoku_py", name="cell_B")
    out_path = tmp_path / "stats.csv"

    result = runner.invoke(
        app,
        [
            "stats",
            str(dir_a),
            str(dir_b),
            "--cycles",
            "1",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    df = pd.read_csv(out_path, index_col=0)
    assert len(df) == 2
    assert sorted(df.index.tolist()) == ["cell_A", "cell_B"]


def test_missing_directory_surfaces_as_bad_parameter(tmp_path: Path) -> None:
    """A non-existent directory argument yields a ``BadParameter``-style error."""
    missing = tmp_path / "does_not_exist"
    out_path = tmp_path / "stats.csv"

    result = runner.invoke(
        app,
        ["stats", str(missing), "--cycles", "1", "--out", str(out_path)],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output or "Invalid value" in result.output
