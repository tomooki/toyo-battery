# toyo-battery

OSS Python toolkit for TOYO battery cycler data.

Reads output from TOYO System battery charge/discharge testers, computes
cycle capacity, Coulombic efficiency, dQ/dV curves, and summary statistics.
Pure Python, installable from PyPI — including into OriginLab's embedded
Python.

- PyPI: <https://pypi.org/project/toyo-battery/>
- Docs: <https://tomooki.github.io/toyo-battery/>
- Changelog: [CHANGELOG.md](./CHANGELOG.md)

> Status: **pre-alpha (0.0.x)**. Public API is unstable and may change
> without deprecation until 0.1.0.

## Installation

```bash
# Core library only (numpy, pandas, scipy)
pip install toyo-battery

# Pick one or more extras:
pip install "toyo-battery[plot]"    # Matplotlib plotting
pip install "toyo-battery[plotly]"  # Plotly + Kaleido (static image export)
pip install "toyo-battery[cli]"     # toyo-battery CLI (typer + rich)
pip install "toyo-battery[gui]"     # Tk desktop app (matplotlib)

# Everything non-Origin in one shot
pip install "toyo-battery[all]"
```

The `[origin]` extra has no runtime deps — the `toyo_battery.origin` submodule
imports `originpro` lazily, so it only works inside Origin's embedded Python
(see below).

## Quick start

```python
from toyo_battery import Cell

cell = Cell.from_dir("path/to/cell_dir")

cell.chdis_df.to_csv("chdis.csv")   # charge/discharge V-Q curves per cycle
cell.cap_df.to_csv("cycle.csv")     # per-cycle capacity + Coulombic efficiency
cell.dqdv_df.to_csv("dqdv.csv")     # dQ/dV per cycle
```

`Cell.from_dir()` auto-detects three TOYO layouts: `連続データ.csv` (native
export), `連続データ_py.csv` (pre-normalized), or 6-digit raw files with a
`*.PTN` pattern file. Mass is read from the PTN file unless you pass
`mass=...` (grams).

## CLI

Install with `pip install "toyo-battery[cli]"`. All subcommands accept one or
more cell directories plus the shared options `--mass`, `--encoding`,
`--column-lang`.

```bash
# {name}_chdis.csv / {name}_cap.csv / {name}_dqdv.csv per cell
toyo-battery process cell_A cell_B --out ./csvs

# chdis / cycle / dqdv PNGs (one figure per kind, one Axes per cell)
toyo-battery plot cell_A cell_B --out ./pngs \
    --kinds chdis,cycle,dqdv --cycles 1,10,50

# Single stat_table CSV spanning all cells at the given target cycles
toyo-battery stats cell_A cell_B --cycles 10,50 --out stats.csv
```

## GUI

Install with `pip install "toyo-battery[gui]"` and launch:

```bash
python -m toyo_battery.gui
```

Tk app for interactive directory selection, plot-kind toggles, per-axis
ranges, and Savitzky–Golay window tuning for dQ/dV.

## Plotting from Python

Matplotlib and Plotly backends expose the same three functions — pick by
import path:

```python
from toyo_battery import Cell
from toyo_battery.plotting.matplotlib_backend import plot_chdis, plot_cycle, plot_dqdv
# or: from toyo_battery.plotting.plotly_backend import plot_chdis, plot_cycle, plot_dqdv

cells = [Cell.from_dir(p) for p in ("cell_A", "cell_B")]
fig = plot_cycle(cells)
fig.savefig("cycle.png", dpi=150, bbox_inches="tight")  # matplotlib
# fig.write_image("cycle.png")                          # plotly
```

## Installing into Origin's embedded Python

Open Origin, then in the Origin Python console:

```python
import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "toyo-battery[origin]"]
)
```

The `[origin]` extra pulls in matplotlib so the Tk GUI works out of the box;
the three `.otpu` graph templates ship inside the wheel. After install, one
line opens the directory picker and pushes results into the active Origin
project:

```python
from toyo_battery.origin import launch_gui
launch_gui()
```

For scripted use, `toyo_battery.origin.push_to_origin(cells, ...)` populates
the current Origin project with per-cell worksheets, template graphs, and a
stat sheet. It imports `originpro` lazily, so the submodule is importable
outside Origin (`push_to_origin` itself raises there).

See [docs/ORIGIN_SETUP.md](./docs/ORIGIN_SETUP.md) for the full workflow.

## License

MIT — see [LICENSE](./LICENSE).
