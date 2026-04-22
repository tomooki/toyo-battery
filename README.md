# toyo-battery

OSS Python toolkit for TOYO battery cycler data.

Reads output from TOYO System battery charge/discharge testers, computes
cycle capacity, Coulombic efficiency, dQ/dV curves, and summary statistics.
Pure Python, installable from PyPI — including into OriginLab's embedded
Python.

> Status: **0.0.1 — pre-alpha scaffold.** Core logic is being ported from the
> private `TOYO_origin` scripts. Public API is unstable.

## Installation

```bash
# Core only
pip install toyo-battery

# With Matplotlib plotting
pip install "toyo-battery[plot]"

# Everything except Origin
pip install "toyo-battery[all]"
```

### CLI

With the `[cli]` extra (`pip install "toyo-battery[cli]"`) the `toyo-battery`
command offers batch operations over one or more cell directories:

```bash
toyo-battery process cell_A cell_B --out ./csvs        # chdis/cap/dqdv CSVs
toyo-battery plot    cell_A cell_B --out ./pngs        # chdis/cycle/dqdv PNGs
toyo-battery stats   cell_A cell_B --cycles 10,50 --out stats.csv
```

### Installing into Origin's embedded Python

Open Origin, go to **Connectivity → Python Packages**, then in the Origin
Python console:

```python
import pip
pip.main(["install", "toyo-battery"])
```

The `toyo_battery.origin` submodule uses `originpro` (shipped with Origin)
and is only importable inside Origin's Python environment.

## Quick start

```python
from toyo_battery import Cell

cell = Cell.from_dir("path/to/cell_dir")
cell.cap_df.to_csv("cycle.csv")
```

## License

MIT — see [LICENSE](./LICENSE).
