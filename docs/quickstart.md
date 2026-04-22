# Quick start

This page walks through loading one cell directory and inspecting the
derived tables that `echemplot` exposes.

## Install

```bash
pip install "echemplot[plot]"
```

The `[plot]` extra pulls in Matplotlib, which is required for the plotting
helpers shown below. Core data ingestion works without it.

## Load a cell

`Cell.from_dir` (see [Cell API](api/cell.md)) is the one-call entry
point — it discovers the TOYO files in the directory, picks the right
reader, resolves the active-material mass, and returns a `Cell` instance.

```python
from echemplot.core.cell import Cell

cell = Cell.from_dir("path/to/cell_dir")
print(cell.name, cell.mass_g)
```

The directory is expected to contain one of:

- `連続データ.csv` — native export from the TOYO tester
- `連続データ_py.csv` — already-normalized output from a previous run
- A set of 6-digit raw files plus a `*.PTN` file carrying the mass

See the [IO API](api/io.md) for the full discovery / mass-resolution
contract.

## Derived tables

All derived tables are [`cached_property`](https://docs.python.org/3/library/functools.html#functools.cached_property)
on the `Cell`, so the first access computes them and subsequent accesses
are free.

```python
# Per-cycle charge/discharge segments (wide, 3-level column MultiIndex)
chdis = cell.chdis_df

# Per-cycle summary: Q_ch / Q_dis / Coulombic efficiency
cap = cell.cap_df
cap.to_csv("cycle.csv")

# dQ/dV with default parameters
dqdv = cell.dqdv_df
```

For non-default dQ/dV parameters (`inter_num`, `window_length`, `polyorder`)
call `get_dqdv_df` directly on `cell.chdis_df` (see
[Core API](api/core.md)).

## Summary table across multiple cells

`stat_table` (see [Core API](api/core.md)) aggregates a list of `Cell`
objects into a single wide summary table — one row per cell, with
deterministic capacity / efficiency / retention columns keyed off a
caller-supplied `target_cycles`.

```python
from echemplot.core.stats import stat_table

cells = [Cell.from_dir(p) for p in cell_dirs]
summary = stat_table(cells, target_cycles=(10, 50, 100))
summary.to_csv("summary.csv")
```

## Plot

The Matplotlib backend exposes three helpers that each return a
`matplotlib.figure.Figure` — the caller handles `savefig` / `close`.

```python
from echemplot.plotting.matplotlib_backend import (
    plot_chdis,
    plot_cycle,
    plot_dqdv,
)

fig = plot_chdis([cell])
fig.savefig("chdis.png", dpi=150)
```

Plot every cycle for the cell (cycle 1 is drawn in red, other cycles in
black — the TOYO legacy convention). Pass `cycles=[1, 5, 10]` to restrict
the cycle set.

For multi-cell overlay plots, pass a list: `plot_chdis([cell_a, cell_b])`
produces one subplot per cell in a near-square grid.

See the [Plotting API](api/plotting.md) for the full function signatures.
