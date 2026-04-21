# toyo-battery

OSS Python toolkit for TOYO battery cycler data.

Reads output from TOYO System battery charge/discharge testers, computes
cycle capacity, Coulombic efficiency, dQ/dV curves, and summary statistics.
Pure Python, installable from PyPI — including into OriginLab's embedded
Python.

!!! warning "Pre-alpha"
    Status: **0.0.1 — pre-alpha scaffold.** Core logic is being ported from
    the private `TOYO_origin` scripts. Public API is unstable.

## Installation

```bash
# Core only
pip install toyo-battery

# With Matplotlib plotting
pip install "toyo-battery[plot]"

# Everything except Origin
pip install "toyo-battery[all]"
```

The `toyo_battery.origin` submodule uses `originpro` (shipped with Origin)
and is only importable inside Origin's Python environment. See
[Origin setup](ORIGIN_SETUP.md) for installing into Origin's embedded Python.

## Next steps

- [Quick start](quickstart.md) — load a cell directory and inspect derived
  tables.
- API reference — auto-generated from the source docstrings:
    - [Cell](api/cell.md)
    - [IO](api/io.md)
    - [Core](api/core.md)
    - [Plotting](api/plotting.md)

## License

MIT.
