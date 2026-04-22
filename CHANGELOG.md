# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3] - 2026-04-22

### Fixed
- `push_to_origin` opened graph windows from the bundled `.otpu` templates
  but rendered them without any data lines. Root cause: `Layer.add_plot`
  was called without `colx` / `coly`, so `originpro` left the plot with
  no column designation. Secondary cause: `cap_df` was written with its
  `cycle` index still as an index, so the `cycle_efficiency` template
  had no `cycle` column to bind as X.
- `toyo_battery.origin` now passes explicit 0-based `colx` / `coly`
  indices on every `add_plot` call (one call per `(cycle, side)` pair
  for `chdis` / `dqdv`, plus dual-Y bindings for the `cycle_efficiency`
  template — `graph[0]` = discharge capacity, `graph[1]` = Coulombic
  efficiency). Worksheets additionally call `wks.cols_axis(...)` after
  `from_df` so each column's X/Y role is designated up front.

## [0.0.2] - 2026-04-22

### Added
- `toyo_battery.gui.launch_gui()` public entry point for launching the Tk GUI
  from a host Python process (including Origin's embedded Python Console).
  Documented in `docs/ORIGIN_SETUP.md`.
- `toyo_battery.origin.launch_gui()` — one-liner for Origin users that opens
  the Tk directory picker and routes Run output into the active Origin
  project via `push_to_origin`, no PNG/CSV intermediates.
- Bundled the three `.otpu` graph templates (`charge_discharge`,
  `cycle_efficiency`, `dqdv`) inside the wheel so `push_to_origin` works
  out of the box from a fresh `pip install`. `TOYO_ORIGIN_TEMPLATE_DIR`
  remains as an override path.
- Internal `toyo_battery.gui._controller.RunResult` dataclass exposing
  both the loaded `Cell` instances and the generated figures from
  `run()`, so the Origin launcher can hand cells straight to
  `push_to_origin` without re-loading from disk. Private to the package
  — there is no supported public import path.

### Changed
- Renamed internal `toyo_battery.gui.main` → `launch_gui`. The CLI entry
  `python -m toyo_battery.gui` is unaffected.
- `toyo_battery.gui._controller.run` now returns `RunResult(cells, figures)`
  instead of a bare list of figures. The Tk view (`tk_app.py`) is the only
  in-tree caller and has been updated; the module is private (leading
  underscore) so no external code is expected to depend on it.
- `[origin]` extra now declares `matplotlib>=3.7` so a single
  `pip install toyo-battery[origin]` is enough to run `launch_gui` from
  inside Origin (previously required adding `[gui]` separately).

## [0.0.1] - 2026-04-22

### Added
- Initial public scaffold ported from the private `TOYO_origin` scripts.
- `toyo_battery.io` reader for TOYO System cycler output (Shift-JIS CSV + PTN).
- `toyo_battery.core`: `Cell` class, capacity / charge-discharge / dQdV / stats helpers.
- `toyo_battery.plotting`: Matplotlib and Plotly backends (`plot_chdis`, `plot_dqdv`, `plot_cycle`).
- `toyo_battery.cli` (`toyo-battery process | plot | stats`) — requires `[cli]` extra.
- `toyo_battery.gui` Tk app (`python -m toyo_battery.gui`) — requires `[gui]` extra.
- `toyo_battery.origin` adapter for OriginLab's embedded Python.
- MkDocs + mkdocstrings documentation, deployed via the `Docs` workflow.

### Notes
- **Pre-alpha.** Public API is unstable and may change without deprecation in
  0.0.x releases.

[Unreleased]: https://github.com/tomooki/toyo-battery/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/tomooki/toyo-battery/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/tomooki/toyo-battery/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/tomooki/toyo-battery/releases/tag/v0.0.1
