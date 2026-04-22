# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `toyo_battery.gui.launch_gui()` public entry point for launching the Tk GUI
  from a host Python process (including Origin's embedded Python Console).
  Documented in `docs/ORIGIN_SETUP.md`.

### Changed
- Renamed internal `toyo_battery.gui.main` → `launch_gui`. The CLI entry
  `python -m toyo_battery.gui` is unaffected.

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

[Unreleased]: https://github.com/tomooki/toyo-battery/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/tomooki/toyo-battery/releases/tag/v0.0.1
