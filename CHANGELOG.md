# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `read_cell_dir` no longer raises `ValueError: unknown 状態 codes in
  source: [9]` on raw 6-digit cell directories whose last data row is
  the TOYO end-of-test sentinel (state code `9` with zero 経過時間 and
  zero 電流). The `_finalize` step now drops a contiguous trailing
  block of such sentinel rows (with a `logger.debug` notice) before
  state-code mapping. Unknown state codes that appear mid-file or with
  non-zero flow still raise, and the error now reports the offending
  row index and a link to file an issue. ([#91])

[#91]: https://github.com/tomooki/toyo-battery/issues/91

## [0.1.6] - 2026-04-23

### Changed
- Default Savitzky-Golay `window_length` for dQ/dV smoothing restored to
  `11` (reverting [#69]). Applies uniformly across
  `echemplot.core.dqdv.get_dqdv_df`, `Cell.dqdv_df`, `plot_dqdv`,
  `echemplot.origin.push_to_origin`, and the Tk GUI's `SG window_length`
  entry. Callers that explicitly pass `window_length=21` / `sg_window=21`
  are unaffected. ([#88])

## [0.1.5] - 2026-04-23

### Fixed
- `echemplot.core.cell` (and anything importing it, including
  `push_to_origin` and the Origin-mode GUI) no longer raises
  `ImportError: cannot import name 'NDArray' from 'numpy.typing'` on
  Origin's embedded Python 3.11. Origin ships a NumPy that predates
  `numpy.typing.NDArray` (NumPy < 1.21) and is not reliably upgradable
  from inside Origin's site-packages. `NDArray` was used only for
  static type hints in `core/dqdv.py` and `core/stats.py`; both modules
  already carry `from __future__ import annotations`, so the imports —
  and the module-scope `_SegPair` alias in `dqdv.py` — now live under
  `TYPE_CHECKING`, eliminating the runtime dependency on
  `numpy.typing.NDArray`. ([#86])

## [0.1.4] - 2026-04-23

### Fixed
- `push_to_origin` (and the Origin-mode GUI completion hook) no longer
  raises `AttributeError: 'StringDtype' object has no attribute 'char'`
  when the host pandas has `future.infer_string` enabled (or defaults
  it on in a future pandas release). `originpro.worksheet.from_df`
  dispatches per-column storage via `.dtype.char`, which pandas
  extension dtypes don't expose; `echemplot.origin._worksheets` now
  pins the flattened column `Index` to `dtype=object` and applies a
  `_coerce_for_originpro` preprocessing step that rebuilds any
  extension-dtype column Index as numpy-object and converts string-like
  extension columns to object — closing both the `_flatten_columns`
  Index path and the `stat_table.reset_index()` `cell` column path.
  ([#75], [#80])

### Changed
- `echemplot.origin.launch_gui` now closes the Tk window automatically
  after a successful Run. The launcher's `mainloop()` runs inside the
  Origin Python Console, so previously the user had to click the window
  close button to release the Console; the one-shot batch UX now simply
  ends when the push completes. Error dialogs still keep the window open
  so the inputs can be corrected and re-Run. ([#79])

## [0.1.3] - 2026-04-23

### Fixed
- Origin-mode graphs now autoscale after binding. Every template-backed
  layer receives a `layer.rescale()` call at the tail of the bind
  helpers (`_bind_xy_pairs`, `_bind_cycle`) so the three per-cell and
  three comparison graphs fit their data instead of inheriting the
  template's default axis window. The issue #61 shared-range path
  (`_set_axis_limits` under explicit `ranges`) continues to override
  the rescaled values. ([#73])

## [0.1.2] - 2026-04-22

### Changed
- Default Savitzky-Golay `window_length` for dQ/dV smoothing raised from
  `11` to `21`. Applies uniformly across `echemplot.core.dqdv.get_dqdv_df`,
  `Cell.dqdv_df`, `plot_dqdv`, `echemplot.origin.push_to_origin`, and the
  Tk GUI's `SG window_length` entry. Callers that explicitly pass
  `window_length=11` / `sg_window=11` are unaffected. ([#69])

## [0.1.1] - 2026-04-22

### Added
- Multi-folder drag-and-drop support on the GUI's cell-directory list
  when `tkinterdnd2` is installed (shipped in the `[gui]` / `[all]`
  extras as an optional dependency). Dropping multiple folders from the
  host file manager queues them all at once. When `tkinterdnd2` or the
  underlying tkdnd Tcl extension is unavailable the GUI silently falls
  back to Add-button-only; no user-visible error surfaces. ([#63])
- `origin_mode` keyword on `echemplot.gui.launch_gui`. When `True`, the
  plot-kind checkboxes, cycles entry, and voltage/capacity/dQ-dV range
  entries are greyed out and annotated with an inline note explaining
  that only `SG window_length` reaches the Origin push path.
  `echemplot.origin.launch_gui` now sets this automatically. ([#60])
- `sg_window` / `sg_polyorder` parameters on
  `echemplot.origin.push_to_origin`. At the defaults the cached
  `Cell.dqdv_df` is reused as before; non-default values trigger a
  one-off `get_dqdv_df` recompute per cell so the dQ/dV worksheet
  reflects the caller's choice. The Tk view's `OnComplete` callback
  signature gained a third `sg_window: int` argument so the Origin
  launcher can forward the GUI value. ([#60])

### Changed
- Origin template graphs now autoscale to a shared axis range computed
  across all input cells. With a single cell the range collapses to
  that cell's data; with multiple cells every per-cell graph and the
  comparison overlay share the same scale for direct comparison.
  Rescaling is a no-op when the data yields no finite values, so
  graphs fall back to the template's baked-in scale. ([#61])
- The GUI folder-picker no longer loops. Clicking `Add...` opens one
  `askdirectory` dialog and returns; cancel is a no-op. Users add
  multiple directories either by clicking `Add...` repeatedly or via
  the new drag-and-drop path above. ([#62])
- `echemplot.origin._plots` documents a new originpro API assumption:
  `layer.axis(axis).begin/end` attribute access for axis limits. A
  LabTalk fallback via `op.lt_exec` is in place for environments where
  the attribute path doesn't propagate to Origin, guarded by a
  round-trip readback.

## [0.1.0] - 2026-04-22

### Changed
- **BREAKING**: Renamed the package from `toyo-battery` (PyPI distribution)
  / `toyo_battery` (import) to `echemplot` for both. All user code must
  replace `from toyo_battery ...` with `from echemplot ...`, `pip install
  toyo-battery[...]` with `pip install echemplot[...]`, and the CLI
  command `toyo-battery` with `echemplot`. No compatibility shim is
  provided — the old `toyo_battery` import no longer resolves. The
  `toyo-battery` project on PyPI remains available for installing past
  0.0.x releases; new versions are published under `echemplot`.
- **BREAKING**: Renamed the Origin template override env var
  `TOYO_ORIGIN_TEMPLATE_DIR` → `ECHEMPLOT_ORIGIN_TEMPLATE_DIR`. Users who
  set the override must update their environment; the old name is no
  longer read.

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

[Unreleased]: https://github.com/tomooki/toyo-battery/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/tomooki/toyo-battery/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/tomooki/toyo-battery/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/tomooki/toyo-battery/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/tomooki/toyo-battery/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/tomooki/toyo-battery/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/tomooki/toyo-battery/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tomooki/toyo-battery/compare/v0.0.3...v0.1.0
[#60]: https://github.com/tomooki/toyo-battery/issues/60
[#61]: https://github.com/tomooki/toyo-battery/issues/61
[#62]: https://github.com/tomooki/toyo-battery/issues/62
[#63]: https://github.com/tomooki/toyo-battery/issues/63
[#69]: https://github.com/tomooki/toyo-battery/pull/69
[#73]: https://github.com/tomooki/toyo-battery/pull/73
[#75]: https://github.com/tomooki/toyo-battery/issues/75
[#79]: https://github.com/tomooki/toyo-battery/pull/79
[#80]: https://github.com/tomooki/toyo-battery/pull/80
[#86]: https://github.com/tomooki/toyo-battery/pull/86
[#88]: https://github.com/tomooki/toyo-battery/pull/88
[0.0.3]: https://github.com/tomooki/toyo-battery/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/tomooki/toyo-battery/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/tomooki/toyo-battery/releases/tag/v0.0.1
