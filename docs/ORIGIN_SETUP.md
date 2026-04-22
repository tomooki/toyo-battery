# Installing echemplot into Origin's embedded Python

`echemplot` is a **pure-Python** package published on PyPI. It installs into
OriginLab's embedded Python the same way as numpy/pandas/scipy.

## Steps

1. Open Origin.
2. **Connectivity → Python Packages** (or Python Console → `pip`).
3. Install with the `[origin]` extra — it pulls in matplotlib for the Tk GUI
   path so a single command is enough:

   ```python
   import subprocess
   import sys

   subprocess.check_call(
       [sys.executable, "-m", "pip", "install", "--upgrade", "echemplot[origin]"]
   )
   ```

4. Confirm:

   ```python
   import echemplot
   print(echemplot.__version__)
   ```

5. Optional — check the Origin adapter is wired up:

   ```python
   from echemplot.origin import push_to_origin
   ```

   (Inside Origin this succeeds. Outside Origin, importing `echemplot.origin`
   still works — calling `push_to_origin` will raise `ImportError` with a clear
   message.)

## Launching the GUI from Origin

After step 3 above, the recommended one-liner runs the Tk directory picker
and pushes results straight into the active Origin project:

```python
from echemplot.origin import launch_gui
launch_gui()
```

Pick one or more cell directories in the dialog, set the plot kinds /
parameters you want, and click **Run**. For each cell the launcher creates
`{name}_chdis`, `{name}_cycle`, `{name}_dqdv` worksheets, the matching
template-backed graphs, and a `stat_table` worksheet — no PNG/CSV
intermediates, no manual `push_to_origin(...)` call.

To override the cycles used for the stat sheet, or to open a project file
on disk, pass kwargs:

```python
launch_gui(project_path=r"C:\Users\me\Documents\toyo.opju", stat_cycles=(10, 50, 100))
```

If you want the standalone GUI behaviour (matplotlib figures in `Toplevel`
windows, no Origin write-back) instead, call `echemplot.gui.launch_gui()`
directly.

Notes:

- `launch_gui()` runs Tk's `mainloop()` **in the Origin process**, so the
  Origin Python Console is blocked while the window is open. After a
  successful Run the window closes automatically and the Console prompt
  returns; if the Run surfaces an error dialog, close the window manually
  (or fix the inputs and Run again) to release the Console.
- Origin's embedded CPython must include `_tkinter` (OriginLab 2022 and later
  do). If it doesn't, the call raises `ImportError` from `import tkinter`; use
  a separate system Python in that case.
- The three `.otpu` graph templates ship inside the wheel — no extra setup
  is needed. Set `ECHEMPLOT_ORIGIN_TEMPLATE_DIR` only if you want to substitute
  your own templates.

## Origin Python version check (please report back)

Please run this inside Origin's Python Console and paste the result into the
repo as a comment on the P0 tracking issue:

```python
import sys, platform
print(sys.version)
print(platform.python_implementation(), platform.python_version())
```

This tells us the exact CPython version shipped with your Origin release, so
we can confirm the `requires-python = ">=3.9"` floor in `pyproject.toml` is
correct. If your Origin ships 3.8, we'll lower the floor.
