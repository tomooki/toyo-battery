# Installing toyo-battery into Origin's embedded Python

`toyo-battery` is a **pure-Python** package published on PyPI. It installs into
OriginLab's embedded Python the same way as numpy/pandas/scipy.

## Steps

1. Open Origin.
2. **Connectivity → Python Packages** (or Python Console → `pip`).
3. Install:

   ```python
   import subprocess
   import sys

   subprocess.check_call(
       [sys.executable, "-m", "pip", "install", "--upgrade", "toyo-battery"]
   )
   ```

4. Confirm:

   ```python
   import toyo_battery
   print(toyo_battery.__version__)
   ```

5. Optional — check the Origin adapter is wired up:

   ```python
   from toyo_battery.origin import push_to_origin
   ```

   (Inside Origin this succeeds. Outside Origin, importing `toyo_battery.origin`
   still works — calling `push_to_origin` will raise `ImportError` with a clear
   message.)

## Launching the GUI from Origin

With the `[gui]` extra installed (`pip install "toyo-battery[gui]"`), the Tk
GUI can be launched directly from Origin's Python Console:

```python
from toyo_battery.gui import launch_gui
launch_gui()
```

Notes:

- `launch_gui()` runs Tk's `mainloop()` **in the Origin process**. The Origin
  Python Console is blocked until you close the GUI window.
- Origin's embedded CPython must include `_tkinter` (OriginLab 2022 and later
  do). If it doesn't, the call raises `ImportError` from `import tkinter`; use
  a separate system Python in that case.

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
