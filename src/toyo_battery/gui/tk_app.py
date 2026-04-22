"""Tk application. Requires the [gui] extra.

Launch with ``python -m toyo_battery.gui.tk_app`` (or
``python -m toyo_battery.gui`` via the module's ``__main__``).

This module is the view only: it parses widget state, hands a
:class:`toyo_battery.gui._controller.GuiRequest` to
:func:`toyo_battery.gui._controller.run`, and displays the resulting
:class:`matplotlib.figure.Figure` objects in one ``Toplevel`` per plot.
All non-UI logic — directory loading, plot dispatch, axis-range
application, Savitzky-Golay window propagation — lives in the
controller so it can be tested without a display.

Matplotlib and tkinter are imported at module scope so an
``ImportError`` raised when the ``[gui]`` extra is missing propagates
immediately (the expected UX for an optional feature).

The matplotlib / tkinter stubs are permissive and emit ``no-untyped-call``
or ``arg-type`` errors for a handful of standard invocations
(``FigureCanvasTkAgg``, ``NavigationToolbar2Tk``, ``Listbox.curselection``).
We silence those locally with ``# type: ignore`` rather than relaxing
mypy project-wide or adding a pyproject override — this file is the only
place those calls live.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

from matplotlib.backends.backend_tkagg import (  # type: ignore[attr-defined]
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from toyo_battery.gui._controller import GuiRequest, run

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_DEFAULT_CYCLES_TEXT = "1 10 50"
_DEFAULT_SG_WINDOW_TEXT = "11"
_PADX = 6
_PADY = 4


def _parse_cycles(text: str) -> list[int]:
    """Parse a space-separated cycle list.

    Empty/whitespace-only input → empty list (plot every available cycle
    per the controller's convention). Non-integer tokens raise
    ``ValueError``.
    """
    tokens = text.split()
    if not tokens:
        return []
    out: list[int] = []
    for tok in tokens:
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise ValueError(f"cycles entry must be space-separated integers; got {tok!r}") from exc
    return out


def _parse_range(text: str, field_name: str) -> tuple[float, float] | None:
    """Parse an optional ``"lo hi"`` axis range.

    Empty/whitespace-only text → ``None`` (use matplotlib's auto-scale).
    Anything else must be exactly two whitespace-separated floats with
    ``lo < hi``; violations raise ``ValueError``.
    """
    tokens = text.split()
    if not tokens:
        return None
    if len(tokens) != 2:
        raise ValueError(f"{field_name} must be two space-separated numbers 'lo hi' or blank")
    try:
        lo = float(tokens[0])
        hi = float(tokens[1])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric; got {text!r}") from exc
    if not lo < hi:
        raise ValueError(f"{field_name} must satisfy lo < hi; got {lo} {hi}")
    return (lo, hi)


def _parse_sg_window(text: str) -> int:
    """Parse the SG window_length entry.

    Must be a positive odd integer. Even or non-positive values raise
    ``ValueError`` with a message the caller surfaces via ``messagebox``.
    """
    try:
        n = int(text.strip())
    except ValueError as exc:
        raise ValueError(f"SG window_length must be a positive odd integer; got {text!r}") from exc
    if n < 1:
        raise ValueError(f"SG window_length must be >= 1, got {n}")
    if n % 2 == 0:
        raise ValueError(f"SG window_length must be odd, got {n}")
    return n


class _App:
    """The single Tk window that owns all widgets and event handlers.

    Kept as a class (rather than a pile of module-level closures) so
    ``main`` can construct one instance per ``mainloop`` call and all
    inter-widget state lives on ``self``.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("toyo-battery GUI")

        # State owned by the instance (not the widgets) so it round-trips
        # through the directory list reliably — Listbox stores only the
        # display strings.
        self._dirs: list[Path] = []

        self._build_widgets()

    # ----- layout ---------------------------------------------------

    def _build_widgets(self) -> None:
        # Directories section
        dirs_frame = ttk.LabelFrame(self.root, text="Cell directories")
        dirs_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=_PADX, pady=_PADY)

        self._dirs_list = tk.Listbox(dirs_frame, height=6, width=60)
        self._dirs_list.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=_PADX, pady=_PADY)
        ttk.Button(dirs_frame, text="Add...", command=self._on_add).grid(
            row=0, column=1, sticky="ew", padx=_PADX, pady=_PADY
        )
        ttk.Button(dirs_frame, text="Remove selected", command=self._on_remove).grid(
            row=1, column=1, sticky="ew", padx=_PADX, pady=_PADY
        )
        ttk.Button(dirs_frame, text="Clear", command=self._on_clear).grid(
            row=2, column=1, sticky="ew", padx=_PADX, pady=_PADY
        )

        # Plot kinds
        kinds_frame = ttk.LabelFrame(self.root, text="Plot kinds")
        kinds_frame.grid(row=1, column=0, sticky="nsew", padx=_PADX, pady=_PADY)
        self._var_chdis = tk.BooleanVar(value=True)
        self._var_cycle = tk.BooleanVar(value=True)
        self._var_dqdv = tk.BooleanVar(value=False)
        ttk.Checkbutton(kinds_frame, text="chdis", variable=self._var_chdis).grid(
            row=0, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        ttk.Checkbutton(kinds_frame, text="cycle", variable=self._var_cycle).grid(
            row=1, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        ttk.Checkbutton(kinds_frame, text="dQ/dV", variable=self._var_dqdv).grid(
            row=2, column=0, sticky="w", padx=_PADX, pady=_PADY
        )

        # Parameters
        params_frame = ttk.LabelFrame(self.root, text="Parameters")
        params_frame.grid(row=1, column=1, sticky="nsew", padx=_PADX, pady=_PADY)

        ttk.Label(params_frame, text="Cycles (space-separated):").grid(
            row=0, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        self._entry_cycles = ttk.Entry(params_frame, width=24)
        self._entry_cycles.insert(0, _DEFAULT_CYCLES_TEXT)
        self._entry_cycles.grid(row=0, column=1, sticky="ew", padx=_PADX, pady=_PADY)

        ttk.Label(params_frame, text="SG window_length:").grid(
            row=1, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        self._entry_sg = ttk.Entry(params_frame, width=24)
        self._entry_sg.insert(0, _DEFAULT_SG_WINDOW_TEXT)
        self._entry_sg.grid(row=1, column=1, sticky="ew", padx=_PADX, pady=_PADY)

        ttk.Label(params_frame, text="Voltage range (lo hi):").grid(
            row=2, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        self._entry_vrange = ttk.Entry(params_frame, width=24)
        self._entry_vrange.grid(row=2, column=1, sticky="ew", padx=_PADX, pady=_PADY)

        ttk.Label(params_frame, text="Capacity range (lo hi):").grid(
            row=3, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        self._entry_qrange = ttk.Entry(params_frame, width=24)
        self._entry_qrange.grid(row=3, column=1, sticky="ew", padx=_PADX, pady=_PADY)

        ttk.Label(params_frame, text="dQ/dV range (lo hi):").grid(
            row=4, column=0, sticky="w", padx=_PADX, pady=_PADY
        )
        self._entry_drange = ttk.Entry(params_frame, width=24)
        self._entry_drange.grid(row=4, column=1, sticky="ew", padx=_PADX, pady=_PADY)

        # Run + status
        ttk.Button(self.root, text="Run", command=self._on_run).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=_PADX, pady=_PADY
        )
        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self._status, relief=tk.SUNKEN, anchor="w").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=_PADX, pady=_PADY
        )

    # ----- directory list handlers ----------------------------------

    def _on_add(self) -> None:
        """Open ``askdirectory`` in a loop so the user can add multiple dirs.

        An empty return from ``askdirectory`` (the user cancelled) breaks
        the loop; otherwise each selection is appended to both the state
        list and the display listbox.
        """
        while True:
            selected = filedialog.askdirectory(
                parent=self.root, title="Select a cell directory (cancel to stop)"
            )
            if not selected:
                break
            path = Path(selected)
            self._dirs.append(path)
            self._dirs_list.insert(tk.END, str(path))

    def _on_remove(self) -> None:
        # Walk selected indices in reverse so the remaining indices stay
        # valid as we pop from the lists.
        selection = self._dirs_list.curselection()  # type: ignore[no-untyped-call]
        for idx in reversed(selection):
            self._dirs_list.delete(idx)
            del self._dirs[int(idx)]

    def _on_clear(self) -> None:
        self._dirs_list.delete(0, tk.END)
        self._dirs.clear()

    # ----- run --------------------------------------------------------

    def _collect_kinds(self) -> frozenset[str]:
        kinds: set[str] = set()
        if self._var_chdis.get():
            kinds.add("chdis")
        if self._var_cycle.get():
            kinds.add("cycle")
        if self._var_dqdv.get():
            kinds.add("dqdv")
        return frozenset(kinds)

    def _on_run(self) -> None:
        try:
            request = GuiRequest(
                dirs=tuple(self._dirs),
                kinds=self._collect_kinds(),
                cycles=tuple(_parse_cycles(self._entry_cycles.get())),
                sg_window=_parse_sg_window(self._entry_sg.get()),
                voltage_range=_parse_range(self._entry_vrange.get(), "Voltage range"),
                capacity_range=_parse_range(self._entry_qrange.get(), "Capacity range"),
                dqdv_range=_parse_range(self._entry_drange.get(), "dQ/dV range"),
            )
        except ValueError as exc:
            self._fail(f"Invalid input: {exc}")
            return

        # Controller-level validation (empty dirs / no kinds) raises too.
        try:
            figures = run(request)
        except ValueError as exc:
            self._fail(f"Invalid request: {exc}")
            return
        except Exception as exc:
            self._fail(f"Error running pipeline: {exc}")
            return

        for fig in figures:
            self._show_figure(fig)
        self._status.set(f"Ran {len(figures)} figure(s).")

    def _show_figure(self, fig: Figure) -> None:
        """Embed ``fig`` in a new ``Toplevel`` with a matplotlib toolbar.

        Closing the Toplevel also closes the matplotlib figure via
        ``plt.close`` so a long-running session doesn't accumulate figures
        in pyplot's global registry (which would eventually trip the
        max_open_warning).
        """
        import matplotlib.pyplot as plt  # local: pyplot only at show-time

        top = tk.Toplevel(self.root)
        top.title("toyo-battery plot")
        canvas = FigureCanvasTkAgg(fig, master=top)  # type: ignore[no-untyped-call]
        canvas.draw()  # type: ignore[no-untyped-call]
        toolbar = NavigationToolbar2Tk(canvas, top)  # type: ignore[no-untyped-call]
        toolbar.update()
        canvas.get_tk_widget().pack(  # type: ignore[no-untyped-call]
            side=tk.TOP, fill=tk.BOTH, expand=True
        )

        # Default args bind ``fig`` and ``top`` by value — avoids late-binding
        # bugs if ``_show_figure`` is called multiple times in quick succession.
        def _on_close(fig: Figure = fig, top: tk.Toplevel = top) -> None:
            plt.close(fig)
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", _on_close)

    def _fail(self, message: str) -> None:
        """Surface ``message`` via a modal error and the status bar."""
        messagebox.showerror("toyo-battery", message, parent=self.root)
        self._status.set(message)


def main() -> None:
    """Entry point for ``python -m toyo_battery.gui.tk_app``.

    Switches matplotlib to the TkAgg backend at call time (not import
    time): a module-level ``matplotlib.use("TkAgg")`` would crash
    headless test collection on Linux CI runners that have no Tk
    toolkit, even though the test merely imports a sibling controller
    module that doesn't need matplotlib at all.
    """
    import matplotlib

    matplotlib.use("TkAgg")
    root = tk.Tk()
    _App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
