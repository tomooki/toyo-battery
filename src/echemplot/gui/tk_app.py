"""Tk application. Requires the [gui] extra.

Launch with ``python -m echemplot.gui.tk_app`` (or
``python -m echemplot.gui`` via the module's ``__main__``).

This module is the view only: it parses widget state, hands a
:class:`echemplot.gui._controller.GuiRequest` to
:func:`echemplot.gui._controller.run`, and displays the resulting
:class:`matplotlib.figure.Figure` objects in one ``Toplevel`` per plot.
All non-UI logic — directory loading, plot dispatch, axis-range
application, Savitzky-Golay window propagation — lives in the
controller so it can be tested without a display.

Matplotlib and tkinter are imported at module scope so an
``ImportError`` raised when the ``[gui]`` extra is missing propagates
immediately (the expected UX for an optional feature).

``tkinterdnd2`` is an optional dependency (also shipped via the ``[gui]``
extra): when present the directory Listbox accepts multi-folder
drag-and-drop from the host file manager; when absent the GUI silently
falls back to the Add-button-only workflow.

The matplotlib / tkinter stubs are permissive and emit ``no-untyped-call``
or ``arg-type`` errors for a handful of standard invocations
(``FigureCanvasTkAgg``, ``NavigationToolbar2Tk``, ``Listbox.curselection``).
We silence those locally with ``# type: ignore`` rather than relaxing
mypy project-wide or adding a pyproject override — this file is the only
place those calls live.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Callable

from matplotlib.backends.backend_tkagg import (  # type: ignore[attr-defined]
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from echemplot.gui._controller import GuiRequest, run

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from echemplot.core.cell import Cell

OnComplete = Callable[[Sequence["Cell"], Sequence["Figure"], int], None]

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
    ``launch_gui`` can construct one instance per ``mainloop`` call and
    all inter-widget state lives on ``self``.
    """

    def __init__(
        self,
        root: tk.Tk,
        *,
        on_complete: OnComplete | None = None,
        origin_mode: bool = False,
    ) -> None:
        self.root = root
        root.title("echemplot GUI")

        # State owned by the instance (not the widgets) so it round-trips
        # through the directory list reliably — Listbox stores only the
        # display strings.
        self._dirs: list[Path] = []

        # Hook fired after a successful Run. ``None`` keeps the standalone
        # behaviour of opening one Toplevel per figure; the Origin
        # launcher injects a callback that pushes results into the active
        # Origin project instead.
        self._on_complete: OnComplete | None = on_complete

        # When ``True``, widgets whose values never reach Origin's push
        # path are disabled and annotated with a note. See
        # :func:`echemplot.origin.launch_gui` for the motivation (issue
        # #60): ``push_to_origin`` always writes all three worksheets and
        # the full per-cell DataFrames, and the matplotlib figures the
        # controller returns are closed before their axis-range overrides
        # can affect any visible plot. ``SG window_length`` stays
        # editable because the ``on_complete`` callback forwards it to
        # :func:`echemplot.origin.push_to_origin`, which recomputes the
        # dQ/dV DataFrame on non-default values so the user's choice
        # actually lands in the worksheet.
        self._origin_mode = origin_mode

        self._build_widgets()

    # ----- layout ---------------------------------------------------

    def _build_widgets(self) -> None:
        # Directories section
        dirs_frame = ttk.LabelFrame(self.root, text="Cell directories")
        dirs_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=_PADX, pady=_PADY)

        self._dirs_list = tk.Listbox(dirs_frame, height=6, width=60)
        self._dirs_list.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=_PADX, pady=_PADY)

        # Enable multi-folder drag-and-drop when tkinterdnd2 is installed AND
        # the tkdnd Tcl extension actually loaded into this interpreter.
        # Importing tkinterdnd2 monkey-patches ``drop_target_register`` onto
        # every Tk widget regardless of whether the root is a ``TkinterDnD.Tk``
        # instance, so ``hasattr`` / ``callable`` checks are not sufficient —
        # calling the method on a plain ``tk.Tk`` root raises ``TclError:
        # invalid command name "tkdnd::drop_target"``. We wrap the
        # registration in ``try/except`` so the GUI still launches if
        # ``_make_root`` fell back to plain ``tk.Tk`` (tkdnd shared library
        # unavailable on the host).
        if _DND_AVAILABLE:
            try:
                self._dirs_list.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
                self._dirs_list.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]
            except tk.TclError:
                pass

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
        # Kept as instance attributes so ``_build_widgets`` can toggle
        # their ``state`` in origin-mode below (and so tests can assert
        # the disabled-state without reaching into grid slaves).
        self._chk_chdis = ttk.Checkbutton(kinds_frame, text="chdis", variable=self._var_chdis)
        self._chk_chdis.grid(row=0, column=0, sticky="w", padx=_PADX, pady=_PADY)
        self._chk_cycle = ttk.Checkbutton(kinds_frame, text="cycle", variable=self._var_cycle)
        self._chk_cycle.grid(row=1, column=0, sticky="w", padx=_PADX, pady=_PADY)
        self._chk_dqdv = ttk.Checkbutton(kinds_frame, text="dQ/dV", variable=self._var_dqdv)
        self._chk_dqdv.grid(row=2, column=0, sticky="w", padx=_PADX, pady=_PADY)

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

        # Origin-mode note + grey-out. The note shifts Run/status down one
        # row; we thread ``run_row`` / ``status_row`` through the two
        # ``.grid()`` calls below so the non-origin layout stays pixel
        # identical to what it was before this switch existed.
        if self._origin_mode:
            note = ttk.Label(
                self.root,
                text=("Origin mode: only SG window_length is applied. Other options are disabled."),
                foreground="#666",
                wraplength=520,
            )
            note.grid(row=2, column=0, columnspan=2, sticky="ew", padx=_PADX, pady=_PADY)
            run_row = 3
            status_row = 4

            for widget in (
                self._chk_chdis,
                self._chk_cycle,
                self._chk_dqdv,
                self._entry_cycles,
                self._entry_vrange,
                self._entry_qrange,
                self._entry_drange,
            ):
                widget.configure(state="disabled")
        else:
            run_row = 2
            status_row = 3

        # Run + status
        ttk.Button(self.root, text="Run", command=self._on_run).grid(
            row=run_row, column=0, columnspan=2, sticky="ew", padx=_PADX, pady=_PADY
        )
        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self._status, relief=tk.SUNKEN, anchor="w").grid(
            row=status_row, column=0, columnspan=2, sticky="ew", padx=_PADX, pady=_PADY
        )

    # ----- directory list handlers ----------------------------------

    def _on_add(self) -> None:
        """Open ``askdirectory`` once and append the chosen directory.

        Cancel (empty return from ``askdirectory``) is a no-op. Users who
        want to add multiple directories can either click Add repeatedly
        or drag-and-drop multiple folders onto the list when
        ``tkinterdnd2`` is installed.
        """
        selected = filedialog.askdirectory(parent=self.root, title="Select a cell directory")
        if not selected:
            return
        self._add_dir(Path(selected))

    def _on_drop(self, event: object) -> None:
        """Handle a ``<<Drop>>`` event from tkinterdnd2.

        ``event.data`` is a Tcl list string (brace-quoted for paths with
        spaces); ``tk.splitlist`` unpacks it into a tuple of tokens.
        Non-directory tokens are silently dropped so stray file drops
        don't pollute the cell-directory list.
        """
        raw = getattr(event, "data", "")
        for token in self.root.tk.splitlist(raw):
            path = Path(token)
            if path.is_dir():
                self._add_dir(path)

    def _add_dir(self, path: Path) -> None:
        """Append ``path`` to the state list and Listbox if not already present.

        Listbox allows duplicate rows but they make Remove-selected
        ambiguous and produce redundant Cell loads at Run time, so we
        de-dupe here.
        """
        if path in self._dirs:
            return
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
            if self._origin_mode:
                # Skip the disabled widgets entirely so a stale value
                # left behind from a toggle can't feed into the request
                # (and so an accidentally non-odd SG placeholder in a
                # disabled entry can't fail validation). ``push_to_origin``
                # always writes all three worksheets and the full
                # DataFrames, so the kinds / cycles / range fields have no
                # effect on the Origin output regardless.
                request = GuiRequest(
                    dirs=tuple(self._dirs),
                    kinds=frozenset({"chdis", "cycle", "dqdv"}),
                    cycles=(),
                    sg_window=_parse_sg_window(self._entry_sg.get()),
                    voltage_range=None,
                    capacity_range=None,
                    dqdv_range=None,
                )
            else:
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
            self._fail_and_maybe_close(f"Invalid input: {exc}")
            return

        # Controller-level validation (empty dirs / no kinds) raises too.
        try:
            result = run(request)
        except ValueError as exc:
            self._fail_and_maybe_close(f"Invalid request: {exc}")
            return
        except Exception as exc:
            self._fail_and_maybe_close(f"Error running pipeline: {exc}")
            return

        if self._on_complete is not None:
            try:
                # ``sg_window`` is surfaced so the Origin launcher can
                # propagate a non-default Savitzky-Golay window into the
                # worksheet data — ``Cell.dqdv_df`` is cached at defaults
                # and ignores per-run overrides, so without this argument
                # the Origin-mode ``SG window_length`` field would have no
                # effect on the pushed output. See issue #60.
                self._on_complete(result.cells, result.figures, request.sg_window)
            except Exception as exc:
                # Drop the traceback chain BEFORE the modal blocks: in
                # origin_mode the chain transitively pins partial Origin
                # COM proxies created by ``push_to_origin`` before it
                # raised, and Origin's "is a Python script running?"
                # probe can fire while ``messagebox.showerror`` is up.
                # Implicit ``del exc`` at end-of-except runs too late
                # (after ``root.destroy``). See issue history for the
                # "stop external script execution" dialog bug.
                msg = f"Error in completion hook: {exc}"
                exc = None  # type: ignore[assignment]
                self._fail_and_maybe_close(msg)
                return
            self._status.set(f"Ran {len(result.figures)} figure(s); pushed via hook.")
            if self._origin_mode:
                # Origin-mode is a one-shot batch: ``launch_gui`` runs Tk's
                # ``mainloop()`` inside the Origin Python Console, so the
                # Console stays blocked until this root is destroyed. After
                # a successful push there is nothing left to do, so we close
                # the window automatically instead of forcing the user to
                # click the OS close button to regain the Console.
                self.root.destroy()
        else:
            for fig in result.figures:
                self._show_figure(fig)
            self._status.set(f"Ran {len(result.figures)} figure(s).")

    def _show_figure(self, fig: Figure) -> None:
        """Embed ``fig`` in a new ``Toplevel`` with a matplotlib toolbar.

        Closing the Toplevel also closes the matplotlib figure via
        ``plt.close`` so a long-running session doesn't accumulate figures
        in pyplot's global registry (which would eventually trip the
        max_open_warning).
        """
        import matplotlib.pyplot as plt  # local: pyplot only at show-time

        top = tk.Toplevel(self.root)
        top.title("echemplot plot")
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
        messagebox.showerror("echemplot", message, parent=self.root)
        self._status.set(message)

    def _fail_and_maybe_close(self, message: str) -> None:
        """Surface error, then auto-close the root in ``origin_mode``.

        In origin_mode the Tk mainloop blocks the Origin Python Console
        (``launch_gui()`` is a one-shot batch). If we leave the window
        open after a failed Run, Origin's "is a Python script running?"
        check trips when the user later clicks the Origin close button
        and shows a confusing "stop external script execution" warning.
        ``messagebox.showerror`` is modal, so by the time we get here the
        user has already dismissed the dialog; auto-closing the window
        afterwards lets them re-invoke ``launch_gui()`` from the Console.
        ``gc.collect()`` releases any Origin COM proxies that participate
        in reference cycles before mainloop exits.
        """
        self._fail(message)
        if self._origin_mode:
            import gc

            gc.collect()
            self.root.destroy()


def launch_gui(
    *,
    on_complete: OnComplete | None = None,
    origin_mode: bool = False,
) -> None:
    """Start the Tk GUI and run its event loop.

    Callable two ways:

    * ``python -m echemplot.gui`` — standalone CLI (via
      :mod:`echemplot.gui.__main__`).
    * ``from echemplot.gui import launch_gui; launch_gui()`` — direct
      call from any host Python process, including Origin's embedded
      Python Console. The call blocks until the window is closed.

    Parameters
    ----------
    on_complete
        Optional callback invoked after each successful Run with
        ``(cells, figures)``. ``None`` (the default) preserves the
        standalone behaviour of opening one ``Toplevel`` per figure;
        :func:`echemplot.origin.launch_gui` injects a callback that
        forwards to :func:`echemplot.origin.push_to_origin` instead,
        so figures are not shown and the results land directly in the
        active Origin project.
    origin_mode
        When ``True``, the GUI greys out the options that have no effect
        on the Origin push path (the plot-kind checkboxes, the cycles
        entry, and the voltage/capacity/dQ-dV range entries) and shows
        an inline note explaining why. Only ``SG window_length`` stays
        editable because it flows into the dQ/dV worksheet the Origin
        push writes. After a successful Run the window is destroyed
        automatically so the Origin Python Console (which is blocked on
        this ``mainloop()``) is released without a manual window close.
        :func:`echemplot.origin.launch_gui` sets this; standalone callers
        should leave it at the ``False`` default.

    Switches matplotlib to the TkAgg backend at call time (not import
    time): a module-level ``matplotlib.use("TkAgg")`` would crash
    headless test collection on Linux CI runners that have no Tk
    toolkit, even though the test merely imports a sibling controller
    module that doesn't need matplotlib at all.
    """
    import matplotlib

    matplotlib.use("TkAgg")
    root = _make_root()
    _App(root, on_complete=on_complete, origin_mode=origin_mode)
    root.mainloop()


def _make_root() -> tk.Tk:
    """Return a Tk root, preferring ``TkinterDnD.Tk`` when the extra is installed.

    ``TkinterDnD.Tk`` is a drop-in subclass that loads the tkdnd Tcl
    extension into the interpreter so widgets can register as drop
    targets. We fall back to plain ``tk.Tk`` silently when the package
    isn't importable so the GUI still launches without the optional
    drag-and-drop feature.
    """
    if _DND_AVAILABLE:
        try:
            return TkinterDnD.Tk()  # type: ignore[no-any-return]
        except tk.TclError:
            # The Python package imported but the underlying tkdnd Tcl
            # extension failed to load (missing shared library on the
            # host). Fall through to plain Tk rather than crash.
            pass
    return tk.Tk()


if __name__ == "__main__":
    launch_gui()
