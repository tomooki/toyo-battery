"""Command-line interface. Requires the ``[cli]`` extra (typer + rich).

Three subcommands, all operating over one or more cell directories:

- ``toyo-battery process <dirs>... [--out DIR]`` — load each cell via
  :meth:`toyo_battery.core.cell.Cell.from_dir` and write the three derived
  CSVs (``chdis`` / ``cap`` / ``dqdv``) per cell.
- ``toyo-battery plot <dirs>... --out DIR`` — render matplotlib PNGs
  (``chdis`` / ``cycle`` / ``dqdv``) across all cells. Matplotlib is
  lazy-imported inside the command so ``process`` / ``stats`` stay fast.
- ``toyo-battery stats <dirs>... --cycles N,M --out PATH`` — write the
  :func:`toyo_battery.core.stats.stat_table` summary CSV.

The ``app`` object is a Typer instance, which is directly callable — so the
``[project.scripts]`` entry ``toyo-battery = toyo_battery.cli:app`` works
without wrapping.

Note: this module intentionally omits ``from __future__ import annotations``.
On Python 3.9, stringified annotations (``future-annotations`` mode) do not
round-trip cleanly through Typer 0.12's ``get_type_hints(include_extras=True)``
call for ``Annotated[...]`` parameter metadata — the ``typer.Option`` /
``typer.Argument`` marker gets dropped, causing options to be silently
reinterpreted as positional arguments. Evaluating annotations eagerly at
function-definition time sidesteps the round-trip entirely.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from toyo_battery.core.cell import Cell
from toyo_battery.core.stats import stat_table
from toyo_battery.io.schema import ColumnLang

app = typer.Typer(
    add_completion=False,
    help="TOYO battery cycler toolkit — batch operations.",
)

# Errors / progress go to stderr so stdout stays clean for any future
# machine-readable emissions (JSON, etc.).
_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared option annotations
# ---------------------------------------------------------------------------
# Typer 0.12+ idiom: use ``Annotated[T, typer.Option(...)]`` so the type
# annotation stays the canonical source of truth and option metadata lives
# alongside it.

# ``Optional[float]`` (not ``float | None``) is mandatory here: Typer resolves
# these annotations at runtime via ``get_type_hints``, and on Python 3.9 the
# union-pipe syntax is not a valid type expression even under
# ``from __future__ import annotations`` when the annotation is stored on a
# module-level variable (and thus eagerly constructed by ``Annotated``).
MassOption = Annotated[
    Optional[float],
    typer.Option(
        "--mass",
        help="Active-material mass override [g]. Default: read from cell dir.",
    ),
]
EncodingOption = Annotated[
    str,
    typer.Option(
        "--encoding",
        help="Text encoding of the TOYO source CSVs.",
    ),
]
ColumnLangOption = Annotated[
    str,
    typer.Option(
        "--column-lang",
        help="Column-name language for derived tables: 'ja' or 'en'.",
    ),
]
DirsArgument = Annotated[
    list[Path],
    typer.Argument(
        ...,
        help="One or more TOYO cell directories.",
        exists=False,  # we validate & re-raise as BadParameter ourselves
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_column_lang(value: str) -> ColumnLang:
    """Coerce the ``--column-lang`` string to a typed ``ColumnLang`` literal.

    ``typer.Option`` with a raw ``str`` annotation accepts any input; this
    helper converts the two valid values and raises ``BadParameter`` for
    anything else. Keeps the public signature a plain ``str`` (which plays
    cleanly with ``Annotated`` under Python 3.9) while still surfacing a
    typed literal to downstream calls.
    """
    if value == "ja":
        return "ja"
    if value == "en":
        return "en"
    raise typer.BadParameter(
        f"must be 'ja' or 'en'; got {value!r}",
        param_hint="--column-lang",
    )


def _parse_cycles(raw: str) -> list[int]:
    """Parse a comma-separated cycle list (e.g. ``"1,10,50"``) → ``[1, 10, 50]``.

    Whitespace around entries is tolerated. Empty entries and non-numeric
    tokens raise ``typer.BadParameter`` with the offending token in the
    message so callers can find their typo quickly.
    """
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise typer.BadParameter(
            f"expected a non-empty comma-separated list of integers; got {raw!r}",
            param_hint="--cycles",
        )
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as exc:
            raise typer.BadParameter(
                f"invalid cycle value {p!r} (expected integer)",
                param_hint="--cycles",
            ) from exc
    return out


def _parse_kinds(raw: str) -> list[str]:
    """Parse ``--kinds`` into a list, validating against the allowed set.

    Duplicates are preserved in caller order but a duplicated kind would
    overwrite its own PNG, so callers almost certainly mean a typo — we
    reject duplicates with ``BadParameter`` to surface the mistake.
    """
    allowed = {"chdis", "cycle", "dqdv"}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter(
            f"expected a non-empty comma-separated list of kinds; got {raw!r}",
            param_hint="--kinds",
        )
    seen: set[str] = set()
    for p in parts:
        if p not in allowed:
            raise typer.BadParameter(
                f"unknown kind {p!r}; choose from {sorted(allowed)}",
                param_hint="--kinds",
            )
        if p in seen:
            raise typer.BadParameter(
                f"duplicate kind {p!r}",
                param_hint="--kinds",
            )
        seen.add(p)
    return parts


def _load_cell(
    path: Path,
    *,
    mass: Optional[float],
    encoding: str,
    column_lang: ColumnLang,
) -> Cell:
    """Wrap :meth:`Cell.from_dir` so a missing directory surfaces as a
    ``BadParameter`` (with the directory name inline) rather than a bare
    ``FileNotFoundError`` traceback.
    """
    if not path.exists():
        raise typer.BadParameter(
            f"directory does not exist: {path}",
            param_hint="DIRS",
        )
    if not path.is_dir():
        raise typer.BadParameter(
            f"not a directory: {path}",
            param_hint="DIRS",
        )
    try:
        return Cell.from_dir(path, mass=mass, encoding=encoding, column_lang=column_lang)
    except FileNotFoundError as exc:
        # Reader raises FileNotFoundError for missing required files *inside*
        # an otherwise-extant dir — rewrite to BadParameter so the error
        # origin (which directory, which path) is immediately visible.
        raise typer.BadParameter(
            f"could not load cell from {path}: {exc}",
            param_hint="DIRS",
        ) from exc


def _make_progress() -> Progress:
    """Shared progress-bar configuration for all subcommands."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("process")
def process(
    dirs: DirsArgument,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory for the derived CSVs. Defaults to cwd.",
        ),
    ] = Path("."),
    mass: MassOption = None,
    encoding: EncodingOption = "shift_jis",
    column_lang: ColumnLangOption = "ja",
) -> None:
    """Load each cell dir and write ``{name}_chdis.csv`` / ``_cap.csv`` / ``_dqdv.csv``."""
    col_lang = _validate_column_lang(column_lang)
    out.mkdir(parents=True, exist_ok=True)

    with _make_progress() as progress:
        task = progress.add_task("Processing cells", total=len(dirs))
        for d in dirs:
            progress.update(task, description=f"Processing {d.name}")
            cell = _load_cell(d, mass=mass, encoding=encoding, column_lang=col_lang)
            cell.chdis_df.to_csv(out / f"{cell.name}_chdis.csv")
            cell.cap_df.to_csv(out / f"{cell.name}_cap.csv")
            cell.dqdv_df.to_csv(out / f"{cell.name}_dqdv.csv")
            progress.advance(task)


@app.command("plot")
def plot(
    dirs: DirsArgument,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory for the PNGs.",
        ),
    ],
    kinds: Annotated[
        str,
        typer.Option(
            "--kinds",
            help="Comma-separated plot kinds: any subset of 'chdis,cycle,dqdv'.",
        ),
    ] = "chdis,cycle,dqdv",
    cycles: Annotated[
        str,
        typer.Option(
            "--cycles",
            help="Comma-separated cycle indices (ignored for 'cycle' kind).",
        ),
    ] = "1,10,50",
    mass: MassOption = None,
    encoding: EncodingOption = "shift_jis",
    column_lang: ColumnLangOption = "ja",
) -> None:
    """Render PNGs across all cells; one figure per kind, one Axes per cell."""
    col_lang = _validate_column_lang(column_lang)
    kinds_list = _parse_kinds(kinds)
    cycles_list = _parse_cycles(cycles)
    out.mkdir(parents=True, exist_ok=True)

    # Lazy-import the backend so ``process`` / ``stats`` don't pay the
    # matplotlib import cost (and don't fail when the ``[plot]`` extra is
    # absent but only ``process`` is being invoked).
    from toyo_battery.plotting.matplotlib_backend import (
        plot_chdis,
        plot_cycle,
        plot_dqdv,
    )

    cells: list[Cell] = []
    with _make_progress() as progress:
        task = progress.add_task("Loading cells", total=len(dirs))
        for d in dirs:
            progress.update(task, description=f"Loading {d.name}")
            cells.append(_load_cell(d, mass=mass, encoding=encoding, column_lang=col_lang))
            progress.advance(task)

    with _make_progress() as progress:
        task = progress.add_task("Rendering plots", total=len(kinds_list))
        for kind in kinds_list:
            progress.update(task, description=f"Rendering {kind}")
            if kind == "chdis":
                fig = plot_chdis(cells, cycles=cycles_list)
            elif kind == "cycle":
                fig = plot_cycle(cells)
            else:  # "dqdv" — guarded by _parse_kinds
                fig = plot_dqdv(cells, cycles=cycles_list)
            fig.savefig(out / f"{kind}.png", dpi=150, bbox_inches="tight")
            # Explicit close frees the figure immediately; matplotlib would
            # otherwise hold on to every figure until process exit and trip
            # the max_open_warning on multi-kind invocations.
            import matplotlib.pyplot as plt

            plt.close(fig)
            progress.advance(task)


@app.command("stats")
def stats(
    dirs: DirsArgument,
    cycles: Annotated[
        str,
        typer.Option(
            "--cycles",
            help="Comma-separated target cycles for the stat table (e.g. '10,50').",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output CSV path for the stat table.",
        ),
    ],
    mass: MassOption = None,
    encoding: EncodingOption = "shift_jis",
    column_lang: ColumnLangOption = "ja",
) -> None:
    """Write a single ``stat_table`` CSV spanning all the supplied cells."""
    col_lang = _validate_column_lang(column_lang)
    cycles_list = _parse_cycles(cycles)

    cells: list[Cell] = []
    with _make_progress() as progress:
        task = progress.add_task("Loading cells", total=len(dirs))
        for d in dirs:
            progress.update(task, description=f"Loading {d.name}")
            cells.append(_load_cell(d, mass=mass, encoding=encoding, column_lang=col_lang))
            progress.advance(task)

    tbl = stat_table(cells, target_cycles=cycles_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out)
