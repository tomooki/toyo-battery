"""Command-line interface. Requires the [cli] extra (typer)."""

from __future__ import annotations


def app() -> None:
    try:
        import typer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The CLI requires the [cli] extra: pip install 'toyo-battery[cli]'"
        ) from e
    raise NotImplementedError("CLI will be implemented in P2")
