"""Origin adapter submodule.

`originpro` is shipped with OriginLab's embedded Python; it is NOT listed as a
PyPI dependency. Importing this submodule outside Origin raises at call time.
"""

from __future__ import annotations


def _require_originpro() -> object:
    try:
        import originpro as op
    except ImportError as e:
        raise ImportError(
            "toyo_battery.origin requires `originpro`, which is provided by "
            "OriginLab's embedded Python. Run this module from Origin's Python."
        ) from e
    return op


def push_to_origin(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("push_to_origin will be implemented in P5")
