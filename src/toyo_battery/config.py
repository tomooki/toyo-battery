"""Configuration loader. Full pydantic validation lands with P3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file (network_setting.json successor)."""
    with Path(path).open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data
