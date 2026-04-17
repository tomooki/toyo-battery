"""SMB remote directory access. Requires the [smb] extra (pysmb)."""

from __future__ import annotations


def connect(*args: object, **kwargs: object) -> object:
    raise NotImplementedError("SMB support will be implemented in P3")
