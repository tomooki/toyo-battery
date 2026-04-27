"""Core processing: Cell class and per-step transformations."""

from __future__ import annotations


class DataIntegrityWarning(UserWarning):
    """Emitted when core processing silently drops or alters input rows.

    Currently raised by :func:`echemplot.core.chdis.get_chdis_df` when the
    running-max filter discards rows whose absolute capacity falls below
    the segment's running maximum. For real TOYO data this is expected
    (CC→CV sub-step boundaries) and the warning is informational. For
    hand-preprocessed inputs it surfaces previously-silent data loss so
    callers can audit their pipeline.
    """


__all__ = ["DataIntegrityWarning"]
