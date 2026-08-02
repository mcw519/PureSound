"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.bank.loader` during R6 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.bank.loader import (
    PreGeneratedReleaseBank,
    PreGeneratedRoomBank,
)

__all__ = [
    "PreGeneratedReleaseBank",
    "PreGeneratedRoomBank",
]
