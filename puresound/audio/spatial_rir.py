"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.render.spatial` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.render.spatial import (
    SPATIAL_ROOM_RIR_POLICY,
    SpatialRoomRIRRender,
    render_room_scene_spatial_rir,
)

__all__ = [
    "SPATIAL_ROOM_RIR_POLICY",
    "SpatialRoomRIRRender",
    "render_room_scene_spatial_rir",
]
