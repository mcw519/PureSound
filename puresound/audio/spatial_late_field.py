"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.render.spatial_late_field` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.render.spatial_late_field import (
    SPATIAL_EARLY_LATE_COUPLING_POLICY,
    SPATIAL_LATE_FIELD_POLICY,
    SpatialEarlyLateRender,
    SpatialLateFieldRender,
    couple_receiver_array_early_late,
    fibonacci_sphere_directions,
    render_path_events_ambisonic,
    render_spatial_fdn_late_field,
)

__all__ = [
    "SPATIAL_EARLY_LATE_COUPLING_POLICY",
    "SPATIAL_LATE_FIELD_POLICY",
    "SpatialEarlyLateRender",
    "SpatialLateFieldRender",
    "couple_receiver_array_early_late",
    "fibonacci_sphere_directions",
    "render_path_events_ambisonic",
    "render_spatial_fdn_late_field",
]
