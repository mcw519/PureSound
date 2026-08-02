"""Compatibility shim for the coherent path-event pipeline.

The implementation moved to :mod:`puresound.audio.rir.path_events` during R3
of ``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged;
prefer the new path in new code.
"""

from puresound.audio.rir.path_events import (
    FRACTIONAL_DELAY_POLICY,
    OBJECT_VISIBILITY_POLICY,
    PATH_EVENT_SCHEMA_VERSION,
    PATH_EVENT_SET_SCHEMA_VERSION,
    PATH_GAIN_SCHEMA_VERSION,
    ComplexPathGainSpectrum,
    NormalizedAdmittanceModel,
    PathEvent,
    PathEventSet,
    apply_scene_object_visibility,
    augment_scene_path_events_with_interactions,
    causal_fractional_delay_kernel,
    directivity_pressure_gain,
    generate_scene_shoebox_path_events,
    generate_shoebox_path_events,
    locally_reacting_reflection_coefficient,
    material_absorption_relaxation_models,
    orientation_forward_unit,
    partition_path_events_by_arrival,
    render_path_events,
    segment_intersects_scene_object,
    segment_scene_object_intersection_interval,
)

__all__ = [
    "ComplexPathGainSpectrum",
    "FRACTIONAL_DELAY_POLICY",
    "OBJECT_VISIBILITY_POLICY",
    "NormalizedAdmittanceModel",
    "PATH_EVENT_SCHEMA_VERSION",
    "PATH_EVENT_SET_SCHEMA_VERSION",
    "PATH_GAIN_SCHEMA_VERSION",
    "PathEvent",
    "PathEventSet",
    "apply_scene_object_visibility",
    "augment_scene_path_events_with_interactions",
    "causal_fractional_delay_kernel",
    "directivity_pressure_gain",
    "generate_scene_shoebox_path_events",
    "generate_shoebox_path_events",
    "locally_reacting_reflection_coefficient",
    "material_absorption_relaxation_models",
    "orientation_forward_unit",
    "partition_path_events_by_arrival",
    "render_path_events",
    "segment_intersects_scene_object",
    "segment_scene_object_intersection_interval",
]
