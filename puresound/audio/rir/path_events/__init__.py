"""Coherent path events: schema, geometry, generation, and rendering.

Layer 2 of the RIR package.  A ``PathEvent`` is an inspectable physical
arrival — distance, fractional delay, ordered surfaces, complex gain — that
stays identifiable until the final waveform assembly.
"""

from puresound.audio.rir.path_events.directivity import (
    directivity_pressure_gain,
    orientation_forward_unit,
)
from puresound.audio.rir.path_events.generator import (
    generate_scene_shoebox_path_events,
    generate_shoebox_path_events,
)
from puresound.audio.rir.path_events.geometry import (
    OBJECT_VISIBILITY_POLICY,
    apply_scene_object_visibility,
    segment_intersects_scene_object,
    segment_scene_object_intersection_interval,
)
from puresound.audio.rir.path_events.interactions import (
    augment_scene_path_events_with_interactions,
)
from puresound.audio.rir.path_events.renderer import (
    FRACTIONAL_DELAY_POLICY,
    causal_fractional_delay_kernel,
    partition_path_events_by_arrival,
    render_path_events,
)
from puresound.audio.rir.path_events.schema import (
    PATH_EVENT_SCHEMA_VERSION,
    PATH_EVENT_SET_SCHEMA_VERSION,
    PATH_GAIN_SCHEMA_VERSION,
    ComplexPathGainSpectrum,
    NormalizedAdmittanceModel,
    PathEvent,
    PathEventSet,
    locally_reacting_reflection_coefficient,
    material_absorption_relaxation_models,
)

__all__ = [
    "FRACTIONAL_DELAY_POLICY",
    "OBJECT_VISIBILITY_POLICY",
    "PATH_EVENT_SCHEMA_VERSION",
    "PATH_EVENT_SET_SCHEMA_VERSION",
    "PATH_GAIN_SCHEMA_VERSION",
    "ComplexPathGainSpectrum",
    "NormalizedAdmittanceModel",
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
