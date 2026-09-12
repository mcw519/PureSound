"""High-level opt-in M4 spatial RIR synthesis for material-first scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from puresound.audio.rir.render.binaural import (
    AmbisonicBinauralDecoder,
    BinauralBRIRRender,
    render_ambisonic_brir,
)
from puresound.audio.rir.render.multiband_fdn import design_multiband_fdn
from puresound.audio.rir.physics.propagation import (
    air_adjusted_rt60_s,
    apply_air_absorption,
)
from puresound.audio.rir.path_events import (
    PathEventSet,
    generate_scene_shoebox_path_events,
    material_absorption_relaxation_models,
    render_path_events,
)
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.render.spatial_late_field import (
    SpatialEarlyLateRender,
    SpatialLateFieldRender,
    couple_receiver_array_early_late,
    render_path_events_ambisonic,
    render_spatial_fdn_late_field,
)


SPATIAL_ROOM_RIR_POLICY = "puresound.spatial_room_rir.v1"


@dataclass(frozen=True)
class SpatialRoomRIRRender:
    """Complete receiver-array/FOA RIRs and optional binaural BRIR."""

    receiver_rirs: np.ndarray
    ambisonic_acn_sn3d: np.ndarray
    coherent_receiver_rirs: np.ndarray
    coherent_ambisonic_acn_sn3d: np.ndarray
    spatial_late_field: SpatialLateFieldRender
    receiver_coupling: SpatialEarlyLateRender
    ambisonic_coupling: SpatialEarlyLateRender
    binaural: BinauralBRIRRender | None
    metadata: Mapping[str, Any]


def _material_boundary_models(
    scene: RoomSceneV2,
    reference_frequency_hz: float,
) -> dict[str, Any]:
    models, _metadata = material_absorption_relaxation_models(
        scene,
        reference_frequency_hz=float(reference_frequency_hz),
    )
    return models


def _path_events_for_receiver(
    scene: RoomSceneV2,
    source_index: int,
    receiver_index: int,
    *,
    max_order: int,
    boundary_models: Mapping[str, Any],
) -> PathEventSet:
    return generate_scene_shoebox_path_events(
        scene,
        source_index=source_index,
        receiver_index=receiver_index,
        max_order=max_order,
        edge_corner_policy="exclude",
        resolve_object_visibility=True,
        include_scene_interactions=True,
        boundary_admittance_models=boundary_models,
        reflection_frequencies_hz=(
            60.0,
            125.0,
            250.0,
            500.0,
            1000.0,
            2000.0,
            4000.0,
            8000.0,
        ),
    )


def _direct_sample(event_set: PathEventSet, sample_rate: int) -> int:
    direct = next(
        (event for event in event_set.events if event.path_type == "direct"),
        None,
    )
    if direct is None:
        raise ValueError("PathEvent set has no direct event")
    return int(round(direct.delay_s * sample_rate))


def render_room_scene_spatial_rir(
    scene: RoomSceneV2,
    *,
    sample_rate: int = 16000,
    duration_s: float = 1.2,
    source_index: int = 0,
    max_order: int = 4,
    mixing_time_s: float = 0.024,
    transition_duration_s: float = 0.016,
    delay_line_count: int = 16,
    plane_wave_count: int = 128,
    seed: int = 20260731,
    minimum_fdn_center_hz: float = 500.0,
    material_reference_frequency_hz: float = 1000.0,
    decoder: AmbisonicBinauralDecoder | None = None,
) -> SpatialRoomRIRRender:
    """Synthesize synchronized array, FOA, and optional binaural room responses.

    This is an explicit M4 API and does not alter the legacy production
    generator's Pyroomacoustics default.  Coherent direct/early paths are
    generated per receiver.  A single passive material-derived FDN drives one
    shared isotropic plane-wave late field, which is then coupled with exact
    early preservation and per-channel post-transition energy matching.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not 0 <= int(source_index) < len(scene.sources):
        raise ValueError("source_index is outside the scene source list")
    if not 0 <= int(max_order) <= 20:
        raise ValueError("max_order must be in [0, 20]")
    sample_count = max(3, int(round(float(duration_s) * sample_rate)))
    boundary_models = _material_boundary_models(
        scene,
        float(material_reference_frequency_hz),
    )
    surface_models = {
        surface.surface_id: boundary_models[surface.boundary]
        for surface in scene.surfaces
    }

    event_sets = [
        _path_events_for_receiver(
            scene,
            int(source_index),
            receiver_index,
            max_order=int(max_order),
            boundary_models=boundary_models,
        )
        for receiver_index in range(len(scene.receivers))
    ]
    coherent_receivers = np.vstack(
        [
            render_path_events(
                event_set,
                sample_rate_hz=sample_rate,
                num_samples=sample_count,
                surface_admittance_models=surface_models,
                maximum_boundary_filter_tail_samples=max(
                    1,
                    int(round(0.024 * sample_rate)),
                ),
            )
            for event_set in event_sets
        ]
    )
    source_position = np.asarray(
        scene.sources[int(source_index)].pose.position_m,
        dtype=np.float64,
    )
    coherent_receivers = np.vstack(
        [
            apply_air_absorption(
                channel,
                sample_rate,
                float(
                    np.linalg.norm(
                        np.asarray(receiver.pose.position_m, dtype=np.float64)
                        - source_position
                    )
                ),
                temperature_c=float(scene.environment.temperature_c),
                relative_humidity_percent=float(
                    scene.environment.relative_humidity_percent
                ),
                pressure_pa=float(scene.environment.pressure_pa),
            )[0]
            for channel, receiver in zip(
                coherent_receivers,
                scene.receivers,
            )
        ]
    )
    direct_samples = [
        _direct_sample(event_set, sample_rate) for event_set in event_sets
    ]

    # FOA early paths use an omnidirectional reference at the array centroid,
    # matching the fixed reference used by the plane-wave late field.
    receiver_centroid = np.mean(
        np.asarray(
            [receiver.pose.position_m for receiver in scene.receivers],
            dtype=np.float64,
        ),
        axis=0,
    )
    reference_receiver = replace(
        scene.receivers[0],
        transducer_id=f"{scene.receivers[0].transducer_id}:foa-centroid",
        directivity_id="omnidirectional",
        pose=replace(
            scene.receivers[0].pose,
            position_m=receiver_centroid.astype(float).tolist(),
        ),
    )
    reference_scene = replace(scene, receivers=[reference_receiver])
    ambisonic_event_set = _path_events_for_receiver(
        reference_scene,
        int(source_index),
        0,
        max_order=int(max_order),
        boundary_models=boundary_models,
    )
    coherent_ambisonic = render_path_events_ambisonic(
        ambisonic_event_set,
        sample_rate_hz=sample_rate,
        num_samples=sample_count,
        surface_admittance_models=surface_models,
        maximum_boundary_filter_tail_samples=max(
            1,
            int(round(0.024 * sample_rate)),
        ),
    )
    coherent_ambisonic = np.vstack(
        [
            apply_air_absorption(
                channel,
                sample_rate,
                float(np.linalg.norm(receiver_centroid - source_position)),
                temperature_c=float(scene.environment.temperature_c),
                relative_humidity_percent=float(
                    scene.environment.relative_humidity_percent
                ),
                pressure_pa=float(scene.environment.pressure_pa),
            )[0]
            for channel in coherent_ambisonic
        ]
    )
    ambisonic_direct_sample = _direct_sample(ambisonic_event_set, sample_rate)

    target_centers = (500.0, 1000.0, 2000.0, 4000.0)
    target_rt60 = {
        float(key): air_adjusted_rt60_s(
            float(value),
            float(key),
            float(scene.environment.sound_speed_m_s),
            temperature_c=float(scene.environment.temperature_c),
            relative_humidity_percent=float(
                scene.environment.relative_humidity_percent
            ),
            pressure_pa=float(scene.environment.pressure_pa),
        )
        for key, value in scene.predicted_octave_rt60_s(target_centers).items()
        if float(key) >= float(minimum_fdn_center_hz)
        and float(key) * math.sqrt(2.0) < 0.5 * sample_rate * 0.99
    }
    design = design_multiband_fdn(
        int(sample_rate),
        target_rt60,
        target_mixing_time_s=float(mixing_time_s),
        delay_line_count=int(delay_line_count),
        seed=int(seed),
    )
    excitation = np.zeros(sample_count, dtype=np.float64)
    excitation[min(direct_samples)] = 1.0
    late_field = render_spatial_fdn_late_field(
        design,
        excitation,
        [receiver.pose.position_m for receiver in scene.receivers],
        sound_speed_m_s=float(scene.environment.sound_speed_m_s),
        receiver_directivity_ids=[
            receiver.directivity_id for receiver in scene.receivers
        ],
        receiver_orientations_ypr_deg=[
            receiver.pose.orientation_ypr_deg for receiver in scene.receivers
        ],
        plane_wave_count=int(plane_wave_count),
        seed=int(seed) + 1,
    )
    receiver_coupling = couple_receiver_array_early_late(
        coherent_receivers,
        late_field.receiver_rirs,
        sample_rate,
        direct_samples,
        mixing_time_s=float(mixing_time_s),
        transition_duration_s=float(transition_duration_s),
        target_rt60_s_by_hz=target_rt60,
    )
    ambisonic_coupling = couple_receiver_array_early_late(
        coherent_ambisonic,
        late_field.ambisonic_acn_sn3d,
        sample_rate,
        [ambisonic_direct_sample] * 4,
        mixing_time_s=float(mixing_time_s),
        transition_duration_s=float(transition_duration_s),
        target_rt60_s_by_hz=target_rt60,
    )
    binaural = (
        render_ambisonic_brir(
            ambisonic_coupling.rir,
            sample_rate,
            decoder,
        )
        if decoder is not None
        else None
    )
    metadata = {
        "policy": SPATIAL_ROOM_RIR_POLICY,
        "scope": "explicit_opt_in_m4_spatial_renderer",
        "scene_id": scene.scene_id,
        "source_id": scene.sources[int(source_index)].transducer_id,
        "source_index": int(source_index),
        "receiver_ids": [
            receiver.transducer_id for receiver in scene.receivers
        ],
        "sample_rate": int(sample_rate),
        "sample_count": int(sample_count),
        "duration_s": float(duration_s),
        "max_path_event_order": int(max_order),
        "direct_samples": direct_samples,
        "ambisonic_origin_receiver_id": reference_receiver.transducer_id,
        "ambisonic_origin_position_m": receiver_centroid.astype(float).tolist(),
        "ambisonic_direct_sample": int(ambisonic_direct_sample),
        "target_rt60_s_by_hz": {
            f"{center:g}": value for center, value in target_rt60.items()
        },
        "mixing_time_s": float(mixing_time_s),
        "transition_duration_s": float(transition_duration_s),
        "plane_wave_count": int(plane_wave_count),
        "seed": int(seed),
        "late_excitation": "unit_impulse_at_earliest_receiver_direct_sample",
        "receiver_path_event_counts": [
            len(event_set.events) for event_set in event_sets
        ],
        "ambisonic_path_event_count": len(ambisonic_event_set.events),
        "late_field": late_field.metadata,
        "receiver_coupling": receiver_coupling.metadata,
        "ambisonic_coupling": ambisonic_coupling.metadata,
        "binaural": binaural.metadata if binaural is not None else None,
        "production_default_changed": False,
        "finite_output": bool(
            np.all(np.isfinite(receiver_coupling.rir))
            and np.all(np.isfinite(ambisonic_coupling.rir))
            and (
                binaural is None or np.all(np.isfinite(binaural.brir))
            )
        ),
    }
    return SpatialRoomRIRRender(
        receiver_rirs=receiver_coupling.rir,
        ambisonic_acn_sn3d=ambisonic_coupling.rir,
        coherent_receiver_rirs=coherent_receivers,
        coherent_ambisonic_acn_sn3d=coherent_ambisonic,
        spatial_late_field=late_field,
        receiver_coupling=receiver_coupling,
        ambisonic_coupling=ambisonic_coupling,
        binaural=binaural,
        metadata=metadata,
    )


__all__ = [
    "SPATIAL_ROOM_RIR_POLICY",
    "SpatialRoomRIRRender",
    "render_room_scene_spatial_rir",
]
