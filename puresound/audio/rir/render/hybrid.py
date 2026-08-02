"""Hybrid low-frequency wave and high-frequency geometric RIR generation.

The module is intentionally split into small, testable pieces.  Scene
sampling, obstacle geometry, crossover filtering, and file writing work without
optional simulators installed.  Real generation uses a low-frequency backend
such as pytARD and a high-frequency Pyroomacoustics backend.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Optional

import numpy as np
import torch

from puresound.audio.rir.bank.storage import write_hybrid_rir_dataset_item
from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.metrics import analyze_rir, valid_octave_centers
from puresound.audio.rir.render.arrays import (
    coerce_rir_array as _coerce_rir_array,
    pad_or_trim as _pad_or_trim,
)
from puresound.audio.rir.render.backend import RIRBackend
from puresound.audio.rir.render.crossover import (
    align_high_band_direct as _align_high_band_direct,
    clip_rir_before_physical_arrival as _clip_rir_before_physical_arrival,
    effective_crossover_match_band as _effective_crossover_match_band,
    hybrid_crossover,
    hybrid_crossover_with_metadata as _hybrid_crossover_with_metadata,
    match_low_band_to_high_band as _match_low_band_to_high_band,
)
from puresound.audio.rir.render.high_frequency import (
    PathEventFDNHighFrequencyBackend,
    PathEventHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
    apply_obstacle_high_frequency_effects,
    obstacle_effects_metadata,
)
from puresound.audio.rir.render.low_frequency import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    ImpedanceModalLowFrequencyBackend,
    PytARDWaveBackend,
    material_modal_damping_metadata,
)
from puresound.audio.rir.render.low_frequency.modal_damping import (
    material_modal_decay_rates as _material_modal_decay_rates,
)
from puresound.audio.rir.render.low_frequency.pytard import (
    apply_rt60_decay_envelope as _apply_rt60_decay_envelope,
    calibrate_pytard_signal as _calibrate_pytard_signal,
    pytard_green_delta_excitation as _pytard_green_delta_excitation,
    solve_modal_ard as _solve_modal_ard,
)
from puresound.audio.rir.scene.geometry import (
    clip_position_to_room as _clip_position_to_room,
    distance_point_to_polygon as _distance_point_to_polygon,
    distance_point_to_segment as _distance_point_to_segment,
    max_room_distance_from_point as _max_room_distance_from_point,
    max_room_horizontal_distance_from_point as _max_room_horizontal_distance_from_point,  # noqa: E501
    point_in_polygon as _point_in_polygon,
    polygon_area as _polygon_area,
    polygon_distance as _polygon_distance,
    polygons_overlap as _polygons_overlap,
    segments_intersect as _segments_intersect,
)
from puresound.audio.rir.scene.sampling import (
    HybridRIRScene,
    PolygonObstacle,
    min_feasible_rt60 as _min_feasible_rt60,
    obstacle_floor_coverage as _obstacle_floor_coverage,
    sample_hybrid_rir_scene,
    sample_material_first_rir_scene,
    sample_point as _sample_point,
    sample_polygon_obstacles,
    sample_source_in_horizontal_shell as _sample_source_in_horizontal_shell,
    upgrade_hybrid_scene_to_v2,
)
from puresound.audio.rir.scene.schema import RoomSceneV2


def generate_hybrid_rir(
    config: HybridRIRConfig,
    scene: Optional[HybridRIRScene | RoomSceneV2] = None,
    low_backend: Optional[RIRBackend] = None,
    high_backend: Optional[RIRBackend] = None,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scene = scene or sample_hybrid_rir_scene(config=config, seed=seed)
    low_backend = low_backend or GpuARDPytARDBackend()
    high_backend = high_backend or PyroomacousticsHighFrequencyBackend()

    if config.output_mode not in {"peak_normalized", "calibrated"}:
        raise ValueError(
            "HybridRIRConfig.output_mode must be 'peak_normalized' or 'calibrated'"
        )
    effective_config = config
    if isinstance(scene, RoomSceneV2):
        if len(scene.sources) != config.num_sources:
            raise ValueError(
                f"RoomSceneV2 contains {len(scene.sources)} sources but config "
                f"expects {config.num_sources}"
            )
        if len(scene.receivers) != 1:
            raise ValueError(
                "generate_hybrid_rir produces source channels for exactly one "
                "receiver; use the M4 spatial renderer for receiver arrays"
            )
        effective_config = replace(
            config, sound_speed=float(scene.environment.sound_speed_m_s)
        )

    low = low_backend.simulate(scene, effective_config)
    # The finite modal/voxel low-frequency solve can leave a small numerical
    # precursor before the geometric wavefront.  It is not a physical path and
    # must not be allowed into the hybrid signal (or into M6 causality QC).
    # Apply the same discrete arrival-bin contract used by the high-band
    # alignment: samples n < floor(distance / c * fs) are exactly zero.
    low = _clip_rir_before_physical_arrival(low, scene, effective_config)
    high = high_backend.simulate(scene, effective_config)
    if isinstance(scene, RoomSceneV2):
        gains = scene.transducer_channel_gains(
            reference_source_spl_db=config.calibrated_reference_source_spl_db
        ).reshape(-1, 1)
        low = np.asarray(low, dtype=np.float64) * gains
        high = np.asarray(high, dtype=np.float64) * gains
    material_low_damping = bool(
        getattr(low_backend, "material_modal_damping", False)
    )
    impedance_low_boundary = bool(
        getattr(low_backend, "impedance_boundary_model", False)
    )
    low_modal_metadata = getattr(low_backend, "last_modal_metadata", None)
    source_convention_matched = bool(
        isinstance(low_modal_metadata, dict)
        and low_modal_metadata.get(
            "direct_path_source_convention_matched",
            False,
        )
    )
    energy_matching_requested = bool(
        effective_config.match_crossover_energy
    )
    preserve_source_convention = bool(
        source_convention_matched
        and effective_config.preserve_source_convention_at_crossover
    )
    energy_matching_applied = bool(
        energy_matching_requested and not preserve_source_convention
    )
    crossover_config = replace(
        effective_config,
        match_crossover_energy=energy_matching_applied,
    )
    rir, crossover_metadata = _hybrid_crossover_with_metadata(
        low,
        high,
        crossover_config,
    )
    crossover_metadata.update(
        {
            "energy_matching_requested": energy_matching_requested,
            "energy_matching_applied": energy_matching_applied,
            "source_convention_matched_before_crossover": (
                source_convention_matched
            ),
            "source_convention_preservation_enabled": bool(
                effective_config.preserve_source_convention_at_crossover
            ),
            "source_convention_preserved": bool(
                source_convention_matched and not energy_matching_applied
            ),
            "policy": (
                "preserve_validated_source_convention"
                if preserve_source_convention
                else "energy_rms_match"
                if energy_matching_applied
                else "no_energy_match"
            ),
        }
    )
    if impedance_low_boundary:
        low_mode_excitation_model = (
            "separable_complex_eigenfunction_source_receiver_coupling"
        )
    elif hasattr(low_backend, "physical_mode_coupling"):
        low_mode_excitation_model = (
            "rectangular_eigenfunction_source_receiver_coupling"
            if bool(getattr(low_backend, "physical_mode_coupling"))
            else "legacy_rank_amplitude_and_position_phase"
        )
    else:
        low_mode_excitation_model = "voxel_dct_source_receiver_coupling"
    if isinstance(high_backend, PathEventHighFrequencyBackend):
        obstacle_metadata = {
            "policy": "per_path_geometry_visibility_transmission_diffraction",
            "legacy_whole_rir_post_effect_applied": False,
            "object_count": len(scene.objects) if isinstance(scene, RoomSceneV2) else 0,
            "source_count": len(scene.sources) if isinstance(scene, RoomSceneV2) else 0,
        }
    else:
        obstacle_metadata = obstacle_effects_metadata(scene, config)
    metadata = {
        "config": _config_metadata(config),
        "scene": scene.to_metadata(),
        "obstacle_effects": obstacle_metadata,
        "bands": {
            "low": {
                "backend": low_backend.__class__.__name__,
                "frequency_hz": [config.low_fmin_hz, config.low_fmax_hz],
                "boundary_model": (
                    "separable_rational_impedance_eigenproblem"
                    if impedance_low_boundary
                    else "per_mode_surface_material_damping"
                    if material_low_damping
                    else "global_rt60_envelope"
                ),
                "global_rt60_envelope_applied": not (
                    material_low_damping or impedance_low_boundary
                ),
                "mode_excitation_model": low_mode_excitation_model,
                "solver_excitation": getattr(
                    low_backend,
                    "last_excitation_metadata",
                    None,
                ),
                "causality_policy": (
                    "zero_samples_before_floor_distance_over_sound_speed"
                ),
            },
            "high": {
                "backend": high_backend.__class__.__name__,
                "frequency_hz": [config.crossover_hz, config.sample_rate / 2.0],
                "rng_seed": getattr(high_backend, "rng_seed", None),
                "source_directivity_policy": (
                    "scene_v2_first_order_pressure_pattern"
                    if isinstance(scene, RoomSceneV2)
                    else "legacy_omnidirectional"
                ),
                "boundary_model": (
                    "frequency_dependent_surface_materials"
                    if isinstance(scene, RoomSceneV2)
                    else "uniform_broadband_inverse_sabine"
                ),
            },
        },
        "output_calibration": {
            "mode": config.output_mode,
            "peak_target": (
                float(config.normalize_peak)
                if config.output_mode == "peak_normalized"
                else None
            ),
            "reference_source_spl_db": float(
                config.calibrated_reference_source_spl_db
            ),
            "per_item_peak_normalized": config.output_mode == "peak_normalized",
        },
        "crossover": crossover_metadata,
    }
    late_field_metadata = getattr(
        high_backend,
        "last_late_field_metadata",
        None,
    )
    if isinstance(late_field_metadata, dict):
        metadata["bands"]["high"]["late_field"] = late_field_metadata
    boundary_metadata = getattr(
        high_backend,
        "last_boundary_metadata",
        None,
    )
    if isinstance(boundary_metadata, dict):
        metadata["bands"]["high"]["surface_boundary_prior"] = (
            boundary_metadata
        )
    air_absorption_metadata = getattr(
        high_backend,
        "last_air_absorption_metadata",
        None,
    )
    if isinstance(air_absorption_metadata, dict):
        metadata["bands"]["high"]["air_absorption"] = (
            air_absorption_metadata
        )
    if isinstance(low_backend, AnalyticModalLowFrequencyBackend):
        metadata["bands"]["low"]["analytic_mode_index_limit"] = int(
            low_backend.num_modes_per_axis
        )
        metadata["bands"]["low"]["analytic_max_modes"] = (
            int(low_backend.max_modes)
            if low_backend.max_modes is not None
            else None
        )
    if isinstance(low_backend, ImpedanceModalLowFrequencyBackend):
        metadata["bands"]["low"]["analytic_mode_index_limit"] = int(
            low_backend.num_modes_per_axis
        )
        metadata["bands"]["low"]["analytic_max_modes"] = (
            int(low_backend.max_modes)
            if low_backend.max_modes is not None
            else None
        )
        metadata["bands"]["low"]["impedance_modes"] = (
            low_backend.last_modal_metadata
        )
    if material_low_damping and isinstance(scene, RoomSceneV2):
        modal_loss_scale = float(
            getattr(low_backend, "material_modal_loss_scale", 1.0)
        )
        metadata["bands"]["low"]["modal_damping"] = (
            material_modal_damping_metadata(
                scene,
                effective_config,
                loss_scale=modal_loss_scale,
            )
        )
    if isinstance(scene, RoomSceneV2) and config.record_realized_metrics:
        metadata["realized_acoustics"] = _realized_acoustics_metadata(
            rir, scene, effective_config
        )
    return torch.as_tensor(rir, dtype=torch.float32), metadata


def _realized_acoustics_metadata(
    rir: np.ndarray,
    scene: RoomSceneV2,
    config: HybridRIRConfig,
) -> dict[str, Any]:
    """Measure generated broadband and octave decay for every source channel."""
    centers = valid_octave_centers(
        config.sample_rate,
        next(iter(scene.materials.values())).absorption.center_frequencies_hz,
    )
    distances = scene.source_distances()
    channels = []
    for index, channel in enumerate(np.asarray(rir, dtype=np.float64)):
        direct_index = int(
            round(
                distances[index]
                / max(float(config.sound_speed), 1e-6)
                * float(config.sample_rate)
            )
        )
        channels.append(
            {
                "channel": index,
                "label": scene.source_labels[index],
                "distance_m": distances[index],
                "metrics": analyze_rir(
                    channel,
                    sample_rate=config.sample_rate,
                    direct_index=direct_index,
                    octave_centers_hz=centers,
                ),
            }
        )
    return {
        "method": "puresound.audio.rir.metrics.analyze_rir",
        "octave_centers_hz": centers,
        "channels": channels,
    }




























def _config_metadata(config: HybridRIRConfig) -> dict[str, Any]:
    data = asdict(config)
    data["num_samples"] = config.num_samples
    data["num_sources"] = config.num_sources
    return data






































































#: The public surface of this module, frozen by R0 and enforced by
#: ``test/test_rir_r0_api_inventory.py``.  Symbols re-exported from
#: ``puresound.audio.rir`` during the modularization stay listed here so the
#: legacy import path keeps working.
__all__ = [
    "AnalyticModalLowFrequencyBackend",
    "GpuARDPytARDBackend",
    "GpuARDPytARDCuPyBackend",
    "HybridRIRConfig",
    "HybridRIRScene",
    "ImpedanceModalLowFrequencyBackend",
    "PathEventFDNHighFrequencyBackend",
    "PathEventHighFrequencyBackend",
    "PolygonObstacle",
    "PyroomacousticsHighFrequencyBackend",
    "PytARDWaveBackend",
    "RIRBackend",
    "apply_obstacle_high_frequency_effects",
    "generate_hybrid_rir",
    "hybrid_crossover",
    "material_modal_damping_metadata",
    "obstacle_effects_metadata",
    "sample_hybrid_rir_scene",
    "sample_material_first_rir_scene",
    "sample_polygon_obstacles",
    "upgrade_hybrid_scene_to_v2",
    "write_hybrid_rir_dataset_item",
]
