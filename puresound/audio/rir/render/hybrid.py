"""Hybrid RIR generation: the orchestration that composes the render layers.

``generate_hybrid_rir`` samples or accepts a scene, runs a low-frequency and a
high-frequency backend, applies the causality clip and the crossover, and
assembles the metadata.  It owns no geometry, no solver and no file writing —
those live in ``scene``, ``render.low_frequency`` / ``render.high_frequency``,
``render.crossover`` and ``bank.storage`` respectively.

Backends are injected: pass any object satisfying
``puresound.audio.rir.render.backend.RIRBackend``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Optional

import numpy as np
import torch

from puresound.audio.rir.bank.storage import write_hybrid_rir_dataset_item
from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.metrics import analyze_rir, valid_octave_centers
from puresound.audio.rir.render.backend import RIRBackend
from puresound.audio.rir.render.crossover import (
    clip_rir_before_physical_arrival,
    hybrid_crossover_with_metadata,
)
from puresound.audio.rir.render.high_frequency import (
    PathEventHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
    obstacle_effects_metadata,
)
from puresound.audio.rir.render.low_frequency import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    ImpedanceModalLowFrequencyBackend,
    material_modal_damping_metadata,
)
from puresound.audio.rir.scene.sampling import (
    HybridRIRScene,
    sample_hybrid_rir_scene,
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
    low = clip_rir_before_physical_arrival(low, scene, effective_config)
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
    rir, crossover_metadata = hybrid_crossover_with_metadata(
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






































































__all__ = ["generate_hybrid_rir"]
