"""Deterministic synchronized array and Ambisonic rendering for M4.5.

The late field is represented as a finite isotropic plane-wave quadrature.
Every plane wave receives an independent noise carrier whose octave-band
envelope is taken from the same passive multiband FDN.  Receiver delays,
receiver directivity, and first-order Ambisonic channels are projections of
those shared plane waves; channels are therefore synchronized views of one
field rather than independently generated mono tails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from puresound.audio.multiband_fdn import (
    MultibandFDNDesign,
    render_multiband_fdn,
)
from puresound.audio.rir_late_coupling import (
    energy_preserving_diffuse_gain,
    equal_power_transition_weights,
    extrapolated_path_tail_energy_target,
    transition_samples,
)
from puresound.audio.rir_metrics import octave_band_rir
from puresound.audio.rir_path_events import (
    PathEventSet,
    causal_fractional_delay_kernel,
    directivity_pressure_gain,
    render_path_events,
)


SPATIAL_LATE_FIELD_POLICY = "puresound.spatial_late_field.v1"
SPATIAL_EARLY_LATE_COUPLING_POLICY = (
    "puresound.spatial_early_late_coupling.v1"
)


@dataclass(frozen=True)
class SpatialLateFieldRender:
    """Synchronized receiver and first-order Ambisonic late-field RIRs."""

    receiver_rirs: np.ndarray
    ambisonic_acn_sn3d: np.ndarray
    directions_unit: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SpatialEarlyLateRender:
    """PathEvent early fields coupled to synchronized spatial FDN tails."""

    rir: np.ndarray
    coherent_component: np.ndarray
    diffuse_component: np.ndarray
    early_weight: np.ndarray
    late_weight: np.ndarray
    metadata: Mapping[str, Any]


def fibonacci_sphere_directions(
    count: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return deterministic near-uniform unit directions with a seeded rotation."""

    direction_count = int(count)
    if direction_count != count or direction_count < 16:
        raise ValueError("plane-wave count must be an integer of at least 16")
    index = np.arange(direction_count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / direction_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - np.square(z)))
    azimuth = index * math.pi * (3.0 - math.sqrt(5.0))
    directions = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )

    rng = np.random.default_rng(int(seed))
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    rotated = directions @ rotation.T
    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    return np.asarray(rotated, dtype=np.float64)


def _receiver_positions(receiver_positions_m: Any) -> np.ndarray:
    positions = np.asarray(receiver_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] != 3:
        raise ValueError("receiver_positions_m must have shape [receiver, 3]")
    if not np.all(np.isfinite(positions)):
        raise ValueError("receiver positions must be finite")
    return positions


def _causal_rms_envelope(signal: np.ndarray, window_samples: int) -> np.ndarray:
    squared = np.square(np.asarray(signal, dtype=np.float64))
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(squared, dtype=np.float64))
    )
    end = np.arange(1, squared.size + 1, dtype=np.int64)
    start = np.maximum(0, end - int(window_samples))
    count = end - start
    mean_energy = (cumulative[end] - cumulative[start]) / count
    return np.sqrt(np.maximum(mean_energy, 0.0))


def _causal_fractional_shift(
    signal: np.ndarray,
    delay_samples: float,
    *,
    order: int,
) -> np.ndarray:
    start, kernel = causal_fractional_delay_kernel(
        delay_samples,
        order=order,
    )
    result = np.zeros(signal.size, dtype=np.float64)
    if start >= signal.size:
        return result
    filtered = np.convolve(signal, kernel, mode="full")
    available = signal.size - start
    result[start:] = filtered[:available]
    return result


def _render_plane_waves(
    design: MultibandFDNDesign,
    excitation: np.ndarray,
    *,
    plane_wave_count: int,
    seed: int,
    envelope_window_ms: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    fdn = render_multiband_fdn(
        design,
        excitation,
        filter_order=filter_order,
    )
    window_samples = max(
        1,
        int(round(1e-3 * float(envelope_window_ms) * design.sample_rate)),
    )
    directions = fibonacci_sphere_directions(
        plane_wave_count,
        seed=seed,
    )
    planes = np.zeros((plane_wave_count, excitation.size), dtype=np.float64)
    rng = np.random.default_rng(int(seed) + 1)
    for center_hz in design.centers_hz:
        envelope = _causal_rms_envelope(
            np.asarray(fdn.band_rirs[center_hz], dtype=np.float64),
            window_samples,
        )
        for direction_index in range(plane_wave_count):
            carrier = octave_band_rir(
                rng.standard_normal(excitation.size),
                design.sample_rate,
                center_hz,
                filter_order=filter_order,
            )
            carrier_rms = math.sqrt(float(np.mean(np.square(carrier))))
            if carrier_rms > np.finfo(np.float64).tiny:
                carrier /= carrier_rms
            planes[direction_index] += envelope * carrier

    normalization = math.sqrt(float(plane_wave_count))
    ambisonic_w = np.sum(planes, axis=0) / normalization
    target_energy = float(np.dot(fdn.rir, fdn.rir))
    realized_energy = float(np.dot(ambisonic_w, ambisonic_w))
    gain = (
        math.sqrt(target_energy / realized_energy)
        if target_energy > 0.0 and realized_energy > 0.0
        else 0.0
    )
    planes *= gain
    return planes, directions, float(gain)


def render_spatial_fdn_late_field(
    design: MultibandFDNDesign,
    excitation: Any,
    receiver_positions_m: Any,
    *,
    sound_speed_m_s: float = 343.0,
    receiver_directivity_ids: Sequence[str] | None = None,
    receiver_orientations_ypr_deg: Sequence[Iterable[float]] | None = None,
    plane_wave_count: int = 128,
    seed: int = 0,
    envelope_window_ms: float = 10.0,
    fractional_delay_order: int = 3,
    filter_order: int = 4,
) -> SpatialLateFieldRender:
    """Render one shared diffuse field to an array and ACN/SN3D FOA.

    ``directions_unit`` stores propagation directions.  Receiver patterns and
    Ambisonic encoding use the opposite direction, i.e. the direction of
    arrival seen from the receiver.
    """

    signal = np.asarray(excitation, dtype=np.float64).squeeze()
    if signal.ndim == 0:
        signal = signal.reshape(1)
    if signal.ndim != 1 or signal.size < 1 or not np.all(np.isfinite(signal)):
        raise ValueError("excitation must be a non-empty finite mono signal")
    if design.sample_rate <= 0:
        raise ValueError("FDN design sample rate must be positive")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and positive")
    if not np.isfinite(envelope_window_ms) or envelope_window_ms <= 0.0:
        raise ValueError("envelope_window_ms must be finite and positive")
    positions = _receiver_positions(receiver_positions_m)
    receiver_count = positions.shape[0]
    directivities = (
        ["omnidirectional"] * receiver_count
        if receiver_directivity_ids is None
        else [str(value) for value in receiver_directivity_ids]
    )
    orientations = (
        [[0.0, 0.0, 0.0] for _ in range(receiver_count)]
        if receiver_orientations_ypr_deg is None
        else [list(value) for value in receiver_orientations_ypr_deg]
    )
    if len(directivities) != receiver_count or len(orientations) != receiver_count:
        raise ValueError("receiver pattern metadata must match receiver count")

    planes, propagation_directions, plane_gain = _render_plane_waves(
        design,
        signal,
        plane_wave_count=int(plane_wave_count),
        seed=int(seed),
        envelope_window_ms=float(envelope_window_ms),
        filter_order=int(filter_order),
    )
    arrival_directions = -propagation_directions
    normalization = math.sqrt(float(plane_wave_count))

    reference_position = np.mean(positions, axis=0, keepdims=True)
    centered_positions = positions - reference_position
    aperture_radius_m = float(
        np.max(np.linalg.norm(centered_positions, axis=1))
    )
    causal_margin_samples = (
        aperture_radius_m / float(sound_speed_m_s) * design.sample_rate
    )
    receiver_rirs = np.zeros((receiver_count, signal.size), dtype=np.float64)
    maximum_relative_delay_samples = 0.0
    for direction_index, propagation in enumerate(propagation_directions):
        delays_samples = causal_margin_samples + (
            centered_positions @ propagation
            / float(sound_speed_m_s)
            * design.sample_rate
        )
        delays_samples = np.maximum(delays_samples, 0.0)
        maximum_relative_delay_samples = max(
            maximum_relative_delay_samples,
            float(np.max(delays_samples)),
        )
        for receiver_index in range(receiver_count):
            pattern_gain = directivity_pressure_gain(
                directivities[receiver_index],
                orientations[receiver_index],
                arrival_directions[direction_index],
            )
            shifted = _causal_fractional_shift(
                planes[direction_index],
                float(delays_samples[receiver_index]),
                order=int(fractional_delay_order),
            )
            receiver_rirs[receiver_index] += pattern_gain * shifted / normalization

    # ACN channel order 0,1,2,3 and SN3D basis 1,y,z,x.  The shared
    # aperture margin places FOA at the same centroid reference as the array.
    reference_planes = np.vstack(
        [
            _causal_fractional_shift(
                plane,
                causal_margin_samples,
                order=int(fractional_delay_order),
            )
            for plane in planes
        ]
    )
    ambisonic = np.vstack(
        (
            np.sum(reference_planes, axis=0),
            np.sum(
                arrival_directions[:, 1, None] * reference_planes,
                axis=0,
            ),
            np.sum(
                arrival_directions[:, 2, None] * reference_planes,
                axis=0,
            ),
            np.sum(
                arrival_directions[:, 0, None] * reference_planes,
                axis=0,
            ),
        )
    ) / normalization
    pair_distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )
    metadata = {
        "policy": SPATIAL_LATE_FIELD_POLICY,
        "sample_rate": int(design.sample_rate),
        "sample_count": int(signal.size),
        "seed": int(seed),
        "plane_wave_count": int(plane_wave_count),
        "plane_wave_quadrature": "seeded_rotated_fibonacci_sphere",
        "carrier": "independent_gaussian_nominal_octave_noise",
        "envelope": "causal_rms_of_passive_multiband_fdn_band",
        "envelope_window_ms": float(envelope_window_ms),
        "plane_energy_normalization_gain": float(plane_gain),
        "receiver_positions_m": positions.astype(float).tolist(),
        "spatial_reference": "receiver_array_centroid",
        "spatial_reference_position_m": reference_position[0]
        .astype(float)
        .tolist(),
        "causal_aperture_margin_samples": float(causal_margin_samples),
        "causal_aperture_margin_m": float(aperture_radius_m),
        "receiver_pair_distances_m": pair_distances.astype(float).tolist(),
        "receiver_directivity_ids": directivities,
        "receiver_orientations_ypr_deg": orientations,
        "sound_speed_m_s": float(sound_speed_m_s),
        "fractional_delay_order": int(fractional_delay_order),
        "maximum_relative_delay_samples": float(maximum_relative_delay_samples),
        "ambisonic": {
            "order": 1,
            "channel_order": "ACN",
            "normalization": "SN3D",
            "channel_labels": ["W", "Y", "Z", "X"],
            "basis": ["1", "arrival_y", "arrival_z", "arrival_x"],
        },
        "fdn_design": design.to_dict(include_coefficients=False),
        "finite_output": bool(
            np.all(np.isfinite(receiver_rirs))
            and np.all(np.isfinite(ambisonic))
        ),
    }
    return SpatialLateFieldRender(
        receiver_rirs=np.asarray(receiver_rirs, dtype=np.float64),
        ambisonic_acn_sn3d=np.asarray(ambisonic, dtype=np.float64),
        directions_unit=np.asarray(propagation_directions, dtype=np.float64),
        metadata=metadata,
    )


def render_path_events_ambisonic(
    event_set: PathEventSet,
    *,
    sample_rate_hz: float,
    num_samples: int,
    fractional_delay_order: int = 3,
    surface_admittance_models: Mapping[str, Any] | None = None,
    maximum_boundary_filter_tail_samples: int | None = None,
) -> np.ndarray:
    """Encode coherent PathEvents as ACN/SN3D first-order Ambisonics.

    The event arrival vector points along propagation into the receiver, so
    its negation is used as the direction of arrival.  Each event is rendered
    with the same causal fractional-delay and boundary-gain contract as the
    mono PathEvent renderer before applying the ``[1, y, z, x]`` basis.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    ambisonic = np.zeros((4, int(num_samples)), dtype=np.float64)
    for event in event_set.events:
        event_rir = render_path_events(
            [event],
            sample_rate_hz=float(sample_rate_hz),
            num_samples=int(num_samples),
            fractional_delay_order=int(fractional_delay_order),
            surface_admittance_models=surface_admittance_models,
            maximum_boundary_filter_tail_samples=(
                maximum_boundary_filter_tail_samples
            ),
        )
        arrival = -np.asarray(event.arrival_direction_unit, dtype=np.float64)
        basis = np.asarray(
            [1.0, arrival[1], arrival[2], arrival[0]],
            dtype=np.float64,
        )
        ambisonic += basis[:, None] * event_rir[None, :]
    return ambisonic


def couple_receiver_array_early_late(
    coherent_rirs: Any,
    spatial_late_rirs: Any,
    sample_rate: int,
    direct_samples: Sequence[int],
    *,
    mixing_time_s: float = 0.024,
    transition_duration_s: float = 0.016,
    target_rt60_s_by_hz: Mapping[float, float] | None = None,
) -> SpatialEarlyLateRender:
    """Couple synchronized receiver PathEvents to synchronized spatial tails."""

    coherent_input = np.asarray(coherent_rirs, dtype=np.float64)
    late_input = np.asarray(spatial_late_rirs, dtype=np.float64)
    if coherent_input.ndim != 2 or coherent_input.shape != late_input.shape:
        raise ValueError("coherent and late RIR arrays must share [receiver, sample]")
    if not np.all(np.isfinite(coherent_input)) or not np.all(np.isfinite(late_input)):
        raise ValueError("coherent and late RIR arrays must be finite")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(direct_samples) != coherent_input.shape[0]:
        raise ValueError("direct_samples must match receiver count")

    early_weights = np.zeros_like(coherent_input)
    late_weights = np.zeros_like(coherent_input)
    coherent_components = np.zeros_like(coherent_input)
    diffuse_components = np.zeros_like(coherent_input)
    channels: list[dict[str, Any]] = []
    transition_rows: list[tuple[int, int, int]] = []
    target_energies: list[float] = []
    for receiver_index, direct_sample in enumerate(direct_samples):
        center, start, end = transition_samples(
            coherent_input.shape[1],
            int(sample_rate),
            int(direct_sample),
            float(mixing_time_s),
            float(transition_duration_s),
        )
        early, late = equal_power_transition_weights(
            coherent_input.shape[1],
            start,
            end,
        )
        coherent = coherent_input[receiver_index] * early
        diffuse_before_gain = late_input[receiver_index] * late
        finite_target = float(
            np.dot(
                coherent_input[receiver_index, start:],
                coherent_input[receiver_index, start:],
            )
        )
        if target_rt60_s_by_hz is None:
            target_energy = finite_target
            extrapolation = None
        else:
            target_energy, extrapolation = extrapolated_path_tail_energy_target(
                coherent_input[receiver_index],
                start,
                int(sample_rate),
                target_rt60_s_by_hz,
            )
        target_energies.append(float(target_energy))
        early_weights[receiver_index] = early
        late_weights[receiver_index] = late
        coherent_components[receiver_index] = coherent
        diffuse_components[receiver_index] = diffuse_before_gain
        transition_rows.append((center, start, end))
        channels.append(
            {
                "receiver_index": int(receiver_index),
                "direct_sample": int(direct_sample),
                "transition_center_sample": int(center),
                "transition_start_sample": int(start),
                "transition_end_sample": int(end),
                "target_post_transition_energy": float(target_energy),
                "finite_coherent_post_transition_energy": finite_target,
                "path_tail_extrapolation": extrapolation,
            }
        )

    original_tails = np.concatenate(
        [
            coherent_input[index, start:]
            for index, (_center, start, _end) in enumerate(transition_rows)
        ]
    )
    coherent_tails = np.concatenate(
        [
            coherent_components[index, start:]
            for index, (_center, start, _end) in enumerate(transition_rows)
        ]
    )
    diffuse_tails = np.concatenate(
        [
            diffuse_components[index, start:]
            for index, (_center, start, _end) in enumerate(transition_rows)
        ]
    )
    shared_gain, aggregate_energy = energy_preserving_diffuse_gain(
        original_tails,
        coherent_tails,
        diffuse_tails,
        0,
        target_energy=float(sum(target_energies)),
    )
    diffuse_components *= shared_gain
    output = coherent_components + diffuse_components
    for receiver_index, (_center, start, _end) in enumerate(transition_rows):
        channel = output[receiver_index]
        realized = float(np.dot(channel[start:], channel[start:]))
        channels[receiver_index].update(
            {
                "pre_transition_max_abs_error": float(
                    np.max(
                        np.abs(
                            channel[: start + 1]
                            - coherent_input[receiver_index, : start + 1]
                        )
                    )
                ),
                "realized_post_transition_energy": realized,
                "shared_diffuse_gain": float(shared_gain),
            }
        )
    metadata = {
        "policy": SPATIAL_EARLY_LATE_COUPLING_POLICY,
        "sample_rate": int(sample_rate),
        "receiver_count": int(coherent_input.shape[0]),
        "sample_count": int(coherent_input.shape[1]),
        "mixing_time_s_after_direct": float(mixing_time_s),
        "transition_duration_s": float(transition_duration_s),
        "crossfade": "per_receiver_complementary_equal_power_cosine_sine",
        "energy_policy": "one_shared_array_gain_preserves_spatial_ratios",
        "shared_diffuse_gain": float(shared_gain),
        "aggregate_energy": aggregate_energy,
        "channels": channels,
        "finite_output": bool(np.all(np.isfinite(output))),
    }
    return SpatialEarlyLateRender(
        rir=output,
        coherent_component=coherent_components,
        diffuse_component=diffuse_components,
        early_weight=early_weights,
        late_weight=late_weights,
        metadata=metadata,
    )


__all__ = [
    "SPATIAL_EARLY_LATE_COUPLING_POLICY",
    "SPATIAL_LATE_FIELD_POLICY",
    "SpatialEarlyLateRender",
    "SpatialLateFieldRender",
    "couple_receiver_array_early_late",
    "fibonacci_sphere_directions",
    "render_spatial_fdn_late_field",
    "render_path_events_ambisonic",
]
