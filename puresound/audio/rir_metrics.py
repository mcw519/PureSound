"""Compatibility shim for the RIR metrics library.

The implementation moved to :mod:`puresound.audio.rir.metrics` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.metrics import (
    ABEL_ECHO_DENSITY_POLICY,
    DEFAULT_OCTAVE_CENTERS_HZ,
    DIFFUSE_FIELD_COHERENCE_POLICY,
    IACC_POLICY,
    MULTIBAND_LATE_FIELD_POLICY,
    DecayEstimate,
    NoiseFloorEstimate,
    abel_normalized_echo_density_profile,
    analyze_array_spatial_coherence,
    analyze_binaural_iacc,
    analyze_echo_density,
    analyze_multiband_late_field,
    analyze_rir,
    clarity_db,
    compute_drr_db,
    diffuse_field_coherence,
    direct_sample,
    estimate_abel_mixing_time,
    estimate_decay_time,
    estimate_noise_floor_lundeby,
    interaural_cross_correlation,
    noise_compensated_schroeder_decay_db,
    octave_band_rir,
    schroeder_decay_db,
    spectral_tilt_db_per_octave,
    valid_octave_centers,
)

__all__ = [
    "ABEL_ECHO_DENSITY_POLICY",
    "DEFAULT_OCTAVE_CENTERS_HZ",
    "DIFFUSE_FIELD_COHERENCE_POLICY",
    "IACC_POLICY",
    "MULTIBAND_LATE_FIELD_POLICY",
    "DecayEstimate",
    "NoiseFloorEstimate",
    "abel_normalized_echo_density_profile",
    "analyze_array_spatial_coherence",
    "analyze_binaural_iacc",
    "analyze_echo_density",
    "analyze_multiband_late_field",
    "analyze_rir",
    "clarity_db",
    "compute_drr_db",
    "diffuse_field_coherence",
    "direct_sample",
    "estimate_abel_mixing_time",
    "estimate_decay_time",
    "estimate_noise_floor_lundeby",
    "interaural_cross_correlation",
    "noise_compensated_schroeder_decay_db",
    "octave_band_rir",
    "schroeder_decay_db",
    "spectral_tilt_db_per_octave",
    "valid_octave_centers",
]
