"""RIR analysis metrics.

Layer 2.  Every function takes array-like input plus an explicit analysis
configuration; nothing here knows which renderer produced the signal.
"""

from puresound.audio.rir.metrics.core import (
    DecayEstimate,
    NoiseFloorEstimate,
    direct_sample,
)
from puresound.audio.rir.metrics.density import (
    ABEL_ECHO_DENSITY_POLICY,
    MULTIBAND_LATE_FIELD_POLICY,
    abel_normalized_echo_density_profile,
    analyze_echo_density,
    analyze_multiband_late_field,
    estimate_abel_mixing_time,
)
from puresound.audio.rir.metrics.report import analyze_rir
from puresound.audio.rir.metrics.spatial import (
    DIFFUSE_FIELD_COHERENCE_POLICY,
    IACC_POLICY,
    analyze_array_spatial_coherence,
    analyze_binaural_iacc,
    diffuse_field_coherence,
    interaural_cross_correlation,
)
from puresound.audio.rir.metrics.spectral import (
    DEFAULT_OCTAVE_CENTERS_HZ,
    octave_band_rir,
    spectral_tilt_db_per_octave,
    valid_octave_centers,
)
from puresound.audio.rir.metrics.temporal import (
    clarity_db,
    compute_drr_db,
    estimate_decay_time,
    estimate_noise_floor_lundeby,
    noise_compensated_schroeder_decay_db,
    schroeder_decay_db,
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
