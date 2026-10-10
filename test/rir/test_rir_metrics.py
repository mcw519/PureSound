import json
import math

import numpy as np
import pytest
import torch

from puresound.audio.impulse_response import compute_drr_db as legacy_compute_drr_db
from puresound.audio.rir.metrics import (
    DEFAULT_OCTAVE_CENTERS_HZ,
    abel_normalized_echo_density_profile,
    analyze_array_spatial_coherence,
    analyze_binaural_iacc,
    analyze_echo_density,
    analyze_multiband_late_field,
    analyze_rir,
    clarity_db,
    compute_drr_db,
    diffuse_field_coherence,
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


def _exponential_decay(sample_rate, rt60_s, duration_s=1.0):
    time_s = np.arange(int(duration_s * sample_rate), dtype=np.float64) / sample_rate
    return np.exp(-math.log(1000.0) * time_s / rt60_s)


def test_energy_ratio_metrics_follow_their_definitions():
    rir = torch.zeros(1, 400)
    rir[0, 10] = 1.0
    rir[0, 100] = 0.3
    measured = compute_drr_db(rir, sample_rate=16000, direct_window_ms=2.5)
    assert measured == pytest.approx(10.0 * math.log10(1.0 / 0.09), abs=1e-5)
    assert legacy_compute_drr_db(rir, 16000, 2.5) == pytest.approx(measured)

    clarity = np.zeros(2000, dtype=np.float64)
    clarity[10] = 1.0
    clarity[10 + 400] = 0.5  # 25 ms at 16 kHz: early for C50.
    clarity[10 + 1200] = 0.25  # 75 ms: late for C50, early for C80.
    assert clarity_db(clarity, 16000, 50.0) == pytest.approx(
        10.0 * math.log10(1.25 / 0.0625)
    )
    assert math.isinf(clarity_db(clarity, 16000, 80.0))

    impulse = np.zeros(1024, dtype=np.float64)
    impulse[0] = 1.0
    assert spectral_tilt_db_per_octave(impulse, sample_rate=16000) == pytest.approx(
        0.0, abs=1e-10
    )


def test_decay_estimates_recover_rt60_and_fail_explicitly_without_a_tail():
    sample_rate = 16000
    target_rt60 = 0.5
    rir = _exponential_decay(sample_rate, target_rt60)

    for start_db, end_db in ((0.0, -10.0), (-5.0, -25.0), (-5.0, -35.0)):
        estimate = estimate_decay_time(rir, sample_rate, start_db, end_db)
        assert estimate is not None
        assert estimate.rt60_s == pytest.approx(target_rt60, rel=2e-3)
    assert estimate.r_squared > 0.999

    anechoic = np.zeros(1600, dtype=np.float64)
    anechoic[20] = 1.0
    assert estimate_decay_time(anechoic, 16000, -5.0, -35.0) is None
    assert np.isnan(schroeder_decay_db(np.zeros(32))).all()


def test_lundeby_corrects_a_noise_floor_but_not_a_clean_tail():
    sample_rate = 16000
    target_rt60 = 0.5
    clean = _exponential_decay(sample_rate, target_rt60, duration_s=2.0)
    rir = clean + np.random.default_rng(2).normal(0.0, 0.003, clean.size)

    estimate = estimate_noise_floor_lundeby(rir, sample_rate, direct_index=0)
    raw = estimate_decay_time(
        rir, sample_rate, -5.0, -35.0, direct_index=0, noise_compensation=False
    )
    corrected = estimate_decay_time(
        rir, sample_rate, -5.0, -35.0, direct_index=0, noise_compensation=True
    )
    curve, curve_estimate = noise_compensated_schroeder_decay_db(
        rir, sample_rate, direct_index=0
    )

    assert estimate.correction_applied
    # The noise sits about 50 dB below the direct path and crosses the decay
    # near 0.42 s (the time a 0.5 s RT60 needs to fall that far).
    assert estimate.dynamic_range_db == pytest.approx(49.8, abs=1.0)
    assert estimate.intersection_time_s == pytest.approx(0.42, abs=0.03)
    assert curve_estimate == estimate
    assert np.all(np.diff(curve) <= 1e-12)
    assert raw is not None and raw.rt60_s > 1.0
    assert corrected is not None
    assert corrected.rt60_s == pytest.approx(target_rt60, rel=0.03)
    assert corrected.r_squared > 0.99

    clean_one_second = _exponential_decay(sample_rate, target_rt60)
    untouched = estimate_noise_floor_lundeby(clean_one_second, sample_rate)
    assert not untouched.correction_applied
    assert untouched.reason == "tail_is_not_stationary_noise"
    assert untouched.truncation_sample == clean_one_second.size


def test_analyze_rir_is_strict_json_with_octave_bands_below_nyquist():
    sample_rate = 16000
    target_rt60 = 0.4
    rir = _exponential_decay(sample_rate, target_rt60)

    result = analyze_rir(
        rir,
        sample_rate,
        octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    )

    assert result["t30_s"] == pytest.approx(target_rt60, rel=2e-3)
    assert result["noise_floor"]["correction_applied"] is False
    assert set(result["octave_bands"]) == {"63", "125", "250", "500", "1000", "2000", "4000"}
    json.dumps(result, allow_nan=False)
    assert valid_octave_centers(8000) == [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]


def test_analyze_rir_reports_unbounded_ratios_as_null_and_rejects_multichannel():
    rir = np.zeros(1000, dtype=np.float64)
    rir[10] = 1.0

    result = analyze_rir(rir, 16000)

    assert result["drr_db"] is None
    assert result["c50_db"] is None
    assert result["t30"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="one channel"):
        analyze_rir(np.zeros((2, 100)), 16000)


def test_an_empty_response_is_analyzed_to_null_metrics_in_every_band():
    """Empty input is a defined case for every metric, band filtering included."""
    empty = np.zeros(0, dtype=np.float64)

    assert octave_band_rir(empty, 16000, 1000.0).shape == (0,)
    result = analyze_rir(
        empty, 16000, octave_centers_hz=(250.0, 1000.0), echo_density=True
    )
    json.dumps(result, allow_nan=False)
    assert set(result["octave_bands"]) == {"250", "1000"}
    assert result["octave_bands"]["1000"]["t20_s"] is None
    assert analyze_binaural_iacc(empty, empty, 16000)["iacc_e4"] is None


def test_abel_echo_density_is_scale_invariant_unity_for_noise_and_low_for_sparse_trains():
    sample_rate = 8000
    rir = np.random.default_rng(21).normal(0.0, 1.0, 4 * sample_rate)

    times_s, density = abel_normalized_echo_density_profile(
        rir, sample_rate, direct_index=0, window_ms=40.0, hop_ms=2.0
    )
    scaled_times_s, scaled_density = abel_normalized_echo_density_profile(
        0.013 * rir, sample_rate, direct_index=0, window_ms=40.0, hop_ms=2.0
    )

    assert np.array_equal(times_s, scaled_times_s)
    assert np.array_equal(density, scaled_density)
    assert np.median(density) == pytest.approx(1.0, abs=0.04)

    sparse = np.zeros(2 * sample_rate, dtype=np.float64)
    sparse[::80] = 1.0
    sparse_times_s, sparse_density = abel_normalized_echo_density_profile(
        sparse, sample_rate, direct_index=0, window_ms=40.0
    )
    assert sparse_density.size == sparse_times_s.size
    assert np.max(sparse_density) < 0.1
    assert estimate_abel_mixing_time(sparse_times_s, sparse_density) is None


def test_abel_mixing_time_detects_the_sparse_to_dense_transition_and_validates_profiles():
    sample_rate = 8000
    transition_s = 0.15
    rir = np.zeros(2 * sample_rate, dtype=np.float64)
    rir[0] = 5.0
    rir[80 : int(transition_s * sample_rate) : 160] = 0.2
    rir[int(transition_s * sample_rate) :] = np.random.default_rng(22).normal(
        0.0, 0.05, rir.size - int(transition_s * sample_rate)
    )

    times_s, density = abel_normalized_echo_density_profile(
        rir, sample_rate, direct_index=0, window_ms=20.0
    )
    mixing_time_s = estimate_abel_mixing_time(
        times_s, density, threshold=0.9, minimum_sustain_ms=10.0
    )

    assert mixing_time_s is not None
    assert mixing_time_s == pytest.approx(transition_s, abs=0.03)
    with pytest.raises(ValueError, match="equal length"):
        estimate_abel_mixing_time([0.0], [0.8, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_abel_mixing_time([0.0, 0.0], [0.8, 1.0])
    with pytest.raises(ValueError, match="positive"):
        abel_normalized_echo_density_profile([1.0, 0.0], 8000, window_ms=0.0)


def test_echo_density_summary_is_json_safe_and_opt_in_to_analyze_rir():
    sample_rate = 8000
    rir = np.random.default_rng(23).normal(0.0, 0.1, sample_rate)
    rir[0] = 3.0

    summary = analyze_echo_density(rir, sample_rate, direct_index=0)
    result = analyze_rir(
        rir,
        sample_rate,
        direct_index=0,
        echo_density=True,
    )

    assert summary["policy"] == "puresound.abel_huang_echo_density.v1"
    assert summary["time_reference"] == "seconds_after_direct_sample"
    assert summary["mixing_time_s"] is not None
    assert result["echo_density"] == summary
    json.dumps(result, allow_nan=False)


def test_multiband_late_field_uses_cycle_aware_windows_and_stops_at_the_noise_floor():
    sample_rate = 8000
    time_s = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    rir = np.random.default_rng(31).normal(0.0, 1.0, time_s.size)
    rir *= np.exp(-math.log(1000.0) * time_s / 0.7)
    rir[0] = 5.0

    result = analyze_multiband_late_field(
        rir,
        sample_rate,
        direct_index=0,
        centers_hz=(63.0, 500.0, 1000.0),
    )

    assert result["policy"] == "puresound.multiband_late_field.v1"
    assert set(result["bands"]) == {"63", "500", "1000"}
    assert result["bands"]["63"]["analysis_window_ms"] > 80.0
    assert result["bands"]["500"]["analysis_window_ms"] == pytest.approx(20.0)
    assert result["bands"]["63"]["echo_density"][
        "normalized_density_at_ms"
    ]["50"] is not None
    assert result["bands"]["63"]["t20_s"] is not None
    json.dumps(result, allow_nan=False)

    noisy_rate = 16000
    noisy_time_s = np.arange(2 * noisy_rate, dtype=np.float64) / noisy_rate
    decay = np.random.default_rng(32).normal(0.0, 1.0, noisy_time_s.size)
    decay *= np.exp(-math.log(1000.0) * noisy_time_s / 0.5)
    noisy = decay + np.random.default_rng(33).normal(0.0, 0.002, noisy_time_s.size)
    noisy[0] = 5.0

    band = analyze_multiband_late_field(
        noisy,
        noisy_rate,
        direct_index=0,
        centers_hz=(1000.0,),
    )["bands"]["1000"]

    assert band["noise_floor"]["correction_applied"] is True
    assert band["echo_density_truncated_at_noise_intersection"] is True
    assert band["analysis_end_sample"] < noisy.size


def test_iacc_recovers_submillisecond_delay_and_separates_independent_channels():
    sample_rate = 16000
    delay_samples = 5
    left = np.random.default_rng(34).normal(0.0, 1.0, sample_rate)
    right = np.zeros_like(left)
    right[delay_samples:] = left[:-delay_samples]

    correlation = interaural_cross_correlation(
        left, right, sample_rate, direct_index=0, start_ms=0.0, end_ms=500.0
    )
    identical = analyze_binaural_iacc(left, left, sample_rate, direct_index=0)

    assert correlation["iacc"] == pytest.approx(1.0, abs=0.002)
    assert abs(correlation["lag_samples"]) == delay_samples
    assert identical["broadband"]["early"]["iacc"] == pytest.approx(1.0)
    assert identical["broadband"]["late"]["iacc"] == pytest.approx(1.0)
    assert identical["iacc_e4"] == pytest.approx(1.0)
    assert identical["iacc_l4"] == pytest.approx(1.0)
    json.dumps(identical, allow_nan=False)

    independent = interaural_cross_correlation(
        np.random.default_rng(35).normal(0.0, 1.0, 2 * sample_rate),
        np.random.default_rng(36).normal(0.0, 1.0, 2 * sample_rate),
        sample_rate,
        direct_index=0,
        start_ms=80.0,
        end_ms=None,
    )
    assert independent["valid"] is True
    assert independent["iacc"] < 0.04


def test_diffuse_field_coherence_target_pair_analysis_and_input_validation():
    sound_speed = 340.0
    spacing = 0.17
    frequencies = np.asarray([0.0, sound_speed / (2.0 * spacing)])
    target = diffuse_field_coherence(frequencies, spacing, sound_speed)
    signal = np.random.default_rng(37).normal(0.0, 1.0, 2 * 16000)

    result = analyze_array_spatial_coherence(
        signal,
        signal,
        16000,
        microphone_spacing_m=0.0,
        direct_index=0,
        start_ms=80.0,
    )

    assert target[0] == pytest.approx(1.0)
    assert target[1] == pytest.approx(0.0, abs=1e-12)
    assert result["valid"] is True
    assert result["complex_rmse"] < 1e-10
    assert result["measured_imaginary_rms"] < 1e-10
    json.dumps(result, allow_nan=False)

    with pytest.raises(ValueError, match="non-negative"):
        diffuse_field_coherence([100.0], -0.1)
    with pytest.raises(ValueError, match="greater than"):
        interaural_cross_correlation(
            [1.0, 0.0], [1.0, 0.0], 8000, start_ms=80.0, end_ms=20.0
        )
    short = analyze_array_spatial_coherence(
        [1.0, 0.0],
        [1.0, 0.0],
        8000,
        microphone_spacing_m=0.1,
        direct_index=0,
    )
    assert short["valid"] is False
    assert short["reason"] == "insufficient_late_samples"
