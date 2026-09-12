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
    schroeder_decay_db,
    spectral_tilt_db_per_octave,
    valid_octave_centers,
)


def test_drr_matches_energy_ratio_and_legacy_entrypoint():
    rir = torch.zeros(1, 400)
    rir[0, 10] = 1.0
    rir[0, 100] = 0.3
    expected = 10.0 * math.log10(1.0 / 0.09)

    measured = compute_drr_db(rir, sample_rate=16000, direct_window_ms=2.5)

    assert measured == pytest.approx(expected, abs=1e-5)
    assert legacy_compute_drr_db(rir, 16000, 2.5) == pytest.approx(measured)


def test_clarity_uses_requested_early_late_boundary():
    rir = np.zeros(2000, dtype=np.float64)
    rir[10] = 1.0
    rir[10 + 400] = 0.5  # 25 ms at 16 kHz: early for C50.
    rir[10 + 1200] = 0.25  # 75 ms: late for C50, early for C80.

    assert clarity_db(rir, 16000, 50.0) == pytest.approx(
        10.0 * math.log10(1.25 / 0.0625)
    )
    assert math.isinf(clarity_db(rir, 16000, 80.0))


def test_decay_estimates_exponential_rt60():
    sample_rate = 16000
    target_rt60 = 0.5
    time_s = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rir = np.exp(-math.log(1000.0) * time_s / target_rt60)

    edt = estimate_decay_time(rir, sample_rate, 0.0, -10.0)
    t20 = estimate_decay_time(rir, sample_rate, -5.0, -25.0)
    t30 = estimate_decay_time(rir, sample_rate, -5.0, -35.0)

    assert edt is not None and edt.rt60_s == pytest.approx(target_rt60, rel=2e-3)
    assert t20 is not None and t20.rt60_s == pytest.approx(target_rt60, rel=2e-3)
    assert t30 is not None and t30.rt60_s == pytest.approx(target_rt60, rel=2e-3)
    assert t30.r_squared > 0.999


def test_lundeby_correction_recovers_decay_from_stationary_noise():
    sample_rate = 16000
    target_rt60 = 0.5
    time_s = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    clean = np.exp(-math.log(1000.0) * time_s / target_rt60)
    noise = np.random.default_rng(2).normal(0.0, 0.003, time_s.size)
    rir = clean + noise

    estimate = estimate_noise_floor_lundeby(rir, sample_rate, direct_index=0)
    raw = estimate_decay_time(
        rir,
        sample_rate,
        -5.0,
        -35.0,
        direct_index=0,
        noise_compensation=False,
    )
    corrected = estimate_decay_time(
        rir,
        sample_rate,
        -5.0,
        -35.0,
        direct_index=0,
        noise_compensation=True,
    )
    curve, curve_estimate = noise_compensated_schroeder_decay_db(
        rir, sample_rate, direct_index=0
    )

    assert estimate.correction_applied
    assert estimate.dynamic_range_db == pytest.approx(49.8, abs=1.0)
    assert estimate.intersection_time_s == pytest.approx(0.42, abs=0.03)
    assert curve_estimate == estimate
    assert np.all(np.diff(curve) <= 1e-12)
    assert raw is not None and raw.rt60_s > 1.0
    assert corrected is not None
    assert corrected.rt60_s == pytest.approx(target_rt60, rel=0.03)
    assert corrected.r_squared > 0.99


def test_lundeby_does_not_mistake_clean_decay_tail_for_noise():
    sample_rate = 16000
    time_s = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rir = np.exp(-math.log(1000.0) * time_s / 0.5)

    estimate = estimate_noise_floor_lundeby(rir, sample_rate)

    assert not estimate.correction_applied
    assert estimate.reason == "tail_is_not_stationary_noise"
    assert estimate.truncation_sample == rir.size


def test_decay_failure_is_explicit_for_anechoic_rir():
    rir = np.zeros(1600, dtype=np.float64)
    rir[20] = 1.0

    assert estimate_decay_time(rir, 16000, -5.0, -35.0) is None
    curve = schroeder_decay_db(np.zeros(32))
    assert np.isnan(curve).all()


def test_flat_impulse_has_zero_spectral_tilt():
    rir = np.zeros(1024, dtype=np.float64)
    rir[0] = 1.0

    tilt = spectral_tilt_db_per_octave(rir, sample_rate=16000)

    assert tilt == pytest.approx(0.0, abs=1e-10)


def test_analyze_rir_is_strict_json_and_includes_octave_bands():
    sample_rate = 16000
    target_rt60 = 0.4
    time_s = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rir = np.exp(-math.log(1000.0) * time_s / target_rt60)

    result = analyze_rir(
        rir,
        sample_rate,
        octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    )

    assert result["t30_s"] == pytest.approx(target_rt60, rel=2e-3)
    assert result["noise_floor"]["correction_applied"] is False
    assert set(result["octave_bands"]) == {"63", "125", "250", "500", "1000", "2000", "4000"}
    json.dumps(result, allow_nan=False)


def test_analyze_anechoic_rir_uses_null_for_unbounded_ratios():
    rir = np.zeros(1000, dtype=np.float64)
    rir[10] = 1.0

    result = analyze_rir(rir, 16000)

    assert result["drr_db"] is None
    assert result["c50_db"] is None
    assert result["t30"] is None
    json.dumps(result, allow_nan=False)


def test_metrics_reject_real_multichannel_input():
    with pytest.raises(ValueError, match="one channel"):
        analyze_rir(np.zeros((2, 100)), 16000)


def test_octave_centers_stop_below_nyquist():
    assert valid_octave_centers(8000) == [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]


def test_abel_echo_density_is_scale_invariant_and_gaussian_is_unity():
    sample_rate = 8000
    rir = np.random.default_rng(21).normal(0.0, 1.0, 4 * sample_rate)

    times_s, density = abel_normalized_echo_density_profile(
        rir,
        sample_rate,
        direct_index=0,
        window_ms=40.0,
        hop_ms=2.0,
    )
    scaled_times_s, scaled_density = abel_normalized_echo_density_profile(
        0.013 * rir,
        sample_rate,
        direct_index=0,
        window_ms=40.0,
        hop_ms=2.0,
    )

    assert np.array_equal(times_s, scaled_times_s)
    assert np.array_equal(density, scaled_density)
    assert np.median(density) == pytest.approx(1.0, abs=0.04)


def test_abel_echo_density_keeps_sparse_impulse_train_below_threshold():
    sample_rate = 8000
    rir = np.zeros(2 * sample_rate, dtype=np.float64)
    rir[::80] = 1.0

    times_s, density = abel_normalized_echo_density_profile(
        rir,
        sample_rate,
        direct_index=0,
        window_ms=40.0,
    )

    assert density.size == times_s.size
    assert np.max(density) < 0.1
    assert estimate_abel_mixing_time(times_s, density) is None


def test_abel_mixing_time_detects_sparse_to_dense_transition():
    sample_rate = 8000
    transition_s = 0.15
    rir = np.zeros(2 * sample_rate, dtype=np.float64)
    rir[0] = 5.0
    rir[80 : int(transition_s * sample_rate) : 160] = 0.2
    tail = np.random.default_rng(22).normal(
        0.0,
        0.05,
        rir.size - int(transition_s * sample_rate),
    )
    rir[int(transition_s * sample_rate) :] = tail

    times_s, density = abel_normalized_echo_density_profile(
        rir,
        sample_rate,
        direct_index=0,
        window_ms=20.0,
    )
    mixing_time_s = estimate_abel_mixing_time(
        times_s,
        density,
        threshold=0.9,
        minimum_sustain_ms=10.0,
    )

    assert mixing_time_s is not None
    assert mixing_time_s == pytest.approx(transition_s, abs=0.03)


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


def test_abel_mixing_time_validates_profile_contract():
    with pytest.raises(ValueError, match="equal length"):
        estimate_abel_mixing_time([0.0], [0.8, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_abel_mixing_time([0.0, 0.0], [0.8, 1.0])
    with pytest.raises(ValueError, match="positive"):
        abel_normalized_echo_density_profile([1.0, 0.0], 8000, window_ms=0.0)


def test_multiband_late_field_uses_cycle_aware_windows_and_strict_json():
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


def test_multiband_late_field_stops_echo_density_at_reliable_noise_floor():
    sample_rate = 16000
    time_s = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    decay = np.random.default_rng(32).normal(0.0, 1.0, time_s.size)
    decay *= np.exp(-math.log(1000.0) * time_s / 0.5)
    noise = np.random.default_rng(33).normal(0.0, 0.002, time_s.size)
    rir = decay + noise
    rir[0] = 5.0

    result = analyze_multiband_late_field(
        rir,
        sample_rate,
        direct_index=0,
        centers_hz=(1000.0,),
    )
    band = result["bands"]["1000"]

    assert band["noise_floor"]["correction_applied"] is True
    assert band["echo_density_truncated_at_noise_intersection"] is True
    assert band["analysis_end_sample"] < rir.size


def test_iacc_recovers_submillisecond_delay_and_identical_binaural_bands():
    sample_rate = 16000
    delay_samples = 5
    left = np.random.default_rng(34).normal(0.0, 1.0, sample_rate)
    right = np.zeros_like(left)
    right[delay_samples:] = left[:-delay_samples]

    correlation = interaural_cross_correlation(
        left,
        right,
        sample_rate,
        direct_index=0,
        start_ms=0.0,
        end_ms=500.0,
    )
    identical = analyze_binaural_iacc(
        left,
        left,
        sample_rate,
        direct_index=0,
    )

    assert correlation["iacc"] == pytest.approx(1.0, abs=0.002)
    assert abs(correlation["lag_samples"]) == delay_samples
    assert identical["broadband"]["early"]["iacc"] == pytest.approx(1.0)
    assert identical["broadband"]["late"]["iacc"] == pytest.approx(1.0)
    assert identical["iacc_e4"] == pytest.approx(1.0)
    assert identical["iacc_l4"] == pytest.approx(1.0)
    json.dumps(identical, allow_nan=False)


def test_iacc_distinguishes_independent_late_channels():
    sample_rate = 16000
    first = np.random.default_rng(35).normal(0.0, 1.0, 2 * sample_rate)
    second = np.random.default_rng(36).normal(0.0, 1.0, 2 * sample_rate)

    result = interaural_cross_correlation(
        first,
        second,
        sample_rate,
        direct_index=0,
        start_ms=80.0,
        end_ms=None,
    )

    assert result["valid"] is True
    assert result["iacc"] < 0.04


def test_diffuse_field_coherence_target_and_pair_analysis():
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


def test_spatial_metrics_validate_geometry_and_empty_windows():
    with pytest.raises(ValueError, match="non-negative"):
        diffuse_field_coherence([100.0], -0.1)
    with pytest.raises(ValueError, match="greater than"):
        interaural_cross_correlation(
            [1.0, 0.0],
            [1.0, 0.0],
            8000,
            start_ms=80.0,
            end_ms=20.0,
        )
    result = analyze_array_spatial_coherence(
        [1.0, 0.0],
        [1.0, 0.0],
        8000,
        microphone_spacing_m=0.1,
        direct_index=0,
    )
    assert result["valid"] is False
    assert result["reason"] == "insufficient_late_samples"
