import json
import math

import numpy as np
import pytest

from puresound.audio.rir.render.multiband_fdn import (
    analyze_fdn_coloration,
    delay_proportional_loop_gains,
    design_multiband_fdn,
    fdn_filterbank_power_response,
    randomized_hadamard_matrix,
    render_multiband_fdn,
    render_multiband_fdn_impulse,
    select_prime_delay_lengths,
)
from puresound.audio.rir.metrics import analyze_echo_density, estimate_decay_time


def _render_scalar_reference(design, excitation):
    signal = np.asarray(excitation, dtype=np.float64)
    band_count = design.band_count
    delay_count = design.delay_line_count
    buffers = [
        np.zeros((band_count, delay), dtype=np.float64)
        for delay in design.delay_lengths_samples
    ]
    pointers = np.zeros(delay_count, dtype=np.int64)
    raw = np.zeros((band_count, signal.size), dtype=np.float64)
    for sample_index, sample in enumerate(signal):
        delayed = np.empty((band_count, delay_count), dtype=np.float64)
        for delay_index, buffer in enumerate(buffers):
            delayed[:, delay_index] = buffer[:, pointers[delay_index]]
        raw[:, sample_index] = np.sum(design.output_matrix * delayed, axis=1)
        writes = (delayed * design.loop_gains) @ design.feedback_matrix.T
        writes += (
            design.band_weights[:, None]
            * sample
            * design.input_vector[None, :]
        )
        for delay_index, buffer in enumerate(buffers):
            pointer = int(pointers[delay_index])
            buffer[:, pointer] = writes[:, delay_index]
            pointers[delay_index] = (pointer + 1) % buffer.shape[1]
    return raw


def test_fdn_building_blocks_are_seeded_and_exact():
    # Delay lengths: distinct, pairwise coprime primes, reproducible per seed.
    first = select_prime_delay_lengths(16000, 16, 3.0, 18.0, seed=11)
    assert first == select_prime_delay_lengths(16000, 16, 3.0, 18.0, seed=11)
    assert first != select_prime_delay_lengths(16000, 16, 3.0, 18.0, seed=12)
    assert len(first) == len(set(first)) == 16
    assert all(math.gcd(a, b) == 1 for index, a in enumerate(first) for b in first[index + 1 :])

    # Feedback matrix: exactly orthogonal, reproducible per seed.
    matrix = randomized_hadamard_matrix(16, seed=13)
    assert np.array_equal(matrix, randomized_hadamard_matrix(16, seed=13))
    assert not np.array_equal(matrix, randomized_hadamard_matrix(16, seed=14))
    assert np.linalg.norm(matrix.T @ matrix - np.eye(16), ord=2) < 1e-14
    with pytest.raises(ValueError, match="power of two"):
        randomized_hadamard_matrix(12)

    # Loop gain: a delay line reaches -60 dB after RT60 seconds of traversals.
    sample_rate, delay, rt60_s = 16000, 160, 0.5
    gain = delay_proportional_loop_gains([delay], sample_rate, rt60_s)[0]
    assert 0.0 < gain < 1.0
    assert gain ** (sample_rate * rt60_s / delay) == pytest.approx(1e-3, rel=1e-12)


def test_fdn_design_is_contractive_json_safe_and_rejects_invalid_input():
    design = design_multiband_fdn(
        8000,
        {500.0: 0.4, 1000.0: 0.35, 2000.0: 0.3},
        target_mixing_time_s=0.024,
        delay_line_count=16,
        seed=15,
    )
    metadata = design.to_dict(include_coefficients=True)

    assert metadata["policy"] == "puresound.multiband_fdn.v2"
    assert metadata["delays_are_distinct_primes"] is True
    assert metadata["feedback_orthogonality_error_2norm"] < 1e-12
    assert metadata["band_weight_energy_sum"] == pytest.approx(1.0)
    assert all(
        band["feedback_operator_2norm"] < 1.0
        for band in metadata["bands"].values()
    )
    json.dumps(metadata, allow_nan=False)

    with pytest.raises(ValueError, match="power of two"):
        design_multiband_fdn(8000, {1000.0: 0.4}, delay_line_count=12)
    with pytest.raises(ValueError, match="positive"):
        design_multiband_fdn(8000, {1000.0: 0.0})
    with pytest.raises(ValueError, match="one-dimensional"):
        render_multiband_fdn(
            design_multiband_fdn(8000, {1000.0: 0.4}), np.zeros((2, 100))
        )


def test_fdn_render_is_causal_deterministic_and_matches_the_sample_recurrence():
    design = design_multiband_fdn(
        8000,
        {500.0: 0.4, 1000.0: 0.35},
        seed=16,
    )
    first = render_multiband_fdn_impulse(design, 0.8)
    repeated = render_multiband_fdn_impulse(design, 0.8)
    zero = render_multiband_fdn(design, np.zeros(6400))
    first_nonzero = int(np.flatnonzero(np.abs(first.rir) > 1e-14)[0])

    assert np.array_equal(first.rir, repeated.rir)
    assert np.all(np.isfinite(first.rir))
    assert np.all(first.rir[: min(design.delay_lengths_samples)] == 0.0)
    assert first_nonzero >= min(design.delay_lengths_samples)
    assert np.array_equal(zero.rir, np.zeros(6400))
    assert set(first.band_rirs) == {500.0, 1000.0}

    # The block renderer is an optimization of the per-sample recurrence.
    small = design_multiband_fdn(
        8000,
        {500.0: 0.4, 1000.0: 0.35},
        delay_line_count=4,
        delay_range_ms=(1.0, 2.5),
        seed=116,
    )
    excitation = np.random.default_rng(117).normal(size=257)
    rendered = render_multiband_fdn(small, excitation)
    blocked_raw = np.vstack(
        [rendered.raw_band_rirs[center] for center in small.centers_hz]
    )
    scalar_raw = _render_scalar_reference(small, excitation)

    np.testing.assert_allclose(blocked_raw, scalar_raw, rtol=1e-13, atol=1e-14)


def test_fdn_high_bands_follow_rt60_and_build_density():
    sample_rate = 8000
    targets = {500.0: 0.45, 1000.0: 0.38, 2000.0: 0.32}
    design = design_multiband_fdn(
        sample_rate,
        targets,
        target_mixing_time_s=0.024,
        seed=17,
    )
    rendered = render_multiband_fdn_impulse(design, 1.2)

    for center_hz, target_rt60_s in targets.items():
        band = rendered.band_rirs[center_hz]
        estimate = estimate_decay_time(
            band,
            sample_rate,
            -5.0,
            -25.0,
            direct_index=0,
        )
        density = analyze_echo_density(
            band,
            sample_rate,
            direct_index=0,
            window_ms=20.0,
            hop_ms=2.0,
        )
        assert estimate is not None
        assert estimate.rt60_s == pytest.approx(target_rt60_s, rel=0.12)
        assert density["mixing_time_s"] is not None
        assert density["mixing_time_s"] <= 0.05


def test_fdn_filterbank_covers_dc_to_nyquist_and_coloration_report_is_strict_json():
    frequencies, power = fdn_filterbank_power_response(
        16000,
        (500.0, 1000.0, 2000.0, 4000.0),
        filter_order=4,
    )
    analysis = (frequencies >= 20.0) & (frequencies <= 7900.0)
    response_db = 10.0 * np.log10(np.maximum(power[analysis], 1e-20))

    assert float(np.min(response_db)) > -3.1
    assert float(np.max(response_db)) < 0.2
    for frequency in (6000.0, 7000.0, 7900.0):
        index = int(np.argmin(np.abs(frequencies - frequency)))
        assert 10.0 * np.log10(power[index]) > -0.5

    design = design_multiband_fdn(
        16000,
        {500.0: 0.5, 1000.0: 0.45, 2000.0: 0.4, 4000.0: 0.35},
        seed=19,
    )
    rendered = render_multiband_fdn_impulse(design, 0.5)
    assert rendered.filterbank_metadata["coverage_hz"] == [0.0, 8000.0]

    report = analyze_fdn_coloration(rendered.rir, 16000, design.centers_hz)
    assert set(report["bands"]) == {"500", "1000", "2000", "4000"}
    assert all(
        0.0 < band["spectral_flatness"] <= 1.0
        for band in report["bands"].values()
    )
    json.dumps(report, allow_nan=False)
