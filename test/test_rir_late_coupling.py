import json

import numpy as np
import pytest

from puresound.audio.rir_late_coupling import (
    PATH_EVENT_FDN_COUPLING_POLICY,
    couple_path_event_rir_with_fdn,
    equal_power_transition_weights,
)


def _path_event_fixture(sample_rate=8000, duration_s=0.8):
    signal = np.zeros(int(round(sample_rate * duration_s)), dtype=np.float64)
    direct = 80
    for offset, amplitude in (
        (0, 1.0),
        (31, -0.45),
        (79, 0.32),
        (151, -0.24),
        (239, 0.18),
        (397, -0.12),
    ):
        if direct + offset < signal.size:
            signal[direct + offset] = amplitude
    return signal, direct


def test_equal_power_weights_are_complementary_and_exact_at_endpoints():
    early, late = equal_power_transition_weights(100, 20, 60)

    assert np.array_equal(early[:21], np.ones(21))
    assert np.array_equal(late[:21], np.zeros(21))
    assert np.array_equal(early[61:], np.zeros(39))
    assert np.array_equal(late[61:], np.ones(39))
    assert np.max(np.abs(early**2 + late**2 - 1.0)) < 5e-16


def test_coupling_preserves_early_samples_and_post_transition_energy():
    signal, direct = _path_event_fixture()
    result = couple_path_event_rir_with_fdn(
        signal,
        sample_rate=8000,
        direct_sample=direct,
        target_rt60_s_by_hz={500.0: 0.45, 1000.0: 0.38, 2000.0: 0.32},
        mixing_time_s=0.024,
        transition_duration_s=0.016,
        seed=21,
    )
    metadata = result.metadata
    start = metadata["transition_start_sample"]
    end = metadata["transition_end_sample"]

    assert metadata["policy"] == PATH_EVENT_FDN_COUPLING_POLICY
    assert np.array_equal(result.rir[: start + 1], signal[: start + 1])
    assert np.count_nonzero(result.diffuse_component[: start + 1]) == 0
    assert np.count_nonzero(result.coherent_component[end + 1 :]) == 0
    assert metadata["pre_transition_max_abs_error"] == 0.0
    assert metadata["energy"]["relative_energy_error"] < 1e-12
    assert np.all(np.isfinite(result.rir))
    assert np.count_nonzero(result.rir[:direct]) == 0
    json.dumps(metadata, allow_nan=False)


def test_coupling_is_same_seed_deterministic_and_seed_sensitive():
    signal, direct = _path_event_fixture()
    kwargs = {
        "sample_rate": 8000,
        "direct_sample": direct,
        "target_rt60_s_by_hz": {500.0: 0.4, 1000.0: 0.35, 2000.0: 0.3},
        "mixing_time_s": 0.024,
        "transition_duration_s": 0.016,
    }
    first = couple_path_event_rir_with_fdn(signal, seed=22, **kwargs)
    repeated = couple_path_event_rir_with_fdn(signal, seed=22, **kwargs)
    changed = couple_path_event_rir_with_fdn(signal, seed=23, **kwargs)
    start = first.metadata["transition_start_sample"]

    assert np.array_equal(first.rir, repeated.rir)
    assert not np.array_equal(first.rir[start + 1 :], changed.rir[start + 1 :])
    assert np.array_equal(first.rir[: start + 1], changed.rir[: start + 1])


def test_long_rt60_extrapolates_more_energy_than_short_rt60():
    signal, direct = _path_event_fixture(duration_s=1.0)
    common = {
        "sample_rate": 8000,
        "direct_sample": direct,
        "mixing_time_s": 0.024,
        "transition_duration_s": 0.016,
        "seed": 24,
    }
    short = couple_path_event_rir_with_fdn(
        signal,
        target_rt60_s_by_hz={500.0: 0.3, 1000.0: 0.3, 2000.0: 0.3},
        **common,
    )
    long = couple_path_event_rir_with_fdn(
        signal,
        target_rt60_s_by_hz={500.0: 1.6, 1000.0: 1.6, 2000.0: 1.6},
        **common,
    )
    short_energy = short.metadata["energy"]
    long_energy = long.metadata["energy"]

    assert short_energy["target_original_post_transition_energy"] > short_energy[
        "finite_original_post_transition_energy"
    ]
    assert long_energy["target_original_post_transition_energy"] > short_energy[
        "target_original_post_transition_energy"
    ]
    assert long.metadata["path_tail_extrapolation"]["extrapolated_energy"] > (
        short.metadata["path_tail_extrapolation"]["extrapolated_energy"]
    )


def test_coupling_rejects_short_or_invalid_transition():
    signal, direct = _path_event_fixture(duration_s=0.04)
    with pytest.raises(ValueError, match="too short"):
        couple_path_event_rir_with_fdn(
            signal,
            sample_rate=8000,
            direct_sample=direct,
            target_rt60_s_by_hz={1000.0: 0.4},
            mixing_time_s=0.03,
            transition_duration_s=0.03,
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        couple_path_event_rir_with_fdn(
            np.zeros((2, 1000)),
            sample_rate=8000,
            direct_sample=10,
            target_rt60_s_by_hz={1000.0: 0.4},
        )
