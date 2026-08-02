import json

import numpy as np
import pytest

from puresound.audio.rir_calibration import (
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
    causality_penalty,
    decay_growth_penalty,
    spatial_coherence_distance,
)


def _stereo_rir(sample_rate=16000, sample_count=4096):
    rng = np.random.default_rng(20260801)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    envelope = np.exp(-6.0 * time)
    common = rng.normal(size=sample_count) * envelope * 0.025
    independent = rng.normal(size=sample_count) * envelope * 0.008
    rir = np.zeros((2, sample_count), dtype=np.float64)
    rir[0, 32:] = common[: sample_count - 32]
    rir[1, 35:] = 0.8 * common[: sample_count - 35] + independent[: sample_count - 35]
    rir[0, 32] = 1.0
    rir[1, 35] = 0.9
    return rir, (32, 35)


def test_identical_rirs_have_zero_data_loss_and_strict_json():
    rir, direct = _stereo_rir()
    weights = CalibrationLossWeights(decay_regularization=0.0)

    report = analyze_rir_calibration_loss(
        rir,
        rir.copy(),
        16000,
        measured_direct_samples=direct,
        synthetic_direct_samples=direct,
        weights=weights,
    )

    assert report.total == pytest.approx(0.0, abs=1e-12)
    assert all(
        value == pytest.approx(0.0, abs=1e-12) for value in report.terms.values()
    )
    assert report.diagnostics["spatial_coherence"]["evaluable"] is True
    json.dumps(report.to_dict(), allow_nan=False)


def test_delay_and_coloration_increase_calibration_loss():
    measured, direct = _stereo_rir()
    synthetic = np.zeros_like(measured)
    synthetic[:, 8:] = measured[:, :-8]
    synthetic[1] *= 0.6

    report = analyze_rir_calibration_loss(
        measured,
        synthetic,
        16000,
        measured_direct_samples=direct,
        synthetic_direct_samples=(40, 43),
        weights=CalibrationLossWeights(decay_regularization=0.0),
    )

    assert report.total > 0.0
    assert report.terms["arrival_timing"] == pytest.approx(0.5)
    assert report.terms["multiresolution_stft"] > 0.0
    assert report.terms["octave_acoustics"] > 0.0


def test_causality_and_spatial_terms_are_independently_auditable():
    measured, direct = _stereo_rir()
    noncausal = measured.copy()
    noncausal[:, 4] = 0.25
    causality = causality_penalty(noncausal, direct)
    assert causality["distance"] > 0.0

    decorrelated = measured.copy()
    rng = np.random.default_rng(7)
    decorrelated[1, 35:] += rng.normal(size=decorrelated.shape[1] - 35) * 0.05
    spatial = spatial_coherence_distance(
        measured,
        decorrelated,
        16000,
        direct,
        direct,
    )
    assert spatial["evaluable"] is True
    assert spatial["distance"] > 0.0


def test_mono_spatial_loss_is_explicitly_not_evaluable():
    rir, direct = _stereo_rir()
    report = spatial_coherence_distance(
        rir[0],
        rir[0],
        16000,
        direct[:1],
        direct[:1],
    )
    assert report["distance"] == 0.0
    assert report["evaluable"] is False
    assert report["reason"] == "fewer_than_two_synchronized_receivers"


def test_invalid_loss_contracts_are_rejected():
    rir, direct = _stereo_rir()
    with pytest.raises(ValueError, match="weights"):
        CalibrationLossWeights(causality=-1.0)
    with pytest.raises(ValueError, match="direct_samples"):
        decay_growth_penalty(rir, 16000, direct[:1])
    with pytest.raises(ValueError, match="equal channel count"):
        analyze_rir_calibration_loss(rir, rir[:1], 16000)
