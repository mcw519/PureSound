import json

import numpy as np
import pytest

from puresound.audio.rir.calibration.synthetic_recovery import (
    SyntheticMeasurementPerturbation,
    SyntheticRecoveryObjectiveConfig,
    SyntheticRecoveryParameters,
    build_synthetic_recovery_observation,
    evaluate_synthetic_recovery,
    fit_synthetic_recovery_parameters,
    perturb_synthetic_recovery_measurement,
    render_synthetic_recovery_rir,
)


CENTERS = (500.0, 1000.0, 2000.0, 4000.0)
TRUTH = SyntheticRecoveryParameters(
    mixing_time_s=0.032,
    early_reflection_gain_db=-1.5,
    rt60_s_by_hz={500.0: 0.82, 1000.0: 0.68, 2000.0: 0.56, 4000.0: 0.44},
    late_gain_db_by_hz={500.0: -5.0, 1000.0: -4.0, 2000.0: -3.0, 4000.0: -2.0},
)
INITIAL = SyntheticRecoveryParameters(
    mixing_time_s=0.054,
    early_reflection_gain_db=4.0,
    rt60_s_by_hz={500.0: 0.45, 1000.0: 1.1, 2000.0: 0.85, 4000.0: 0.75},
    late_gain_db_by_hz={500.0: 1.0, 1000.0: -9.0, 2000.0: 2.0, 4000.0: -8.0},
)


def _observation(name, distance, seed):
    return build_synthetic_recovery_observation(
        name,
        16000,
        4096,
        distance,
        centers_hz=CENTERS,
        seed=seed,
    )


def test_synthetic_recovery_finds_hidden_shared_parameters_and_holdout():
    train = (_observation("train-a", 1.1, 10), _observation("train-b", 2.4, 11))
    targets = tuple(render_synthetic_recovery_rir(item, TRUTH) for item in train)

    fit = fit_synthetic_recovery_parameters(train, targets, INITIAL)

    assert fit.success is True
    assert fit.locally_full_rank is True
    assert fit.final_cost < 1e-16 * fit.initial_cost
    assert fit.parameters.to_vector() == pytest.approx(TRUTH.to_vector(), abs=1e-8)

    holdout = (_observation("holdout", 3.0, 99),)
    holdout_targets = (render_synthetic_recovery_rir(holdout[0], TRUTH),)
    initial_loss = evaluate_synthetic_recovery(holdout, holdout_targets, INITIAL)[0]
    fitted_loss = evaluate_synthetic_recovery(
        holdout,
        holdout_targets,
        fit.parameters,
    )[0]
    assert fitted_loss["total"] < 1e-8 * initial_loss["total"]
    json.dumps(fit.to_dict(), allow_nan=False)


def test_recovery_renderer_is_causal_and_rejects_mismatched_bands():
    observation = _observation("causal", 1.5, 3)
    rir = render_synthetic_recovery_rir(observation, TRUTH)
    assert np.count_nonzero(rir[: observation.direct_sample]) == 0
    assert (
        rir[observation.direct_sample]
        == observation.direct_component[observation.direct_sample]
    )

    invalid = SyntheticRecoveryParameters(
        mixing_time_s=0.03,
        early_reflection_gain_db=0.0,
        rt60_s_by_hz={500.0: 0.5},
        late_gain_db_by_hz={500.0: -3.0},
    )
    with pytest.raises(ValueError, match="centers"):
        render_synthetic_recovery_rir(observation, invalid)


def test_known_nuisance_is_corrected_while_noise_and_mismatch_remain():
    observation = _observation("perturbed", 2.0, 8)
    perturbation = SyntheticMeasurementPerturbation(
        snr_db=35.0,
        gain_error_db=1.25,
        latency_offset_samples=7,
        early_model_mismatch_fraction=0.08,
        late_model_mismatch_fraction=0.10,
        seed=44,
    )

    measurement = perturb_synthetic_recovery_measurement(
        observation,
        TRUTH,
        perturbation,
    )

    assert measurement.metadata["windowed_detected_raw_direct_sample"] == (
        observation.direct_sample + 7
    )
    assert measurement.metadata["windowed_detected_corrected_direct_sample"] == (
        observation.direct_sample
    )
    assert np.linalg.norm(measurement.mismatch_component) > 0.0
    assert np.linalg.norm(measurement.noise_component_raw) > 0.0
    assert not np.array_equal(measurement.corrected_rir, measurement.clean_rir)
    json.dumps(measurement.to_dict(), allow_nan=False)


def test_noise_aware_multiterm_objective_recovers_perturbed_parameters():
    observations = (
        _observation("robust-a", 1.1, 10),
        _observation("robust-b", 2.4, 11),
    )
    perturbations = (
        SyntheticMeasurementPerturbation(35.0, 1.2, 7, 0.08, 0.10, 100),
        SyntheticMeasurementPerturbation(33.0, -0.8, -4, 0.10, 0.12, 101),
    )
    targets = tuple(
        perturb_synthetic_recovery_measurement(
            observation, TRUTH, perturbation
        ).corrected_rir
        for observation, perturbation in zip(observations, perturbations)
    )
    objective = SyntheticRecoveryObjectiveConfig(
        mode="m4_multiterm_v2",
        waveform_weight=1.0,
        early_waveform_weight=1.0,
        broadband_decay_weight=0.25,
        octave_decay_weight=0.25,
        noise_margin_db=20.0,
    )

    fit = fit_synthetic_recovery_parameters(
        observations,
        targets,
        INITIAL,
        objective=objective,
        maximum_evaluations=100,
    )

    assert fit.success is True
    assert fit.objective.mode == "m4_multiterm_v2"
    assert abs(fit.parameters.mixing_time_s - TRUTH.mixing_time_s) <= 0.001
    assert (
        max(
            abs(fit.parameters.rt60_s_by_hz[center] - TRUTH.rt60_s_by_hz[center])
            / TRUTH.rt60_s_by_hz[center]
            for center in CENTERS
        )
        <= 0.08
    )
