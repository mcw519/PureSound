"""Inverse calibration: synthetic parameter recovery, the path-event/FDN
parameter profile, and the identifiability and spatial-selection checks."""

import json
from dataclasses import replace

import numpy as np
import pytest

from puresound.audio.rir.calibration.inverse_m4 import (
    M4_PROFILE_CONVERGENCE_POLICY,
    M4InverseObservation,
    M4InverseParameters,
    M4ProfileObjectiveConfig,
    fit_m4_parameter_profile,
    render_m4_inverse_observation,
)
from puresound.audio.rir.calibration.inverse_m5 import (
    analyze_local_identifiability,
    select_spatial_calibration_candidate,
)
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


def test_an_unidentifiable_recovery_reports_a_finite_condition_number():
    """A parameter with no effect on the response must read as rank deficient.

    The condition number of that fit is unbounded, but the report has to stay
    strict JSON, so it saturates at the largest finite float.
    """
    centers = (500.0, 1000.0)
    observation = build_synthetic_recovery_observation(
        "dead-band", 16000, 8000, 2.0, centers_hz=centers, seed=1
    )
    late = dict(observation.late_basis_by_hz)
    late[1000.0] = np.zeros_like(late[1000.0])
    observation = replace(observation, late_basis_by_hz=late)
    truth = SyntheticRecoveryParameters(
        0.03, 1.0, {500.0: 0.5, 1000.0: 0.6}, {500.0: -3.0, 1000.0: -2.0}
    )
    initial = SyntheticRecoveryParameters(
        0.04, 0.0, {500.0: 0.7, 1000.0: 0.7}, {500.0: -6.0, 1000.0: -6.0}
    )

    fit = fit_synthetic_recovery_parameters(
        (observation,),
        (render_synthetic_recovery_rir(observation, truth),),
        initial,
        maximum_evaluations=20,
    )

    assert fit.locally_full_rank is False
    assert np.isfinite(fit.scaled_jacobian_condition_number)
    json.dumps(fit.to_dict(), allow_nan=False)


def test_recovery_renderer_is_causal_and_perturbation_corrects_only_known_nuisance():
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

    # Gain and latency are known nuisances and are corrected; noise and model
    # mismatch are not and must remain in the corrected measurement.
    perturbed = _observation("perturbed", 2.0, 8)
    measurement = perturb_synthetic_recovery_measurement(
        perturbed,
        TRUTH,
        SyntheticMeasurementPerturbation(
            snr_db=35.0,
            gain_error_db=1.25,
            latency_offset_samples=7,
            early_model_mismatch_fraction=0.08,
            late_model_mismatch_fraction=0.10,
            seed=44,
        ),
    )
    assert measurement.metadata["windowed_detected_raw_direct_sample"] == (
        perturbed.direct_sample + 7
    )
    assert measurement.metadata["windowed_detected_corrected_direct_sample"] == (
        perturbed.direct_sample
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


# --- path-event / FDN parameter profile -----------------------------------

PROFILE_TRUTH = M4InverseParameters(
    mixing_time_s=0.024,
    coherent_reflection_gain_db=-1.5,
    target_rt60_s_by_hz={500.0: 0.68, 1000.0: 0.54, 2000.0: 0.42},
)
PROFILE_INITIAL = M4InverseParameters(
    mixing_time_s=0.028,
    coherent_reflection_gain_db=3.0,
    target_rt60_s_by_hz={500.0: 0.40, 1000.0: 0.85, 2000.0: 0.72},
)


def _profile_observation(name="fixture", seed=10):
    signal = np.zeros(3200, dtype=np.float64)
    direct = 40
    for offset, amplitude in (
        (0, 1.0),
        (17, -0.45),
        (43, 0.34),
        (71, -0.25),
        (121, 0.18),
        (203, -0.12),
    ):
        signal[direct + offset] = amplitude
    return M4InverseObservation(
        observation_id=name,
        path_event_rir=signal,
        sample_rate=8000,
        direct_sample=direct,
        fdn_seed=seed,
        delay_line_count=4,
    )


def test_profile_recovers_exact_parameters_and_rejects_a_truncated_fit():
    """The convergence verdict must still be able to say no.

    ``converged`` replaces ``least_squares``'s ``success`` flag because a rough
    objective can exhaust the budget at a point no further optimization
    improves. The replacement is only worth anything if a genuinely truncated
    fit -- one still descending when the budget ran out -- is still rejected.
    """
    observations = (_profile_observation("a", 10), _profile_observation("b", 11))
    targets = tuple(
        render_m4_inverse_observation(observation, PROFILE_TRUTH).rir
        for observation in observations
    )

    def fit(maximum_evaluations):
        return fit_m4_parameter_profile(
            observations,
            targets,
            PROFILE_INITIAL,
            (PROFILE_TRUTH.mixing_time_s,),
            maximum_evaluations=maximum_evaluations,
        )

    settled_profile = fit(50)
    settled = settled_profile.best
    assert settled.success is True
    assert settled.locally_full_rank is True
    assert settled.cost < 1e-14
    assert settled.parameters.continuous_vector() == pytest.approx(
        PROFILE_TRUTH.continuous_vector(), abs=2e-6
    )
    assert settled.converged is True
    assert settled.to_dict()["convergence"]["converged"] is True
    json.dumps(settled_profile.to_dict(), allow_nan=False)

    truncated = fit(2).best
    assert truncated.converged is False, (
        "a fit stopped two evaluations in is still descending and must not "
        "count as a stable minimum"
    )
    record = truncated.to_dict()["convergence"]
    assert record["policy"] == M4_PROFILE_CONVERGENCE_POLICY
    assert record["initial_solve"]["solver_declared_success"] is False
    assert (
        record["restart_solve"]["relative_improvement"]
        > record["stable_minimum_relative_tolerance"]
    )
    assert settled.cost < truncated.cost


def test_profile_render_is_causal_and_the_contract_rejects_invalid_inputs():
    observation = _profile_observation()
    result = render_m4_inverse_observation(observation, PROFILE_TRUTH)
    transition_start = result.metadata["transition_start_sample"]

    assert np.count_nonzero(result.rir[: observation.direct_sample]) == 0
    # The late-field target cannot reach back into the coherent early part.
    shorter_decay = M4InverseParameters(
        mixing_time_s=PROFILE_TRUTH.mixing_time_s,
        coherent_reflection_gain_db=PROFILE_TRUTH.coherent_reflection_gain_db,
        target_rt60_s_by_hz={center: 0.3 for center in PROFILE_TRUTH.centers_hz},
    )
    assert np.array_equal(
        result.rir[: transition_start + 1],
        render_m4_inverse_observation(observation, shorter_decay).rir[
            : transition_start + 1
        ],
    )
    assert result.metadata["pre_transition_max_abs_error"] == 0.0

    objective = M4ProfileObjectiveConfig(early_window_ms=10.0)
    assert objective.to_dict()["early_window_ms"] == 10.0
    with pytest.raises(ValueError, match="energy and octave"):
        M4ProfileObjectiveConfig(early_window_ms=0.0)
    with pytest.raises(ValueError, match="positive finite"):
        fit_m4_parameter_profile(
            (observation,),
            (np.zeros(3200),),
            PROFILE_INITIAL,
            (0.0,),
        )
    above_nyquist = M4InverseParameters(
        mixing_time_s=0.024,
        coherent_reflection_gain_db=0.0,
        target_rt60_s_by_hz={5000.0: 0.5},
    )
    with pytest.raises(ValueError, match="Nyquist"):
        render_m4_inverse_observation(observation, above_nyquist)


# --- identifiability and spatial selection --------------------------------


def test_identifiability_keeps_priority_column_and_rejects_duplicate():
    jacobian = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )

    report = analyze_local_identifiability(
        jacobian,
        ("effective_reflection", "scattering", "late_decay"),
    )

    assert report.full_column_rank is False
    assert report.numerical_rank == 2
    assert report.accepted_parameters == (
        "effective_reflection",
        "late_decay",
    )
    assert report.rejected_parameters == {
        "scattering": "redundant_with:effective_reflection"
    }
    json.dumps(report.to_dict(), allow_nan=False)


def test_spatial_profile_requires_synchronized_array_and_selects_identity():
    rng = np.random.default_rng(10)
    target = rng.standard_normal((2, 1024)) * np.exp(-np.arange(1024) / 300.0)
    target[:, :20] = 0.0
    target[:, 20] = (1.0, 0.9)
    wrong = target.copy()
    wrong[1, 100:] *= -1.0

    selection = select_spatial_calibration_candidate(
        target,
        {"identity": target, "wrong": wrong},
        8000,
        (20, 20),
        physical_first_samples=(20, 20),
    )

    assert selection.best_candidate_id == "identity"
    assert selection.candidate_reports["identity"]["total"] == 0.0
    with pytest.raises(ValueError, match="synchronized receivers"):
        select_spatial_calibration_candidate(
            target[:1],
            {"identity": target[:1]},
            8000,
            (20,),
        )
