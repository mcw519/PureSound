import json

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


CENTERS = (500.0, 1000.0, 2000.0)
TRUTH = M4InverseParameters(
    mixing_time_s=0.024,
    coherent_reflection_gain_db=-1.5,
    target_rt60_s_by_hz={500.0: 0.68, 1000.0: 0.54, 2000.0: 0.42},
)
INITIAL = M4InverseParameters(
    mixing_time_s=0.028,
    coherent_reflection_gain_db=3.0,
    target_rt60_s_by_hz={500.0: 0.40, 1000.0: 0.85, 2000.0: 0.72},
)


def _observation(name="fixture", seed=10):
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


def test_m4_inner_profile_recovers_exact_continuous_parameters():
    observations = (_observation("a", 10), _observation("b", 11))
    targets = tuple(
        render_m4_inverse_observation(observation, TRUTH).rir
        for observation in observations
    )

    fit = fit_m4_parameter_profile(
        observations,
        targets,
        INITIAL,
        (TRUTH.mixing_time_s,),
        maximum_evaluations=50,
    )

    assert fit.best.success is True
    assert fit.best.locally_full_rank is True
    assert fit.best.cost < 1e-14
    assert fit.best.parameters.continuous_vector() == pytest.approx(
        TRUTH.continuous_vector(), abs=2e-6
    )
    json.dumps(fit.to_dict(), allow_nan=False)


def test_m4_profile_convergence_separates_a_stable_minimum_from_a_truncated_fit():
    """The convergence verdict must still be able to say no.

    ``converged`` replaced ``least_squares``'s ``success`` flag because a
    rough objective can exhaust the budget at a point no further optimization
    improves.  The replacement is only worth anything if a genuinely truncated
    fit — one still descending when the budget ran out — is still rejected.
    """

    observations = (_observation("a", 10), _observation("b", 11))
    targets = tuple(
        render_m4_inverse_observation(observation, TRUTH).rir
        for observation in observations
    )

    truncated = fit_m4_parameter_profile(
        observations,
        targets,
        INITIAL,
        (TRUTH.mixing_time_s,),
        maximum_evaluations=2,
    ).best
    settled = fit_m4_parameter_profile(
        observations,
        targets,
        INITIAL,
        (TRUTH.mixing_time_s,),
        maximum_evaluations=50,
    ).best

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

    assert settled.converged is True
    assert settled.cost < truncated.cost
    assert settled.to_dict()["convergence"]["converged"] is True


def test_m4_inverse_render_preserves_fractional_causality_and_direct_path():
    observation = _observation()
    result = render_m4_inverse_observation(observation, TRUTH)
    transition_start = result.metadata["transition_start_sample"]

    assert np.count_nonzero(result.rir[: observation.direct_sample]) == 0
    assert np.array_equal(
        result.rir[: transition_start + 1],
        render_m4_inverse_observation(
            observation,
            M4InverseParameters(
                mixing_time_s=TRUTH.mixing_time_s,
                coherent_reflection_gain_db=TRUTH.coherent_reflection_gain_db,
                target_rt60_s_by_hz={center: 0.3 for center in TRUTH.centers_hz},
            ),
        ).rir[: transition_start + 1],
    )
    assert result.metadata["pre_transition_max_abs_error"] == 0.0


def test_m4_profile_contract_rejects_invalid_inputs_and_serializes_early_window():
    objective = M4ProfileObjectiveConfig(early_window_ms=10.0)
    assert objective.to_dict()["early_window_ms"] == 10.0
    with pytest.raises(ValueError, match="energy and octave"):
        M4ProfileObjectiveConfig(early_window_ms=0.0)
    with pytest.raises(ValueError, match="positive finite"):
        fit_m4_parameter_profile(
            (_observation(),),
            (np.zeros(3200),),
            INITIAL,
            (0.0,),
        )
    invalid_bands = M4InverseParameters(
        mixing_time_s=0.024,
        coherent_reflection_gain_db=0.0,
        target_rt60_s_by_hz={5000.0: 0.5},
    )
    with pytest.raises(ValueError, match="Nyquist"):
        render_m4_inverse_observation(_observation(), invalid_bands)
