import json

import numpy as np
import pytest

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir.physics.wave.fdtd import (
    FDTDReferenceConfig,
    simulate_fdtd_reference,
)
from puresound.audio.rir.physics.impedance.priors import (
    IMPEDANCE_PRIOR_CATALOG_VERSION,
    fit_first_order_relaxation,
    reference_impedance_priors,
)
from puresound.audio.rir.physics.impedance.modes import (
    RECTANGULAR_BOUNDARIES,
    RectangularImpedanceBoundaryConfig,
    solve_rectangular_impedance_modes,
)
from puresound.audio.rir.physics.wave.low_frequency import estimate_low_frequency_modes


def test_reference_priors_have_provenance_and_conservative_validity():
    priors = reference_impedance_priors()
    light = priors["glass_wool_14kgm3_100mm_normal"]
    thin = priors["glass_wool_14kgm3_50mm_normal"]
    dense = priors["glass_wool_30kgm3_100mm_normal"]

    assert light.validity_frequency_range_hz == pytest.approx((58.8, 5880.0))
    assert dense.validity_frequency_range_hz == pytest.approx((155.0, 15500.0))
    assert thin.thickness_m == pytest.approx(0.05)
    assert thin.flow_resistivity_pa_s_m2 == pytest.approx(
        light.flow_resistivity_pa_s_m2
    )
    assert light.metadata()["catalog_version"] == IMPEDANCE_PRIOR_CATALOG_VERSION
    assert len(light.provenance_urls) == 2
    assert light.metadata()["production_material_mapping_enabled"] is False
    json.dumps(light.metadata(), allow_nan=False)


def test_50mm_installation_variant_has_accepted_complex_reflection_fit():
    prior = reference_impedance_priors()["glass_wool_14kgm3_50mm_normal"]

    fit = fit_first_order_relaxation(prior, 60.0, 300.0)

    assert fit.accepted
    assert fit.maximum_complex_reflection_error < 0.006
    assert fit.model.normalized_admittance_zero > 0.0


def test_miki_rigid_backed_prior_is_complex_passive_and_phase_aware():
    prior = reference_impedance_priors()["glass_wool_14kgm3_100mm_normal"]

    impedances = np.asarray(
        [
            prior.surface_impedance_pa_s_m(frequency, 1.204, 343.0)
            for frequency in (60.0, 125.0, 250.0, 300.0)
        ]
    )
    reflections = np.asarray(
        [
            prior.reflection_coefficient(frequency, 1.204, 343.0)
            for frequency in (60.0, 125.0, 250.0, 300.0)
        ]
    )

    assert np.all(impedances.real >= 0.0)
    assert np.all(np.abs(reflections) <= 1.0)
    assert np.any(np.abs(impedances.imag) > 1.0)
    assert np.any(np.abs(np.angle(reflections)) > 0.1)
    assert prior.absorption_coefficient(250.0, 1.204, 343.0) > (
        prior.absorption_coefficient(60.0, 1.204, 343.0)
    )
    with pytest.raises(ValueError, match="outside prior validity"):
        prior.surface_impedance_pa_s_m(35.0, 1.204, 343.0)


@pytest.mark.parametrize(
    ("prior_id", "minimum_frequency_hz", "maximum_error"),
    (
        ("glass_wool_14kgm3_100mm_normal", 60.0, 0.06),
        ("glass_wool_30kgm3_100mm_normal", 155.0, 0.02),
    ),
)
def test_first_order_fit_preserves_phase_aware_reflection(
    prior_id,
    minimum_frequency_hz,
    maximum_error,
):
    prior = reference_impedance_priors()[prior_id]

    fit = fit_first_order_relaxation(
        prior,
        minimum_frequency_hz,
        300.0,
    )

    assert fit.accepted
    assert fit.maximum_complex_reflection_error < maximum_error
    assert fit.model.normalized_admittance_relaxation < 0.0
    assert fit.model.normalized_admittance_zero >= 0.0
    json.dumps(fit.metadata(), allow_nan=False)


def test_fit_rejects_extrapolation_below_model_validity():
    prior = reference_impedance_priors()["glass_wool_30kgm3_100mm_normal"]

    with pytest.raises(ValueError, match="validity"):
        fit_first_order_relaxation(prior, 60.0, 300.0)


def test_phase_aware_prior_shifts_fdtd_mode_against_magnitude_only_boundary():
    prior = reference_impedance_priors()["glass_wool_14kgm3_100mm_normal"]
    fit = fit_first_order_relaxation(prior, 60.0, 300.0)
    reference_frequency_hz = 80.0
    reflection_magnitude = abs(
        prior.reflection_coefficient(
            reference_frequency_hz,
            1.204,
            343.0,
        )
    )
    magnitude_only_admittance = (
        1.0 - reflection_magnitude
    ) / (1.0 + reflection_magnitude)
    magnitude_only = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=magnitude_only_admittance,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    config = FDTDReferenceConfig(
        room_dim_m=(2.0, 1.2, 1.0),
        duration_s=0.8,
        grid_spacing_m=0.12,
        source_position_m=(0.5, 0.4, 0.35),
        receiver_position_m=(1.5, 0.8, 0.65),
        source_center_hz=120.0,
    )

    phase_result = simulate_fdtd_reference(
        config,
        boundary_admittance=fit.model,
    )
    magnitude_result = simulate_fdtd_reference(
        config,
        boundary_admittance=magnitude_only,
    )

    def dominant_mode(result):
        analysis = estimate_low_frequency_modes(
            result.rir,
            result.sample_rate_hz,
            min_frequency_hz=50.0,
            max_frequency_hz=120.0,
            analysis_start_s=0.06,
            analysis_duration_s=0.65,
            min_prominence_db=3.0,
        )
        return max(analysis.peaks, key=lambda peak: peak.prominence_db)

    phase_mode = dominant_mode(phase_result)
    magnitude_mode = dominant_mode(magnitude_result)
    boundary_config = RectangularImpedanceBoundaryConfig(
        reference_id="glass_wool_fdtd_crosscheck",
        boundaries={
            boundary: fit.model for boundary in RECTANGULAR_BOUNDARIES
        },
        source={
            "evidence_tier": "measured_flow_resistivity_plus_miki_model"
        },
        applicability={"scope": "controlled_fdtd_crosscheck"},
    )
    impedance_mode = solve_rectangular_impedance_modes(
        config.room_dim_m,
        boundary_config,
        mode_indices=((1, 0, 0),),
        sound_speed_m_s=config.sound_speed_m_s,
    )[0]

    assert np.all(np.isfinite(phase_result.rir))
    assert phase_result.boundary_time_step_limit_s < (
        phase_result.interior_cfl_time_step_s
    )
    assert phase_mode.frequency_hz == pytest.approx(64.0, abs=1.0)
    assert magnitude_mode.frequency_hz == pytest.approx(85.7, abs=1.0)
    assert abs(phase_mode.frequency_hz - magnitude_mode.frequency_hz) > 15.0
    assert phase_mode.q_factor is not None
    assert magnitude_mode.q_factor is not None
    assert impedance_mode.frequency_hz == pytest.approx(
        phase_mode.frequency_hz,
        abs=1.0,
    )
    assert impedance_mode.q_factor == pytest.approx(
        phase_mode.q_factor,
        rel=0.15,
    )
