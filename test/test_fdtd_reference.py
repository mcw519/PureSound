import math

import numpy as np
import pytest

from egs.rir_generation.phases.m3_wave_path.scripts.validate_corrected_fdtd_boundary import (
    _simulate_1d_plane_wave,
)
from puresound.audio.acoustic_impedance import (
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    characteristic_impedance_pa_s_m,
)
from puresound.audio.fdtd_reference import (
    BOUNDARIES,
    FDTDReferenceConfig,
    absorption_to_impedance,
    fdtd_discrete_plane_wave_reflection,
    impedance_to_absorption,
    simulate_reciprocal_fdtd_reference,
    simulate_fdtd_reference,
)
from puresound.audio.low_frequency_modes import (
    estimate_low_frequency_modes,
    rigid_rectangular_room_modes,
)


def test_absorption_impedance_conversion_round_trips():
    for absorption in (0.0, 0.03, 0.2, 0.8, 1.0):
        impedance = absorption_to_impedance(absorption, 1.204, 343.0)
        assert impedance_to_absorption(impedance, 1.204, 343.0) == pytest.approx(
            absorption
        )


def test_modal_peak_estimator_recovers_damped_sinusoid_frequency_and_q():
    sample_rate = 4000
    frequency_hz = 80.0
    amplitude_decay_rate = 10.0
    time_s = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    signal = np.exp(-amplitude_decay_rate * time_s) * np.sin(
        2.0 * math.pi * frequency_hz * time_s
    )

    analysis = estimate_low_frequency_modes(
        signal,
        sample_rate,
        min_frequency_hz=50.0,
        max_frequency_hz=110.0,
        analysis_duration_s=None,
        min_prominence_db=10.0,
    )

    assert len(analysis.peaks) == 1
    peak = analysis.peaks[0]
    expected_q = 2.0 * math.pi * frequency_hz / (2.0 * amplitude_decay_rate)
    assert peak.frequency_hz == pytest.approx(frequency_hz, abs=0.1)
    assert peak.q_factor == pytest.approx(expected_q, rel=0.02)


def test_independent_fdtd_recovers_first_axial_frequencies_and_boundary_q():
    absorption = 0.08
    config = FDTDReferenceConfig(
        duration_s=0.8,
        grid_spacing_m=0.15,
        source_center_hz=120.0,
    )

    result = simulate_fdtd_reference(config, absorption)
    analysis = estimate_low_frequency_modes(
        result.rir,
        result.sample_rate_hz,
        min_frequency_hz=35.0,
        max_frequency_hz=80.0,
        analysis_start_s=0.06,
        analysis_duration_s=None,
        min_prominence_db=4.0,
    )
    expected_modes = rigid_rectangular_room_modes(
        config.room_dim_m,
        max_frequency_hz=80.0,
        sound_speed_m_s=config.sound_speed_m_s,
    )
    axial_x = next(mode for mode in expected_modes if mode.indices == (1, 0, 0))
    axial_y = next(mode for mode in expected_modes if mode.indices == (0, 1, 0))

    assert np.all(np.isfinite(result.rir))
    assert len(analysis.peaks) == 2
    for measured, expected in zip(analysis.peaks, (axial_x, axial_y)):
        assert measured.frequency_hz == pytest.approx(
            expected.frequency_hz, rel=0.01
        )

    # For a real locally reacting impedance, replace the first-order absorption
    # coefficient in the modal loss formula with -ln(1-alpha).
    loss = -math.log1p(-absorption)
    lx, ly, lz = config.room_dim_m
    expected_decay_rates = (
        config.sound_speed_m_s
        / 8.0
        * (
            4.0 * loss / lx
            + 2.0 * loss / ly
            + 2.0 * loss / lz
        ),
        config.sound_speed_m_s
        / 8.0
        * (
            2.0 * loss / lx
            + 4.0 * loss / ly
            + 2.0 * loss / lz
        ),
    )
    for measured, expected, decay_rate in zip(
        analysis.peaks,
        (axial_x, axial_y),
        expected_decay_rates,
    ):
        expected_q = (
            2.0 * math.pi * expected.frequency_hz / (2.0 * decay_rate)
        )
        assert measured.q_factor == pytest.approx(expected_q, rel=0.2)


def test_frequency_independent_admittance_matches_real_impedance_fdtd():
    absorption = 0.12
    config = FDTDReferenceConfig(
        duration_s=0.15,
        grid_spacing_m=0.20,
        source_center_hz=140.0,
    )
    impedance = absorption_to_impedance(
        absorption,
        config.air_density_kg_m3,
        config.sound_speed_m_s,
    )
    z0 = characteristic_impedance_pa_s_m(
        config.air_density_kg_m3,
        config.sound_speed_m_s,
    )
    constant_admittance = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=z0 / impedance,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )

    real_result = simulate_fdtd_reference(config, absorption)
    admittance_result = simulate_fdtd_reference(
        config,
        boundary_admittance=constant_admittance,
    )

    np.testing.assert_allclose(
        admittance_result.rir,
        real_result.rir,
        rtol=0.0,
        atol=1e-12,
    )
    assert real_result.boundary_model["west"]["model"] == (
        "frequency_independent_real_impedance"
    )
    assert admittance_result.boundary_model["west"]["model"] == (
        "first_order_relaxation_admittance"
    )


def test_discrete_plane_wave_rigid_reflection_is_exactly_one():
    rigid = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )

    result = fdtd_discrete_plane_wave_reflection(
        rigid,
        frequency_hz=240.0,
        incident_direction_unit=(math.sqrt(0.75), 0.0, 0.5),
        normal_axis=2,
        grid_spacing_xyz_m=(0.06, 0.06, 0.06),
        time_step_s=1.0 / 30000.0,
    )

    assert result.reflection_coefficient == pytest.approx(1.0 + 0.0j)
    assert result.requested_incidence_cosine == pytest.approx(0.5)
    assert 0.0 < result.discrete_incidence_cosine <= 1.0


def test_discrete_plane_wave_reflection_converges_to_continuous_boundary():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )
    frequency_hz = 300.0
    cosine = 0.5
    direction = (math.sqrt(1.0 - cosine**2), 0.0, cosine)
    analog_admittance = model.normalized_admittance(frequency_hz)
    continuous = (
        cosine - analog_admittance
    ) / (
        cosine + analog_admittance
    )
    errors = []
    for spacing_m in (0.12, 0.06, 0.03, 0.015):
        time_step_s = (
            0.7 * spacing_m / (343.0 * math.sqrt(3.0))
        )
        discrete = fdtd_discrete_plane_wave_reflection(
            model,
            frequency_hz=frequency_hz,
            incident_direction_unit=direction,
            normal_axis=2,
            grid_spacing_xyz_m=(spacing_m,) * 3,
            time_step_s=time_step_s,
        )
        errors.append(abs(discrete.reflection_coefficient - continuous))

    assert all(
        later < earlier for earlier, later in zip(errors, errors[1:])
    )
    assert errors[-1] < 0.02


def test_face_time_boundary_reduces_mid_angle_discrete_phase_error():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )
    frequency_hz = 300.0
    cosine = 0.5
    direction = (math.sqrt(1.0 - cosine**2), 0.0, cosine)
    admittance = model.normalized_admittance(frequency_hz)
    continuous = (cosine - admittance) / (cosine + admittance)
    common = {
        "frequency_hz": frequency_hz,
        "incident_direction_unit": direction,
        "normal_axis": 2,
        "grid_spacing_xyz_m": (0.06, 0.06, 0.06),
        "time_step_s": 1.0 / 30000.0,
    }

    legacy = fdtd_discrete_plane_wave_reflection(
        model,
        **common,
        boundary_pressure_scheme="cell_center",
    )
    corrected = fdtd_discrete_plane_wave_reflection(
        model,
        **common,
        boundary_pressure_scheme="face_time_extrapolated",
    )

    assert abs(corrected.reflection_coefficient - continuous) < abs(
        legacy.reflection_coefficient - continuous
    )
    assert corrected.boundary_pressure_scheme == "face_time_extrapolated"


def test_fourth_order_symbol_reduces_grazing_characteristic_error():
    rigid = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    cosine = 0.1
    common = {
        "frequency_hz": 300.0,
        "incident_direction_unit": (
            math.sqrt(1.0 - cosine**2),
            0.0,
            cosine,
        ),
        "normal_axis": 2,
        "grid_spacing_xyz_m": (0.06, 0.06, 0.06),
        "time_step_s": 1.0 / 30000.0,
    }

    second_order = fdtd_discrete_plane_wave_reflection(
        rigid,
        **common,
        spatial_derivative_order=2,
    )
    fourth_order = fdtd_discrete_plane_wave_reflection(
        rigid,
        **common,
        spatial_derivative_order=4,
    )

    assert abs(fourth_order.discrete_incidence_cosine - cosine) < abs(
        second_order.discrete_incidence_cosine - cosine
    )
    assert fourth_order.spatial_derivative_order == 4
    assert fourth_order.time_domain_implemented is True


def test_fourth_order_quadratic_candidate_meets_harmonic_gate():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )
    complex_errors = []
    magnitude_errors_db = []
    phase_errors_deg = []
    magnitudes = []
    for cosine in (0.056, 0.1, 0.25, 0.5, 0.75, 1.0):
        direction = (math.sqrt(1.0 - cosine**2), 0.0, cosine)
        for frequency_hz in (168.0, 240.0, 300.0):
            candidate = fdtd_discrete_plane_wave_reflection(
                model,
                frequency_hz=frequency_hz,
                incident_direction_unit=direction,
                normal_axis=2,
                grid_spacing_xyz_m=(0.06, 0.06, 0.06),
                time_step_s=1.0 / 30000.0,
                boundary_pressure_scheme=(
                    "face_quadratic_time_quadratic"
                ),
                spatial_derivative_order=4,
            )
            admittance = model.normalized_admittance(frequency_hz)
            continuous = (
                cosine - admittance
            ) / (
                cosine + admittance
            )
            complex_errors.append(
                abs(candidate.reflection_coefficient - continuous)
            )
            magnitude_errors_db.append(
                abs(
                    20.0
                    * math.log10(
                        abs(candidate.reflection_coefficient)
                        / abs(continuous)
                    )
                )
            )
            phase_errors_deg.append(
                abs(
                    math.degrees(
                        np.angle(
                            candidate.reflection_coefficient / continuous
                        )
                    )
                )
            )
            magnitudes.append(abs(candidate.reflection_coefficient))

    assert max(complex_errors) < 0.02
    assert max(magnitude_errors_db) < 0.10
    assert max(phase_errors_deg) < 1.0
    assert max(magnitudes) <= 1.0 + 1e-12


def test_fourth_order_quadratic_candidate_converges_under_grid_refinement():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )
    frequency_hz = 300.0
    cosine = 0.1
    direction = (math.sqrt(1.0 - cosine**2), 0.0, cosine)
    admittance = model.normalized_admittance(frequency_hz)
    continuous = (cosine - admittance) / (cosine + admittance)
    errors = []
    for spacing_m in (0.12, 0.06, 0.03, 0.015):
        time_step_s = 0.33 * spacing_m / (343.0 * math.sqrt(3.0))
        candidate = fdtd_discrete_plane_wave_reflection(
            model,
            frequency_hz=frequency_hz,
            incident_direction_unit=direction,
            normal_axis=2,
            grid_spacing_xyz_m=(spacing_m,) * 3,
            time_step_s=time_step_s,
            boundary_pressure_scheme="face_quadratic_time_quadratic",
            spatial_derivative_order=4,
        )
        errors.append(abs(candidate.reflection_coefficient - continuous))

    assert all(
        later < earlier for earlier, later in zip(errors, errors[1:])
    )
    assert errors[-1] < 0.001


def test_discrete_plane_wave_passive_reference_stays_bounded():
    model = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.01,
        pole_frequencies_hz=(70.0, 240.0),
        normalized_admittance_lowpass=(0.02, 0.01),
        normalized_admittance_highpass=(0.25, 0.55),
    )

    magnitudes = [
        abs(
            fdtd_discrete_plane_wave_reflection(
                model,
                frequency_hz=frequency_hz,
                incident_direction_unit=(
                    math.sqrt(1.0 - cosine**2),
                    0.0,
                    cosine,
                ),
                normal_axis=2,
                grid_spacing_xyz_m=(0.06, 0.06, 0.06),
                time_step_s=1.0 / 30000.0,
            ).reflection_coefficient
        )
        for cosine in (0.1, 0.25, 0.5, 0.75, 1.0)
        for frequency_hz in (80.0, 168.0, 240.0, 300.0)
    ]

    assert max(magnitudes) <= 1.0 + 1e-12


@pytest.mark.parametrize(
    "scheme",
    ("face_extrapolated", "face_time_extrapolated"),
)
def test_experimental_face_pressure_fdtd_remains_finite(scheme):
    config = FDTDReferenceConfig(
        room_dim_m=(0.8, 0.7, 0.6),
        duration_s=0.04,
        grid_spacing_m=0.10,
        source_position_m=(0.22, 0.24, 0.23),
        receiver_position_m=(0.58, 0.46, 0.37),
        source_center_hz=500.0,
        source_delay_s=0.008,
        boundary_pressure_scheme=scheme,
    )
    admittance = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )

    result = simulate_fdtd_reference(
        config,
        boundary_admittance=admittance,
    )

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert result.metadata()["config"]["boundary_pressure_scheme"] == scheme


def test_fourth_order_quadratic_3d_fdtd_remains_finite_and_serialized():
    config = FDTDReferenceConfig(
        room_dim_m=(0.8, 0.7, 0.6),
        duration_s=0.04,
        grid_spacing_m=0.10,
        source_position_m=(0.22, 0.24, 0.23),
        receiver_position_m=(0.58, 0.46, 0.37),
        source_center_hz=500.0,
        source_delay_s=0.008,
        boundary_pressure_scheme="face_quadratic_time_quadratic",
        spatial_derivative_order=4,
        near_wall_closure="third_order_one_sided",
    )
    admittance = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )

    result = simulate_fdtd_reference(
        config,
        boundary_admittance=admittance,
    )
    metadata = result.metadata()

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert metadata["config"]["spatial_derivative_order"] == 4
    assert metadata["config"]["near_wall_closure"] == (
        "third_order_one_sided"
    )
    assert 0.0 < metadata["effective_interior_cfl"] <= 0.25
    assert metadata["boundary_time_step_limit_s"] <= (
        metadata["interior_cfl_time_step_s"]
    )
    assert metadata["source_spatial_weights_used"] is False
    assert metadata["receiver_spatial_weights_used"] is False


def test_fourth_order_3d_fdtd_accepts_plane_mode_spatial_weights():
    config = FDTDReferenceConfig(
        room_dim_m=(0.6, 0.6, 0.6),
        duration_s=0.02,
        grid_spacing_m=0.10,
        source_position_m=(0.25, 0.25, 0.25),
        receiver_position_m=(0.45, 0.35, 0.35),
        source_center_hz=500.0,
        source_delay_s=0.004,
        boundary_pressure_scheme="face_quadratic_time_quadratic",
        spatial_derivative_order=4,
        near_wall_closure="third_order_one_sided",
    )
    weights = np.zeros((6, 6, 6), dtype=np.float64)
    weights[:, :, 2] = 1.0
    receiver = np.zeros_like(weights)
    receiver[:, :, 4] = 1.0 / 36.0

    result = simulate_fdtd_reference(
        config,
        boundary_absorption=0.1,
        source_spatial_weights_zyx=weights,
        receiver_spatial_weights_zyx=receiver,
    )

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert result.source_spatial_weights_used is True
    assert result.receiver_spatial_weights_used is True


def test_fourth_order_reciprocal_transfer_matches_exchanged_problem():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.9,
        normalized_admittance_relaxation=0.4,
        relaxation_frequency_hz=180.0,
    )
    common = {
        "room_dim_m": (0.8, 0.7, 0.6),
        "duration_s": 0.025,
        "grid_spacing_m": 0.10,
        "source_center_hz": 500.0,
        "source_delay_s": 0.006,
        "boundary_pressure_scheme": (
            "face_quadratic_time_quadratic"
        ),
        "spatial_derivative_order": 4,
        "near_wall_closure": "third_order_one_sided",
    }
    forward_config = FDTDReferenceConfig(
        source_position_m=(0.15, 0.15, 0.15),
        receiver_position_m=(0.65, 0.55, 0.45),
        **common,
    )
    reverse_config = FDTDReferenceConfig(
        source_position_m=forward_config.receiver_position_m,
        receiver_position_m=forward_config.source_position_m,
        **common,
    )

    forward = simulate_reciprocal_fdtd_reference(
        forward_config,
        boundary_admittance=model,
        remove_output_dc=False,
    )
    reverse = simulate_reciprocal_fdtd_reference(
        reverse_config,
        boundary_admittance=model,
        remove_output_dc=False,
    )

    np.testing.assert_allclose(
        forward.rir,
        reverse.rir,
        rtol=0.0,
        atol=0.0,
    )
    assert forward.reciprocity_averaged is True
    assert forward.raw_reciprocity_nrmse is not None
    assert forward.raw_reciprocity_nrmse >= 0.0
    assert forward.metadata()["reciprocity_averaged"] is True


def test_fourth_order_uniform_3d_plane_mode_reduces_to_1d_update():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )
    rigid = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    config = FDTDReferenceConfig(
        room_dim_m=(1.2, 0.6, 0.6),
        duration_s=0.02,
        grid_spacing_m=0.10,
        source_position_m=(0.75, 0.3, 0.3),
        receiver_position_m=(0.45, 0.3, 0.3),
        source_center_hz=500.0,
        source_delay_s=0.004,
        boundary_pressure_scheme="face_quadratic_time_quadratic",
        spatial_derivative_order=4,
        near_wall_closure="third_order_one_sided",
    )
    source = np.zeros((6, 6, 12), dtype=np.float64)
    source[:, :, 7] = 1.0
    receiver = np.zeros_like(source)
    receiver[:, :, 4] = 1.0 / 36.0
    boundaries = {boundary: rigid for boundary in BOUNDARIES}
    boundaries["west"] = model

    result_3d = simulate_fdtd_reference(
        config,
        boundary_admittance=boundaries,
        source_spatial_weights_zyx=source,
        receiver_spatial_weights_zyx=receiver,
        remove_output_dc=False,
    )
    result_1d, _geometry = _simulate_1d_plane_wave(
        model,
        boundary_pressure_scheme="face_quadratic_time_quadratic",
        sample_rate_hz=result_3d.sample_rate_hz,
        grid_spacing_m=0.10,
        duration_s=config.duration_s,
        room_length_m=1.2,
        source_position_m=0.75,
        receiver_position_m=0.45,
        source_center_hz=config.source_center_hz,
        source_delay_s=config.source_delay_s,
        spatial_derivative_order=4,
        near_wall_closure="third_order_one_sided",
    )
    common_samples = min(result_3d.rir.size, result_1d.size)

    np.testing.assert_allclose(
        result_3d.rir[:common_samples],
        result_1d[:common_samples],
        rtol=0.0,
        atol=1e-12,
    )
    assert result_3d.output_dc_removed is False


def test_fourth_order_3d_config_requires_explicit_matching_closure():
    with pytest.raises(ValueError, match="fourth-order closure"):
        FDTDReferenceConfig(spatial_derivative_order=4)
    with pytest.raises(ValueError, match="fourth-order interior"):
        FDTDReferenceConfig(
            boundary_pressure_scheme="face_quadratic_time_quadratic",
        )


def test_frequency_dependent_admittance_fdtd_is_stable_and_serialized():
    config = FDTDReferenceConfig(
        duration_s=0.25,
        grid_spacing_m=0.20,
        source_center_hz=140.0,
    )
    admittance = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.18,
        normalized_admittance_relaxation=0.55,
        relaxation_frequency_hz=100.0,
    )

    result = simulate_fdtd_reference(
        config,
        boundary_admittance=admittance,
    )
    metadata = result.metadata()

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert all(
        value is None for value in result.boundary_impedance_pa_s_m.values()
    )
    assert metadata["boundary_model"]["west"] == admittance.metadata()
    assert metadata["boundary_model"]["west"]["passive"] is True
    assert metadata["boundary_model"]["west"]["causal"] is True


def test_passive_multi_pole_admittance_runs_with_per_pole_wall_state():
    config = FDTDReferenceConfig(
        duration_s=0.20,
        grid_spacing_m=0.20,
        source_center_hz=140.0,
    )
    admittance = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.01,
        pole_frequencies_hz=(70.0, 240.0),
        normalized_admittance_lowpass=(0.02, 0.01),
        normalized_admittance_highpass=(0.25, 0.55),
    )

    result = simulate_fdtd_reference(
        config,
        boundary_admittance=admittance,
    )

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert result.boundary_model["west"]["model"] == (
        "passive_multi_pole_admittance"
    )
    assert len(result.boundary_model["west"]["pole_frequencies_hz"]) == 2
    assert result.boundary_time_step_limit_s < (
        result.interior_cfl_time_step_s
    )


def test_passive_resonant_admittance_runs_with_per_wall_biquad_state():
    config = FDTDReferenceConfig(
        room_dim_m=(0.6, 0.5, 0.4),
        duration_s=0.025,
        grid_spacing_m=0.10,
        source_position_m=(0.18, 0.18, 0.16),
        receiver_position_m=(0.42, 0.32, 0.24),
        source_center_hz=1600.0,
        source_delay_s=0.004,
    )
    admittance = PassiveResonantAdmittance(
        normalized_admittance_static=0.03,
        resonance_frequencies_hz=(1600.0,),
        quality_factors=(12.0,),
        peak_normalized_admittances=(7.5,),
    )

    result = simulate_fdtd_reference(
        config,
        boundary_admittance=admittance,
    )

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert result.boundary_model["west"]["model"] == (
        "passive_resonant_admittance"
    )
    assert result.boundary_time_step_limit_s < (
        result.interior_cfl_time_step_s
    )
