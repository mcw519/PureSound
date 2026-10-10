import math

import numpy as np
import pytest

from egs.rir_generation.phases.m3_wave_path.scripts.validate_corrected_fdtd_boundary import (
    _simulate_1d_plane_wave,
)
from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    characteristic_impedance_pa_s_m,
)
from puresound.audio.rir.physics.wave.fdtd import (
    BOUNDARIES,
    FDTDReferenceConfig,
    absorption_to_impedance,
    fdtd_discrete_plane_wave_reflection,
    impedance_to_absorption,
    simulate_fdtd_reference,
    simulate_reciprocal_fdtd_reference,
)
from puresound.audio.rir.physics.wave.low_frequency import (
    estimate_low_frequency_modes,
    rigid_rectangular_room_modes,
)


def _relaxation(infinite=0.08, relaxation=0.6, frequency_hz=120.0):
    return FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=infinite,
        normalized_admittance_relaxation=relaxation,
        relaxation_frequency_hz=frequency_hz,
    )


RIGID = _relaxation(0.0, 0.0, 100.0)
PASSIVE_MULTI_POLE = PassiveMultiPoleAdmittance(
    normalized_admittance_static=0.01,
    pole_frequencies_hz=(70.0, 240.0),
    normalized_admittance_lowpass=(0.02, 0.01),
    normalized_admittance_highpass=(0.25, 0.55),
)


def _direction(cosine):
    return (math.sqrt(1.0 - cosine**2), 0.0, cosine)


def _continuous_reflection(model, frequency_hz, cosine):
    admittance = model.normalized_admittance(frequency_hz)
    return (cosine - admittance) / (cosine + admittance)


def _plane_wave(model, frequency_hz, cosine, **kwargs):
    options = {
        "normal_axis": 2,
        "grid_spacing_xyz_m": (0.06, 0.06, 0.06),
        "time_step_s": 1.0 / 30000.0,
    }
    options.update(kwargs)
    return fdtd_discrete_plane_wave_reflection(
        model,
        frequency_hz=frequency_hz,
        incident_direction_unit=_direction(cosine),
        **options,
    )


FOURTH_ORDER = {
    "boundary_pressure_scheme": "face_quadratic_time_quadratic",
    "spatial_derivative_order": 4,
}
FOURTH_ORDER_3D = {**FOURTH_ORDER, "near_wall_closure": "third_order_one_sided"}


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


def test_modal_peak_estimator_reports_no_peaks_when_the_band_holds_no_fft_bin():
    # A 16-sample gate zero-padded to 128 bins has a 125 Hz bin spacing at
    # 16 kHz, so a 20-100 Hz band contains no bin at all.
    signal = np.random.default_rng(0).standard_normal(16)

    analysis = estimate_low_frequency_modes(
        signal, 16000, min_frequency_hz=20.0, max_frequency_hz=100.0
    )

    assert analysis.peaks == ()
    assert analysis.fft_bin_spacing_hz == pytest.approx(125.0)


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
        config.sound_speed_m_s / 8.0 * (4.0 * loss / lx + 2.0 * loss / ly + 2.0 * loss / lz),
        config.sound_speed_m_s / 8.0 * (2.0 * loss / lx + 4.0 * loss / ly + 2.0 * loss / lz),
    )
    for measured, expected, decay_rate in zip(
        analysis.peaks,
        (axial_x, axial_y),
        expected_decay_rates,
    ):
        expected_q = 2.0 * math.pi * expected.frequency_hz / (2.0 * decay_rate)
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
    constant_admittance = _relaxation(z0 / impedance, 0.0, 100.0)

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


def test_discrete_plane_wave_reflection_is_exact_for_rigid_and_bounded_for_passive():
    rigid = _plane_wave(RIGID, 240.0, 0.5)

    assert rigid.reflection_coefficient == pytest.approx(1.0 + 0.0j)
    assert rigid.requested_incidence_cosine == pytest.approx(0.5)
    assert 0.0 < rigid.discrete_incidence_cosine <= 1.0

    magnitudes = [
        abs(_plane_wave(PASSIVE_MULTI_POLE, frequency_hz, cosine).reflection_coefficient)
        for cosine in (0.1, 0.25, 0.5, 0.75, 1.0)
        for frequency_hz in (80.0, 168.0, 240.0, 300.0)
    ]
    assert max(magnitudes) <= 1.0 + 1e-12


@pytest.mark.parametrize(
    "scheme_kwargs,cfl,cosine,final_tolerance",
    [
        ({}, 0.7, 0.5, 0.02),
        (FOURTH_ORDER, 0.33, 0.1, 0.001),
    ],
    ids=["default", "fourth_order_quadratic"],
)
def test_discrete_plane_wave_reflection_converges_to_the_continuous_boundary(
    scheme_kwargs, cfl, cosine, final_tolerance
):
    model = _relaxation()
    continuous = _continuous_reflection(model, 300.0, cosine)
    errors = []
    for spacing_m in (0.12, 0.06, 0.03, 0.015):
        discrete = _plane_wave(
            model,
            300.0,
            cosine,
            grid_spacing_xyz_m=(spacing_m,) * 3,
            time_step_s=cfl * spacing_m / (343.0 * math.sqrt(3.0)),
            **scheme_kwargs,
        )
        errors.append(abs(discrete.reflection_coefficient - continuous))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < final_tolerance


def test_refined_boundary_schemes_reduce_discrete_error():
    # Face-time extrapolation beats the cell-centre boundary pressure at a mid
    # angle ...
    model = _relaxation()
    continuous = _continuous_reflection(model, 300.0, 0.5)
    legacy = _plane_wave(model, 300.0, 0.5, boundary_pressure_scheme="cell_center")
    corrected = _plane_wave(
        model, 300.0, 0.5, boundary_pressure_scheme="face_time_extrapolated"
    )
    assert abs(corrected.reflection_coefficient - continuous) < abs(
        legacy.reflection_coefficient - continuous
    )
    assert corrected.boundary_pressure_scheme == "face_time_extrapolated"

    # ... and a fourth-order symbol beats second order at grazing incidence.
    second_order = _plane_wave(RIGID, 300.0, 0.1, spatial_derivative_order=2)
    fourth_order = _plane_wave(RIGID, 300.0, 0.1, spatial_derivative_order=4)
    assert abs(fourth_order.discrete_incidence_cosine - 0.1) < abs(
        second_order.discrete_incidence_cosine - 0.1
    )
    assert fourth_order.spatial_derivative_order == 4
    assert fourth_order.time_domain_implemented is True


def test_fourth_order_quadratic_reflection_is_accurate_and_passive_at_every_angle():
    model = _relaxation()
    complex_errors = []
    magnitude_errors_db = []
    phase_errors_deg = []
    magnitudes = []
    for cosine in (0.056, 0.1, 0.25, 0.5, 0.75, 1.0):
        for frequency_hz in (168.0, 240.0, 300.0):
            candidate = _plane_wave(model, frequency_hz, cosine, **FOURTH_ORDER)
            continuous = _continuous_reflection(model, frequency_hz, cosine)
            ratio = candidate.reflection_coefficient / continuous
            complex_errors.append(abs(candidate.reflection_coefficient - continuous))
            magnitude_errors_db.append(abs(20.0 * math.log10(abs(ratio))))
            phase_errors_deg.append(abs(math.degrees(np.angle(ratio))))
            magnitudes.append(abs(candidate.reflection_coefficient))

    assert max(complex_errors) < 0.02
    assert max(magnitude_errors_db) < 0.10
    assert max(phase_errors_deg) < 1.0
    assert max(magnitudes) <= 1.0 + 1e-12


@pytest.mark.parametrize(
    "scheme_kwargs",
    [
        {"boundary_pressure_scheme": "face_extrapolated"},
        {"boundary_pressure_scheme": "face_time_extrapolated"},
        FOURTH_ORDER_3D,
    ],
    ids=["face_extrapolated", "face_time_extrapolated", "fourth_order_quadratic"],
)
def test_3d_fdtd_boundary_schemes_remain_finite_and_serialized(scheme_kwargs):
    config = FDTDReferenceConfig(
        room_dim_m=(0.8, 0.7, 0.6),
        duration_s=0.04,
        grid_spacing_m=0.10,
        source_position_m=(0.22, 0.24, 0.23),
        receiver_position_m=(0.58, 0.46, 0.37),
        source_center_hz=500.0,
        source_delay_s=0.008,
        **scheme_kwargs,
    )

    result = simulate_fdtd_reference(config, boundary_admittance=_relaxation())
    metadata = result.metadata()

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    for key, value in scheme_kwargs.items():
        assert metadata["config"][key] == value
    assert 0.0 < metadata["effective_interior_cfl"]
    assert metadata["boundary_time_step_limit_s"] <= metadata["interior_cfl_time_step_s"]
    assert metadata["source_spatial_weights_used"] is False
    assert metadata["receiver_spatial_weights_used"] is False
    if scheme_kwargs is FOURTH_ORDER_3D:
        assert metadata["effective_interior_cfl"] <= 0.25


def test_fourth_order_reciprocal_transfer_matches_exchanged_problem():
    model = _relaxation(0.9, 0.4, 180.0)
    common = {
        "room_dim_m": (0.8, 0.7, 0.6),
        "duration_s": 0.025,
        "grid_spacing_m": 0.10,
        "source_center_hz": 500.0,
        "source_delay_s": 0.006,
        **FOURTH_ORDER_3D,
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

    np.testing.assert_allclose(forward.rir, reverse.rir, rtol=0.0, atol=0.0)
    assert forward.reciprocity_averaged is True
    assert forward.raw_reciprocity_nrmse is not None
    assert forward.raw_reciprocity_nrmse >= 0.0
    assert forward.metadata()["reciprocity_averaged"] is True


def test_fourth_order_uniform_3d_plane_mode_reduces_to_1d_update():
    model = _relaxation()
    config = FDTDReferenceConfig(
        room_dim_m=(1.2, 0.6, 0.6),
        duration_s=0.02,
        grid_spacing_m=0.10,
        source_position_m=(0.75, 0.3, 0.3),
        receiver_position_m=(0.45, 0.3, 0.3),
        source_center_hz=500.0,
        source_delay_s=0.004,
        **FOURTH_ORDER_3D,
    )
    # Plane-wave source and receiver weights excite and read one x-mode.
    source = np.zeros((6, 6, 12), dtype=np.float64)
    source[:, :, 7] = 1.0
    receiver = np.zeros_like(source)
    receiver[:, :, 4] = 1.0 / 36.0
    boundaries = {boundary: RIGID for boundary in BOUNDARIES}
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
    assert result_3d.source_spatial_weights_used is True
    assert result_3d.receiver_spatial_weights_used is True


def test_fourth_order_3d_config_requires_explicit_matching_closure():
    with pytest.raises(ValueError, match="fourth-order closure"):
        FDTDReferenceConfig(spatial_derivative_order=4)
    with pytest.raises(ValueError, match="fourth-order interior"):
        FDTDReferenceConfig(
            boundary_pressure_scheme="face_quadratic_time_quadratic",
        )


_ROOM_CONFIG = {"duration_s": 0.20, "grid_spacing_m": 0.20, "source_center_hz": 140.0}
_SMALL_ROOM_CONFIG = {
    "room_dim_m": (0.6, 0.5, 0.4),
    "duration_s": 0.025,
    "grid_spacing_m": 0.10,
    "source_position_m": (0.18, 0.18, 0.16),
    "receiver_position_m": (0.42, 0.32, 0.24),
    "source_center_hz": 1600.0,
    "source_delay_s": 0.004,
}


@pytest.mark.parametrize(
    "admittance,config_kwargs,model_name,stiff_wall_state",
    [
        (
            _relaxation(0.18, 0.55, 100.0),
            _ROOM_CONFIG,
            "first_order_relaxation_admittance",
            False,
        ),
        (PASSIVE_MULTI_POLE, _ROOM_CONFIG, "passive_multi_pole_admittance", True),
        (
            PassiveResonantAdmittance(
                normalized_admittance_static=0.03,
                resonance_frequencies_hz=(1600.0,),
                quality_factors=(12.0,),
                peak_normalized_admittances=(7.5,),
            ),
            _SMALL_ROOM_CONFIG,
            "passive_resonant_admittance",
            True,
        ),
    ],
    ids=["relaxation", "multi_pole", "resonant"],
)
def test_frequency_dependent_admittance_fdtd_is_stable_and_serialized(
    admittance, config_kwargs, model_name, stiff_wall_state
):
    result = simulate_fdtd_reference(
        FDTDReferenceConfig(**config_kwargs),
        boundary_admittance=admittance,
    )
    metadata = result.metadata()

    assert np.all(np.isfinite(result.rir))
    assert np.max(np.abs(result.rir)) > 0.0
    assert all(value is None for value in result.boundary_impedance_pa_s_m.values())
    assert metadata["boundary_model"]["west"] == admittance.metadata()
    assert metadata["boundary_model"]["west"]["passive"] is True
    assert result.boundary_model["west"]["model"] == model_name
    if stiff_wall_state:
        # Per-pole and per-branch wall states are stiffer than the interior
        # update, so they, not the CFL limit, set the time step.
        assert result.boundary_time_step_limit_s < result.interior_cfl_time_step_s
