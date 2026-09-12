import math

import numpy as np
import pytest
from scipy.signal import fftconvolve

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir.physics.wave.fdtd import (
    FDTDReferenceConfig,
    ricker_source,
    simulate_fdtd_reference,
)
from puresound.audio.rir.physics.wave.source_convention import (
    convert_pressure_state_modal_residue,
    fdtd_cell_center_position,
    fdtd_pressure_cell_free_field_direct,
    fdtd_pressure_cell_to_free_field_input,
)


def test_matched_boundary_fdtd_direct_path_matches_source_convention():
    config = FDTDReferenceConfig(
        room_dim_m=(1.0, 1.0, 1.0),
        grid_spacing_m=0.05,
        duration_s=0.04,
        source_position_m=(0.275, 0.525, 0.525),
        receiver_position_m=(0.725, 0.525, 0.525),
        source_center_hz=400.0,
        source_delay_s=0.012,
    )
    matched_boundary = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=1.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=1.0,
    )
    result = simulate_fdtd_reference(
        config,
        boundary_admittance=matched_boundary,
    )
    source_position = np.asarray(
        fdtd_cell_center_position(
            result.source_cell_zyx,
            result.grid_spacing_xyz_m,
        )
    )
    receiver_position = np.asarray(
        fdtd_cell_center_position(
            result.receiver_cell_zyx,
            result.grid_spacing_xyz_m,
        )
    )
    distance_m = float(np.linalg.norm(receiver_position - source_position))
    source = ricker_source(
        result.rir.size,
        result.time_step_s,
        config.source_center_hz,
        config.source_delay_s,
    )
    predicted = fdtd_pressure_cell_free_field_direct(
        source,
        time_step_s=result.time_step_s,
        cell_volume_m3=float(np.prod(result.grid_spacing_xyz_m)),
        distance_m=distance_m,
        sound_speed_m_s=config.sound_speed_m_s,
    )
    time_s = np.arange(result.rir.size) * result.time_step_s
    direct_center_s = (
        config.source_delay_s + distance_m / config.sound_speed_m_s
    )
    direct_window = np.abs(time_s - direct_center_s) < 0.004
    amplitude_scale = float(
        np.dot(result.rir[direct_window], predicted[direct_window])
        / np.dot(predicted[direct_window], predicted[direct_window])
    )
    correlation = float(
        np.corrcoef(
            result.rir[direct_window],
            predicted[direct_window],
        )[0, 1]
    )

    assert correlation > 0.94
    assert amplitude_scale == pytest.approx(1.0, abs=0.12)


@pytest.mark.parametrize("sample_rate_hz", [4000.0, 8000.0, 32000.0])
def test_residue_transform_reproduces_pressure_state_modal_response(
    sample_rate_hz,
):
    duration_s = 0.3
    time_step_s = 1.0 / sample_rate_hz
    time_s = np.arange(round(duration_s * sample_rate_hz)) * time_step_s
    sound_speed_m_s = 343.0
    cell_volume_m3 = 0.04**3
    pole = -20.0 + 1j * 2.0 * math.pi * 120.0
    continuous_coupling = 0.3 + 0.1j
    source_argument = math.pi * 150.0 * (time_s - 0.025)
    source = (
        (1.0 - 2.0 * source_argument**2)
        * np.exp(-(source_argument**2))
    )

    pressure_state_response = np.real(
        cell_volume_m3
        * continuous_coupling
        * np.exp(pole * time_s)
    )
    expected = fftconvolve(source, pressure_state_response)[: time_s.size]
    free_field_input = fdtd_pressure_cell_to_free_field_input(
        source,
        time_step_s=time_step_s,
        cell_volume_m3=cell_volume_m3,
        sound_speed_m_s=sound_speed_m_s,
    )
    digital_residue = convert_pressure_state_modal_residue(
        continuous_coupling,
        pole,
        sample_rate_hz=sample_rate_hz,
        sound_speed_m_s=sound_speed_m_s,
    )
    digital_rir_mode = np.real(digital_residue * np.exp(pole * time_s))
    actual = fftconvolve(free_field_input, digital_rir_mode)[: time_s.size]
    comparison = time_s >= 0.05
    nrmse = float(
        np.linalg.norm(actual[comparison] - expected[comparison])
        / np.linalg.norm(expected[comparison])
    )

    assert nrmse < 0.007


def test_transformed_digital_rir_frequency_response_is_sample_rate_stable():
    sound_speed_m_s = 343.0
    distance_m = sound_speed_m_s * 0.0015
    frequencies_hz = np.linspace(60.0, 240.0, 37)
    poles = (
        -14.0 + 1j * 2.0 * math.pi * 90.0,
        -22.0 + 1j * 2.0 * math.pi * 160.0,
    )
    residues = (0.5 + 0.1j, -0.2 + 0.35j)

    def response(sample_rate_hz):
        time_s = (
            np.arange(round(0.8 * sample_rate_hz), dtype=np.float64)
            / sample_rate_hz
        )
        rir = np.zeros(time_s.size, dtype=np.float64)
        rir[round(distance_m / sound_speed_m_s * sample_rate_hz)] = (
            1.0 / distance_m
        )
        for pole, residue in zip(poles, residues):
            converted = convert_pressure_state_modal_residue(
                residue,
                pole,
                sample_rate_hz=sample_rate_hz,
                sound_speed_m_s=sound_speed_m_s,
            )
            rir += np.real(converted * np.exp(pole * time_s))
        return np.asarray(
            [
                np.sum(rir * np.exp(-2j * math.pi * frequency * time_s))
                for frequency in frequencies_hz
            ]
        )

    reference = response(32000.0)
    candidate = response(4000.0)
    relative_error = np.abs(candidate - reference) / np.maximum(
        np.abs(reference),
        1e-12,
    )

    assert float(np.max(relative_error)) < 0.05
