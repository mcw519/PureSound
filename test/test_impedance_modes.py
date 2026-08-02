import json
import math

import pytest

from puresound.audio.acoustic_impedance import (
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
)
from puresound.audio.impedance_modes import (
    RECTANGULAR_BOUNDARIES,
    RectangularImpedanceBoundaryConfig,
    solve_1d_impedance_cavity_modes,
    solve_rectangular_impedance_modes,
)


def test_static_boundary_cavity_mode_matches_closed_form_frequency_decay_and_q():
    length_m = 2.0
    sound_speed_m_s = 343.0
    normalized_admittance = 0.02
    boundary = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=normalized_admittance,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    reflection = (1.0 - normalized_admittance) / (
        1.0 + normalized_admittance
    )

    mode = solve_1d_impedance_cavity_modes(
        length_m,
        boundary,
        sound_speed_m_s=sound_speed_m_s,
        mode_indices=(1,),
    )[0]

    expected_frequency = sound_speed_m_s / (2.0 * length_m)
    expected_decay = -sound_speed_m_s / length_m * math.log(reflection)
    assert mode.frequency_hz == pytest.approx(expected_frequency, rel=1e-9)
    assert mode.amplitude_decay_rate_per_s == pytest.approx(
        expected_decay,
        rel=1e-9,
    )
    assert mode.q_factor == pytest.approx(
        2.0
        * math.pi
        * expected_frequency
        / (2.0 * expected_decay),
        rel=1e-9,
    )
    assert mode.residual < 1e-9


def test_phase_aware_multi_pole_boundary_shifts_complex_cavity_mode():
    length_m = 2.0
    phase_aware = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.005,
        pole_frequencies_hz=(80.0, 240.0),
        normalized_admittance_lowpass=(0.0, 0.0),
        normalized_admittance_highpass=(0.25, 0.55),
    )
    reference_frequency = 85.75
    reflection_magnitude = abs(
        phase_aware.reflection_coefficient(reference_frequency)
    )
    magnitude_only_admittance = (
        1.0 - reflection_magnitude
    ) / (1.0 + reflection_magnitude)
    magnitude_only = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=magnitude_only_admittance,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )

    phase_mode = solve_1d_impedance_cavity_modes(
        length_m,
        phase_aware,
    )[0]
    magnitude_mode = solve_1d_impedance_cavity_modes(
        length_m,
        magnitude_only,
    )[0]

    assert abs(phase_mode.frequency_hz - magnitude_mode.frequency_hz) > 5.0
    assert phase_mode.amplitude_decay_rate_per_s > 0.0
    assert phase_mode.q_factor > 0.0
    json.dumps(phase_mode.metadata(), allow_nan=False)


def test_measured_liner_resonance_changes_complex_cavity_decay():
    boundary = PassiveResonantAdmittance(
        normalized_admittance_static=0.03862573737771217,
        resonance_frequencies_hz=(1646.6617905770727,),
        quality_factors=(11.693489607418242,),
        peak_normalized_admittances=(7.69153609500207,),
    )
    length_m = 0.10415
    rigid_frequency_hz = 343.0 / (2.0 * length_m)
    reflection_magnitude = abs(
        boundary.reflection_coefficient(rigid_frequency_hz)
    )
    magnitude_only = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=(
            (1.0 - reflection_magnitude)
            / (1.0 + reflection_magnitude)
        ),
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )

    resonant_mode = solve_1d_impedance_cavity_modes(
        length_m,
        boundary,
    )[0]
    magnitude_only_mode = solve_1d_impedance_cavity_modes(
        length_m,
        magnitude_only,
    )[0]

    assert resonant_mode.frequency_hz == pytest.approx(1646.2, abs=0.5)
    assert resonant_mode.q_factor > 2.0 * magnitude_only_mode.q_factor
    assert resonant_mode.residual < 1e-8


def _boundary_config(
    x_boundary,
    *,
    y_boundary=None,
    z_boundary=None,
):
    rigid = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    y_boundary = rigid if y_boundary is None else y_boundary
    z_boundary = rigid if z_boundary is None else z_boundary
    return RectangularImpedanceBoundaryConfig(
        reference_id="synthetic_validation",
        boundaries={
            "west": x_boundary,
            "east": x_boundary,
            "south": y_boundary,
            "north": y_boundary,
            "floor": z_boundary,
            "ceiling": z_boundary,
        },
        source={"evidence_tier": "synthetic_validation"},
        applicability={
            "scope": "solver_validation_only",
            "automatic_scene_catalog_mapping": False,
        },
    )


def test_3d_static_boundary_reduces_to_exact_1d_axis_mode():
    boundary = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.02,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    one_dimensional = solve_1d_impedance_cavity_modes(
        2.0,
        boundary,
    )[0]
    three_dimensional = solve_rectangular_impedance_modes(
        (2.0, 3.0, 4.0),
        _boundary_config(boundary),
        mode_indices=((1, 0, 0),),
    )[0]

    assert three_dimensional.frequency_hz == pytest.approx(
        one_dimensional.frequency_hz,
        rel=1e-10,
    )
    assert three_dimensional.amplitude_decay_rate_per_s == pytest.approx(
        one_dimensional.amplitude_decay_rate_per_s,
        rel=1e-10,
    )
    assert three_dimensional.q_factor == pytest.approx(
        one_dimensional.q_factor,
        rel=1e-10,
    )
    assert three_dimensional.residual < 1e-8
    assert abs(
        three_dimensional.eigenfunction_at((0.5, 1.5, 2.0))
    ) > 0.0
    json.dumps(three_dimensional.metadata(), allow_nan=False)


def test_3d_uniform_cube_preserves_axis_permutation_degeneracy():
    boundary = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.01,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    config = RectangularImpedanceBoundaryConfig(
        reference_id="uniform_cube",
        boundaries={name: boundary for name in RECTANGULAR_BOUNDARIES},
        source={"evidence_tier": "synthetic_validation"},
        applicability={"scope": "solver_validation_only"},
    )

    modes = solve_rectangular_impedance_modes(
        (3.0, 3.0, 3.0),
        config,
        mode_indices=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    assert len(modes) == 3
    assert max(mode.frequency_hz for mode in modes) == pytest.approx(
        min(mode.frequency_hz for mode in modes),
        rel=1e-10,
    )
    assert max(
        mode.amplitude_decay_rate_per_s for mode in modes
    ) == pytest.approx(
        min(mode.amplitude_decay_rate_per_s for mode in modes),
        rel=1e-10,
    )
    assert all(mode.residual < 1e-8 for mode in modes)


def test_3d_phase_aware_axis_matches_1d_nonlinear_boundary():
    phase_aware = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.005,
        pole_frequencies_hz=(80.0, 240.0),
        normalized_admittance_lowpass=(0.0, 0.0),
        normalized_admittance_highpass=(0.25, 0.55),
    )

    one_dimensional = solve_1d_impedance_cavity_modes(
        2.0,
        phase_aware,
    )[0]
    three_dimensional = solve_rectangular_impedance_modes(
        (2.0, 3.0, 4.0),
        _boundary_config(phase_aware),
        mode_indices=((1, 0, 0),),
    )[0]

    assert three_dimensional.frequency_hz == pytest.approx(
        one_dimensional.frequency_hz,
        rel=1e-8,
    )
    assert three_dimensional.amplitude_decay_rate_per_s == pytest.approx(
        one_dimensional.amplitude_decay_rate_per_s,
        rel=1e-8,
    )


def test_rectangular_boundary_config_json_round_trip(tmp_path):
    boundary = PassiveResonantAdmittance(
        normalized_admittance_static=0.04,
        resonance_frequencies_hz=(160.0,),
        quality_factors=(8.0,),
        peak_normalized_admittances=(1.5,),
    )
    expected = RectangularImpedanceBoundaryConfig(
        reference_id="resonant_fixture",
        boundaries={name: boundary for name in RECTANGULAR_BOUNDARIES},
        source={"evidence_tier": "synthetic_validation"},
        applicability={
            "scope": "solver_validation_only",
            "automatic_scene_catalog_mapping": False,
        },
    )
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(expected.metadata()), encoding="utf-8")

    loaded = RectangularImpedanceBoundaryConfig.from_json(path)

    assert loaded.reference_id == expected.reference_id
    assert loaded.metadata() == expected.metadata()
