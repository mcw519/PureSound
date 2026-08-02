import cmath
import json
import math

import numpy as np
import pytest

from puresound.audio.acoustic_impedance import (
    DigitalBoundaryReflectionFilter,
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    characteristic_impedance_pa_s_m,
    digital_locally_reacting_reflection_filter,
    digital_normalized_admittance_filter,
    impedance_from_absorption_and_phase,
    impedance_from_normal_incidence_reflection,
    normal_incidence_absorption_coefficient,
    normal_incidence_reflection_coefficient,
)


AIR_DENSITY = 1.204
SOUND_SPEED = 343.0


def test_characteristic_impedance_and_matched_surface():
    z0 = characteristic_impedance_pa_s_m(AIR_DENSITY, SOUND_SPEED)

    assert z0 == pytest.approx(AIR_DENSITY * SOUND_SPEED)
    assert normal_incidence_reflection_coefficient(
        z0,
        AIR_DENSITY,
        SOUND_SPEED,
    ) == pytest.approx(0.0j)
    assert normal_incidence_absorption_coefficient(
        z0,
        AIR_DENSITY,
        SOUND_SPEED,
    ) == pytest.approx(1.0)


def test_complex_reflection_and_impedance_round_trip():
    expected_reflection = 0.62 * cmath.exp(0.73j)

    impedance = impedance_from_normal_incidence_reflection(
        expected_reflection,
        AIR_DENSITY,
        SOUND_SPEED,
    )
    actual_reflection = normal_incidence_reflection_coefficient(
        impedance,
        AIR_DENSITY,
        SOUND_SPEED,
    )

    assert impedance.real > 0.0
    assert actual_reflection == pytest.approx(expected_reflection)
    assert normal_incidence_absorption_coefficient(
        impedance,
        AIR_DENSITY,
        SOUND_SPEED,
    ) == pytest.approx(1.0 - abs(expected_reflection) ** 2)


def test_absorption_does_not_determine_reflection_phase_or_impedance():
    absorption = 0.35
    phases = (0.0, 0.8)
    impedances = [
        impedance_from_absorption_and_phase(
            absorption,
            phase,
            AIR_DENSITY,
            SOUND_SPEED,
        )
        for phase in phases
    ]

    assert impedances[0] != pytest.approx(impedances[1])
    assert impedances[0].imag == pytest.approx(0.0)
    assert impedances[1].imag != pytest.approx(0.0)
    for phase, impedance in zip(phases, impedances):
        reflection = normal_incidence_reflection_coefficient(
            impedance,
            AIR_DENSITY,
            SOUND_SPEED,
        )
        assert abs(reflection) == pytest.approx(math.sqrt(1.0 - absorption))
        assert cmath.phase(reflection) == pytest.approx(phase)
        assert normal_incidence_absorption_coefficient(
            impedance,
            AIR_DENSITY,
            SOUND_SPEED,
        ) == pytest.approx(absorption)


def test_rigid_pressure_release_and_active_boundaries():
    assert normal_incidence_reflection_coefficient(
        float("inf"),
        AIR_DENSITY,
        SOUND_SPEED,
    ) == pytest.approx(1.0 + 0.0j)
    assert normal_incidence_reflection_coefficient(
        0.0,
        AIR_DENSITY,
        SOUND_SPEED,
    ) == pytest.approx(-1.0 + 0.0j)
    with pytest.raises(ValueError, match="negative real"):
        normal_incidence_reflection_coefficient(
            complex(-1.0, 4.0),
            AIR_DENSITY,
            SOUND_SPEED,
        )
    with pytest.raises(ValueError, match="magnitude"):
        impedance_from_normal_incidence_reflection(
            1.01,
            AIR_DENSITY,
            SOUND_SPEED,
        )


def test_relaxation_admittance_is_passive_and_has_expected_limits():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.2,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )

    assert model.normalized_admittance(0.0) == pytest.approx(0.8 + 0.0j)
    assert model.normalized_admittance(1e12) == pytest.approx(
        0.2 + 0.0j,
        abs=1e-9,
    )
    for frequency_hz in np.linspace(0.0, 2000.0, 1001):
        impedance = model.surface_impedance_pa_s_m(
            frequency_hz,
            AIR_DENSITY,
            SOUND_SPEED,
        )
        reflection = model.reflection_coefficient(frequency_hz)
        assert impedance.real >= 0.0
        assert abs(reflection) <= 1.0 + 1e-12
        assert 0.0 <= model.absorption_coefficient(frequency_hz) <= 1.0


def test_relaxation_reflection_filter_matches_analog_low_band():
    sample_rate_hz = 16000
    num_samples = sample_rate_hz
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.18,
        normalized_admittance_relaxation=0.55,
        relaxation_frequency_hz=110.0,
    )

    impulse = model.reflection_impulse_response(sample_rate_hz, num_samples)
    spectrum = np.fft.rfft(impulse)
    frequencies_hz = np.fft.rfftfreq(num_samples, 1.0 / sample_rate_hz)

    assert np.all(np.isfinite(impulse))
    for frequency_hz in (0.0, 40.0, 110.0, 300.0):
        index = int(np.argmin(np.abs(frequencies_hz - frequency_hz)))
        discrete = model.digital_reflection_coefficient(
            frequency_hz,
            sample_rate_hz,
        )
        analog = model.reflection_coefficient(frequency_hz)
        assert spectrum[index] == pytest.approx(discrete, abs=1e-10)
        assert discrete == pytest.approx(analog, abs=0.003)

    numerator, denominator = model.digital_reflection_filter(sample_rate_hz)
    pole = -denominator[1]
    assert abs(pole) < 1.0
    assert numerator != denominator


def test_relaxation_model_rejects_non_passive_or_invalid_parameters():
    with pytest.raises(ValueError, match="cannot be negative"):
        FirstOrderRelaxationAdmittance(-0.1, 0.2, 100.0)
    compliant = FirstOrderRelaxationAdmittance(0.2, -0.15, 100.0)
    assert compliant.normalized_admittance_zero == pytest.approx(0.05)
    with pytest.raises(ValueError, match="zero-frequency"):
        FirstOrderRelaxationAdmittance(0.1, -0.2, 100.0)
    with pytest.raises(ValueError, match="must be positive"):
        FirstOrderRelaxationAdmittance(0.1, 0.2, 0.0)


def test_multi_pole_admittance_is_passive_and_has_expected_endpoints():
    model = PassiveMultiPoleAdmittance(
        normalized_admittance_static=0.02,
        pole_frequencies_hz=(80.0, 400.0),
        normalized_admittance_lowpass=(0.10, 0.03),
        normalized_admittance_highpass=(0.40, 0.70),
    )

    assert model.normalized_admittance(0.0) == pytest.approx(0.15 + 0.0j)
    assert model.normalized_admittance(1e12) == pytest.approx(
        1.12 + 0.0j,
        abs=1e-8,
    )
    assert model.normalized_admittance_zero == pytest.approx(0.15)
    assert model.normalized_admittance_infinite == pytest.approx(1.12)
    for frequency_hz in np.linspace(0.0, 4000.0, 2001):
        admittance = model.normalized_admittance(frequency_hz)
        reflection = model.reflection_coefficient(frequency_hz)
        assert admittance.real >= 0.0
        assert abs(reflection) <= 1.0 + 1e-12
    assert len(model.digital_lowpass_coefficients(16000.0)) == 2
    assert model.metadata()["passive"] is True


def test_multi_pole_admittance_rejects_invalid_sections():
    with pytest.raises(ValueError, match="at least one pole"):
        PassiveMultiPoleAdmittance(0.1, (), (), ())
    with pytest.raises(ValueError, match="lengths"):
        PassiveMultiPoleAdmittance(0.1, (100.0,), (), (0.2,))
    with pytest.raises(ValueError, match="strictly increasing"):
        PassiveMultiPoleAdmittance(
            0.1,
            (200.0, 100.0),
            (0.0, 0.0),
            (0.2, 0.2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        PassiveMultiPoleAdmittance(
            0.1,
            (100.0,),
            (-0.1,),
            (0.2,),
        )


def test_resonant_admittance_is_passive_and_bilinear_sections_are_stable():
    model = PassiveResonantAdmittance(
        normalized_admittance_static=0.03,
        resonance_frequencies_hz=(1600.0,),
        quality_factors=(12.0,),
        peak_normalized_admittances=(7.5,),
    )

    assert model.normalized_admittance(0.0) == pytest.approx(0.03 + 0.0j)
    assert model.normalized_admittance(1600.0).real == pytest.approx(7.53)
    assert model.normalized_admittance(1600.0).imag == pytest.approx(0.0)
    assert model.maximum_normalized_admittance_bound == pytest.approx(7.53)
    for frequency_hz in np.linspace(1.0, 6000.0, 3001):
        admittance = model.normalized_admittance(frequency_hz)
        assert admittance.real >= -1e-12
        assert abs(model.reflection_coefficient(frequency_hz)) <= 1.0 + 1e-12

    sample_rate_hz = 192000.0
    for frequency_hz in (500.0, 1600.0, 2500.0):
        assert model.digital_normalized_admittance(
            frequency_hz,
            sample_rate_hz,
        ) == pytest.approx(
            model.normalized_admittance(frequency_hz),
            abs=0.01,
        )
    _, _, _, a1, a2 = model.digital_biquad_coefficients(sample_rate_hz)[0]
    assert np.max(np.abs(np.roots((1.0, a1, a2)))) < 1.0
    assert model.metadata()["analog_branch"] == "series_r_l_c"


def test_resonant_admittance_rejects_invalid_sections():
    with pytest.raises(ValueError, match="at least one branch"):
        PassiveResonantAdmittance(0.0, (), (), ())
    with pytest.raises(ValueError, match="lengths"):
        PassiveResonantAdmittance(0.0, (1000.0,), (), (1.0,))
    with pytest.raises(ValueError, match="quality factors"):
        PassiveResonantAdmittance(0.0, (1000.0,), (0.0,), (1.0,))
    with pytest.raises(ValueError, match="non-negative"):
        PassiveResonantAdmittance(0.0, (1000.0,), (10.0,), (-1.0,))


@pytest.mark.parametrize(
    "model,sample_rate_hz,frequencies_hz",
    [
        (
            FirstOrderRelaxationAdmittance(0.18, 0.55, 110.0),
            16000.0,
            (0.0, 40.0, 110.0, 300.0, 1000.0),
        ),
        (
            PassiveMultiPoleAdmittance(
                normalized_admittance_static=0.02,
                pole_frequencies_hz=(80.0, 400.0),
                normalized_admittance_lowpass=(0.10, 0.03),
                normalized_admittance_highpass=(0.40, 0.70),
            ),
            16000.0,
            (0.0, 80.0, 240.0, 1000.0, 3000.0),
        ),
        (
            PassiveResonantAdmittance(
                normalized_admittance_static=0.03,
                resonance_frequencies_hz=(1600.0,),
                quality_factors=(12.0,),
                peak_normalized_admittances=(7.5,),
            ),
            192000.0,
            (0.0, 500.0, 1600.0, 2500.0, 6000.0),
        ),
    ],
)
def test_angle_reflection_filter_is_causal_stable_and_bounded_real(
    model,
    sample_rate_hz,
    frequencies_hz,
):
    incidence_cosine = 0.37
    admittance_numerator, admittance_denominator = (
        digital_normalized_admittance_filter(model, sample_rate_hz)
    )
    reflection_filter = digital_locally_reacting_reflection_filter(
        model,
        incidence_cosine,
        sample_rate_hz,
    )

    assert reflection_filter.maximum_pole_magnitude < 1.0
    assert reflection_filter.to_dict()["passive_by_construction"] is True
    for frequency_hz in frequencies_hz:
        delay = np.exp(-2j * np.pi * frequency_hz / sample_rate_hz)
        digital_admittance = (
            sum(
                value * delay**index
                for index, value in enumerate(admittance_numerator)
            )
            / sum(
                value * delay**index
                for index, value in enumerate(admittance_denominator)
            )
        )
        expected = (
            incidence_cosine - digital_admittance
        ) / (
            incidence_cosine + digital_admittance
        )
        actual = reflection_filter.frequency_response(frequency_hz)

        assert actual == pytest.approx(expected, abs=1e-11)
        assert abs(actual) <= 1.0 + 1e-10

    sweep = np.linspace(0.0, 0.5 * sample_rate_hz, 2049)
    assert max(
        abs(reflection_filter.frequency_response(float(frequency)))
        for frequency in sweep
    ) <= 1.0 + 1e-10


def test_angle_reflection_filter_impulse_matches_its_transfer_function():
    sample_rate_hz = 16000.0
    num_samples = 16384
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.18,
        normalized_admittance_relaxation=0.55,
        relaxation_frequency_hz=110.0,
    )
    reflection_filter = digital_locally_reacting_reflection_filter(
        model,
        0.42,
        sample_rate_hz,
    )
    restored = DigitalBoundaryReflectionFilter.from_dict(
        json.loads(json.dumps(reflection_filter.to_dict()))
    )
    impulse = reflection_filter.impulse_response(num_samples)
    spectrum = np.fft.rfft(impulse)
    frequencies = np.fft.rfftfreq(num_samples, 1.0 / sample_rate_hz)

    assert np.all(np.isfinite(impulse))
    assert np.count_nonzero(impulse[:1]) == 1
    assert restored.to_dict() == reflection_filter.to_dict()
    for frequency_hz in (0.0, 40.0, 110.0, 300.0):
        index = int(np.argmin(np.abs(frequencies - frequency_hz)))
        assert spectrum[index] == pytest.approx(
            reflection_filter.frequency_response(float(frequencies[index])),
            abs=1e-10,
        )
