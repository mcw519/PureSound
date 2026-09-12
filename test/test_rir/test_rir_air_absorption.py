import math

import numpy as np
import pytest

from puresound.audio.rir.physics.propagation import (
    air_adjusted_rt60_s,
    atmospheric_absorption_db_per_m,
    minimum_phase_air_absorption_filter,
)


def test_iso_air_absorption_is_finite_and_increases_toward_high_frequency():
    frequencies = np.asarray([0.0, 1000.0, 2000.0, 4000.0, 8000.0])
    attenuation = atmospheric_absorption_db_per_m(
        frequencies,
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )

    assert np.all(np.isfinite(attenuation))
    assert attenuation[0] == 0.0
    assert np.all(np.diff(attenuation) > 0.0)


#: ISO 9613-1 pure-tone atmospheric attenuation, dB/m, at 101.325 kPa, over the
#: temperature and humidity range ``upgrade_hybrid_scene_to_v2`` samples.
#:
#: The 20 C / 50% row is the one to trust independently: it agrees with the
#: standard's published tabulation (0.0013, 0.0028, 0.0047, 0.0099, 0.029,
#: 0.105 dB/m) to three figures.  The other two rows come from
#: ``_iso_9613_1_reference`` below rather than from a table, so they are a
#: cross-check on transcription, not a second independent source.
ISO_9613_1_REFERENCE_DB_PER_M = {
    (20.0, 50.0): {
        250.0: 0.001310, 500.0: 0.002728, 1000.0: 0.004665,
        2000.0: 0.009887, 4000.0: 0.029666, 8000.0: 0.105291,
    },
    (15.0, 30.0): {
        250.0: 0.001207, 500.0: 0.002228, 1000.0: 0.005454,
        2000.0: 0.017769, 4000.0: 0.062494, 8000.0: 0.191755,
    },
    (25.0, 70.0): {
        250.0: 0.001056, 500.0: 0.003069, 1000.0: 0.006186,
        2000.0: 0.010399, 4000.0: 0.022006, 8000.0: 0.066240,
    },
}


def _iso_9613_1_reference(
    frequency_hz: float,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float = 101325.0,
) -> float:
    """ISO 9613-1 §6.2, transcribed separately from the implementation.

    Deliberately a second transcription rather than a call into
    ``propagation``: the defect this guards against was a unit slip on the
    molar water-vapour concentration, which every internal caller would have
    inherited.  Two independent readings of the standard disagree on that; one
    reading compared with itself does not.
    """

    temperature_k = temperature_c + 273.15
    reference_k = 293.15
    triple_point_k = 273.16
    pressure_ratio = pressure_pa / 101325.0
    temperature_ratio = temperature_k / reference_k
    saturation_ratio = 10.0 ** (
        -6.8346 * (triple_point_k / temperature_k) ** 1.261 + 4.6151
    )
    # h is a percentage, and so is the relative humidity feeding it.
    molar_water = relative_humidity_percent * saturation_ratio / pressure_ratio
    oxygen_hz = pressure_ratio * (
        24.0
        + 4.04e4 * molar_water * (0.02 + molar_water) / (0.391 + molar_water)
    )
    nitrogen_hz = (
        pressure_ratio
        * temperature_ratio**-0.5
        * (
            9.0
            + 280.0
            * molar_water
            * math.exp(-4.170 * (temperature_ratio ** (-1.0 / 3.0) - 1.0))
        )
    )
    classical = 1.84e-11 / pressure_ratio * math.sqrt(temperature_ratio)
    squared = frequency_hz * frequency_hz
    molecular = temperature_ratio**-2.5 * (
        0.01275
        * math.exp(-2239.1 / temperature_k)
        / (oxygen_hz + squared / oxygen_hz)
        + 0.1068
        * math.exp(-3352.0 / temperature_k)
        / (nitrogen_hz + squared / nitrogen_hz)
    )
    return 8.686 * squared * (classical + molecular)


@pytest.mark.parametrize("temperature_c", [15.0, 20.0, 25.0])
@pytest.mark.parametrize("relative_humidity_percent", [30.0, 50.0, 70.0])
@pytest.mark.parametrize("frequency_hz", [250.0, 1000.0, 4000.0, 8000.0])
def test_iso_air_absorption_agrees_with_an_independent_transcription(
    temperature_c, relative_humidity_percent, frequency_hz
):
    """Cross-check the shipped curve against a separate reading of the standard."""

    value = atmospheric_absorption_db_per_m(
        np.asarray([frequency_hz]),
        temperature_c=temperature_c,
        relative_humidity_percent=relative_humidity_percent,
        pressure_pa=101325.0,
    )[0]
    expected = _iso_9613_1_reference(
        frequency_hz, temperature_c, relative_humidity_percent
    )

    assert value == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("condition", sorted(ISO_9613_1_REFERENCE_DB_PER_M))
def test_iso_air_absorption_matches_the_standard_tabulation(condition):
    """Pin the magnitude, not just the shape.

    The shape assertions above — finite, zero at DC, monotonically rising —
    all passed while the humidity unit was wrong by a factor of 100, which
    left attenuation 5x too low at 4 kHz and 7.8x too low at 8 kHz. A curve
    that rises is not the same as a curve that rises by the right amount, and
    air absorption is the mechanism that gives a room its high-frequency
    rolloff, so the error showed up as a spectral tilt in every rendered bank.
    """

    temperature_c, relative_humidity = condition
    reference = ISO_9613_1_REFERENCE_DB_PER_M[condition]
    frequencies = sorted(reference)
    attenuation = atmospheric_absorption_db_per_m(
        np.asarray(frequencies, dtype=float),
        temperature_c=temperature_c,
        relative_humidity_percent=relative_humidity,
        pressure_pa=101325.0,
    )

    for frequency, value in zip(frequencies, attenuation):
        expected = reference[frequency]
        assert value == pytest.approx(expected, rel=0.02), (
            f"{frequency:g} Hz at {temperature_c} C / {relative_humidity}% RH: "
            f"got {value:.6f} dB/m, ISO 9613-1 gives {expected:.6f}"
        )


def test_iso_air_absorption_spans_the_full_audio_dynamic_range():
    """A too-flat curve is the signature the unit error left behind.

    From 250 Hz to 8 kHz the standard spans about eighty-fold. The broken
    version spanned 4.3x, because with the relaxation frequencies collapsed
    only the classical term survived — and a flat curve cannot produce the
    high-frequency rolloff a real room has.
    """

    low, high = atmospheric_absorption_db_per_m(
        np.asarray([250.0, 8000.0]),
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )

    assert high / low > 40.0, (
        f"250 Hz to 8 kHz spans only {high / low:.1f}x; ISO gives about 80x"
    )


def test_air_filter_is_causal_and_longer_distance_has_more_high_frequency_loss():
    near, _ = minimum_phase_air_absorption_filter(
        16000,
        1.0,
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )
    far, metadata = minimum_phase_air_absorption_filter(
        16000,
        20.0,
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )
    near_high = abs(np.fft.rfft(near, 4096)[-1])
    far_high = abs(np.fft.rfft(far, 4096)[-1])

    assert near[0] != 0.0 and far[0] != 0.0
    assert far_high < near_high
    assert metadata["distance_m"] == 20.0


def test_air_loss_shortens_material_rt60():
    adjusted = air_adjusted_rt60_s(
        1.2,
        8000.0,
        343.0,
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )

    assert 0.0 < adjusted < 1.2
