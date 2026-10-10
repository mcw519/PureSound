import math

import numpy as np
import pytest

from puresound.audio.rir.physics.propagation import (
    air_adjusted_rt60_s,
    atmospheric_absorption_db_per_m,
    minimum_phase_air_absorption_filter,
)


#: ISO 9613-1 pure-tone atmospheric attenuation at 20 C, 50% RH and 101.325 kPa,
#: dB/m, from the standard's published tabulation.
ISO_9613_1_TABULATION_20C_50RH_DB_PER_M = {
    250.0: 0.001310, 500.0: 0.002728, 1000.0: 0.004665,
    2000.0: 0.009887, 4000.0: 0.029666, 8000.0: 0.105291,
}


def _iso_9613_1_reference(
    frequency_hz: float,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float = 101325.0,
) -> float:
    """ISO 9613-1 §6.2, transcribed separately from the implementation.

    Deliberately a second transcription rather than a call into
    ``propagation``: a unit slip in the implementation (the molar water-vapour
    concentration is the easy one to get wrong) would be inherited by every
    internal caller. Two independent readings of the standard disagree on such
    a slip; one reading compared with itself does not.
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


def test_iso_air_absorption_matches_the_standard_tabulation_in_magnitude_and_shape():
    """Pin the magnitude, not just the shape.

    A curve that is finite, zero at DC and rising can still be wrong by a large
    factor at high frequency; air absorption is what gives a room its
    high-frequency rolloff, so such an error shows up as a spectral tilt in
    every rendered bank.
    """

    frequencies = sorted(ISO_9613_1_TABULATION_20C_50RH_DB_PER_M)
    attenuation = atmospheric_absorption_db_per_m(
        np.asarray([0.0, *frequencies], dtype=float),
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )

    assert np.all(np.isfinite(attenuation))
    assert attenuation[0] == 0.0
    assert np.all(np.diff(attenuation) > 0.0)
    for frequency, value in zip(frequencies, attenuation[1:]):
        expected = ISO_9613_1_TABULATION_20C_50RH_DB_PER_M[frequency]
        assert value == pytest.approx(expected, rel=0.02), (
            f"{frequency:g} Hz: got {value:.6f} dB/m, ISO 9613-1 gives {expected:.6f}"
        )


def test_air_filter_is_causal_distance_dependent_and_shortens_rt60():
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

    adjusted = air_adjusted_rt60_s(
        1.2,
        8000.0,
        343.0,
        temperature_c=20.0,
        relative_humidity_percent=50.0,
        pressure_pa=101325.0,
    )

    assert 0.0 < adjusted < 1.2
