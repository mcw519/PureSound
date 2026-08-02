import numpy as np

from puresound.audio.rir_air_absorption import (
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
