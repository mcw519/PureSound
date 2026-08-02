"""Causal atmospheric absorption utilities for geometric RIR rendering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import firwin2, lfilter, minimum_phase


AIR_ABSORPTION_POLICY = "puresound.iso9613_1.minimum_phase_direct.v1"


def atmospheric_absorption_db_per_m(
    frequencies_hz: Any,
    *,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float,
) -> np.ndarray:
    """Return ISO 9613-1 atmospheric attenuation in dB/m."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("air-absorption frequencies must be finite and non-negative")
    temperature_k = float(temperature_c) + 273.15
    pressure_ratio = float(pressure_pa) / 101325.0
    humidity_fraction = float(relative_humidity_percent) / 100.0
    if not 223.15 <= temperature_k <= 333.15:
        raise ValueError("temperature is outside the supported ISO range")
    if not 0.0 <= humidity_fraction <= 1.0:
        raise ValueError("relative humidity must lie in [0, 100] percent")
    if not math.isfinite(pressure_ratio) or pressure_ratio <= 0.0:
        raise ValueError("pressure must be finite and positive")

    reference_temperature_k = 293.15
    saturation_pressure_ratio = 10.0 ** (
        -6.8346 * (273.16 / temperature_k) ** 1.261 + 4.6151
    )
    molar_water_concentration = (
        humidity_fraction * saturation_pressure_ratio / pressure_ratio
    )
    oxygen_relaxation_hz = pressure_ratio * (
        24.0
        + 4.04e4
        * molar_water_concentration
        * (0.02 + molar_water_concentration)
        / (0.391 + molar_water_concentration)
    )
    nitrogen_relaxation_hz = (
        pressure_ratio
        * (temperature_k / reference_temperature_k) ** -0.5
        * (
            9.0
            + 280.0
            * molar_water_concentration
            * math.exp(
                -4.17
                * (
                    (temperature_k / reference_temperature_k) ** (-1.0 / 3.0)
                    - 1.0
                )
            )
        )
    )
    classical = (
        1.84e-11
        * pressure_ratio**-1.0
        * math.sqrt(temperature_k / reference_temperature_k)
    )
    molecular = (
        (temperature_k / reference_temperature_k) ** -2.5
        * (
            0.01275
            * math.exp(-2239.1 / temperature_k)
            / (
                oxygen_relaxation_hz
                + np.square(frequencies) / oxygen_relaxation_hz
            )
            + 0.1068
            * math.exp(-3352.0 / temperature_k)
            / (
                nitrogen_relaxation_hz
                + np.square(frequencies) / nitrogen_relaxation_hz
            )
        )
    )
    return np.asarray(
        8.686 * np.square(frequencies) * (classical + molecular),
        dtype=np.float64,
    )


def minimum_phase_air_absorption_filter(
    sample_rate: int,
    distance_m: float,
    *,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float,
    num_taps: int = 129,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Design a causal minimum-phase FIR for one direct propagation distance."""

    if int(sample_rate) <= 0:
        raise ValueError("sample rate must be positive")
    distance = float(distance_m)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("air-absorption distance must be finite and non-negative")
    taps = int(num_taps)
    if taps < 33 or taps % 2 == 0:
        raise ValueError("air-absorption FIR length must be odd and at least 33")
    nyquist = 0.5 * float(sample_rate)
    frequencies = np.linspace(0.0, nyquist, 257, dtype=np.float64)
    attenuation = atmospheric_absorption_db_per_m(
        frequencies,
        temperature_c=temperature_c,
        relative_humidity_percent=relative_humidity_percent,
        pressure_pa=pressure_pa,
    )
    target_gain = np.power(10.0, -attenuation * distance / 20.0)
    linear_phase = firwin2(
        taps,
        frequencies,
        np.square(target_gain),
        fs=float(sample_rate),
    )
    causal = minimum_phase(linear_phase, method="homomorphic", half=True)
    dc_gain = float(np.sum(causal))
    if not math.isfinite(dc_gain) or abs(dc_gain) <= 1e-12:
        raise ValueError("air-absorption filter has invalid DC gain")
    causal = np.asarray(causal / dc_gain, dtype=np.float64)
    metadata = {
        "policy": AIR_ABSORPTION_POLICY,
        "distance_m": distance,
        "temperature_c": float(temperature_c),
        "relative_humidity_percent": float(relative_humidity_percent),
        "pressure_pa": float(pressure_pa),
        "filter_taps": int(causal.size),
        "attenuation_db_per_m": {
            f"{frequency:g}": float(value)
            for frequency, value in zip(
                (1000.0, 2000.0, 4000.0, min(8000.0, nyquist)),
                atmospheric_absorption_db_per_m(
                    (1000.0, 2000.0, 4000.0, min(8000.0, nyquist)),
                    temperature_c=temperature_c,
                    relative_humidity_percent=relative_humidity_percent,
                    pressure_pa=pressure_pa,
                ),
            )
        },
    }
    return causal, metadata


def apply_air_absorption(
    signal: Any,
    sample_rate: int,
    distance_m: float,
    *,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float,
    num_taps: int = 129,
) -> tuple[np.ndarray, dict[str, Any]]:
    coefficients, metadata = minimum_phase_air_absorption_filter(
        sample_rate,
        distance_m,
        temperature_c=temperature_c,
        relative_humidity_percent=relative_humidity_percent,
        pressure_pa=pressure_pa,
        num_taps=num_taps,
    )
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("air absorption expects one RIR channel")
    return np.asarray(lfilter(coefficients, [1.0], values)), metadata


def air_adjusted_rt60_s(
    material_rt60_s: float,
    frequency_hz: float,
    sound_speed_m_s: float,
    *,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_pa: float,
) -> float:
    """Combine material decay with atmospheric dB loss along a growing path."""

    rt60 = float(material_rt60_s)
    sound_speed = float(sound_speed_m_s)
    if not math.isfinite(rt60) or rt60 <= 0.0:
        raise ValueError("material RT60 must be finite and positive")
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    attenuation = float(
        atmospheric_absorption_db_per_m(
            (frequency_hz,),
            temperature_c=temperature_c,
            relative_humidity_percent=relative_humidity_percent,
            pressure_pa=pressure_pa,
        )[0]
    )
    return float(60.0 / (60.0 / rt60 + attenuation * sound_speed))


__all__ = [
    "AIR_ABSORPTION_POLICY",
    "air_adjusted_rt60_s",
    "apply_air_absorption",
    "atmospheric_absorption_db_per_m",
    "minimum_phase_air_absorption_filter",
]
