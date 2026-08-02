"""Locally reacting acoustic-impedance primitives.

The functions in this module deliberately keep reflection phase explicit.
An energy absorption coefficient determines only ``abs(reflection)`` and
cannot uniquely determine a complex surface impedance without an additional
phase assumption.

All impedance values use SI units of Pa*s/m. Material measurements remain
normal-incidence or explicitly model-derived; the digital local-reaction
renderer converts a normalized admittance into an angle-aware passive causal
reflection filter without inventing a new material phase.
"""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from scipy.signal import lfilter


DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION = (
    "puresound.digital_boundary_reflection_filter.v1"
)


class RationalAdmittanceModel(Protocol):
    """Passive rational admittance interface accepted by digital rendering."""

    def normalized_admittance(self, frequency_hz: float) -> complex:
        """Evaluate the continuous-time normalized admittance."""

    def metadata(self) -> dict[str, Any]:
        """Return versionable model metadata."""


@dataclass(frozen=True)
class FirstOrderRelaxationAdmittance:
    """Passive causal one-pole normalized surface-admittance model.

    The continuous-time model is

    ``y(s) = g_infinite + g_relaxation / (1 + s*tau)``,

    where ``y = rho*c*Y_surface`` is dimensionless.  The relaxation term may
    have either sign, but both endpoint admittances must be non-negative:
    ``g_zero = g_infinite + g_relaxation >= 0`` and ``g_infinite >= 0``.
    The real part at every intermediate frequency is a convex interpolation
    of those endpoints, so the model remains positive-real.
    """

    normalized_admittance_infinite: float
    normalized_admittance_relaxation: float
    relaxation_frequency_hz: float

    def __post_init__(self) -> None:
        for name, value in (
            (
                "normalized_admittance_infinite",
                self.normalized_admittance_infinite,
            ),
            (
                "normalized_admittance_relaxation",
                self.normalized_admittance_relaxation,
            ),
            ("relaxation_frequency_hz", self.relaxation_frequency_hz),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.normalized_admittance_infinite < 0.0:
            raise ValueError("infinite-frequency admittance cannot be negative")
        if (
            self.normalized_admittance_infinite
            + self.normalized_admittance_relaxation
            < 0.0
        ):
            raise ValueError("zero-frequency admittance cannot be negative")
        if self.relaxation_frequency_hz <= 0.0:
            raise ValueError("relaxation frequency must be positive")

    @property
    def relaxation_time_s(self) -> float:
        return 1.0 / (2.0 * math.pi * float(self.relaxation_frequency_hz))

    @property
    def normalized_admittance_zero(self) -> float:
        return float(
            self.normalized_admittance_infinite
            + self.normalized_admittance_relaxation
        )

    @property
    def maximum_normalized_admittance_bound(self) -> float:
        """Conservative all-frequency magnitude bound used by explicit solvers."""
        return float(
            max(
                self.normalized_admittance_zero,
                self.normalized_admittance_infinite,
            )
        )

    def normalized_admittance(self, frequency_hz: float) -> complex:
        """Evaluate dimensionless continuous-time admittance at ``j*omega``."""
        frequency = float(frequency_hz)
        if not math.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency must be finite and non-negative")
        ratio = frequency / float(self.relaxation_frequency_hz)
        return complex(
            float(self.normalized_admittance_infinite)
            + float(self.normalized_admittance_relaxation) / (1.0 + 1j * ratio)
        )

    def normalized_admittance_laplace(
        self,
        complex_angular_frequency_rad_s: complex,
    ) -> complex:
        """Evaluate normalized admittance at complex Laplace frequency ``s``."""
        complex_frequency = complex(complex_angular_frequency_rad_s)
        if not cmath.isfinite(complex_frequency):
            raise ValueError("complex angular frequency must be finite")
        return complex(
            float(self.normalized_admittance_infinite)
            + float(self.normalized_admittance_relaxation)
            / (1.0 + complex_frequency * self.relaxation_time_s)
        )

    def reflection_coefficient_laplace(
        self,
        complex_angular_frequency_rad_s: complex,
    ) -> complex:
        admittance = self.normalized_admittance_laplace(
            complex_angular_frequency_rad_s
        )
        return complex((1.0 - admittance) / (1.0 + admittance))

    def surface_impedance_pa_s_m(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> complex:
        """Evaluate surface impedance ``Z = rho*c/y`` in Pa*s/m."""
        normalized_admittance = self.normalized_admittance(frequency_hz)
        if abs(normalized_admittance) <= 1e-15:
            return complex(float("inf"), 0.0)
        return complex(
            characteristic_impedance_pa_s_m(
                air_density_kg_m3,
                sound_speed_m_s,
            )
            / normalized_admittance
        )

    def reflection_coefficient(self, frequency_hz: float) -> complex:
        """Evaluate normal-incidence pressure reflection ``(1-y)/(1+y)``."""
        normalized_admittance = self.normalized_admittance(frequency_hz)
        return complex(
            (1.0 - normalized_admittance)
            / (1.0 + normalized_admittance)
        )

    def absorption_coefficient(self, frequency_hz: float) -> float:
        reflection = self.reflection_coefficient(frequency_hz)
        return float(min(1.0, max(0.0, 1.0 - abs(reflection) ** 2)))

    def digital_lowpass_coefficients(
        self,
        sample_rate_hz: float,
    ) -> tuple[float, float, float]:
        """Return ``b0, b1, a1`` for the Tustin relaxation low-pass.

        The recurrence is ``u[n] = b0*p[n] + b1*p[n-1] - a1*u[n-1]``.
        Tustin maps the stable positive-real analog model into the unit circle.
        """
        sample_rate = float(sample_rate_hz)
        if not math.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("sample rate must be finite and positive")
        k = 2.0 * self.relaxation_time_s * sample_rate
        b = 1.0 / (1.0 + k)
        a1 = (1.0 - k) / (1.0 + k)
        return float(b), float(b), float(a1)

    def relaxation_sections(
        self,
    ) -> tuple[
        float,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        """Return static, pole, low-pass, and high-pass branch parameters."""
        relaxation = float(self.normalized_admittance_relaxation)
        if relaxation >= 0.0:
            return (
                float(self.normalized_admittance_infinite),
                (float(self.relaxation_frequency_hz),),
                (relaxation,),
                (0.0,),
            )
        return (
            float(self.normalized_admittance_zero),
            (float(self.relaxation_frequency_hz),),
            (0.0,),
            (-relaxation,),
        )

    def digital_reflection_filter(
        self,
        sample_rate_hz: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return first-order digital reflection numerator and denominator."""
        b0, b1, a1 = self.digital_lowpass_coefficients(sample_rate_hz)
        g_infinite = float(self.normalized_admittance_infinite)
        g_relaxation = float(self.normalized_admittance_relaxation)
        admittance_b0 = g_infinite + g_relaxation * b0
        admittance_b1 = g_infinite * a1 + g_relaxation * b1
        denominator_0 = 1.0 + admittance_b0
        numerator = (
            (1.0 - admittance_b0) / denominator_0,
            (a1 - admittance_b1) / denominator_0,
        )
        denominator = (
            1.0,
            (a1 + admittance_b1) / denominator_0,
        )
        return numerator, denominator

    def digital_reflection_coefficient(
        self,
        frequency_hz: float,
        sample_rate_hz: float,
    ) -> complex:
        """Evaluate the Tustin-discretized reflection filter."""
        frequency = float(frequency_hz)
        sample_rate = float(sample_rate_hz)
        if (
            not math.isfinite(frequency)
            or frequency < 0.0
            or frequency > 0.5 * sample_rate
        ):
            raise ValueError("frequency must lie between zero and Nyquist")
        numerator, denominator = self.digital_reflection_filter(sample_rate)
        delay = cmath.exp(-2j * math.pi * frequency / sample_rate)
        return complex(
            (numerator[0] + numerator[1] * delay)
            / (denominator[0] + denominator[1] * delay)
        )

    def reflection_impulse_response(
        self,
        sample_rate_hz: float,
        num_samples: int,
    ) -> np.ndarray:
        """Render the causal discrete single-wall reflection response."""
        if int(num_samples) < 1:
            raise ValueError("num_samples must be positive")
        numerator, denominator = self.digital_reflection_filter(sample_rate_hz)
        output = np.zeros(int(num_samples), dtype=np.float64)
        previous_input = 0.0
        previous_output = 0.0
        for sample in range(int(num_samples)):
            current_input = 1.0 if sample == 0 else 0.0
            current_output = (
                numerator[0] * current_input
                + numerator[1] * previous_input
                - denominator[1] * previous_output
            )
            output[sample] = current_output
            previous_input = current_input
            previous_output = current_output
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "first_order_relaxation_admittance",
            "units": {
                "surface_impedance": "Pa*s/m",
                "normalized_admittance": "dimensionless",
            },
            **asdict(self),
            "normalized_admittance_zero": self.normalized_admittance_zero,
            "relaxation_time_s": float(self.relaxation_time_s),
            "passive": True,
            "causal": True,
            "discretization": "bilinear_transform",
        }


@dataclass(frozen=True)
class PassiveMultiPoleAdmittance:
    """Positive-real parallel sum of first-order relaxation branches.

    The dimensionless surface admittance is:

    ``y(s) = g_static
             + sum(g_low[k] / (1 + s*tau[k]))
             + sum(g_high[k] * s*tau[k] / (1 + s*tau[k]))``.

    Every coefficient is non-negative. Each low-pass and high-pass term is
    positive-real, so their parallel sum is passive without a post-fit
    passivity repair.
    """

    normalized_admittance_static: float
    pole_frequencies_hz: tuple[float, ...]
    normalized_admittance_lowpass: tuple[float, ...]
    normalized_admittance_highpass: tuple[float, ...]

    def __post_init__(self) -> None:
        static = float(self.normalized_admittance_static)
        poles = tuple(float(value) for value in self.pole_frequencies_hz)
        lowpass = tuple(
            float(value) for value in self.normalized_admittance_lowpass
        )
        highpass = tuple(
            float(value) for value in self.normalized_admittance_highpass
        )
        if not math.isfinite(static) or static < 0.0:
            raise ValueError("static normalized admittance must be non-negative")
        if not poles:
            raise ValueError("multi-pole admittance requires at least one pole")
        if len(poles) != len(lowpass) or len(poles) != len(highpass):
            raise ValueError("pole and branch coefficient lengths must match")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in poles
        ):
            raise ValueError("pole frequencies must be finite and positive")
        if any(next_value <= value for value, next_value in zip(poles, poles[1:])):
            raise ValueError("pole frequencies must be strictly increasing")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (*lowpass, *highpass)
        ):
            raise ValueError("relaxation branch strengths must be non-negative")
        object.__setattr__(self, "normalized_admittance_static", static)
        object.__setattr__(self, "pole_frequencies_hz", poles)
        object.__setattr__(self, "normalized_admittance_lowpass", lowpass)
        object.__setattr__(self, "normalized_admittance_highpass", highpass)

    @property
    def normalized_admittance_zero(self) -> float:
        return float(
            self.normalized_admittance_static
            + sum(self.normalized_admittance_lowpass)
        )

    @property
    def normalized_admittance_infinite(self) -> float:
        return float(
            self.normalized_admittance_static
            + sum(self.normalized_admittance_highpass)
        )

    @property
    def maximum_normalized_admittance_bound(self) -> float:
        """Conservative all-frequency magnitude bound used by explicit solvers."""
        return float(
            self.normalized_admittance_static
            + sum(
                max(lowpass, highpass)
                for lowpass, highpass in zip(
                    self.normalized_admittance_lowpass,
                    self.normalized_admittance_highpass,
                )
            )
        )

    def normalized_admittance(self, frequency_hz: float) -> complex:
        frequency = float(frequency_hz)
        if not math.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency must be finite and non-negative")
        result = complex(self.normalized_admittance_static, 0.0)
        for pole, lowpass, highpass in zip(
            self.pole_frequencies_hz,
            self.normalized_admittance_lowpass,
            self.normalized_admittance_highpass,
        ):
            scaled = 1j * frequency / pole
            result += lowpass / (1.0 + scaled)
            result += highpass * scaled / (1.0 + scaled)
        return complex(result)

    def normalized_admittance_laplace(
        self,
        complex_angular_frequency_rad_s: complex,
    ) -> complex:
        complex_frequency = complex(complex_angular_frequency_rad_s)
        if not cmath.isfinite(complex_frequency):
            raise ValueError("complex angular frequency must be finite")
        result = complex(self.normalized_admittance_static, 0.0)
        for pole, lowpass, highpass in zip(
            self.pole_frequencies_hz,
            self.normalized_admittance_lowpass,
            self.normalized_admittance_highpass,
        ):
            scaled = complex_frequency / (2.0 * math.pi * pole)
            result += lowpass / (1.0 + scaled)
            result += highpass * scaled / (1.0 + scaled)
        return complex(result)

    def reflection_coefficient_laplace(
        self,
        complex_angular_frequency_rad_s: complex,
    ) -> complex:
        admittance = self.normalized_admittance_laplace(
            complex_angular_frequency_rad_s
        )
        return complex((1.0 - admittance) / (1.0 + admittance))

    def surface_impedance_pa_s_m(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> complex:
        normalized_admittance = self.normalized_admittance(frequency_hz)
        if abs(normalized_admittance) <= 1e-15:
            return complex(float("inf"), 0.0)
        return complex(
            characteristic_impedance_pa_s_m(
                air_density_kg_m3,
                sound_speed_m_s,
            )
            / normalized_admittance
        )

    def reflection_coefficient(self, frequency_hz: float) -> complex:
        normalized_admittance = self.normalized_admittance(frequency_hz)
        return complex(
            (1.0 - normalized_admittance)
            / (1.0 + normalized_admittance)
        )

    def absorption_coefficient(self, frequency_hz: float) -> float:
        reflection = self.reflection_coefficient(frequency_hz)
        return float(min(1.0, max(0.0, 1.0 - abs(reflection) ** 2)))

    def digital_lowpass_coefficients(
        self,
        sample_rate_hz: float,
    ) -> tuple[tuple[float, float, float], ...]:
        sample_rate = float(sample_rate_hz)
        if not math.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("sample rate must be finite and positive")
        result: list[tuple[float, float, float]] = []
        for frequency_hz in self.pole_frequencies_hz:
            relaxation_time = 1.0 / (2.0 * math.pi * frequency_hz)
            k = 2.0 * relaxation_time * sample_rate
            b = 1.0 / (1.0 + k)
            a1 = (1.0 - k) / (1.0 + k)
            result.append((float(b), float(b), float(a1)))
        return tuple(result)

    def relaxation_sections(
        self,
    ) -> tuple[
        float,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        return (
            float(self.normalized_admittance_static),
            tuple(self.pole_frequencies_hz),
            tuple(self.normalized_admittance_lowpass),
            tuple(self.normalized_admittance_highpass),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "passive_multi_pole_admittance",
            "units": {
                "surface_impedance": "Pa*s/m",
                "normalized_admittance": "dimensionless",
            },
            **asdict(self),
            "normalized_admittance_zero": self.normalized_admittance_zero,
            "normalized_admittance_infinite": (
                self.normalized_admittance_infinite
            ),
            "passive": True,
            "causal": True,
            "discretization": "parallel_bilinear_relaxation_sections",
        }


@dataclass(frozen=True)
class PassiveResonantAdmittance:
    """Passive parallel sum of damped series-RLC admittance branches.

    For each branch, with ``u = s / omega_0``,

    ``y_k(s) = (g_peak / Q) * u / (u**2 + u/Q + 1)``.

    This is the normalized admittance of a passive series RLC resonator.
    ``g_peak`` is its real normalized admittance at resonance. Parallel sums
    with a non-negative static conductance remain positive-real, while the
    conjugate pole pair can represent a Helmholtz-like resonance that real
    relaxation poles cannot.
    """

    normalized_admittance_static: float
    resonance_frequencies_hz: tuple[float, ...]
    quality_factors: tuple[float, ...]
    peak_normalized_admittances: tuple[float, ...]

    def __post_init__(self) -> None:
        static = float(self.normalized_admittance_static)
        frequencies = tuple(
            float(value) for value in self.resonance_frequencies_hz
        )
        quality_factors = tuple(float(value) for value in self.quality_factors)
        peaks = tuple(float(value) for value in self.peak_normalized_admittances)
        if not math.isfinite(static) or static < 0.0:
            raise ValueError("static normalized admittance must be non-negative")
        if not frequencies:
            raise ValueError("resonant admittance requires at least one branch")
        if len(frequencies) != len(quality_factors) or len(frequencies) != len(
            peaks
        ):
            raise ValueError("resonance frequency, Q, and peak lengths must match")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in frequencies
        ):
            raise ValueError(
                "resonance frequencies must be finite and positive"
            )
        if any(
            next_value <= value
            for value, next_value in zip(frequencies, frequencies[1:])
        ):
            raise ValueError("resonance frequencies must be strictly increasing")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in quality_factors
        ):
            raise ValueError("resonance quality factors must be finite and positive")
        if any(not math.isfinite(value) or value < 0.0 for value in peaks):
            raise ValueError(
                "peak normalized admittances must be finite and non-negative"
            )
        object.__setattr__(self, "normalized_admittance_static", static)
        object.__setattr__(self, "resonance_frequencies_hz", frequencies)
        object.__setattr__(self, "quality_factors", quality_factors)
        object.__setattr__(self, "peak_normalized_admittances", peaks)

    @property
    def normalized_admittance_zero(self) -> float:
        return float(self.normalized_admittance_static)

    @property
    def normalized_admittance_infinite(self) -> float:
        return float(self.normalized_admittance_static)

    @property
    def maximum_normalized_admittance_bound(self) -> float:
        """Conservative bound from the sum of individual resonance peaks."""
        return float(
            self.normalized_admittance_static
            + sum(self.peak_normalized_admittances)
        )

    def normalized_admittance(self, frequency_hz: float) -> complex:
        frequency = float(frequency_hz)
        if not math.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency must be finite and non-negative")
        return self.normalized_admittance_laplace(
            2j * math.pi * frequency
        )

    def normalized_admittance_laplace(
        self,
        complex_angular_frequency_rad_s: complex,
    ) -> complex:
        complex_frequency = complex(complex_angular_frequency_rad_s)
        if not cmath.isfinite(complex_frequency):
            raise ValueError("complex angular frequency must be finite")
        result = complex(self.normalized_admittance_static, 0.0)
        for frequency_hz, quality_factor, peak in zip(
            self.resonance_frequencies_hz,
            self.quality_factors,
            self.peak_normalized_admittances,
        ):
            scaled = complex_frequency / (
                2.0 * math.pi * frequency_hz
            )
            result += (
                (peak / quality_factor)
                * scaled
                / (
                    scaled * scaled
                    + scaled / quality_factor
                    + 1.0
                )
            )
        return complex(result)

    def reflection_coefficient_laplace(
        self,
        complex_angular_frequency_rad_s: complex,
    ) -> complex:
        admittance = self.normalized_admittance_laplace(
            complex_angular_frequency_rad_s
        )
        return complex((1.0 - admittance) / (1.0 + admittance))

    def surface_impedance_pa_s_m(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> complex:
        normalized_admittance = self.normalized_admittance(frequency_hz)
        if abs(normalized_admittance) <= 1e-15:
            return complex(float("inf"), 0.0)
        return complex(
            characteristic_impedance_pa_s_m(
                air_density_kg_m3,
                sound_speed_m_s,
            )
            / normalized_admittance
        )

    def reflection_coefficient(self, frequency_hz: float) -> complex:
        normalized_admittance = self.normalized_admittance(frequency_hz)
        return complex(
            (1.0 - normalized_admittance)
            / (1.0 + normalized_admittance)
        )

    def absorption_coefficient(self, frequency_hz: float) -> float:
        reflection = self.reflection_coefficient(frequency_hz)
        return float(min(1.0, max(0.0, 1.0 - abs(reflection) ** 2)))

    def relaxation_sections(
        self,
    ) -> tuple[
        float,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        """Return no first-order sections, preserving the FDTD common API."""
        return (
            float(self.normalized_admittance_static),
            (),
            (),
            (),
        )

    def digital_lowpass_coefficients(
        self,
        sample_rate_hz: float,
    ) -> tuple[tuple[float, float, float], ...]:
        sample_rate = float(sample_rate_hz)
        if not math.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("sample rate must be finite and positive")
        return ()

    def digital_biquad_coefficients(
        self,
        sample_rate_hz: float,
    ) -> tuple[tuple[float, float, float, float, float], ...]:
        """Return Tustin ``b0,b1,b2,a1,a2`` for every RLC branch."""
        sample_rate = float(sample_rate_hz)
        if not math.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("sample rate must be finite and positive")
        result: list[tuple[float, float, float, float, float]] = []
        for frequency_hz, quality_factor, peak in zip(
            self.resonance_frequencies_hz,
            self.quality_factors,
            self.peak_normalized_admittances,
        ):
            if frequency_hz >= 0.5 * sample_rate:
                raise ValueError(
                    "resonance frequency must be below the digital Nyquist rate"
                )
            omega = 2.0 * math.pi * frequency_hz
            bilinear_rate = omega / math.tan(
                math.pi * frequency_hz / sample_rate
            )
            damping = omega / quality_factor
            numerator_scale = peak * omega / quality_factor
            denominator_0 = (
                bilinear_rate**2
                + damping * bilinear_rate
                + omega**2
            )
            result.append(
                (
                    float(numerator_scale * bilinear_rate / denominator_0),
                    0.0,
                    float(-numerator_scale * bilinear_rate / denominator_0),
                    float(
                        (-2.0 * bilinear_rate**2 + 2.0 * omega**2)
                        / denominator_0
                    ),
                    float(
                        (
                            bilinear_rate**2
                            - damping * bilinear_rate
                            + omega**2
                        )
                        / denominator_0
                    ),
                )
            )
        return tuple(result)

    def digital_normalized_admittance(
        self,
        frequency_hz: float,
        sample_rate_hz: float,
    ) -> complex:
        """Evaluate the Tustin-discretized normalized admittance."""
        frequency = float(frequency_hz)
        sample_rate = float(sample_rate_hz)
        if (
            not math.isfinite(frequency)
            or frequency < 0.0
            or frequency > 0.5 * sample_rate
        ):
            raise ValueError("frequency must lie between zero and Nyquist")
        delay = cmath.exp(-2j * math.pi * frequency / sample_rate)
        delay_squared = delay * delay
        result = complex(self.normalized_admittance_static, 0.0)
        for b0, b1, b2, a1, a2 in self.digital_biquad_coefficients(
            sample_rate
        ):
            result += (
                b0 + b1 * delay + b2 * delay_squared
            ) / (
                1.0 + a1 * delay + a2 * delay_squared
            )
        return complex(result)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "passive_resonant_admittance",
            "units": {
                "surface_impedance": "Pa*s/m",
                "normalized_admittance": "dimensionless",
            },
            **asdict(self),
            "normalized_admittance_zero": self.normalized_admittance_zero,
            "normalized_admittance_infinite": (
                self.normalized_admittance_infinite
            ),
            "maximum_normalized_admittance_bound": (
                self.maximum_normalized_admittance_bound
            ),
            "passive": True,
            "causal": True,
            "analog_branch": "series_r_l_c",
            "discretization": "parallel_bilinear_biquad_sections",
        }


def _padded_polynomial_add(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    length = max(first.size, second.size)
    output = np.zeros(length, dtype=np.float64)
    output[: first.size] += first
    output[: second.size] += second
    return output


def _parallel_transfer_add(
    first_numerator: np.ndarray,
    first_denominator: np.ndarray,
    second_numerator: np.ndarray,
    second_denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = _padded_polynomial_add(
        np.convolve(first_numerator, second_denominator),
        np.convolve(second_numerator, first_denominator),
    )
    denominator = np.convolve(first_denominator, second_denominator)
    scale = float(denominator[0])
    if not math.isfinite(scale) or abs(scale) <= 1e-15:
        raise ValueError("digital admittance denominator is singular")
    return numerator / scale, denominator / scale


def digital_normalized_admittance_filter(
    model: RationalAdmittanceModel,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return digital ``B(z)/A(z)`` for a passive rational admittance.

    First-order relaxation sections use the ordinary bilinear transform.
    Resonant branches reuse their individually prewarped bilinear biquads.
    Parallel transfer functions are combined exactly as polynomials in
    ``z**-1``.
    """
    sample_rate = float(sample_rate_hz)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample rate must be finite and positive")
    if isinstance(
        model,
        (FirstOrderRelaxationAdmittance, PassiveMultiPoleAdmittance),
    ):
        static, poles, lowpass_gains, highpass_gains = (
            model.relaxation_sections()
        )
        numerator = np.asarray([float(static)], dtype=np.float64)
        denominator = np.asarray([1.0], dtype=np.float64)
        coefficients = (
            (model.digital_lowpass_coefficients(sample_rate),)
            if isinstance(model, FirstOrderRelaxationAdmittance)
            else model.digital_lowpass_coefficients(sample_rate)
        )
        if len(coefficients) != len(poles):
            raise ValueError("digital relaxation section count is inconsistent")
        for (b0, b1, a1), lowpass, highpass in zip(
            coefficients,
            lowpass_gains,
            highpass_gains,
        ):
            branch_numerator = np.asarray(
                [
                    float(lowpass) * b0
                    + float(highpass) * (1.0 - b0),
                    float(lowpass) * b1
                    + float(highpass) * (a1 - b1),
                ],
                dtype=np.float64,
            )
            branch_denominator = np.asarray([1.0, a1], dtype=np.float64)
            numerator, denominator = _parallel_transfer_add(
                numerator,
                denominator,
                branch_numerator,
                branch_denominator,
            )
    elif isinstance(model, PassiveResonantAdmittance):
        numerator = np.asarray(
            [float(model.normalized_admittance_static)],
            dtype=np.float64,
        )
        denominator = np.asarray([1.0], dtype=np.float64)
        for b0, b1, b2, a1, a2 in model.digital_biquad_coefficients(
            sample_rate
        ):
            numerator, denominator = _parallel_transfer_add(
                numerator,
                denominator,
                np.asarray([b0, b1, b2], dtype=np.float64),
                np.asarray([1.0, a1, a2], dtype=np.float64),
            )
    else:
        raise TypeError("unsupported rational admittance model")
    if not np.all(np.isfinite(numerator)) or not np.all(
        np.isfinite(denominator)
    ):
        raise ValueError("digital admittance coefficients must be finite")
    return numerator, denominator


@dataclass(frozen=True)
class DigitalBoundaryReflectionFilter:
    """Causal IIR realization of one angle-aware pressure reflection."""

    numerator: tuple[float, ...]
    denominator: tuple[float, ...]
    sample_rate_hz: float
    incidence_cosine: float
    admittance_model_metadata: dict[str, Any]
    discretization: str = "bilinear_positive_real_cayley_transform"
    schema_version: str = DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported digital boundary filter: {self.schema_version}"
            )
        numerator = tuple(float(value) for value in self.numerator)
        denominator = tuple(float(value) for value in self.denominator)
        if not numerator or not denominator:
            raise ValueError("digital reflection coefficients cannot be empty")
        if not all(
            math.isfinite(value) for value in (*numerator, *denominator)
        ):
            raise ValueError("digital reflection coefficients must be finite")
        if not math.isclose(
            denominator[0], 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("digital reflection denominator must be normalized")
        sample_rate = float(self.sample_rate_hz)
        cosine = float(self.incidence_cosine)
        if not math.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("sample rate must be finite and positive")
        if not math.isfinite(cosine) or not 0.0 < cosine <= 1.0:
            raise ValueError("incidence cosine must lie in (0, 1]")
        poles = (
            np.roots(np.asarray(denominator, dtype=np.float64))
            if len(denominator) > 1
            else np.asarray([], dtype=np.complex128)
        )
        maximum_pole_magnitude = (
            float(np.max(np.abs(poles))) if poles.size else 0.0
        )
        if maximum_pole_magnitude >= 1.0:
            raise ValueError("digital boundary reflection filter is unstable")
        if not isinstance(self.admittance_model_metadata, dict):
            raise ValueError("admittance model metadata must be an object")
        if self.discretization != "bilinear_positive_real_cayley_transform":
            raise ValueError("unsupported digital boundary discretization")
        canonical_metadata = json.loads(
            json.dumps(
                self.admittance_model_metadata,
                allow_nan=False,
            )
        )
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "incidence_cosine", cosine)
        object.__setattr__(
            self,
            "admittance_model_metadata",
            canonical_metadata,
        )

    @property
    def maximum_pole_magnitude(self) -> float:
        if len(self.denominator) == 1:
            return 0.0
        return float(
            np.max(
                np.abs(
                    np.roots(
                        np.asarray(self.denominator, dtype=np.float64)
                    )
                )
            )
        )

    def frequency_response(self, frequency_hz: float) -> complex:
        frequency = float(frequency_hz)
        if (
            not math.isfinite(frequency)
            or frequency < 0.0
            or frequency > 0.5 * self.sample_rate_hz
        ):
            raise ValueError("frequency must lie between zero and Nyquist")
        delay = cmath.exp(
            -2j * math.pi * frequency / self.sample_rate_hz
        )
        numerator = sum(
            value * delay**index
            for index, value in enumerate(self.numerator)
        )
        denominator = sum(
            value * delay**index
            for index, value in enumerate(self.denominator)
        )
        return complex(numerator / denominator)

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("digital boundary filter input must be 1D")
        return np.asarray(
            lfilter(
                np.asarray(self.numerator, dtype=np.float64),
                np.asarray(self.denominator, dtype=np.float64),
                values,
            ),
            dtype=np.float64,
        )

    def impulse_response(self, num_samples: int) -> np.ndarray:
        length = int(num_samples)
        if length != num_samples or length <= 0:
            raise ValueError("num_samples must be a positive integer")
        impulse = np.zeros(length, dtype=np.float64)
        impulse[0] = 1.0
        return self.filter_signal(impulse)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "numerator": list(self.numerator),
            "denominator": list(self.denominator),
            "sample_rate_hz": self.sample_rate_hz,
            "incidence_cosine": self.incidence_cosine,
            "admittance_model": self.admittance_model_metadata,
            "discretization": self.discretization,
            "passive_by_construction": True,
            "causal": True,
            "stable": True,
            "maximum_pole_magnitude": self.maximum_pole_magnitude,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DigitalBoundaryReflectionFilter":
        return cls(
            numerator=tuple(float(value) for value in data["numerator"]),
            denominator=tuple(float(value) for value in data["denominator"]),
            sample_rate_hz=float(data["sample_rate_hz"]),
            incidence_cosine=float(data["incidence_cosine"]),
            admittance_model_metadata=dict(data["admittance_model"]),
            discretization=str(
                data.get(
                    "discretization",
                    "bilinear_positive_real_cayley_transform",
                )
            ),
            schema_version=str(
                data.get(
                    "schema_version",
                    DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION,
                )
            ),
        )


def digital_locally_reacting_reflection_filter(
    model: RationalAdmittanceModel,
    incidence_cosine: float,
    sample_rate_hz: float,
) -> DigitalBoundaryReflectionFilter:
    """Discretize ``Gamma=(cos(theta)-y)/(cos(theta)+y)`` causally.

    Bilinear transformation preserves the positive-real property of the
    admittance.  The Cayley transform from positive-real admittance to pressure
    reflection is therefore bounded-real: a passive input model gives
    ``abs(Gamma) <= 1`` on the unit circle.
    """
    cosine = float(incidence_cosine)
    if not math.isfinite(cosine) or not 0.0 < cosine <= 1.0:
        raise ValueError("incidence cosine must lie in (0, 1]")
    admittance_numerator, admittance_denominator = (
        digital_normalized_admittance_filter(model, sample_rate_hz)
    )
    scaled_denominator = cosine * admittance_denominator
    numerator = _padded_polynomial_add(
        scaled_denominator,
        -admittance_numerator,
    )
    denominator = _padded_polynomial_add(
        scaled_denominator,
        admittance_numerator,
    )
    scale = float(denominator[0])
    if not math.isfinite(scale) or abs(scale) <= 1e-15:
        raise ValueError("digital angle-reflection denominator is singular")
    numerator /= scale
    denominator /= scale
    result = DigitalBoundaryReflectionFilter(
        numerator=tuple(numerator.astype(float).tolist()),
        denominator=tuple(denominator.astype(float).tolist()),
        sample_rate_hz=float(sample_rate_hz),
        incidence_cosine=cosine,
        admittance_model_metadata=model.metadata(),
    )
    return result


def characteristic_impedance_pa_s_m(
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> float:
    """Return the characteristic impedance ``rho * c`` of the medium."""
    density = float(air_density_kg_m3)
    sound_speed = float(sound_speed_m_s)
    if (
        not math.isfinite(density)
        or not math.isfinite(sound_speed)
        or density <= 0.0
        or sound_speed <= 0.0
    ):
        raise ValueError("air density and sound speed must be finite and positive")
    return density * sound_speed


def normal_incidence_reflection_coefficient(
    surface_impedance_pa_s_m: complex | float,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> complex:
    """Return the complex pressure reflection coefficient at normal incidence.

    The locally reacting boundary relation is

    ``Gamma = (Z_surface - rho*c) / (Z_surface + rho*c)``.

    A passive surface must have a non-negative real impedance.  Infinite
    impedance is treated as a rigid boundary with ``Gamma = +1``.
    """
    impedance = complex(surface_impedance_pa_s_m)
    z0 = characteristic_impedance_pa_s_m(
        air_density_kg_m3,
        sound_speed_m_s,
    )
    if math.isnan(impedance.real) or math.isnan(impedance.imag):
        raise ValueError("surface impedance cannot contain NaN")
    if math.isinf(abs(impedance)):
        return complex(1.0, 0.0)
    if impedance.real < 0.0:
        raise ValueError("a passive surface impedance cannot have a negative real part")
    return complex((impedance - z0) / (impedance + z0))


def normal_incidence_absorption_coefficient(
    surface_impedance_pa_s_m: complex | float,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> float:
    """Return ``1 - abs(Gamma)**2`` for a passive, opaque local boundary."""
    reflection = normal_incidence_reflection_coefficient(
        surface_impedance_pa_s_m,
        air_density_kg_m3,
        sound_speed_m_s,
    )
    absorption = 1.0 - abs(reflection) ** 2
    if absorption < -1e-12 or absorption > 1.0 + 1e-12:
        raise ValueError("surface impedance produced a non-passive absorption value")
    return float(min(1.0, max(0.0, absorption)))


def impedance_from_normal_incidence_reflection(
    reflection_coefficient: complex | float,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> complex:
    """Recover surface impedance from a passive complex reflection coefficient."""
    reflection = complex(reflection_coefficient)
    if not cmath.isfinite(reflection):
        raise ValueError("reflection coefficient must be finite")
    if abs(reflection) > 1.0 + 1e-12:
        raise ValueError("a passive reflection coefficient must have magnitude <= 1")
    z0 = characteristic_impedance_pa_s_m(
        air_density_kg_m3,
        sound_speed_m_s,
    )
    if abs(1.0 - reflection) <= 1e-12:
        return complex(float("inf"), 0.0)
    return complex(z0 * (1.0 + reflection) / (1.0 - reflection))


def impedance_from_absorption_and_phase(
    absorption_coefficient: float,
    reflection_phase_rad: float,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> complex:
    """Construct impedance from absorption plus an explicit reflection phase.

    Absorption alone supplies only ``abs(Gamma) = sqrt(1 - alpha)``.  Requiring
    the phase as a separate argument prevents a diffuse-field absorption table
    from being silently reinterpreted as a unique low-frequency impedance.
    """
    absorption = float(absorption_coefficient)
    phase = float(reflection_phase_rad)
    if not 0.0 <= absorption <= 1.0:
        raise ValueError("absorption coefficient must be in [0, 1]")
    if not math.isfinite(phase):
        raise ValueError("reflection phase must be finite")
    reflection = math.sqrt(max(0.0, 1.0 - absorption)) * cmath.exp(1j * phase)
    return impedance_from_normal_incidence_reflection(
        reflection,
        air_density_kg_m3,
        sound_speed_m_s,
    )


__all__ = [
    "DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION",
    "DigitalBoundaryReflectionFilter",
    "FirstOrderRelaxationAdmittance",
    "PassiveMultiPoleAdmittance",
    "PassiveResonantAdmittance",
    "RationalAdmittanceModel",
    "characteristic_impedance_pa_s_m",
    "digital_locally_reacting_reflection_filter",
    "digital_normalized_admittance_filter",
    "impedance_from_absorption_and_phase",
    "impedance_from_normal_incidence_reflection",
    "normal_incidence_absorption_coefficient",
    "normal_incidence_reflection_coefficient",
]
