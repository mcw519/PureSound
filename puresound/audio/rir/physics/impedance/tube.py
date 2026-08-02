"""Reduction of two-microphone impedance-tube transfer measurements.

The coordinate origin is the sample surface and ``x`` increases from the
sample towards the source.  With the ``exp(+i*omega*t)`` convention,

``p(x) = A*exp(+i*k*x) + B*exp(-i*k*x)``,

and the surface pressure-reflection coefficient is ``B/A``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from puresound.audio.acoustic_impedance import (
    characteristic_impedance_pa_s_m,
)


IMPEDANCE_TUBE_TRANSFER_MEASUREMENT_SCHEMA_VERSION = (
    "puresound.impedance_tube_transfer_measurement.v1"
)
RAW_TRANSFER_CSV_COLUMNS = (
    "repeat_id",
    "frequency_hz",
    "h12_real",
    "h12_imag",
    "coherence",
)
MICROPHONE_SWITCH_CSV_COLUMNS = (
    "frequency_hz",
    "h12_original_real",
    "h12_original_imag",
    "h12_swapped_real",
    "h12_swapped_imag",
)


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class TwoMicrophoneTubeGeometry:
    """Circular tube geometry for a normal-incidence two-microphone test."""

    microphone_1_distance_from_sample_m: float
    microphone_2_distance_from_sample_m: float
    tube_diameter_m: float
    sound_speed_m_s: float
    minimum_spacing_sine: float = 0.05

    def __post_init__(self) -> None:
        x1 = _finite_positive(
            self.microphone_1_distance_from_sample_m,
            "microphone 1 distance",
        )
        x2 = _finite_positive(
            self.microphone_2_distance_from_sample_m,
            "microphone 2 distance",
        )
        diameter = _finite_positive(self.tube_diameter_m, "tube diameter")
        sound_speed = _finite_positive(self.sound_speed_m_s, "sound speed")
        minimum_spacing_sine = float(self.minimum_spacing_sine)
        if math.isclose(x1, x2, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("microphone positions must be distinct")
        if (
            not math.isfinite(minimum_spacing_sine)
            or not 0.0 < minimum_spacing_sine < 1.0
        ):
            raise ValueError("minimum spacing sine must be in (0, 1)")
        object.__setattr__(
            self,
            "microphone_1_distance_from_sample_m",
            x1,
        )
        object.__setattr__(
            self,
            "microphone_2_distance_from_sample_m",
            x2,
        )
        object.__setattr__(self, "tube_diameter_m", diameter)
        object.__setattr__(self, "sound_speed_m_s", sound_speed)
        object.__setattr__(
            self,
            "minimum_spacing_sine",
            minimum_spacing_sine,
        )

    @property
    def microphone_spacing_m(self) -> float:
        return abs(
            self.microphone_1_distance_from_sample_m
            - self.microphone_2_distance_from_sample_m
        )

    @property
    def circular_plane_wave_upper_frequency_hz(self) -> float:
        """First non-planar-mode cutoff, ``1.841*c/(pi*D)``."""
        return float(
            1.841
            * self.sound_speed_m_s
            / (math.pi * self.tube_diameter_m)
        )

    def validate_frequencies(self, frequencies_hz: Iterable[float]) -> np.ndarray:
        frequencies = np.asarray(tuple(frequencies_hz), dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size < 3:
            raise ValueError("at least three one-dimensional frequencies are required")
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("frequencies must be finite and positive")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("frequencies must be strictly increasing")
        if np.any(
            frequencies >= self.circular_plane_wave_upper_frequency_hz
        ):
            raise ValueError(
                "frequency reaches the first circular-tube transverse mode "
                f"({self.circular_plane_wave_upper_frequency_hz:.3f} Hz)"
            )
        spacing_phase = (
            2.0
            * math.pi
            * frequencies
            * self.microphone_spacing_m
            / self.sound_speed_m_s
        )
        minimum = float(np.min(np.abs(np.sin(spacing_phase))))
        if minimum < self.minimum_spacing_sine:
            raise ValueError(
                "microphone spacing is ill-conditioned in the selected band: "
                f"minimum abs(sin(k*s))={minimum:.6f}, required "
                f">={self.minimum_spacing_sine:.6f}"
            )
        return frequencies

    def metadata(self) -> dict[str, float | str]:
        return {
            "shape": "circular",
            "tube_diameter_m": self.tube_diameter_m,
            "microphone_1_distance_from_sample_m": (
                self.microphone_1_distance_from_sample_m
            ),
            "microphone_2_distance_from_sample_m": (
                self.microphone_2_distance_from_sample_m
            ),
            "microphone_spacing_m": self.microphone_spacing_m,
            "circular_plane_wave_upper_frequency_hz": (
                self.circular_plane_wave_upper_frequency_hz
            ),
            "minimum_spacing_sine": self.minimum_spacing_sine,
        }


def microphone_switch_calibration_factor(
    h12_original: Iterable[complex],
    h12_swapped: Iterable[complex],
) -> np.ndarray:
    """Return the complex channel mismatch from a microphone-switch test.

    If ``C`` is the channel/microphone mismatch and ``H`` is the true spatial
    transfer function, the two acquisitions are ``C*H`` and ``C/H``.
    Consequently ``C = sqrt(H_original * H_swapped)``.  Unwrapped phase keeps
    the square-root branch continuous over frequency.
    """
    original = np.asarray(tuple(h12_original), dtype=np.complex128)
    swapped = np.asarray(tuple(h12_swapped), dtype=np.complex128)
    if original.ndim != 1 or original.size < 3 or swapped.shape != original.shape:
        raise ValueError(
            "original and swapped calibration arrays must have equal length >= 3"
        )
    if np.any(~np.isfinite(original)) or np.any(~np.isfinite(swapped)):
        raise ValueError("microphone-switch calibration must be finite")
    product = original * swapped
    if np.any(np.abs(product) <= 1e-15):
        raise ValueError("microphone-switch calibration cannot contain zeros")
    phase = np.unwrap(np.angle(product))
    correction = np.sqrt(np.abs(product)) * np.exp(0.5j * phase)
    if float(np.median(correction.real)) < 0.0:
        correction = -correction
    return np.asarray(correction, dtype=np.complex128)


def reflection_from_two_microphone_transfer(
    frequencies_hz: Iterable[float],
    h12: Iterable[complex],
    geometry: TwoMicrophoneTubeGeometry,
) -> np.ndarray:
    """Recover surface reflection from ``H12 = P(x2) / P(x1)``."""
    frequencies = geometry.validate_frequencies(frequencies_hz)
    transfer = np.asarray(tuple(h12), dtype=np.complex128)
    if transfer.shape != frequencies.shape:
        raise ValueError("H12 must match the frequency array")
    if np.any(~np.isfinite(transfer)):
        raise ValueError("H12 must be finite")
    k = 2.0 * math.pi * frequencies / geometry.sound_speed_m_s
    x1 = geometry.microphone_1_distance_from_sample_m
    x2 = geometry.microphone_2_distance_from_sample_m
    numerator = np.exp(1j * k * x2) - transfer * np.exp(1j * k * x1)
    denominator = transfer * np.exp(-1j * k * x1) - np.exp(-1j * k * x2)
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("two-microphone reflection inversion is singular")
    return np.asarray(numerator / denominator, dtype=np.complex128)


def transfer_from_surface_reflection(
    frequencies_hz: Iterable[float],
    reflection: Iterable[complex],
    geometry: TwoMicrophoneTubeGeometry,
) -> np.ndarray:
    """Forward reference used for calibration and numerical round-trip tests."""
    frequencies = geometry.validate_frequencies(frequencies_hz)
    coefficient = np.asarray(tuple(reflection), dtype=np.complex128)
    if coefficient.shape != frequencies.shape:
        raise ValueError("reflection must match the frequency array")
    if np.any(~np.isfinite(coefficient)):
        raise ValueError("reflection must be finite")
    k = 2.0 * math.pi * frequencies / geometry.sound_speed_m_s
    x1 = geometry.microphone_1_distance_from_sample_m
    x2 = geometry.microphone_2_distance_from_sample_m
    pressure_1 = np.exp(1j * k * x1) + coefficient * np.exp(-1j * k * x1)
    pressure_2 = np.exp(1j * k * x2) + coefficient * np.exp(-1j * k * x2)
    if np.any(np.abs(pressure_1) <= 1e-12):
        raise ValueError("microphone 1 lies on a pressure node")
    return np.asarray(pressure_2 / pressure_1, dtype=np.complex128)


@dataclass(frozen=True)
class ImpedanceTubeReduction:
    """Mean complex impedance and repeatability from accepted tube sweeps."""

    frequencies_hz: tuple[float, ...]
    impedance_real_pa_s_m: tuple[float, ...]
    impedance_imag_pa_s_m: tuple[float, ...]
    impedance_real_std_pa_s_m: tuple[float, ...]
    impedance_imag_std_pa_s_m: tuple[float, ...]
    reflection_real: tuple[float, ...]
    reflection_imag: tuple[float, ...]
    mean_coherence: tuple[float, ...]
    num_repeats: int
    calibration_applied: bool
    geometry: TwoMicrophoneTubeGeometry

    @property
    def complex_impedance_pa_s_m(self) -> np.ndarray:
        return np.asarray(self.impedance_real_pa_s_m) + 1j * np.asarray(
            self.impedance_imag_pa_s_m
        )

    @property
    def complex_reflection(self) -> np.ndarray:
        return np.asarray(self.reflection_real) + 1j * np.asarray(
            self.reflection_imag
        )

    def metadata(self) -> dict[str, object]:
        return {
            "reduction": "two_microphone_complex_transfer_v1",
            "phasor_convention": "exp(+i*omega*t)",
            "transfer_definition": "H12=P(x2)/P(x1)",
            "coordinate_definition": (
                "x=0 at sample; positive x points from sample to source"
            ),
            "num_repeats": self.num_repeats,
            "calibration_applied": self.calibration_applied,
            "frequency_range_hz": [
                self.frequencies_hz[0],
                self.frequencies_hz[-1],
            ],
            "minimum_mean_coherence": min(self.mean_coherence),
            "maximum_reflection_magnitude": float(
                np.max(np.abs(self.complex_reflection))
            ),
            "has_repeatability_uncertainty": self.num_repeats >= 2,
            "tube": self.geometry.metadata(),
        }


def reduce_two_microphone_repeats(
    frequencies_hz: Iterable[float],
    h12_repeats: np.ndarray,
    coherence_repeats: np.ndarray,
    geometry: TwoMicrophoneTubeGeometry,
    *,
    air_density_kg_m3: float,
    microphone_correction: Iterable[complex] | None = None,
    minimum_coherence: float = 0.95,
    passivity_tolerance: float = 1e-6,
) -> ImpedanceTubeReduction:
    """Reduce calibrated repeated H12 sweeps to complex surface impedance."""
    frequencies = geometry.validate_frequencies(frequencies_hz)
    transfers = np.asarray(h12_repeats, dtype=np.complex128)
    coherence = np.asarray(coherence_repeats, dtype=np.float64)
    if transfers.ndim != 2 or transfers.shape[1:] != frequencies.shape:
        raise ValueError("H12 repeats must have shape [repeat, frequency]")
    if transfers.shape[0] < 1 or coherence.shape != transfers.shape:
        raise ValueError("coherence must match at least one H12 repeat")
    if np.any(~np.isfinite(transfers)):
        raise ValueError("H12 repeats must be finite")
    if np.any(~np.isfinite(coherence)) or np.any(
        (coherence < 0.0) | (coherence > 1.0)
    ):
        raise ValueError("coherence must be finite and in [0, 1]")
    minimum_coherence = float(minimum_coherence)
    if (
        not math.isfinite(minimum_coherence)
        or not 0.0 <= minimum_coherence <= 1.0
    ):
        raise ValueError("minimum coherence must be in [0, 1]")
    mean_coherence = np.mean(coherence, axis=0)
    if np.any(mean_coherence < minimum_coherence):
        index = int(np.argmin(mean_coherence))
        raise ValueError(
            "coherence gate failed at "
            f"{frequencies[index]:.3f} Hz: {mean_coherence[index]:.6f} "
            f"< {minimum_coherence:.6f}"
        )

    correction_applied = microphone_correction is not None
    if microphone_correction is not None:
        correction = np.asarray(
            tuple(microphone_correction),
            dtype=np.complex128,
        )
        if correction.shape != frequencies.shape:
            raise ValueError("microphone correction must match frequencies")
        if np.any(~np.isfinite(correction)) or np.any(
            np.abs(correction) <= 1e-15
        ):
            raise ValueError("microphone correction must be finite and nonzero")
        transfers = transfers / correction[np.newaxis, :]

    reflections = np.stack(
        [
            reflection_from_two_microphone_transfer(
                frequencies,
                repeat,
                geometry,
            )
            for repeat in transfers
        ]
    )
    denominator = 1.0 - reflections
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("sample reflection is indistinguishable from rigid")
    z0 = characteristic_impedance_pa_s_m(
        air_density_kg_m3,
        geometry.sound_speed_m_s,
    )
    impedances = z0 * (1.0 + reflections) / denominator
    mean_impedance = np.mean(impedances, axis=0)
    if np.any(mean_impedance.real < -abs(float(passivity_tolerance))):
        index = int(np.argmin(mean_impedance.real))
        raise ValueError(
            "reduced impedance violates passivity at "
            f"{frequencies[index]:.3f} Hz"
        )
    mean_impedance = np.maximum(mean_impedance.real, 0.0) + 1j * mean_impedance.imag
    mean_reflection = (mean_impedance - z0) / (mean_impedance + z0)
    if np.any(np.abs(mean_reflection) > 1.0 + abs(float(passivity_tolerance))):
        raise ValueError("reduced impedance violates passive reflection magnitude")

    if transfers.shape[0] >= 2:
        real_std = np.std(impedances.real, axis=0, ddof=1)
        imag_std = np.std(impedances.imag, axis=0, ddof=1)
    else:
        real_std = np.zeros_like(frequencies)
        imag_std = np.zeros_like(frequencies)
    return ImpedanceTubeReduction(
        frequencies_hz=tuple(float(value) for value in frequencies),
        impedance_real_pa_s_m=tuple(float(value) for value in mean_impedance.real),
        impedance_imag_pa_s_m=tuple(float(value) for value in mean_impedance.imag),
        impedance_real_std_pa_s_m=tuple(float(value) for value in real_std),
        impedance_imag_std_pa_s_m=tuple(float(value) for value in imag_std),
        reflection_real=tuple(float(value) for value in mean_reflection.real),
        reflection_imag=tuple(float(value) for value in mean_reflection.imag),
        mean_coherence=tuple(float(value) for value in mean_coherence),
        num_repeats=int(transfers.shape[0]),
        calibration_applied=correction_applied,
        geometry=geometry,
    )


def load_transfer_repeats_csv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load a strict long-form CSV and return aligned repeat arrays."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(RAW_TRANSFER_CSV_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"raw transfer CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("raw transfer CSV contains no rows")

    grouped: dict[str, list[tuple[float, complex, float]]] = {}
    for row in rows:
        try:
            repeat_id = str(row["repeat_id"]).strip()
            frequency = float(row["frequency_hz"])
            transfer = complex(float(row["h12_real"]), float(row["h12_imag"]))
            coherence = float(row["coherence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("raw transfer CSV contains invalid values") from error
        if not repeat_id:
            raise ValueError("repeat_id cannot be empty")
        grouped.setdefault(repeat_id, []).append(
            (frequency, transfer, coherence)
        )

    repeat_ids = tuple(grouped)
    reference_frequencies: np.ndarray | None = None
    transfer_rows = []
    coherence_rows = []
    for repeat_id in repeat_ids:
        samples = grouped[repeat_id]
        frequencies = np.asarray([item[0] for item in samples], dtype=np.float64)
        if reference_frequencies is None:
            reference_frequencies = frequencies
        elif not np.array_equal(reference_frequencies, frequencies):
            raise ValueError("all repeats must use the same ordered frequency grid")
        transfer_rows.append([item[1] for item in samples])
        coherence_rows.append([item[2] for item in samples])
    assert reference_frequencies is not None
    return (
        reference_frequencies,
        np.asarray(transfer_rows, dtype=np.complex128),
        np.asarray(coherence_rows, dtype=np.float64),
        repeat_ids,
    )


def load_microphone_switch_csv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load microphone-switch spectra and return frequencies plus correction."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(MICROPHONE_SWITCH_CSV_COLUMNS).difference(columns)
        if missing:
            raise ValueError(
                f"microphone-switch CSV is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("microphone-switch CSV requires at least three rows")
    try:
        frequencies = np.asarray(
            [float(row["frequency_hz"]) for row in rows],
            dtype=np.float64,
        )
        original = np.asarray(
            [
                complex(
                    float(row["h12_original_real"]),
                    float(row["h12_original_imag"]),
                )
                for row in rows
            ],
            dtype=np.complex128,
        )
        swapped = np.asarray(
            [
                complex(
                    float(row["h12_swapped_real"]),
                    float(row["h12_swapped_imag"]),
                )
                for row in rows
            ],
            dtype=np.complex128,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("microphone-switch CSV contains invalid values") from error
    return frequencies, microphone_switch_calibration_factor(original, swapped)


__all__ = [
    "IMPEDANCE_TUBE_TRANSFER_MEASUREMENT_SCHEMA_VERSION",
    "ImpedanceTubeReduction",
    "MICROPHONE_SWITCH_CSV_COLUMNS",
    "RAW_TRANSFER_CSV_COLUMNS",
    "TwoMicrophoneTubeGeometry",
    "load_microphone_switch_csv",
    "load_transfer_repeats_csv",
    "microphone_switch_calibration_factor",
    "reduce_two_microphone_repeats",
    "reflection_from_two_microphone_transfer",
    "transfer_from_surface_reflection",
]
