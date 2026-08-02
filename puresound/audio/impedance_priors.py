"""Versioned, phase-aware low-frequency surface-impedance priors.

This module intentionally separates three evidence levels:

* measured complex surface impedance;
* a measured non-acoustic parameter passed through a published physical model;
* an unverified engineering prior.

The initial catalog contains only the second level: measured glass-wool flow
resistivity from Tarnow (2002), evaluated as a rigid-backed porous layer with
the positive-real Miki (1990) model.  It is an experimental validation
reference, not an automatic mapping onto the RIR material catalog.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

from puresound.audio.acoustic_impedance import (
    FirstOrderRelaxationAdmittance,
    characteristic_impedance_pa_s_m,
    normal_incidence_reflection_coefficient,
)


IMPEDANCE_PRIOR_CATALOG_VERSION = "puresound-impedance-priors.v0"
MIKI_1990_DOI = "https://doi.org/10.1250/ast.11.19"
TARNOW_2002_DOI = "https://doi.org/10.1121/1.1476686"
EVIDENCE_TIER_MEASURED_PARAMETER_MODEL = (
    "measured_flow_resistivity_plus_miki_model"
)


@dataclass(frozen=True)
class MikiPorousLayerPrior:
    """Rigid-backed porous layer using the Miki 1990 empirical model."""

    prior_id: str
    material_label: str
    flow_resistivity_pa_s_m2: float
    thickness_m: float
    evidence_tier: str
    provenance_urls: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.prior_id or not self.material_label:
            raise ValueError("prior id and material label are required")
        if (
            not math.isfinite(float(self.flow_resistivity_pa_s_m2))
            or self.flow_resistivity_pa_s_m2 <= 0.0
        ):
            raise ValueError("flow resistivity must be finite and positive")
        if not math.isfinite(float(self.thickness_m)) or self.thickness_m <= 0.0:
            raise ValueError("layer thickness must be finite and positive")
        if self.evidence_tier != EVIDENCE_TIER_MEASURED_PARAMETER_MODEL:
            raise ValueError("unsupported impedance-prior evidence tier")
        if not self.provenance_urls:
            raise ValueError("at least one provenance URL is required")

    @property
    def validity_frequency_range_hz(self) -> tuple[float, float]:
        """Conservative Delany-Bazley/Miki range ``0.01 < f/sigma < 1``."""
        sigma = float(self.flow_resistivity_pa_s_m2)
        return 0.01 * sigma, sigma

    def _validate_frequency(self, frequency_hz: float) -> float:
        frequency = float(frequency_hz)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency must be finite and positive")
        minimum, maximum = self.validity_frequency_range_hz
        if not minimum <= frequency <= maximum:
            raise ValueError(
                f"frequency {frequency:g} Hz is outside prior validity "
                f"[{minimum:g}, {maximum:g}] Hz"
            )
        return frequency

    def characteristic_impedance_pa_s_m(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> complex:
        """Return Miki characteristic impedance for ``exp(+j*omega*t)``."""
        frequency = self._validate_frequency(frequency_hz)
        ratio = (
            1000.0 * frequency / float(self.flow_resistivity_pa_s_m2)
        )
        correction = ratio**-0.632
        z0 = characteristic_impedance_pa_s_m(
            air_density_kg_m3,
            sound_speed_m_s,
        )
        return complex(
            z0 * (1.0 + 5.50 * correction - 1j * 8.43 * correction)
        )

    def complex_wavenumber_rad_m(
        self,
        frequency_hz: float,
        sound_speed_m_s: float,
    ) -> complex:
        """Return Miki complex wavenumber for ``exp(+j*omega*t)``."""
        frequency = self._validate_frequency(frequency_hz)
        sound_speed = float(sound_speed_m_s)
        if not math.isfinite(sound_speed) or sound_speed <= 0.0:
            raise ValueError("sound speed must be finite and positive")
        ratio = (
            1000.0 * frequency / float(self.flow_resistivity_pa_s_m2)
        )
        correction = ratio**-0.618
        return complex(
            2.0
            * math.pi
            * frequency
            / sound_speed
            * (1.0 + 7.81 * correction - 1j * 11.41 * correction)
        )

    def surface_impedance_pa_s_m(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> complex:
        """Return rigid-backed layer surface impedance ``-j*Zc*cot(k*d)``."""
        characteristic = self.characteristic_impedance_pa_s_m(
            frequency_hz,
            air_density_kg_m3,
            sound_speed_m_s,
        )
        wavenumber = self.complex_wavenumber_rad_m(
            frequency_hz,
            sound_speed_m_s,
        )
        return complex(
            -1j
            * characteristic
            / cmath.tan(wavenumber * float(self.thickness_m))
        )

    def reflection_coefficient(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> complex:
        return normal_incidence_reflection_coefficient(
            self.surface_impedance_pa_s_m(
                frequency_hz,
                air_density_kg_m3,
                sound_speed_m_s,
            ),
            air_density_kg_m3,
            sound_speed_m_s,
        )

    def absorption_coefficient(
        self,
        frequency_hz: float,
        air_density_kg_m3: float,
        sound_speed_m_s: float,
    ) -> float:
        reflection = self.reflection_coefficient(
            frequency_hz,
            air_density_kg_m3,
            sound_speed_m_s,
        )
        return float(np.clip(1.0 - abs(reflection) ** 2, 0.0, 1.0))

    def metadata(self) -> dict[str, Any]:
        return {
            "catalog_version": IMPEDANCE_PRIOR_CATALOG_VERSION,
            **asdict(self),
            "validity_frequency_range_hz": list(
                self.validity_frequency_range_hz
            ),
            "surface_model": "miki_1990_rigid_backed_porous_layer",
            "incidence": "normal",
            "time_convention": "exp(+j*omega*t)",
            "production_material_mapping_enabled": False,
        }


@dataclass(frozen=True)
class ImpedancePriorFit:
    """Diagnostic fit of a Miki prior to the time-domain relaxation boundary."""

    prior_id: str
    model: FirstOrderRelaxationAdmittance
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    num_frequencies: int
    rms_complex_reflection_error: float
    maximum_complex_reflection_error: float
    maximum_magnitude_error: float
    maximum_phase_error_rad: float
    acceptance_threshold: float

    @property
    def accepted(self) -> bool:
        return self.maximum_complex_reflection_error <= self.acceptance_threshold

    def metadata(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "model": self.model.metadata(),
            "fit_frequency_range_hz": [
                float(self.minimum_frequency_hz),
                float(self.maximum_frequency_hz),
            ],
            "num_frequencies": int(self.num_frequencies),
            "rms_complex_reflection_error": float(
                self.rms_complex_reflection_error
            ),
            "maximum_complex_reflection_error": float(
                self.maximum_complex_reflection_error
            ),
            "maximum_magnitude_error": float(self.maximum_magnitude_error),
            "maximum_phase_error_rad": float(self.maximum_phase_error_rad),
            "acceptance_threshold": float(self.acceptance_threshold),
            "accepted": bool(self.accepted),
            "fit_domain": "complex_pressure_reflection_coefficient",
        }


def reference_impedance_priors() -> dict[str, MikiPorousLayerPrior]:
    """Return immutable phase-aware references; none are auto-applied to scenes."""
    shared = {
        "evidence_tier": EVIDENCE_TIER_MEASURED_PARAMETER_MODEL,
        "provenance_urls": (TARNOW_2002_DOI, MIKI_1990_DOI),
    }
    priors = (
        MikiPorousLayerPrior(
            prior_id="glass_wool_14kgm3_50mm_normal",
            material_label="14 kg/m3 glass wool, 50 mm, normal direction",
            flow_resistivity_pa_s_m2=5_880.0,
            thickness_m=0.05,
            notes=(
                "Flow resistivity measured by Tarnow (2002); a virtual 50 mm "
                "rigid-backed layer is evaluated with Miki (1990), not "
                "directly measured as installed."
            ),
            **shared,
        ),
        MikiPorousLayerPrior(
            prior_id="glass_wool_14kgm3_100mm_normal",
            material_label="14 kg/m3 glass wool, 100 mm, normal direction",
            flow_resistivity_pa_s_m2=5_880.0,
            thickness_m=0.10,
            notes=(
                "Flow resistivity measured by Tarnow (2002); complex surface "
                "impedance predicted with Miki (1990), not directly measured."
            ),
            **shared,
        ),
        MikiPorousLayerPrior(
            prior_id="glass_wool_30kgm3_100mm_normal",
            material_label="30 kg/m3 glass wool, 100 mm, normal direction",
            flow_resistivity_pa_s_m2=15_500.0,
            thickness_m=0.10,
            notes=(
                "Flow resistivity measured by Tarnow (2002); complex surface "
                "impedance predicted with Miki (1990), not directly measured."
            ),
            **shared,
        ),
    )
    return {prior.prior_id: prior for prior in priors}


def fit_first_order_relaxation(
    prior: MikiPorousLayerPrior,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    *,
    air_density_kg_m3: float = 1.204,
    sound_speed_m_s: float = 343.0,
    num_frequencies: int = 128,
    acceptance_threshold: float = 0.08,
) -> ImpedancePriorFit:
    """Fit the causal FDTD boundary to a phase-aware Miki reference."""
    minimum = float(minimum_frequency_hz)
    maximum = float(maximum_frequency_hz)
    valid_minimum, valid_maximum = prior.validity_frequency_range_hz
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError("fit frequency range must be finite, positive, and ordered")
    if minimum < valid_minimum or maximum > valid_maximum:
        raise ValueError(
            "fit range must stay inside the impedance prior validity range"
        )
    if int(num_frequencies) < 8:
        raise ValueError("fit requires at least eight frequencies")
    if not math.isfinite(acceptance_threshold) or acceptance_threshold <= 0.0:
        raise ValueError("acceptance threshold must be finite and positive")

    frequencies = np.geomspace(minimum, maximum, int(num_frequencies))
    target = np.asarray(
        [
            prior.reflection_coefficient(
                float(frequency),
                air_density_kg_m3,
                sound_speed_m_s,
            )
            for frequency in frequencies
        ],
        dtype=np.complex128,
    )

    def model_from_parameters(parameters: Iterable[float]):
        g_zero, g_infinite, relaxation_frequency = np.exp(
            np.asarray(parameters, dtype=np.float64)
        )
        return FirstOrderRelaxationAdmittance(
            normalized_admittance_infinite=float(g_infinite),
            normalized_admittance_relaxation=float(g_zero - g_infinite),
            relaxation_frequency_hz=float(relaxation_frequency),
        )

    def residual(parameters: Iterable[float]) -> np.ndarray:
        model = model_from_parameters(parameters)
        predicted = np.asarray(
            [
                model.reflection_coefficient(float(frequency))
                for frequency in frequencies
            ]
        )
        error = predicted - target
        return np.concatenate((error.real, error.imag))

    center = math.sqrt(minimum * maximum)
    starts = (
        (0.003, 1.5, 3.0 * center),
        (0.01, 0.5, center),
        (0.05, 2.0, 10.0 * center),
    )
    results = [
        least_squares(
            residual,
            np.log(np.asarray(start, dtype=np.float64)),
            bounds=(
                np.log(np.asarray([1e-10, 1e-10, minimum / 100.0])),
                np.log(np.asarray([100.0, 100.0, maximum * 100.0])),
            ),
            max_nfev=20_000,
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.cost))
    model = model_from_parameters(result.x)
    predicted = np.asarray(
        [
            model.reflection_coefficient(float(frequency))
            for frequency in frequencies
        ]
    )
    complex_error = np.abs(predicted - target)
    magnitude_error = np.abs(np.abs(predicted) - np.abs(target))
    phase_error = np.abs(np.angle(predicted / target))
    return ImpedancePriorFit(
        prior_id=prior.prior_id,
        model=model,
        minimum_frequency_hz=minimum,
        maximum_frequency_hz=maximum,
        num_frequencies=int(num_frequencies),
        rms_complex_reflection_error=float(
            np.sqrt(np.mean(np.square(complex_error)))
        ),
        maximum_complex_reflection_error=float(np.max(complex_error)),
        maximum_magnitude_error=float(np.max(magnitude_error)),
        maximum_phase_error_rad=float(np.max(phase_error)),
        acceptance_threshold=float(acceptance_threshold),
    )


__all__ = [
    "EVIDENCE_TIER_MEASURED_PARAMETER_MODEL",
    "IMPEDANCE_PRIOR_CATALOG_VERSION",
    "ImpedancePriorFit",
    "MIKI_1990_DOI",
    "MikiPorousLayerPrior",
    "TARNOW_2002_DOI",
    "fit_first_order_relaxation",
    "reference_impedance_priors",
]
