"""Strict ingestion contract for phase-aware surface-impedance measurements."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from puresound.audio.rir.physics.impedance.admittance import (
    characteristic_impedance_pa_s_m,
    normal_incidence_reflection_coefficient,
)


COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION = (
    "puresound.complex_impedance_measurement.v1"
)
REQUIRED_CSV_COLUMNS = (
    "frequency_hz",
    "impedance_real_pa_s_m",
    "impedance_imag_pa_s_m",
)
OPTIONAL_STD_COLUMNS = (
    "impedance_real_std_pa_s_m",
    "impedance_imag_std_pa_s_m",
)
NORMALIZED_COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION = (
    "puresound.normalized_complex_impedance_measurement.v1"
)
REQUIRED_NORMALIZED_CSV_COLUMNS = (
    "frequency_hz",
    "normalized_impedance_real",
    "normalized_impedance_imag",
)
OPTIONAL_NORMALIZED_STD_COLUMNS = (
    "normalized_impedance_real_std",
    "normalized_impedance_imag_std",
)


@dataclass(frozen=True)
class ComplexImpedanceMeasurement:
    """Normal-incidence complex surface impedance plus measurement provenance."""

    measurement_id: str
    method: str
    incidence: str
    frequencies_hz: tuple[float, ...]
    impedance_real_pa_s_m: tuple[float, ...]
    impedance_imag_pa_s_m: tuple[float, ...]
    air_density_kg_m3: float
    sound_speed_m_s: float
    sample: dict[str, Any]
    provenance: dict[str, Any]
    impedance_real_std_pa_s_m: tuple[float, ...] = ()
    impedance_imag_std_pa_s_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.frequencies_hz)
        real = tuple(float(value) for value in self.impedance_real_pa_s_m)
        imag = tuple(float(value) for value in self.impedance_imag_pa_s_m)
        real_std = tuple(
            float(value) for value in self.impedance_real_std_pa_s_m
        )
        imag_std = tuple(
            float(value) for value in self.impedance_imag_std_pa_s_m
        )
        if not self.measurement_id or not self.method:
            raise ValueError("measurement id and method are required")
        if self.incidence != "normal":
            raise ValueError("only normal-incidence impedance is currently supported")
        if len(frequencies) < 3 or len(real) != len(frequencies) or len(imag) != len(
            frequencies
        ):
            raise ValueError(
                "frequency and complex impedance arrays must have equal length >= 3"
            )
        if any(
            not math.isfinite(value) or value <= 0.0 for value in frequencies
        ):
            raise ValueError("measurement frequencies must be finite and positive")
        if any(
            next_value <= value
            for value, next_value in zip(frequencies, frequencies[1:])
        ):
            raise ValueError("measurement frequencies must be strictly increasing")
        if any(not math.isfinite(value) for value in (*real, *imag)):
            raise ValueError("measured impedance must be finite")
        if any(value < 0.0 for value in real):
            raise ValueError("passive measured impedance cannot have negative real part")
        if bool(real_std) != bool(imag_std):
            raise ValueError("both real and imaginary uncertainty columns are required")
        if real_std and (
            len(real_std) != len(frequencies)
            or len(imag_std) != len(frequencies)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (*real_std, *imag_std)
            )
        ):
            raise ValueError(
                "impedance standard deviations must match data and be non-negative"
            )
        characteristic_impedance_pa_s_m(
            self.air_density_kg_m3,
            self.sound_speed_m_s,
        )
        if not isinstance(self.sample, dict) or not self.sample.get(
            "material_label"
        ):
            raise ValueError("sample.material_label is required")
        if not isinstance(self.provenance, dict) or not self.provenance.get(
            "source_url"
        ):
            raise ValueError("provenance.source_url is required")
        if not self.provenance.get("license"):
            raise ValueError("provenance.license is required")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "impedance_real_pa_s_m", real)
        object.__setattr__(self, "impedance_imag_pa_s_m", imag)
        object.__setattr__(self, "impedance_real_std_pa_s_m", real_std)
        object.__setattr__(self, "impedance_imag_std_pa_s_m", imag_std)

        reflections = self.reflection_coefficients()
        if any(abs(value) > 1.0 + 1e-9 for value in reflections):
            raise ValueError("measured impedance violates passive reflection magnitude")

    @property
    def complex_impedance_pa_s_m(self) -> np.ndarray:
        return np.asarray(self.impedance_real_pa_s_m) + 1j * np.asarray(
            self.impedance_imag_pa_s_m
        )

    def reflection_coefficients(self) -> np.ndarray:
        return np.asarray(
            [
                normal_incidence_reflection_coefficient(
                    impedance,
                    self.air_density_kg_m3,
                    self.sound_speed_m_s,
                )
                for impedance in self.complex_impedance_pa_s_m
            ],
            dtype=np.complex128,
        )

    def metadata(self) -> dict[str, Any]:
        reflection = self.reflection_coefficients()
        return {
            "schema_version": COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
            "measurement_id": self.measurement_id,
            "method": self.method,
            "incidence": self.incidence,
            "environment": {
                "air_density_kg_m3": float(self.air_density_kg_m3),
                "sound_speed_m_s": float(self.sound_speed_m_s),
            },
            "sample": dict(self.sample),
            "provenance": dict(self.provenance),
            "num_frequencies": len(self.frequencies_hz),
            "frequency_range_hz": [
                float(self.frequencies_hz[0]),
                float(self.frequencies_hz[-1]),
            ],
            "maximum_reflection_magnitude": float(np.max(np.abs(reflection))),
            "has_uncertainty": bool(self.impedance_real_std_pa_s_m),
            "units": {
                "frequency": "Hz",
                "surface_impedance": "Pa*s/m",
            },
        }

    @classmethod
    def from_csv_and_metadata(
        cls,
        csv_path: str | Path,
        metadata_path: str | Path,
    ) -> "ComplexImpedanceMeasurement":
        """Load strict CSV samples and a JSON provenance sidecar."""
        csv_file = Path(csv_path)
        metadata_file = Path(metadata_path)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version")
            != COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported complex impedance measurement schema")

        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = set(REQUIRED_CSV_COLUMNS).difference(columns)
            if missing:
                raise ValueError(
                    f"measurement CSV is missing columns: {sorted(missing)}"
                )
            optional_presence = [
                column in columns for column in OPTIONAL_STD_COLUMNS
            ]
            if any(optional_presence) and not all(optional_presence):
                raise ValueError(
                    "measurement CSV must provide both uncertainty columns"
                )
            rows = list(reader)
        if not rows:
            raise ValueError("measurement CSV contains no samples")

        def values(column: str) -> tuple[float, ...]:
            try:
                return tuple(float(row[column]) for row in rows)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"measurement CSV column {column!r} contains invalid values"
                ) from error

        environment = metadata.get("environment", {})
        return cls(
            measurement_id=str(metadata.get("measurement_id", "")),
            method=str(metadata.get("method", "")),
            incidence=str(metadata.get("incidence", "")),
            frequencies_hz=values("frequency_hz"),
            impedance_real_pa_s_m=values("impedance_real_pa_s_m"),
            impedance_imag_pa_s_m=values("impedance_imag_pa_s_m"),
            impedance_real_std_pa_s_m=(
                values("impedance_real_std_pa_s_m")
                if all(optional_presence)
                else ()
            ),
            impedance_imag_std_pa_s_m=(
                values("impedance_imag_std_pa_s_m")
                if all(optional_presence)
                else ()
            ),
            air_density_kg_m3=float(environment.get("air_density_kg_m3", 0.0)),
            sound_speed_m_s=float(environment.get("sound_speed_m_s", 0.0)),
            sample=dict(metadata.get("sample", {})),
            provenance=dict(metadata.get("provenance", {})),
        )


@dataclass(frozen=True)
class NormalizedComplexImpedanceMeasurement:
    """Measured complex surface impedance normalized by ``rho*c``.

    This contract preserves published dimensionless resistance/reactance when
    the source does not report the exact atmosphere used for normalization.
    It also records the acoustic-field geometry separately, so grazing-duct
    impedance eduction cannot be mislabeled as a normal-incidence tube test.
    """

    measurement_id: str
    method: str
    acoustic_field_geometry: str
    phasor_convention: str
    frequencies_hz: tuple[float, ...]
    normalized_impedance_real: tuple[float, ...]
    normalized_impedance_imag: tuple[float, ...]
    mean_flow_mach: float
    source_spl_db: float
    sample: dict[str, Any]
    provenance: dict[str, Any]
    applicability: dict[str, Any]
    normalized_impedance_real_std: tuple[float, ...] = ()
    normalized_impedance_imag_std: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.frequencies_hz)
        real = tuple(float(value) for value in self.normalized_impedance_real)
        imag = tuple(float(value) for value in self.normalized_impedance_imag)
        real_std = tuple(
            float(value) for value in self.normalized_impedance_real_std
        )
        imag_std = tuple(
            float(value) for value in self.normalized_impedance_imag_std
        )
        if not self.measurement_id or not self.method:
            raise ValueError("measurement id and method are required")
        if self.acoustic_field_geometry not in {
            "normal_incidence_tube",
            "grazing_duct",
        }:
            raise ValueError(
                "acoustic field geometry must be normal_incidence_tube "
                "or grazing_duct"
            )
        if self.phasor_convention != "exp(+i*omega*t)":
            raise ValueError("phasor convention must be exp(+i*omega*t)")
        if len(frequencies) < 3 or len(real) != len(frequencies) or len(
            imag
        ) != len(frequencies):
            raise ValueError(
                "frequency and normalized impedance arrays must have "
                "equal length >= 3"
            )
        if any(
            not math.isfinite(value) or value <= 0.0 for value in frequencies
        ):
            raise ValueError("measurement frequencies must be finite and positive")
        if any(
            next_value <= value
            for value, next_value in zip(frequencies, frequencies[1:])
        ):
            raise ValueError("measurement frequencies must be strictly increasing")
        if any(not math.isfinite(value) for value in (*real, *imag)):
            raise ValueError("normalized measured impedance must be finite")
        if any(value < 0.0 for value in real):
            raise ValueError(
                "passive normalized impedance cannot have negative real part"
            )
        if bool(real_std) != bool(imag_std):
            raise ValueError(
                "both normalized real and imaginary uncertainty columns are required"
            )
        if real_std and (
            len(real_std) != len(frequencies)
            or len(imag_std) != len(frequencies)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (*real_std, *imag_std)
            )
        ):
            raise ValueError(
                "normalized impedance standard deviations must match data "
                "and be non-negative"
            )
        mean_flow_mach = float(self.mean_flow_mach)
        source_spl_db = float(self.source_spl_db)
        if not math.isfinite(mean_flow_mach) or mean_flow_mach < 0.0:
            raise ValueError("mean flow Mach number must be finite and non-negative")
        if not math.isfinite(source_spl_db) or source_spl_db <= 0.0:
            raise ValueError("source SPL must be finite and positive")
        if not isinstance(self.sample, dict) or not self.sample.get(
            "material_label"
        ):
            raise ValueError("sample.material_label is required")
        if not isinstance(self.provenance, dict) or not self.provenance.get(
            "source_url"
        ):
            raise ValueError("provenance.source_url is required")
        if not self.provenance.get("license"):
            raise ValueError("provenance.license is required")
        if not isinstance(self.applicability, dict) or not self.applicability.get(
            "scope"
        ):
            raise ValueError("applicability.scope is required")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "normalized_impedance_real", real)
        object.__setattr__(self, "normalized_impedance_imag", imag)
        object.__setattr__(
            self,
            "normalized_impedance_real_std",
            real_std,
        )
        object.__setattr__(
            self,
            "normalized_impedance_imag_std",
            imag_std,
        )
        object.__setattr__(self, "mean_flow_mach", mean_flow_mach)
        object.__setattr__(self, "source_spl_db", source_spl_db)

        cayley = self.cayley_reflection_coefficients()
        if any(abs(value) > 1.0 + 1e-9 for value in cayley):
            raise ValueError(
                "normalized measured impedance violates passive Cayley magnitude"
            )

    @property
    def complex_normalized_impedance(self) -> np.ndarray:
        return np.asarray(self.normalized_impedance_real) + 1j * np.asarray(
            self.normalized_impedance_imag
        )

    def cayley_reflection_coefficients(self) -> np.ndarray:
        """Return bounded ``(z-1)/(z+1)`` fitting coordinates.

        This equals normal-incidence pressure reflection only when a locally
        reacting normal-incidence interpretation is valid. For grazing-duct
        data it is used as a bounded numerical transform, not as a claim about
        the measurement incidence.
        """
        impedance = self.complex_normalized_impedance
        return np.asarray((impedance - 1.0) / (impedance + 1.0))

    def metadata(self) -> dict[str, Any]:
        cayley = self.cayley_reflection_coefficients()
        return {
            "schema_version": (
                NORMALIZED_COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION
            ),
            "measurement_id": self.measurement_id,
            "method": self.method,
            "acoustic_field_geometry": self.acoustic_field_geometry,
            "phasor_convention": self.phasor_convention,
            "conditions": {
                "mean_flow_mach": float(self.mean_flow_mach),
                "source_spl_db": float(self.source_spl_db),
            },
            "sample": dict(self.sample),
            "provenance": dict(self.provenance),
            "applicability": dict(self.applicability),
            "num_frequencies": len(self.frequencies_hz),
            "frequency_range_hz": [
                float(self.frequencies_hz[0]),
                float(self.frequencies_hz[-1]),
            ],
            "maximum_cayley_magnitude": float(np.max(np.abs(cayley))),
            "has_uncertainty": bool(self.normalized_impedance_real_std),
            "units": {
                "frequency": "Hz",
                "normalized_surface_impedance": "dimensionless_Z_over_rho_c",
            },
        }

    @classmethod
    def from_csv_and_metadata(
        cls,
        csv_path: str | Path,
        metadata_path: str | Path,
    ) -> "NormalizedComplexImpedanceMeasurement":
        """Load normalized resistance/reactance plus a provenance sidecar."""
        csv_file = Path(csv_path)
        metadata_file = Path(metadata_path)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version")
            != NORMALIZED_COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported normalized complex impedance measurement schema"
            )

        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = set(REQUIRED_NORMALIZED_CSV_COLUMNS).difference(columns)
            if missing:
                raise ValueError(
                    "normalized measurement CSV is missing columns: "
                    f"{sorted(missing)}"
                )
            optional_presence = [
                column in columns for column in OPTIONAL_NORMALIZED_STD_COLUMNS
            ]
            if any(optional_presence) and not all(optional_presence):
                raise ValueError(
                    "normalized measurement CSV must provide both "
                    "uncertainty columns"
                )
            rows = list(reader)
        if not rows:
            raise ValueError("normalized measurement CSV contains no samples")

        def values(column: str) -> tuple[float, ...]:
            try:
                return tuple(float(row[column]) for row in rows)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"normalized measurement CSV column {column!r} "
                    "contains invalid values"
                ) from error

        conditions = metadata.get("conditions", {})
        return cls(
            measurement_id=str(metadata.get("measurement_id", "")),
            method=str(metadata.get("method", "")),
            acoustic_field_geometry=str(
                metadata.get("acoustic_field_geometry", "")
            ),
            phasor_convention=str(metadata.get("phasor_convention", "")),
            frequencies_hz=values("frequency_hz"),
            normalized_impedance_real=values("normalized_impedance_real"),
            normalized_impedance_imag=values("normalized_impedance_imag"),
            normalized_impedance_real_std=(
                values("normalized_impedance_real_std")
                if all(optional_presence)
                else ()
            ),
            normalized_impedance_imag_std=(
                values("normalized_impedance_imag_std")
                if all(optional_presence)
                else ()
            ),
            mean_flow_mach=float(conditions.get("mean_flow_mach", -1.0)),
            source_spl_db=float(conditions.get("source_spl_db", 0.0)),
            sample=dict(metadata.get("sample", {})),
            provenance=dict(metadata.get("provenance", {})),
            applicability=dict(metadata.get("applicability", {})),
        )


__all__ = [
    "COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION",
    "ComplexImpedanceMeasurement",
    "NORMALIZED_COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION",
    "NormalizedComplexImpedanceMeasurement",
    "OPTIONAL_NORMALIZED_STD_COLUMNS",
    "OPTIONAL_STD_COLUMNS",
    "REQUIRED_NORMALIZED_CSV_COLUMNS",
    "REQUIRED_CSV_COLUMNS",
]
