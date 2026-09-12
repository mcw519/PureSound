"""PathEvent data contracts and their JSON round-trip.

``PathEvent`` keeps the physical delay separate from the complex pressure gain
so propagation phase cannot be counted twice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from puresound.audio.rir.physics.impedance.admittance import FirstOrderRelaxationAdmittance
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.physics.wave.source_convention import (
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
)


PATH_GAIN_SCHEMA_VERSION = "puresound.path_gain_spectrum.v1"
PATH_EVENT_SCHEMA_VERSION = "puresound.path_event.v1"
PATH_EVENT_SET_SCHEMA_VERSION = "puresound.path_event_set.v1"


class NormalizedAdmittanceModel(Protocol):
    """Minimum boundary-model interface used by the shoebox reference."""

    def normalized_admittance(self, frequency_hz: float) -> complex:
        """Return dimensionless locally reacting surface admittance."""


def material_absorption_relaxation_models(
    scene: RoomSceneV2,
    *,
    reference_frequency_hz: float = 1000.0,
) -> tuple[
    dict[str, FirstOrderRelaxationAdmittance],
    dict[str, Any],
]:
    """Build a passive causal broad-band prior from material endpoints.

    Absorption magnitude alone does not determine reflection phase.  This
    helper therefore makes that missing evidence explicit: low- and
    high-frequency absorption endpoints determine the two real admittance
    endpoints of a positive-real one-pole model.  The model is deliberately a
    conservative phase prior, not a claim of measured complex impedance.
    """

    reference = float(reference_frequency_hz)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("material reference frequency must be positive")
    low_frequency_hz = max(60.0, 0.125 * reference)
    high_frequency_hz = min(8000.0, 8.0 * reference)
    if high_frequency_hz <= low_frequency_hz:
        raise ValueError("material admittance endpoints must be increasing")

    models: dict[str, FirstOrderRelaxationAdmittance] = {}
    boundaries: dict[str, Any] = {}
    for boundary, material in scene.effective_boundary_materials().items():
        endpoint_admittance: list[float] = []
        endpoint_properties: list[dict[str, float]] = []
        for frequency_hz in (low_frequency_hz, high_frequency_hz):
            absorption = material.absorption.at(frequency_hz)
            transmission = material.transmission.at(frequency_hz)
            reflected_energy = max(
                0.0,
                1.0 - absorption - transmission,
            )
            pressure_reflection = math.sqrt(reflected_energy)
            normalized_admittance = (
                (1.0 - pressure_reflection)
                / max(1.0 + pressure_reflection, 1e-12)
            )
            endpoint_admittance.append(float(normalized_admittance))
            endpoint_properties.append(
                {
                    "frequency_hz": float(frequency_hz),
                    "absorption": float(absorption),
                    "transmission": float(transmission),
                    "normalized_admittance": float(normalized_admittance),
                }
            )
        low_admittance, high_admittance = endpoint_admittance
        model = FirstOrderRelaxationAdmittance(
            normalized_admittance_infinite=high_admittance,
            normalized_admittance_relaxation=(
                low_admittance - high_admittance
            ),
            relaxation_frequency_hz=reference,
        )
        models[boundary] = model
        boundaries[boundary] = {
            "material_id": material.material_id,
            "endpoint_properties": endpoint_properties,
            "model": model.metadata(),
        }
    metadata = {
        "policy": "puresound.material_absorption_relaxation_prior.v1",
        "phase_evidence": (
            "passive_causal_one_pole_prior_from_absorption_endpoints"
        ),
        "reference_frequency_hz": reference,
        "boundaries": boundaries,
    }
    return models, metadata


def locally_reacting_reflection_coefficient(
    normalized_admittance: complex,
    incidence_cosine: float,
) -> complex:
    """Return angle-aware pressure reflection for a local-reaction boundary.

    ``Gamma(theta, f) = (cos(theta) - y(f)) / (cos(theta) + y(f))``,
    where ``y = rho*c*Y_surface`` is dimensionless.
    """
    admittance = complex(normalized_admittance)
    cosine = float(incidence_cosine)
    if not math.isfinite(admittance.real) or not math.isfinite(admittance.imag):
        raise ValueError("normalized admittance must be finite")
    if not math.isfinite(cosine) or not 0.0 < cosine <= 1.0:
        raise ValueError("incidence cosine must lie in (0, 1]")
    denominator = cosine + admittance
    if abs(denominator) <= 1e-15:
        raise ValueError("reflection coefficient denominator is zero")
    return complex((cosine - admittance) / denominator)


def _finite_float_list(name: str, values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


def _vector3(name: str, values: Iterable[float]) -> list[float]:
    result = _finite_float_list(name, values)
    if len(result) != 3:
        raise ValueError(f"{name} must have three values")
    return result


def _unit_vector3(name: str, values: Iterable[float]) -> list[float]:
    result = _vector3(name, values)
    norm = float(np.linalg.norm(np.asarray(result, dtype=np.float64)))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{name} must be a unit vector")
    return result


@dataclass(frozen=True)
class ComplexPathGainSpectrum:
    """Complex pressure gain sampled in frequency, excluding path delay.

    Real and imaginary components are serialized separately because JSON has
    no complex scalar.  Interpolation is linear in frequency and in the
    complex plane; the samples are an inspectable transfer contract, not by
    themselves a promise that arbitrary interpolation is a causal filter.
    """

    frequencies_hz: list[float]
    real: list[float]
    imag: list[float]
    provenance: str
    quantity: str = "pressure_gain_excluding_propagation_delay"
    interpolation: str = "linear_complex_frequency_edge_hold"
    schema_version: str = PATH_GAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PATH_GAIN_SCHEMA_VERSION:
            raise ValueError(f"unsupported path-gain schema: {self.schema_version}")
        frequencies = _finite_float_list("gain frequency", self.frequencies_hz)
        real = _finite_float_list("gain real", self.real)
        imag = _finite_float_list("gain imaginary", self.imag)
        if (
            not frequencies
            or len(frequencies) != len(real)
            or len(frequencies) != len(imag)
        ):
            raise ValueError("gain frequencies, real, and imaginary must match")
        if frequencies[0] < 0.0:
            raise ValueError("gain frequencies must be non-negative")
        if any(b <= a for a, b in zip(frequencies, frequencies[1:])):
            raise ValueError("gain frequencies must be strictly increasing")
        if not self.provenance:
            raise ValueError("gain provenance is required")
        if self.quantity != "pressure_gain_excluding_propagation_delay":
            raise ValueError("unsupported path gain quantity")
        if self.interpolation != "linear_complex_frequency_edge_hold":
            raise ValueError("unsupported path gain interpolation")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "imag", imag)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.real, dtype=np.float64) + 1j * np.asarray(
            self.imag, dtype=np.float64
        )

    def at(self, frequency_hz: float) -> complex:
        frequency = float(frequency_hz)
        if not math.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        return complex(
            np.interp(
                frequency,
                frequencies,
                np.asarray(self.real, dtype=np.float64),
                left=float(self.real[0]),
                right=float(self.real[-1]),
            ),
            np.interp(
                frequency,
                frequencies,
                np.asarray(self.imag, dtype=np.float64),
                left=float(self.imag[0]),
                right=float(self.imag[-1]),
            ),
        )

    def constant_real_value(self, tolerance: float = 1e-12) -> float:
        """Return a renderable scalar, or reject a non-scalar/complex gain."""
        values = self.values
        if np.max(np.abs(values.imag)) > float(tolerance):
            raise ValueError(
                "the M3.1 renderer cannot render a complex gain spectrum; "
                "realize a causal boundary filter first"
            )
        if np.max(np.abs(values.real - values.real[0])) > float(tolerance):
            raise ValueError(
                "the M3.1 renderer cannot render a frequency-dependent gain "
                "spectrum; realize a causal boundary filter first"
            )
        return float(values.real[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frequencies_hz": list(self.frequencies_hz),
            "real": list(self.real),
            "imag": list(self.imag),
            "provenance": self.provenance,
            "quantity": self.quantity,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComplexPathGainSpectrum":
        return cls(
            frequencies_hz=data["frequencies_hz"],
            real=data["real"],
            imag=data["imag"],
            provenance=str(data["provenance"]),
            quantity=str(
                data.get(
                    "quantity",
                    "pressure_gain_excluding_propagation_delay",
                )
            ),
            interpolation=str(
                data.get(
                    "interpolation",
                    "linear_complex_frequency_edge_hold",
                )
            ),
            schema_version=str(
                data.get("schema_version", PATH_GAIN_SCHEMA_VERSION)
            ),
        )

    @classmethod
    def constant(
        cls,
        gain: complex,
        *,
        provenance: str,
    ) -> "ComplexPathGainSpectrum":
        value = complex(gain)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("constant path gain must be finite")
        return cls(
            frequencies_hz=[0.0],
            real=[float(value.real)],
            imag=[float(value.imag)],
            provenance=provenance,
        )


@dataclass(frozen=True)
class PathEvent:
    """One coherent source-to-receiver propagation path."""

    event_id: str
    source_id: str
    receiver_id: str
    path_type: str
    source_position_m: list[float]
    receiver_position_m: list[float]
    distance_m: float
    delay_s: float
    sound_speed_m_s: float
    departure_direction_unit: list[float]
    arrival_direction_unit: list[float]
    surface_ids: list[str]
    interaction_types: list[str]
    interaction_group_ids: list[int]
    interaction_points_m: list[list[float]]
    incidence_cosines: list[float]
    gain_spectrum: ComplexPathGainSpectrum
    image_order_xyz: list[int] = field(default_factory=list)
    source_directivity_id: str = "omnidirectional"
    receiver_directivity_id: str = "omnidirectional"
    source_directivity_gain: float = 1.0
    receiver_directivity_gain: float = 1.0
    energy_partition_fraction: float = 1.0
    visible: bool = True
    diffraction_model: str = "none"
    scattering_model: str = "specular"
    schema_version: str = PATH_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PATH_EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported path-event schema: {self.schema_version}")
        if not self.event_id or not self.source_id or not self.receiver_id:
            raise ValueError("event, source, and receiver IDs are required")
        if self.path_type not in {
            "direct",
            "specular_reflection",
            "transmission",
            "diffraction",
            "scattering",
        }:
            raise ValueError(f"unsupported path type: {self.path_type}")
        source = _vector3("source position", self.source_position_m)
        receiver = _vector3("receiver position", self.receiver_position_m)
        departure = _unit_vector3(
            "departure direction", self.departure_direction_unit
        )
        arrival = _unit_vector3("arrival direction", self.arrival_direction_unit)
        points = [
            _vector3("interaction point", point)
            for point in self.interaction_points_m
        ]
        surfaces = [str(value) for value in self.surface_ids]
        interaction_types = [str(value) for value in self.interaction_types]
        interaction_groups = [int(value) for value in self.interaction_group_ids]
        cosines = _finite_float_list("incidence cosine", self.incidence_cosines)
        if (
            len(points) != len(surfaces)
            or len(points) != len(interaction_types)
            or len(points) != len(interaction_groups)
            or len(points) != len(cosines)
        ):
            raise ValueError(
                "surface IDs, interaction types, groups, points, and "
                "incidence cosines must match"
            )
        if any(
            converted != value
            for converted, value in zip(
                interaction_groups,
                self.interaction_group_ids,
            )
        ):
            raise ValueError("interaction group IDs must be integers")
        if interaction_groups and (
            interaction_groups[0] != 0
            or any(
                current not in {previous, previous + 1}
                for previous, current in zip(
                    interaction_groups,
                    interaction_groups[1:],
                )
            )
        ):
            raise ValueError(
                "interaction group IDs must be ordered consecutive groups"
            )
        for index in range(1, len(points)):
            if interaction_groups[index] == interaction_groups[index - 1]:
                if not np.allclose(
                    points[index],
                    points[index - 1],
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        "simultaneous interactions must share one point"
                    )
        if any(
            value
            not in {"reflection", "transmission", "diffraction", "scattering"}
            for value in interaction_types
        ):
            raise ValueError("unsupported path interaction type")
        if any(not 0.0 <= value <= 1.0 for value in cosines):
            raise ValueError("incidence cosines must be in [0, 1]")
        if self.path_type == "direct" and points:
            raise ValueError("a direct path cannot have surface interactions")
        if self.path_type == "specular_reflection" and (
            not points or any(value != "reflection" for value in interaction_types)
        ):
            raise ValueError(
                "a specular reflected path requires reflection interactions"
            )
        image_order = [int(value) for value in self.image_order_xyz]
        if any(
            converted != value
            for converted, value in zip(image_order, self.image_order_xyz)
        ):
            raise ValueError("image order values must be integers")
        if len(image_order) not in {0, 3}:
            raise ValueError("image order must be empty or contain x/y/z orders")
        if image_order and sum(abs(value) for value in image_order) != len(
            points
        ):
            raise ValueError(
                "image-order Manhattan norm must match interaction count"
            )
        distance = float(self.distance_m)
        delay = float(self.delay_s)
        sound_speed = float(self.sound_speed_m_s)
        if (
            not math.isfinite(distance)
            or not math.isfinite(delay)
            or not math.isfinite(sound_speed)
            or distance <= 0.0
            or delay <= 0.0
            or sound_speed <= 0.0
        ):
            raise ValueError("path distance, delay, and sound speed must be positive")
        if not math.isclose(
            delay,
            distance / sound_speed,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("path delay must equal distance / sound speed")
        source_gain = float(self.source_directivity_gain)
        receiver_gain = float(self.receiver_directivity_gain)
        if not math.isfinite(source_gain) or not math.isfinite(receiver_gain):
            raise ValueError("directivity gains must be finite")
        energy_partition = float(self.energy_partition_fraction)
        if (
            not math.isfinite(energy_partition)
            or not 0.0 <= energy_partition <= 1.0
        ):
            raise ValueError("energy partition fraction must be in [0, 1]")
        if not self.source_directivity_id or not self.receiver_directivity_id:
            raise ValueError("directivity IDs are required")
        if not self.diffraction_model or not self.scattering_model:
            raise ValueError("diffraction and scattering model names are required")
        object.__setattr__(self, "source_position_m", source)
        object.__setattr__(self, "receiver_position_m", receiver)
        object.__setattr__(self, "departure_direction_unit", departure)
        object.__setattr__(self, "arrival_direction_unit", arrival)
        object.__setattr__(self, "surface_ids", surfaces)
        object.__setattr__(self, "interaction_types", interaction_types)
        object.__setattr__(
            self,
            "interaction_group_ids",
            interaction_groups,
        )
        object.__setattr__(self, "interaction_points_m", points)
        object.__setattr__(self, "incidence_cosines", cosines)
        object.__setattr__(self, "image_order_xyz", image_order)
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "delay_s", delay)
        object.__setattr__(self, "sound_speed_m_s", sound_speed)
        object.__setattr__(self, "source_directivity_gain", source_gain)
        object.__setattr__(self, "receiver_directivity_gain", receiver_gain)
        object.__setattr__(
            self,
            "energy_partition_fraction",
            energy_partition,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "source_id": self.source_id,
            "receiver_id": self.receiver_id,
            "path_type": self.path_type,
            "source_position_m": list(self.source_position_m),
            "receiver_position_m": list(self.receiver_position_m),
            "distance_m": self.distance_m,
            "delay_s": self.delay_s,
            "sound_speed_m_s": self.sound_speed_m_s,
            "departure_direction_unit": list(self.departure_direction_unit),
            "arrival_direction_unit": list(self.arrival_direction_unit),
            "surface_ids": list(self.surface_ids),
            "interaction_types": list(self.interaction_types),
            "interaction_group_ids": list(self.interaction_group_ids),
            "interaction_points_m": [
                list(point) for point in self.interaction_points_m
            ],
            "incidence_cosines": list(self.incidence_cosines),
            "gain_spectrum": self.gain_spectrum.to_dict(),
            "image_order_xyz": list(self.image_order_xyz),
            "source_directivity_id": self.source_directivity_id,
            "receiver_directivity_id": self.receiver_directivity_id,
            "source_directivity_gain": self.source_directivity_gain,
            "receiver_directivity_gain": self.receiver_directivity_gain,
            "energy_partition_fraction": self.energy_partition_fraction,
            "visible": bool(self.visible),
            "diffraction_model": self.diffraction_model,
            "scattering_model": self.scattering_model,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PathEvent":
        return cls(
            event_id=str(data["event_id"]),
            source_id=str(data["source_id"]),
            receiver_id=str(data["receiver_id"]),
            path_type=str(data["path_type"]),
            source_position_m=data["source_position_m"],
            receiver_position_m=data["receiver_position_m"],
            distance_m=float(data["distance_m"]),
            delay_s=float(data["delay_s"]),
            sound_speed_m_s=float(data["sound_speed_m_s"]),
            departure_direction_unit=data["departure_direction_unit"],
            arrival_direction_unit=data["arrival_direction_unit"],
            surface_ids=data["surface_ids"],
            interaction_types=data.get(
                "interaction_types",
                (
                    ["reflection"] * len(data["surface_ids"])
                    if data["path_type"] == "specular_reflection"
                    else []
                ),
            ),
            interaction_group_ids=data.get(
                "interaction_group_ids",
                list(range(len(data["surface_ids"]))),
            ),
            interaction_points_m=data["interaction_points_m"],
            incidence_cosines=data["incidence_cosines"],
            gain_spectrum=ComplexPathGainSpectrum.from_dict(data["gain_spectrum"]),
            image_order_xyz=data.get("image_order_xyz", []),
            source_directivity_id=str(
                data.get("source_directivity_id", "omnidirectional")
            ),
            receiver_directivity_id=str(
                data.get("receiver_directivity_id", "omnidirectional")
            ),
            source_directivity_gain=float(
                data.get("source_directivity_gain", 1.0)
            ),
            receiver_directivity_gain=float(
                data.get("receiver_directivity_gain", 1.0)
            ),
            energy_partition_fraction=float(
                data.get("energy_partition_fraction", 1.0)
            ),
            visible=bool(data.get("visible", True)),
            diffraction_model=str(data.get("diffraction_model", "none")),
            scattering_model=str(data.get("scattering_model", "specular")),
            schema_version=str(
                data.get("schema_version", PATH_EVENT_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True)
class PathEventSet:
    """Versioned collection rendered as one source-receiver RIR channel."""

    scene_id: str
    source_id: str
    receiver_id: str
    events: list[PathEvent]
    generator: str = "puresound.exact_ordered_shoebox_image_source.v2"
    source_convention: str = FREE_FIELD_1_OVER_R_RIR_CONVENTION
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PATH_EVENT_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PATH_EVENT_SET_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported path-event-set schema: {self.schema_version}"
            )
        if not self.scene_id or not self.source_id or not self.receiver_id:
            raise ValueError("scene, source, and receiver IDs are required")
        if not self.events:
            raise ValueError("a path event set cannot be empty")
        if any(
            event.source_id != self.source_id
            or event.receiver_id != self.receiver_id
            for event in self.events
        ):
            raise ValueError("all events must match the collection channel IDs")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("path event IDs must be unique")
        if self.source_convention != FREE_FIELD_1_OVER_R_RIR_CONVENTION:
            raise ValueError("unsupported path-event source convention")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "source_id": self.source_id,
            "receiver_id": self.receiver_id,
            "events": [event.to_dict() for event in self.events],
            "generator": self.generator,
            "source_convention": self.source_convention,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PathEventSet":
        return cls(
            scene_id=str(data["scene_id"]),
            source_id=str(data["source_id"]),
            receiver_id=str(data["receiver_id"]),
            events=[PathEvent.from_dict(item) for item in data["events"]],
            generator=str(
                data.get(
                    "generator",
                    "puresound.exact_ordered_shoebox_image_source.v2",
                )
            ),
            source_convention=str(
                data.get(
                    "source_convention",
                    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
                )
            ),
            metadata=dict(data.get("metadata", {})),
            schema_version=str(
                data.get("schema_version", PATH_EVENT_SET_SCHEMA_VERSION)
            ),
        )

    @classmethod
    def from_json(cls, payload_or_path: str | Path) -> "PathEventSet":
        if isinstance(payload_or_path, Path):
            payload = payload_or_path.read_text(encoding="utf-8")
        else:
            raw = str(payload_or_path)
            candidate = Path(raw)
            payload = (
                candidate.read_text(encoding="utf-8")
                if not raw.lstrip().startswith("{") and candidate.exists()
                else raw
            )
        return cls.from_dict(json.loads(payload))


__all__ = [
    "ComplexPathGainSpectrum",
    "NormalizedAdmittanceModel",
    "PATH_EVENT_SCHEMA_VERSION",
    "PATH_EVENT_SET_SCHEMA_VERSION",
    "PATH_GAIN_SCHEMA_VERSION",
    "PathEvent",
    "PathEventSet",
    "locally_reacting_reflection_coefficient",
    "material_absorption_relaxation_models",
]
