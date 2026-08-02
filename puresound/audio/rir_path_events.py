"""Inspectable direct and early-reflection path events for RIR rendering.

The event representation deliberately separates a physical propagation delay
from the complex pressure gain.  A renderer can therefore change sample rate
without changing geometry, and it cannot accidentally count propagation phase
twice.

M3.1 implements exact shoebox direct and first-order specular paths. M3.2
renders supported passive rational admittance models through a bilinear,
angle-aware causal boundary filter. Arbitrary sampled complex spectra remain
non-renderable because samples alone do not prove a causal realization.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from puresound.audio.acoustic_impedance import (
    FirstOrderRelaxationAdmittance,
    digital_locally_reacting_reflection_filter,
)
from puresound.audio.rir_scene import (
    RoomSceneV2,
    SceneObject,
    SHOEBOX_BOUNDARIES,
)
from puresound.audio.rir_source_convention import (
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
)


PATH_GAIN_SCHEMA_VERSION = "puresound.path_gain_spectrum.v1"
PATH_EVENT_SCHEMA_VERSION = "puresound.path_event.v1"
PATH_EVENT_SET_SCHEMA_VERSION = "puresound.path_event_set.v1"
FRACTIONAL_DELAY_POLICY = "puresound.causal_forward_lagrange.v1"
OBJECT_VISIBILITY_POLICY = (
    "puresound.closed_vertical_prism_segment_visibility.v1"
)


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


_BOUNDARY_GEOMETRY: dict[str, tuple[int, bool, tuple[float, float, float]]] = {
    "west": (0, False, (-1.0, 0.0, 0.0)),
    "east": (0, True, (1.0, 0.0, 0.0)),
    "south": (1, False, (0.0, -1.0, 0.0)),
    "north": (1, True, (0.0, 1.0, 0.0)),
    "floor": (2, False, (0.0, 0.0, -1.0)),
    "ceiling": (2, True, (0.0, 0.0, 1.0)),
}
_BOUNDARY_BY_AXIS_AND_UPPER = {
    (axis, upper): boundary
    for boundary, (axis, upper, _normal) in _BOUNDARY_GEOMETRY.items()
}


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


def _validate_shoebox_position(
    name: str,
    position_m: Iterable[float],
    dimensions_m: np.ndarray,
) -> np.ndarray:
    position = np.asarray(_vector3(name, position_m), dtype=np.float64)
    if np.any(position <= 0.0) or np.any(position >= dimensions_m):
        raise ValueError(f"{name} must lie strictly inside the shoebox")
    return position


def _point_on_segment_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    vector = end - start
    relative = point - start
    cross = float(vector[0] * relative[1] - vector[1] * relative[0])
    if abs(cross) > tolerance * max(1.0, float(np.linalg.norm(vector))):
        return False
    dot = float(np.dot(relative, vector))
    return bool(
        dot >= -tolerance
        and dot <= float(np.dot(vector, vector)) + tolerance
    )


def _segments_intersect_2d(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    def orientation(
        start: np.ndarray,
        end: np.ndarray,
        point: np.ndarray,
    ) -> float:
        first = end - start
        second = point - start
        return float(first[0] * second[1] - first[1] * second[0])

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if (
        ((o1 > tolerance and o2 < -tolerance) or (
            o1 < -tolerance and o2 > tolerance
        ))
        and ((o3 > tolerance and o4 < -tolerance) or (
            o3 < -tolerance and o4 > tolerance
        ))
    ):
        return True
    return any(
        abs(orientation_value) <= tolerance
        and _point_on_segment_2d(
            point,
            segment_start,
            segment_end,
            tolerance=tolerance,
        )
        for orientation_value, point, segment_start, segment_end in (
            (o1, second_start, first_start, first_end),
            (o2, second_end, first_start, first_end),
            (o3, first_start, second_start, second_end),
            (o4, first_end, second_start, second_end),
        )
    )


def segment_intersects_scene_object(
    start_m: Iterable[float],
    end_m: Iterable[float],
    scene_object: SceneObject,
    *,
    tolerance_m: float = 1e-10,
) -> bool:
    """Return whether a closed segment touches a vertical object prism."""
    start = np.asarray(_vector3("segment start", start_m), dtype=np.float64)
    end = np.asarray(_vector3("segment end", end_m), dtype=np.float64)
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("visibility tolerance must be finite and non-negative")
    if float(np.linalg.norm(end - start)) <= tolerance:
        raise ValueError("visibility segment must have nonzero length")

    delta_z = float(end[2] - start[2])
    if abs(delta_z) <= tolerance:
        if (
            start[2] < scene_object.z_min - tolerance
            or start[2] > scene_object.z_max + tolerance
        ):
            return False
        parameter_min = 0.0
        parameter_max = 1.0
    else:
        first = (scene_object.z_min - float(start[2])) / delta_z
        second = (scene_object.z_max - float(start[2])) / delta_z
        parameter_min = max(0.0, min(first, second))
        parameter_max = min(1.0, max(first, second))
        if parameter_min > parameter_max + tolerance:
            return False

    clipped_start = (
        start[:2] + parameter_min * (end[:2] - start[:2])
    )
    clipped_end = (
        start[:2] + parameter_max * (end[:2] - start[:2])
    )
    polygon = np.asarray(scene_object.footprint, dtype=np.float64)
    if (
        scene_object.contains_xy(clipped_start)
        or scene_object.contains_xy(clipped_end)
    ):
        return True
    return any(
        _segments_intersect_2d(
            clipped_start,
            clipped_end,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
            tolerance=tolerance,
        )
        for index in range(len(polygon))
    )


def segment_scene_object_intersection_interval(
    start_m: Iterable[float],
    end_m: Iterable[float],
    scene_object: SceneObject,
    *,
    tolerance_m: float = 1e-10,
) -> tuple[float, float] | None:
    """Return the first/last segment parameters lying inside a prism."""
    start = np.asarray(_vector3("segment start", start_m), dtype=np.float64)
    end = np.asarray(_vector3("segment end", end_m), dtype=np.float64)
    delta = end - start
    tolerance = float(tolerance_m)
    if float(np.linalg.norm(delta)) <= tolerance:
        raise ValueError("intersection segment must have nonzero length")
    candidates = [0.0, 1.0]
    if abs(float(delta[2])) > tolerance:
        for height in (scene_object.z_min, scene_object.z_max):
            parameter = (height - float(start[2])) / float(delta[2])
            if -tolerance <= parameter <= 1.0 + tolerance:
                candidates.append(float(np.clip(parameter, 0.0, 1.0)))

    def cross2(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    polygon = np.asarray(scene_object.footprint, dtype=np.float64)
    ray_xy = delta[:2]
    ray_length_squared = float(np.dot(ray_xy, ray_xy))
    for index in range(len(polygon)):
        edge_start = polygon[index]
        edge_vector = polygon[(index + 1) % len(polygon)] - edge_start
        offset = edge_start - start[:2]
        denominator = cross2(ray_xy, edge_vector)
        if abs(denominator) > tolerance:
            parameter = cross2(offset, edge_vector) / denominator
            edge_parameter = cross2(offset, ray_xy) / denominator
            if (
                -tolerance <= parameter <= 1.0 + tolerance
                and -tolerance <= edge_parameter <= 1.0 + tolerance
            ):
                candidates.append(
                    float(np.clip(parameter, 0.0, 1.0))
                )
        elif (
            ray_length_squared > tolerance**2
            and abs(cross2(offset, ray_xy)) <= tolerance
        ):
            for vertex in (
                edge_start,
                polygon[(index + 1) % len(polygon)],
            ):
                parameter = float(
                    np.dot(vertex - start[:2], ray_xy)
                    / ray_length_squared
                )
                if -tolerance <= parameter <= 1.0 + tolerance:
                    candidates.append(
                        float(np.clip(parameter, 0.0, 1.0))
                    )
    candidates = sorted(candidates)
    unique = []
    for value in candidates:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    inside_intervals = []
    for lower, upper in zip(unique, unique[1:]):
        if upper - lower <= tolerance:
            continue
        midpoint = start + 0.5 * (lower + upper) * delta
        if scene_object.contains_point(midpoint):
            inside_intervals.append((lower, upper))
    if not inside_intervals:
        return None
    return (
        float(inside_intervals[0][0]),
        float(inside_intervals[-1][1]),
    )


def _path_event_vertices(event: PathEvent) -> list[np.ndarray]:
    vertices = [
        np.asarray(event.source_position_m, dtype=np.float64),
    ]
    previous_group: int | None = None
    for group, point in zip(
        event.interaction_group_ids,
        event.interaction_points_m,
    ):
        if group != previous_group:
            vertices.append(np.asarray(point, dtype=np.float64))
            previous_group = group
    vertices.append(
        np.asarray(event.receiver_position_m, dtype=np.float64)
    )
    return vertices


def apply_scene_object_visibility(
    event_set: PathEventSet,
    scene_objects: Iterable[SceneObject],
    *,
    tolerance_m: float = 1e-10,
) -> PathEventSet:
    """Mark paths blocked by any serialized vertical-prism scene object."""
    objects = list(scene_objects)
    blocked_by_event: dict[str, list[str]] = {}
    resolved_events = []
    for event in event_set.events:
        vertices = _path_event_vertices(event)
        blocked = []
        for scene_object in objects:
            if any(
                segment_intersects_scene_object(
                    vertices[index],
                    vertices[index + 1],
                    scene_object,
                    tolerance_m=tolerance_m,
                )
                for index in range(len(vertices) - 1)
            ):
                blocked.append(scene_object.object_id)
        if blocked:
            blocked_by_event[event.event_id] = blocked
        resolved_events.append(
            replace(
                event,
                visible=bool(event.visible and not blocked),
            )
        )
    metadata = dict(event_set.metadata)
    metadata.update(
        {
            "object_visibility_policy": OBJECT_VISIBILITY_POLICY,
            "object_visibility_tolerance_m": float(tolerance_m),
            "object_count": len(objects),
            "blocked_event_count": len(blocked_by_event),
            "visible_event_count": sum(
                event.visible for event in resolved_events
            ),
            "blocked_event_occluder_ids": blocked_by_event,
        }
    )
    return replace(
        event_set,
        events=resolved_events,
        generator=(
            f"{event_set.generator}+vertical_prism_visibility.v1"
        ),
        metadata=metadata,
    )


def _gain_spectrum(
    *,
    distance_m: float,
    incidence_cosines: Iterable[float],
    frequencies_hz: list[float],
    admittance_models: Iterable[NormalizedAdmittanceModel | None],
) -> ComplexPathGainSpectrum:
    cosines = [float(value) for value in incidence_cosines]
    models = list(admittance_models)
    if len(cosines) != len(models):
        raise ValueError("incidence cosines and admittance models must match")
    if not cosines:
        return ComplexPathGainSpectrum.constant(
            1.0 / float(distance_m),
            provenance=(
                "free_field_1_over_r; propagation delay is stored separately"
            ),
        )
    if all(model is None for model in models):
        return ComplexPathGainSpectrum.constant(
            1.0 / float(distance_m),
            provenance=(
                "free_field_1_over_r_times_rigid_specular_reflection_reference"
            ),
        )
    reflections = []
    for frequency in frequencies_hz:
        reflection = complex(1.0, 0.0)
        for incidence_cosine, admittance_model in zip(cosines, models):
            if admittance_model is not None:
                reflection *= locally_reacting_reflection_coefficient(
                    admittance_model.normalized_admittance(frequency),
                    incidence_cosine,
                )
        reflections.append(reflection)
    gains = [reflection / float(distance_m) for reflection in reflections]
    return ComplexPathGainSpectrum(
        frequencies_hz=frequencies_hz,
        real=[float(value.real) for value in gains],
        imag=[float(value.imag) for value in gains],
        provenance=(
            "free_field_1_over_r_times_ordered_angle_aware_locally_reacting_"
            "complex_pressure_reflections"
        ),
    )


def _scaled_gain_spectrum(
    spectrum: ComplexPathGainSpectrum,
    scale: float,
    *,
    provenance_suffix: str,
) -> ComplexPathGainSpectrum:
    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("path gain scale must be finite and non-negative")
    return ComplexPathGainSpectrum(
        frequencies_hz=list(spectrum.frequencies_hz),
        real=[value * item for item in spectrum.real],
        imag=[value * item for item in spectrum.imag],
        provenance=f"{spectrum.provenance}; {provenance_suffix}",
        quantity=spectrum.quantity,
        interpolation=spectrum.interpolation,
    )


def augment_scene_path_events_with_interactions(
    event_set: PathEventSet,
    scene: RoomSceneV2,
    *,
    boundary_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    scattering_reference_hz: float = 1000.0,
    scattering_branch_count: int = 4,
    diffraction_reference_hz: float = 1000.0,
    maximum_diffraction_edges_per_object: int = 2,
    include_transmission: bool = True,
    include_diffraction: bool = True,
    include_scattering: bool = True,
) -> PathEventSet:
    """Add scoped M3 furniture transmission, edge diffraction, and scattering.

    Transmission follows the blocked direct segment through all intersected
    vertical prisms. Diffraction uses the two shortest visible vertical-edge
    detours per blocker with a bounded reference-frequency knife-edge gain.
    First-order wall scattering partitions each parent event's energy between
    its specular path and deterministic nearby surface samples. Higher-order
    paths apply every encountered wall's specular retention; their removed
    diffuse share is delegated to the FDN late field to avoid exponential path
    branching.
    """
    if event_set.scene_id != scene.scene_id:
        raise ValueError("event set and scene IDs must match")
    branch_count = int(scattering_branch_count)
    maximum_edges = int(maximum_diffraction_edges_per_object)
    if branch_count != scattering_branch_count or branch_count < 1:
        raise ValueError("scattering branch count must be a positive integer")
    if maximum_edges != maximum_diffraction_edges_per_object or maximum_edges < 1:
        raise ValueError(
            "maximum diffraction edge count must be a positive integer"
        )
    if (
        not math.isfinite(float(scattering_reference_hz))
        or float(scattering_reference_hz) <= 0.0
        or not math.isfinite(float(diffraction_reference_hz))
        or float(diffraction_reference_hz) <= 0.0
    ):
        raise ValueError("interaction reference frequencies must be positive")
    models = dict(boundary_admittance_models or {})
    surface_by_id = {
        surface.surface_id: surface for surface in scene.surfaces
    }
    events: list[PathEvent] = []
    scattering_events: list[PathEvent] = []
    higher_order_scattering_loss_events = 0
    for event in event_set.events:
        if (
            include_scattering
            and event.visible
            and event.path_type == "specular_reflection"
            and len(event.surface_ids) > 1
            and all(surface_id in surface_by_id for surface_id in event.surface_ids)
        ):
            specular_energy_retention = float(
                np.prod(
                    [
                        1.0
                        - float(
                            scene.materials[
                                surface_by_id[surface_id].material_id
                            ].scattering.at(scattering_reference_hz)
                        )
                        for surface_id in event.surface_ids
                    ]
                )
            )
            if specular_energy_retention < 1.0:
                higher_order_scattering_loss_events += 1
                events.append(
                    replace(
                        event,
                        gain_spectrum=_scaled_gain_spectrum(
                            event.gain_spectrum,
                            math.sqrt(max(0.0, specular_energy_retention)),
                            provenance_suffix=(
                                "multi-order specular retention; diffuse share "
                                "delegated to the FDN late field"
                            ),
                        ),
                        energy_partition_fraction=(
                            event.energy_partition_fraction
                            * specular_energy_retention
                        ),
                        scattering_model=(
                            "puresound.multi_order_specular_retention_fdn_diffuse.v1"
                        ),
                    )
                )
                continue
        if (
            include_scattering
            and event.visible
            and event.path_type == "specular_reflection"
            and len(event.surface_ids) == 1
            and event.surface_ids[0] in surface_by_id
        ):
            surface = surface_by_id[event.surface_ids[0]]
            material = scene.materials[surface.material_id]
            scattering = float(
                material.scattering.at(scattering_reference_hz)
            )
            if scattering > 0.0:
                specular_scale = math.sqrt(max(0.0, 1.0 - scattering))
                events.append(
                    replace(
                        event,
                        gain_spectrum=_scaled_gain_spectrum(
                            event.gain_spectrum,
                            specular_scale,
                            provenance_suffix=(
                                "specular energy partition "
                                f"(1-s)={1.0 - scattering:.9g}"
                            ),
                        ),
                        energy_partition_fraction=(
                            event.energy_partition_fraction
                            * (1.0 - scattering)
                        ),
                    )
                )
                original_point = np.asarray(
                    event.interaction_points_m[0],
                    dtype=np.float64,
                )
                vertices = np.asarray(surface.vertices_m, dtype=np.float64)
                axis, _upper, normal_values = _BOUNDARY_GEOMETRY[
                    surface.boundary
                ]
                tangential_axes = [
                    value for value in range(3) if value != axis
                ]
                spans = np.ptp(vertices[:, tangential_axes], axis=0)
                radius = 0.12 * float(np.min(spans))
                normal = np.asarray(normal_values, dtype=np.float64)
                for branch in range(branch_count):
                    angle = (
                        2.0 * math.pi * (branch + 0.5) / branch_count
                    )
                    point = original_point.copy()
                    point[tangential_axes[0]] += radius * math.cos(angle)
                    point[tangential_axes[1]] += radius * math.sin(angle)
                    for tangent_axis in tangential_axes:
                        lower = float(np.min(vertices[:, tangent_axis]))
                        upper = float(np.max(vertices[:, tangent_axis]))
                        epsilon = min(1e-6, 0.1 * (upper - lower))
                        point[tangent_axis] = float(
                            np.clip(
                                point[tangent_axis],
                                lower + epsilon,
                                upper - epsilon,
                            )
                        )
                    first = point - np.asarray(
                        event.source_position_m,
                        dtype=np.float64,
                    )
                    second = np.asarray(
                        event.receiver_position_m,
                        dtype=np.float64,
                    ) - point
                    first_length = float(np.linalg.norm(first))
                    second_length = float(np.linalg.norm(second))
                    distance = first_length + second_length
                    departure = first / first_length
                    arrival = second / second_length
                    cosine = abs(float(np.dot(departure, normal)))
                    partition = scattering / branch_count
                    base_gain = _gain_spectrum(
                        distance_m=distance,
                        incidence_cosines=[cosine],
                        frequencies_hz=list(
                            event.gain_spectrum.frequencies_hz
                        ),
                        admittance_models=[
                            models.get(event.surface_ids[0])
                        ],
                    )
                    candidate = PathEvent(
                        event_id=(
                            f"{event.event_id}:scatter:{branch:02d}"
                        ),
                        source_id=event.source_id,
                        receiver_id=event.receiver_id,
                        path_type="scattering",
                        source_position_m=list(event.source_position_m),
                        receiver_position_m=list(
                            event.receiver_position_m
                        ),
                        distance_m=distance,
                        delay_s=distance / event.sound_speed_m_s,
                        sound_speed_m_s=event.sound_speed_m_s,
                        departure_direction_unit=departure.tolist(),
                        arrival_direction_unit=arrival.tolist(),
                        surface_ids=list(event.surface_ids),
                        interaction_types=["scattering"],
                        interaction_group_ids=[0],
                        interaction_points_m=[point.tolist()],
                        incidence_cosines=[cosine],
                        gain_spectrum=_scaled_gain_spectrum(
                            base_gain,
                            math.sqrt(partition),
                            provenance_suffix=(
                                "deterministic diffuse energy partition "
                                f"s/N={partition:.9g}"
                            ),
                        ),
                        source_directivity_id=(
                            event.source_directivity_id
                        ),
                        receiver_directivity_id=(
                            event.receiver_directivity_id
                        ),
                        source_directivity_gain=(
                            event.source_directivity_gain
                        ),
                        receiver_directivity_gain=(
                            event.receiver_directivity_gain
                        ),
                        energy_partition_fraction=partition,
                        visible=True,
                        diffraction_model="none",
                        scattering_model=(
                            "puresound.deterministic_surface_partition.v1"
                        ),
                    )
                    candidate_vertices = _path_event_vertices(candidate)
                    blocked = any(
                        segment_intersects_scene_object(
                            candidate_vertices[index],
                            candidate_vertices[index + 1],
                            scene_object,
                        )
                        for scene_object in scene.objects
                        for index in range(
                            len(candidate_vertices) - 1
                        )
                    )
                    scattering_events.append(
                        replace(candidate, visible=not blocked)
                    )
                continue
        events.append(event)
    events.extend(scattering_events)

    direct = next(
        (event for event in event_set.events if event.path_type == "direct"),
        None,
    )
    transmission_events: list[PathEvent] = []
    diffraction_events: list[PathEvent] = []
    if direct is not None and not direct.visible:
        source = np.asarray(direct.source_position_m, dtype=np.float64)
        receiver = np.asarray(
            direct.receiver_position_m,
            dtype=np.float64,
        )
        direct_vector = receiver - source
        intersections = []
        for scene_object in scene.objects:
            interval = segment_scene_object_intersection_interval(
                source,
                receiver,
                scene_object,
            )
            if interval is not None:
                intersections.append((interval, scene_object))
        intersections.sort(key=lambda value: value[0][0])
        if include_transmission and intersections:
            energy_transmission = float(
                np.prod(
                    [
                        scene_object.transmission
                        for _interval, scene_object in intersections
                    ]
                )
            )
            if energy_transmission > 0.0:
                points = []
                surface_ids = []
                for interval, scene_object in intersections:
                    points.extend(
                        [
                            (
                                source + interval[0] * direct_vector
                            ).tolist(),
                            (
                                source + interval[1] * direct_vector
                            ).tolist(),
                        ]
                    )
                    surface_ids.extend(
                        [
                            f"{scene_object.object_id}:entry",
                            f"{scene_object.object_id}:exit",
                        ]
                    )
                pressure_transmission = math.sqrt(energy_transmission)
                transmission_events.append(
                    PathEvent(
                        event_id=(
                            f"{direct.event_id}:object_transmission"
                        ),
                        source_id=direct.source_id,
                        receiver_id=direct.receiver_id,
                        path_type="transmission",
                        source_position_m=list(direct.source_position_m),
                        receiver_position_m=list(
                            direct.receiver_position_m
                        ),
                        distance_m=direct.distance_m,
                        delay_s=direct.delay_s,
                        sound_speed_m_s=direct.sound_speed_m_s,
                        departure_direction_unit=list(
                            direct.departure_direction_unit
                        ),
                        arrival_direction_unit=list(
                            direct.arrival_direction_unit
                        ),
                        surface_ids=surface_ids,
                        interaction_types=["transmission"] * len(points),
                        interaction_group_ids=list(range(len(points))),
                        interaction_points_m=points,
                        incidence_cosines=[1.0] * len(points),
                        gain_spectrum=ComplexPathGainSpectrum.constant(
                            pressure_transmission
                            / direct.distance_m,
                            provenance=(
                                "straight_through_vertical_prisms_times_"
                                "sqrt_product_energy_transmission"
                            ),
                        ),
                        source_directivity_id=(
                            direct.source_directivity_id
                        ),
                        receiver_directivity_id=(
                            direct.receiver_directivity_id
                        ),
                        source_directivity_gain=(
                            direct.source_directivity_gain
                        ),
                        receiver_directivity_gain=(
                            direct.receiver_directivity_gain
                        ),
                        energy_partition_fraction=energy_transmission,
                        visible=True,
                        diffraction_model="none",
                        scattering_model="none",
                    )
                )
        if include_diffraction:
            wavelength = (
                direct.sound_speed_m_s / float(diffraction_reference_hz)
            )
            for _interval, scene_object in intersections:
                candidates = []
                footprint = np.asarray(
                    scene_object.footprint,
                    dtype=np.float64,
                )
                for edge_index, edge_xy in enumerate(footprint):
                    source_horizontal = float(
                        np.linalg.norm(edge_xy - source[:2])
                    )
                    receiver_horizontal = float(
                        np.linalg.norm(edge_xy - receiver[:2])
                    )
                    horizontal_sum = (
                        source_horizontal + receiver_horizontal
                    )
                    if horizontal_sum <= 1e-12:
                        continue
                    edge_z = (
                        receiver_horizontal * source[2]
                        + source_horizontal * receiver[2]
                    ) / horizontal_sum
                    edge_z = float(
                        np.clip(
                            edge_z,
                            scene_object.z_min,
                            scene_object.z_max,
                        )
                    )
                    point = np.asarray(
                        [edge_xy[0], edge_xy[1], edge_z],
                        dtype=np.float64,
                    )
                    first = point - source
                    second = receiver - point
                    first_length = float(np.linalg.norm(first))
                    second_length = float(np.linalg.norm(second))
                    distance = first_length + second_length
                    blocked_by_other = any(
                        segment_intersects_scene_object(
                            endpoint_start,
                            endpoint_stop,
                            other,
                        )
                        for other in scene.objects
                        if other.object_id != scene_object.object_id
                        for endpoint_start, endpoint_stop in (
                            (source, point),
                            (point, receiver),
                        )
                    )
                    if blocked_by_other:
                        continue
                    candidates.append(
                        (
                            distance,
                            edge_index,
                            point,
                            first / first_length,
                            second / second_length,
                        )
                    )
                candidates.sort(key=lambda value: (value[0], value[1]))
                for (
                    distance,
                    edge_index,
                    point,
                    departure,
                    arrival,
                ) in candidates[:maximum_edges]:
                    excess = max(0.0, distance - direct.distance_m)
                    fresnel_v = math.sqrt(
                        max(0.0, 2.0 * excess / wavelength)
                    )
                    available_pressure = math.sqrt(
                        max(
                            0.0,
                            1.0
                            - scene_object.absorption
                            - scene_object.transmission,
                        )
                    )
                    coefficient = (
                        0.5
                        * available_pressure
                        / math.sqrt(1.0 + fresnel_v**2)
                    )
                    diffraction_events.append(
                        PathEvent(
                            event_id=(
                                f"{direct.event_id}:diffraction:"
                                f"{scene_object.object_id}:{edge_index}"
                            ),
                            source_id=direct.source_id,
                            receiver_id=direct.receiver_id,
                            path_type="diffraction",
                            source_position_m=list(
                                direct.source_position_m
                            ),
                            receiver_position_m=list(
                                direct.receiver_position_m
                            ),
                            distance_m=distance,
                            delay_s=distance / direct.sound_speed_m_s,
                            sound_speed_m_s=direct.sound_speed_m_s,
                            departure_direction_unit=departure.tolist(),
                            arrival_direction_unit=arrival.tolist(),
                            surface_ids=[
                                f"{scene_object.object_id}:edge:"
                                f"{edge_index}"
                            ],
                            interaction_types=["diffraction"],
                            interaction_group_ids=[0],
                            interaction_points_m=[point.tolist()],
                            incidence_cosines=[1.0],
                            gain_spectrum=(
                                ComplexPathGainSpectrum.constant(
                                    coefficient / distance,
                                    provenance=(
                                        "bounded_reference_frequency_"
                                        "knife_edge_diffraction"
                                    ),
                                )
                            ),
                            source_directivity_id=(
                                direct.source_directivity_id
                            ),
                            receiver_directivity_id=(
                                direct.receiver_directivity_id
                            ),
                            source_directivity_gain=(
                                direct.source_directivity_gain
                            ),
                            receiver_directivity_gain=(
                                direct.receiver_directivity_gain
                            ),
                            visible=coefficient > 0.0,
                            diffraction_model=(
                                "puresound.bounded_fresnel_edge.v1"
                            ),
                            scattering_model="none",
                        )
                    )
    events.extend(transmission_events)
    events.extend(diffraction_events)
    metadata = dict(event_set.metadata)
    metadata.update(
        {
            "interaction_extension": (
                "puresound.scene_interactions.m3.v1"
            ),
            "transmission_event_count": len(transmission_events),
            "diffraction_event_count": len(diffraction_events),
            "scattering_event_count": len(scattering_events),
            "higher_order_scattering_loss_event_count": (
                higher_order_scattering_loss_events
            ),
            "higher_order_scattering_policy": (
                "per_interaction_specular_retention_with_diffuse_share_"
                "delegated_to_fdn"
            ),
            "scattering_reference_hz": float(
                scattering_reference_hz
            ),
            "scattering_branch_count": branch_count,
            "diffraction_reference_hz": float(
                diffraction_reference_hz
            ),
            "maximum_diffraction_edges_per_object": maximum_edges,
            "interaction_models_are_opt_in": True,
        }
    )
    return replace(
        event_set,
        events=events,
        generator=f"{event_set.generator}+scene_interactions.m3.v1",
        metadata=metadata,
    )


def _fold_shoebox_position(
    unfolded_position_m: np.ndarray,
    dimensions_m: np.ndarray,
) -> np.ndarray:
    period = 2.0 * dimensions_m
    wrapped = np.mod(unfolded_position_m, period)
    return np.where(wrapped <= dimensions_m, wrapped, period - wrapped)


def _shoebox_image_position(
    source_position_m: np.ndarray,
    dimensions_m: np.ndarray,
    image_order_xyz: tuple[int, int, int],
) -> np.ndarray:
    order = np.asarray(image_order_xyz, dtype=np.int64)
    translation = np.floor_divide(order + 1, 2)
    parity = np.where(order % 2 == 0, 1.0, -1.0)
    return 2.0 * translation * dimensions_m + parity * source_position_m


def _ordered_shoebox_image_path(
    *,
    source_position_m: np.ndarray,
    receiver_position_m: np.ndarray,
    dimensions_m: np.ndarray,
    image_order_xyz: tuple[int, int, int],
    edge_corner_policy: str,
    tie_tolerance: float = 1e-10,
) -> dict[str, Any] | None:
    """Fold one unfolded image ray into ordered physical reflection points.

    A simultaneous crossing of two or three unfolded room planes is an
    edge/corner hit. The physical ``exclude`` policy returns ``None`` because a
    sequence of locally reacting face reflections is not defined there. The
    ``sequential_face_product_diagnostic`` policy preserves the old analytic
    image-source approximation by grouping coincident face interactions at one
    point; it is traceable but is not a production corner model.
    """
    if edge_corner_policy not in {
        "exclude",
        "sequential_face_product_diagnostic",
    }:
        raise ValueError("unsupported shoebox edge/corner policy")
    image = _shoebox_image_position(
        source_position_m,
        dimensions_m,
        image_order_xyz,
    )
    unfolded_vector = receiver_position_m - image
    distance = float(np.linalg.norm(unfolded_vector))
    crossings: list[tuple[float, int, int]] = []
    for axis, (image_value, receiver_value, dimension) in enumerate(
        zip(image, receiver_position_m, dimensions_m)
    ):
        lower = min(float(image_value), float(receiver_value))
        upper = max(float(image_value), float(receiver_value))
        first_plane = math.floor(lower / float(dimension)) + 1
        last_plane = math.ceil(upper / float(dimension)) - 1
        for plane_index in range(first_plane, last_plane + 1):
            plane = float(plane_index) * float(dimension)
            parameter = (
                (plane - float(image_value))
                / (float(receiver_value) - float(image_value))
            )
            if 0.0 < parameter < 1.0:
                crossings.append((float(parameter), axis, plane_index))
    expected_order = sum(abs(value) for value in image_order_xyz)
    if len(crossings) != expected_order:
        raise RuntimeError(
            "unfolded shoebox crossing count does not match image order"
        )
    crossings.sort(key=lambda item: item[0])
    crossing_groups: list[list[tuple[float, int, int]]] = []
    for crossing in crossings:
        if (
            crossing_groups
            and abs(crossing[0] - crossing_groups[-1][0][0])
            <= tie_tolerance
        ):
            crossing_groups[-1].append(crossing)
        else:
            crossing_groups.append([crossing])
    has_edge_or_corner_hit = any(
        len(group) > 1 for group in crossing_groups
    )
    if has_edge_or_corner_hit and edge_corner_policy == "exclude":
        return None

    boundaries = []
    points = []
    interaction_group_ids = []
    unique_points = []
    crossing_axes = []
    for group_index, group in enumerate(crossing_groups):
        parameter = float(group[0][0])
        unfolded_point = image + parameter * unfolded_vector
        common_point = _fold_shoebox_position(
            unfolded_point,
            dimensions_m,
        )
        group.sort(key=lambda item: item[1])
        for _parameter, axis, plane_index in group:
            point = common_point.copy()
            upper = bool(plane_index % 2)
            boundary = _BOUNDARY_BY_AXIS_AND_UPPER[(axis, upper)]
            point[axis] = float(dimensions_m[axis]) if upper else 0.0
            common_point[axis] = point[axis]
            boundaries.append(boundary)
            points.append(point)
            crossing_axes.append(axis)
            interaction_group_ids.append(group_index)
        unique_points.append(common_point)
    physical_vertices = [
        source_position_m,
        *unique_points,
        receiver_position_m,
    ]
    segments = [
        physical_vertices[index + 1] - physical_vertices[index]
        for index in range(len(physical_vertices) - 1)
    ]
    segment_lengths = [float(np.linalg.norm(segment)) for segment in segments]
    if any(length <= 0.0 for length in segment_lengths):
        raise RuntimeError("shoebox path contains a zero-length segment")
    folded_distance = float(sum(segment_lengths))
    if not math.isclose(
        folded_distance,
        distance,
        rel_tol=1e-11,
        abs_tol=1e-11,
    ):
        raise RuntimeError("folded and unfolded shoebox distances disagree")
    directions = [
        segment / length for segment, length in zip(segments, segment_lengths)
    ]
    incidence_cosines = [
        abs(float(unfolded_vector[axis])) / distance
        for axis in crossing_axes
    ]
    return {
        "image_position_m": image,
        "distance_m": distance,
        "boundaries": boundaries,
        "interaction_points_m": points,
        "interaction_group_ids": interaction_group_ids,
        "incidence_cosines": incidence_cosines,
        "departure_direction_unit": directions[0],
        "arrival_direction_unit": directions[-1],
        "has_edge_or_corner_hit": has_edge_or_corner_hit,
    }


def generate_shoebox_path_events(
    *,
    dimensions_m: Iterable[float],
    source_position_m: Iterable[float],
    receiver_position_m: Iterable[float],
    sound_speed_m_s: float,
    scene_id: str = "shoebox",
    source_id: str = "source",
    receiver_id: str = "receiver",
    surface_ids: Mapping[str, str] | None = None,
    source_directivity_id: str = "omnidirectional",
    receiver_directivity_id: str = "omnidirectional",
    source_directivity_gain: float = 1.0,
    receiver_directivity_gain: float = 1.0,
    max_order: int = 1,
    edge_corner_policy: str = "exclude",
    boundary_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    reflection_frequencies_hz: Iterable[float] = (
        60.0,
        80.0,
        120.0,
        160.0,
        200.0,
        240.0,
    ),
) -> PathEventSet:
    """Generate exact ordered shoebox image-source events through ``max_order``."""
    dimensions = np.asarray(
        _finite_float_list("shoebox dimension", dimensions_m), dtype=np.float64
    )
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError("shoebox dimensions must contain three positive values")
    source = _validate_shoebox_position(
        "source position", source_position_m, dimensions
    )
    receiver = _validate_shoebox_position(
        "receiver position", receiver_position_m, dimensions
    )
    if np.array_equal(source, receiver):
        raise ValueError("source and receiver positions must differ")
    sound_speed = float(sound_speed_m_s)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    if (
        int(max_order) != max_order
        or int(max_order) < 0
        or int(max_order) > 20
    ):
        raise ValueError("shoebox max_order must be an integer in [0, 20]")
    if edge_corner_policy not in {
        "exclude",
        "sequential_face_product_diagnostic",
    }:
        raise ValueError("unsupported shoebox edge/corner policy")
    frequencies = _finite_float_list(
        "reflection frequency", reflection_frequencies_hz
    )
    if (
        not frequencies
        or frequencies[0] < 0.0
        or any(b <= a for a, b in zip(frequencies, frequencies[1:]))
    ):
        raise ValueError(
            "reflection frequencies must be non-negative and increasing"
        )
    if surface_ids is not None and set(surface_ids) != set(SHOEBOX_BOUNDARIES):
        raise ValueError("surface_ids must define exactly six shoebox boundaries")
    resolved_surface_ids = {
        boundary: (
            str(surface_ids[boundary])
            if surface_ids is not None
            else boundary
        )
        for boundary in SHOEBOX_BOUNDARIES
    }
    admittance_models = dict(boundary_admittance_models or {})
    unknown_boundaries = set(admittance_models).difference(SHOEBOX_BOUNDARIES)
    if unknown_boundaries:
        raise ValueError(
            f"unknown shoebox admittance boundaries: {sorted(unknown_boundaries)}"
        )

    direct_vector = receiver - source
    direct_distance = float(np.linalg.norm(direct_vector))
    direct_direction = direct_vector / direct_distance
    shared = {
        "source_id": source_id,
        "receiver_id": receiver_id,
        "source_position_m": source.astype(float).tolist(),
        "receiver_position_m": receiver.astype(float).tolist(),
        "sound_speed_m_s": sound_speed,
        "source_directivity_id": source_directivity_id,
        "receiver_directivity_id": receiver_directivity_id,
        "source_directivity_gain": float(source_directivity_gain),
        "receiver_directivity_gain": float(receiver_directivity_gain),
        "visible": True,
        "diffraction_model": "none",
        "scattering_model": "specular",
    }
    events = [
        PathEvent(
            event_id=f"{source_id}->{receiver_id}:direct",
            path_type="direct",
            distance_m=direct_distance,
            delay_s=direct_distance / sound_speed,
            departure_direction_unit=direct_direction.astype(float).tolist(),
            arrival_direction_unit=direct_direction.astype(float).tolist(),
            surface_ids=[],
            interaction_types=[],
            interaction_group_ids=[],
            interaction_points_m=[],
            incidence_cosines=[],
            gain_spectrum=_gain_spectrum(
                distance_m=direct_distance,
                incidence_cosines=[],
                frequencies_hz=frequencies,
                admittance_models=[],
            ),
            image_order_xyz=[0, 0, 0],
            **shared,
        )
    ]
    excluded_degenerate_orders: list[list[int]] = []
    included_edge_or_corner_orders: list[list[int]] = []
    reflection_orders = sorted(
        (
            (nx, ny, nz)
            for nx in range(-int(max_order), int(max_order) + 1)
            for ny in range(-int(max_order), int(max_order) + 1)
            for nz in range(-int(max_order), int(max_order) + 1)
            if 0 < abs(nx) + abs(ny) + abs(nz) <= int(max_order)
        ),
        key=lambda order: (
            abs(order[0]) + abs(order[1]) + abs(order[2]),
            order,
        ),
    )
    for image_order in reflection_orders:
        path = _ordered_shoebox_image_path(
            source_position_m=source,
            receiver_position_m=receiver,
            dimensions_m=dimensions,
            image_order_xyz=image_order,
            edge_corner_policy=edge_corner_policy,
        )
        if path is None:
            excluded_degenerate_orders.append(list(image_order))
            continue
        if path["has_edge_or_corner_hit"]:
            included_edge_or_corner_orders.append(list(image_order))
        boundaries = path["boundaries"]
        distance = float(path["distance_m"])
        events.append(
            PathEvent(
                event_id=(
                    f"{source_id}->{receiver_id}:image:"
                    f"{image_order[0]:+d},{image_order[1]:+d},"
                    f"{image_order[2]:+d}"
                ),
                path_type="specular_reflection",
                distance_m=distance,
                delay_s=distance / sound_speed,
                departure_direction_unit=path[
                    "departure_direction_unit"
                ].astype(float).tolist(),
                arrival_direction_unit=path[
                    "arrival_direction_unit"
                ].astype(float).tolist(),
                surface_ids=[
                    resolved_surface_ids[boundary]
                    for boundary in boundaries
                ],
                interaction_types=["reflection"] * len(boundaries),
                interaction_group_ids=path["interaction_group_ids"],
                interaction_points_m=[
                    point.astype(float).tolist()
                    for point in path["interaction_points_m"]
                ],
                incidence_cosines=path["incidence_cosines"],
                gain_spectrum=_gain_spectrum(
                    distance_m=distance,
                    incidence_cosines=path["incidence_cosines"],
                    frequencies_hz=frequencies,
                    admittance_models=[
                        admittance_models.get(boundary)
                        for boundary in boundaries
                    ],
                ),
                image_order_xyz=list(image_order),
                **shared,
            )
        )

    return PathEventSet(
        scene_id=str(scene_id),
        source_id=str(source_id),
        receiver_id=str(receiver_id),
        events=events,
        metadata={
            "geometry": "shoebox_unfolded_image_source_ordered_specular_paths",
            "max_order": int(max_order),
            "dimensions_m": dimensions.astype(float).tolist(),
            "enumerated_image_count": 1 + len(reflection_orders),
            "excluded_edge_or_corner_image_orders": (
                excluded_degenerate_orders
            ),
            "included_diagnostic_edge_or_corner_image_orders": (
                included_edge_or_corner_orders
            ),
            "edge_corner_policy": edge_corner_policy,
            "complex_boundary_spectra_rendered_in_time_domain": False,
        },
    )


def orientation_forward_unit(orientation_ypr_deg: Iterable[float]) -> np.ndarray:
    """Return the forward axis for yaw/pitch/roll Euler metadata."""

    orientation = _finite_float_list("orientation", orientation_ypr_deg)
    if len(orientation) != 3:
        raise ValueError("orientation_ypr_deg must contain yaw, pitch, and roll")
    yaw = math.radians(orientation[0])
    pitch = math.radians(orientation[1])
    return np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )


def directivity_pressure_gain(
    directivity_id: str,
    orientation_ypr_deg: Iterable[float],
    direction_from_transducer_unit: Iterable[float],
) -> float:
    """Evaluate supported real first-order source/receiver pressure patterns."""

    pattern = str(directivity_id)
    if pattern == "omnidirectional":
        return 1.0
    alpha_by_pattern = {
        "speech_cardioid": 0.5,
        "cardioid": 0.5,
        "hypercardioid": 0.25,
        "figure_eight": 0.0,
    }
    if pattern not in alpha_by_pattern:
        raise NotImplementedError(f"unsupported directivity: {pattern}")
    forward = orientation_forward_unit(orientation_ypr_deg)
    direction = np.asarray(
        _unit_vector3("directivity direction", direction_from_transducer_unit),
        dtype=np.float64,
    )
    alpha = alpha_by_pattern[pattern]
    return float(alpha + (1.0 - alpha) * np.dot(forward, direction))


def generate_scene_shoebox_path_events(
    scene: RoomSceneV2,
    *,
    source_index: int = 0,
    receiver_index: int = 0,
    max_order: int = 1,
    edge_corner_policy: str = "exclude",
    resolve_object_visibility: bool = True,
    object_visibility_tolerance_m: float = 1e-10,
    include_scene_interactions: bool = False,
    boundary_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    reflection_frequencies_hz: Iterable[float] = (
        60.0,
        80.0,
        120.0,
        160.0,
        200.0,
        240.0,
    ),
) -> PathEventSet:
    """Adapt a :class:`RoomSceneV2` channel to exact shoebox path events."""
    index = int(source_index)
    if index != source_index or not 0 <= index < len(scene.sources):
        raise ValueError("source_index is outside the scene source list")
    receiver_list_index = int(receiver_index)
    if (
        receiver_list_index != receiver_index
        or not 0 <= receiver_list_index < len(scene.receivers)
    ):
        raise ValueError("receiver_index is outside the scene receiver list")
    source = scene.sources[index]
    receiver = scene.receivers[receiver_list_index]
    if source.directivity_id not in {
        "omnidirectional",
        "speech_cardioid",
        "cardioid",
        "hypercardioid",
        "figure_eight",
    }:
        raise NotImplementedError(
            f"unsupported source directivity: {source.directivity_id}"
        )
    if receiver.directivity_id not in {
        "omnidirectional",
        "cardioid",
        "hypercardioid",
        "figure_eight",
    }:
        raise NotImplementedError(
            f"unsupported receiver directivity: {receiver.directivity_id}"
        )
    boundary_surface_ids = {
        surface.boundary: surface.surface_id for surface in scene.surfaces
    }
    event_set = generate_shoebox_path_events(
        dimensions_m=scene.dimensions_m,
        source_position_m=source.pose.position_m,
        receiver_position_m=receiver.pose.position_m,
        sound_speed_m_s=float(scene.environment.sound_speed_m_s),
        scene_id=scene.scene_id,
        source_id=source.transducer_id,
        receiver_id=receiver.transducer_id,
        surface_ids=boundary_surface_ids,
        source_directivity_id=source.directivity_id,
        receiver_directivity_id=receiver.directivity_id,
        max_order=max_order,
        edge_corner_policy=edge_corner_policy,
        boundary_admittance_models=boundary_admittance_models,
        reflection_frequencies_hz=reflection_frequencies_hz,
    )
    if (
        source.directivity_id != "omnidirectional"
        or receiver.directivity_id != "omnidirectional"
    ):
        event_set = replace(
            event_set,
            events=[
                replace(
                    event,
                    source_directivity_gain=float(
                        directivity_pressure_gain(
                            source.directivity_id,
                            source.pose.orientation_ypr_deg,
                            event.departure_direction_unit,
                        )
                    ),
                    receiver_directivity_gain=float(
                        directivity_pressure_gain(
                            receiver.directivity_id,
                            receiver.pose.orientation_ypr_deg,
                            -np.asarray(
                                event.arrival_direction_unit,
                                dtype=np.float64,
                            ),
                        )
                    ),
                )
                for event in event_set.events
            ],
            metadata={
                **event_set.metadata,
                "source_directivity_model": (
                    "first_order_cardioid_pressure_gain"
                    if source.directivity_id == "speech_cardioid"
                    else "first_order_real_pressure_gain"
                ),
                "source_directivity_id": source.directivity_id,
                "source_orientation_ypr_deg": list(
                    source.pose.orientation_ypr_deg
                ),
                "receiver_directivity_model": (
                    "first_order_real_pressure_gain"
                ),
                "receiver_directivity_id": receiver.directivity_id,
                "receiver_orientation_ypr_deg": list(
                    receiver.pose.orientation_ypr_deg
                ),
            },
        )
    if resolve_object_visibility:
        event_set = apply_scene_object_visibility(
            event_set,
            scene.objects,
            tolerance_m=object_visibility_tolerance_m,
        )
    if include_scene_interactions:
        surface_models = {
            surface.surface_id: boundary_admittance_models[
                surface.boundary
            ]
            for surface in scene.surfaces
            if (
                boundary_admittance_models is not None
                and surface.boundary in boundary_admittance_models
            )
        }
        event_set = augment_scene_path_events_with_interactions(
            event_set,
            scene,
            boundary_admittance_models=surface_models,
        )
    return event_set


def causal_fractional_delay_kernel(
    delay_samples: float,
    *,
    order: int = 3,
) -> tuple[int, np.ndarray]:
    """Return a forward Lagrange kernel and its integer start sample.

    For ``delay_samples = N + mu``, taps are placed at ``N .. N+order`` and
    interpolate the continuous impulse at ``mu``.  Consequently all samples
    before ``floor(delay_samples)`` are exactly zero.  The one-sided filter is
    causal but trades high-frequency accuracy for that strict support rule.
    """
    delay = float(delay_samples)
    interpolation_order = int(order)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("delay_samples must be finite and non-negative")
    if interpolation_order != order or not 1 <= interpolation_order <= 8:
        raise ValueError("fractional-delay order must be an integer in [1, 8]")
    start = int(math.floor(delay))
    fraction = delay - float(start)
    if fraction <= 1e-12:
        fraction = 0.0
    elif 1.0 - fraction <= 1e-12:
        start += 1
        fraction = 0.0
    nodes = np.arange(interpolation_order + 1, dtype=np.float64)
    coefficients = np.ones(interpolation_order + 1, dtype=np.float64)
    for index in range(interpolation_order + 1):
        for other in range(interpolation_order + 1):
            if other != index:
                coefficients[index] *= (
                    (fraction - nodes[other]) / (nodes[index] - nodes[other])
                )
    return start, coefficients


def render_path_events(
    event_set_or_events: PathEventSet | Iterable[PathEvent],
    *,
    sample_rate_hz: float,
    num_samples: int,
    fractional_delay_order: int = 3,
    surface_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    boundary_gain_tolerance: float = 1e-9,
    maximum_boundary_filter_tail_samples: int | None = None,
) -> np.ndarray:
    """Render events into one causal real RIR channel.

    Scalar real gains are rendered directly. A complex specular path is
    renderable only when every interacting surface has a supported passive
    rational admittance model. The stored analog gain samples are checked
    against those models before their causal digital reflection filters are
    applied.
    """
    sample_rate = float(sample_rate_hz)
    length = int(num_samples)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if length != num_samples or length <= 0:
        raise ValueError("num_samples must be a positive integer")
    if isinstance(event_set_or_events, PathEventSet):
        events = event_set_or_events.events
    else:
        events = list(event_set_or_events)
    if not events:
        raise ValueError("at least one path event is required")
    tolerance = float(boundary_gain_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary gain tolerance must be finite and non-negative")
    admittance_models = dict(surface_admittance_models or {})
    if maximum_boundary_filter_tail_samples is None:
        boundary_tail_limit = None
    else:
        boundary_tail_limit = int(maximum_boundary_filter_tail_samples)
        if (
            boundary_tail_limit != maximum_boundary_filter_tail_samples
            or boundary_tail_limit < 1
        ):
            raise ValueError(
                "maximum boundary-filter tail samples must be a positive integer"
            )
    output = np.zeros(length, dtype=np.float64)
    for event in events:
        if not event.visible:
            continue
        start, kernel = causal_fractional_delay_kernel(
            event.delay_s * sample_rate,
            order=fractional_delay_order,
        )
        if start >= length:
            continue
        available = length - start
        direction_gain = (
            float(event.source_directivity_gain)
            * float(event.receiver_directivity_gain)
        )
        interaction_models = [
            admittance_models.get(surface_id)
            for surface_id in event.surface_ids
        ]
        filterable_boundary_path = bool(
            event.path_type in {
                "specular_reflection",
                "scattering",
            }
            and event.interaction_types
            and all(
                interaction in {"reflection", "scattering"}
                for interaction in event.interaction_types
            )
        )
        if filterable_boundary_path and any(
            model is not None for model in interaction_models
        ):
            if any(model is None for model in interaction_models):
                raise ValueError(
                    "every reflection surface in a filtered path requires "
                    "an admittance model"
                )
            expected_gain = []
            for frequency in event.gain_spectrum.frequencies_hz:
                reflection_product = complex(1.0, 0.0)
                for model, incidence_cosine in zip(
                    interaction_models,
                    event.incidence_cosines,
                ):
                    reflection_product *= (
                        locally_reacting_reflection_coefficient(
                            model.normalized_admittance(frequency),
                            incidence_cosine,
                        )
                    )
                expected_gain.append(
                    math.sqrt(event.energy_partition_fraction)
                    * reflection_product
                    / float(event.distance_m)
                )
            maximum_gain_error = float(
                np.max(
                    np.abs(
                        event.gain_spectrum.values
                        - np.asarray(expected_gain, dtype=np.complex128)
                    )
                )
            )
            if maximum_gain_error > tolerance:
                raise ValueError(
                    "stored path gain spectrum does not match the supplied "
                    f"boundary model (maximum error {maximum_gain_error:.3g})"
                )
            local_length = available
            if boundary_tail_limit is not None:
                local_length = min(
                    available,
                    int(kernel.size) + boundary_tail_limit,
                )
            local_signal = np.zeros(local_length, dtype=np.float64)
            local_signal[: min(local_length, kernel.size)] = kernel[:local_length]
            for model, incidence_cosine in zip(
                interaction_models,
                event.incidence_cosines,
            ):
                boundary_filter = digital_locally_reacting_reflection_filter(
                    model,
                    incidence_cosine,
                    sample_rate,
                )
                local_signal = boundary_filter.filter_signal(local_signal)
            output[start : start + local_signal.size] += (
                direction_gain
                * math.sqrt(event.energy_partition_fraction)
                / float(event.distance_m)
                * local_signal
            )
        else:
            gain = (
                event.gain_spectrum.constant_real_value()
                * direction_gain
            )
            stop = min(length, start + kernel.size)
            output[start:stop] += gain * kernel[: stop - start]
    return output


def partition_path_events_by_arrival(
    event_set_or_events: PathEventSet | Iterable[PathEvent],
    *,
    early_window_s: float = 0.050,
) -> dict[str, list[PathEvent]]:
    """Partition paths into direct, early-reflection, and later buckets.

    The early/later threshold is relative to the direct arrival. Boundary
    filter tails remain owned by the path that generated them; this is a
    geometric arrival partition, not a time-domain sample window.
    """
    if isinstance(event_set_or_events, PathEventSet):
        events = list(event_set_or_events.events)
    else:
        events = list(event_set_or_events)
    window = float(early_window_s)
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError("early_window_s must be finite and positive")
    direct = [event for event in events if event.path_type == "direct"]
    if len(direct) != 1:
        raise ValueError("arrival partition requires exactly one direct path")
    direct_delay = float(direct[0].delay_s)
    reflected = [event for event in events if event.path_type != "direct"]
    early = [
        event
        for event in reflected
        if event.delay_s <= direct_delay + window
    ]
    later = [
        event
        for event in reflected
        if event.delay_s > direct_delay + window
    ]
    if len(direct) + len(early) + len(later) != len(events):
        raise RuntimeError("path arrival partition did not preserve all events")
    return {
        "direct": direct,
        "early_reflections": early,
        "later_reflections": later,
    }


__all__ = [
    "ComplexPathGainSpectrum",
    "FRACTIONAL_DELAY_POLICY",
    "OBJECT_VISIBILITY_POLICY",
    "NormalizedAdmittanceModel",
    "PATH_EVENT_SCHEMA_VERSION",
    "PATH_EVENT_SET_SCHEMA_VERSION",
    "PATH_GAIN_SCHEMA_VERSION",
    "PathEvent",
    "PathEventSet",
    "apply_scene_object_visibility",
    "augment_scene_path_events_with_interactions",
    "causal_fractional_delay_kernel",
    "directivity_pressure_gain",
    "generate_scene_shoebox_path_events",
    "generate_shoebox_path_events",
    "locally_reacting_reflection_coefficient",
    "orientation_forward_unit",
    "partition_path_events_by_arrival",
    "render_path_events",
    "segment_intersects_scene_object",
    "segment_scene_object_intersection_interval",
]
