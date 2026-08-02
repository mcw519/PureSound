"""Versioned, material-first room scene schema for RIR generation.

The v2 schema keeps physical causes (geometry, materials, environment, and
transducers) in metadata.  Compatibility properties expose the small legacy
``HybridRIRScene`` interface so existing low-frequency backends can be migrated
incrementally; notably, ``rt60`` is predicted from the surface materials rather
than stored as a requested room parameter.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


SCENE_SCHEMA_VERSION = "rir_scene.v2"
SHOEBOX_BOUNDARIES = ("west", "east", "south", "north", "floor", "ceiling")


def _float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _check_unit_interval(name: str, values: Iterable[float]) -> None:
    if any(not 0.0 <= float(value) <= 1.0 for value in values):
        raise ValueError(f"{name} coefficients must be in [0, 1]")


@dataclass(frozen=True)
class MaterialSpectrum:
    """Values sampled at octave or third-octave center frequencies."""

    center_frequencies_hz: list[float]
    values: list[float]
    uncertainty_std: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        centers = _float_list(self.center_frequencies_hz)
        values = _float_list(self.values)
        uncertainty = _float_list(self.uncertainty_std)
        if not centers or len(centers) != len(values):
            raise ValueError("material spectrum centers and values must have equal length")
        if any(center <= 0.0 for center in centers):
            raise ValueError("material spectrum center frequencies must be positive")
        if any(b <= a for a, b in zip(centers, centers[1:])):
            raise ValueError("material spectrum center frequencies must be increasing")
        if not all(math.isfinite(value) for value in (*centers, *values, *uncertainty)):
            raise ValueError("material spectrum values must be finite")
        if uncertainty and len(uncertainty) != len(values):
            raise ValueError("material uncertainty must be empty or match spectrum length")
        if any(value < 0.0 for value in uncertainty):
            raise ValueError("material uncertainty cannot be negative")
        object.__setattr__(self, "center_frequencies_hz", centers)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainty_std", uncertainty)

    def at(self, frequency_hz: float) -> float:
        """Interpolate in log-frequency and hold the edge values constant."""
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        centers = np.asarray(self.center_frequencies_hz, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        return float(
            np.interp(
                math.log2(float(frequency_hz)),
                np.log2(centers),
                values,
                left=float(values[0]),
                right=float(values[-1]),
            )
        )

    def to_pra_dict(self) -> dict[str, list[float]]:
        return {
            "coeffs": list(self.values),
            "center_freqs": list(self.center_frequencies_hz),
        }

    @classmethod
    def constant(
        cls,
        value: float,
        center_frequencies_hz: Iterable[float],
        uncertainty_std: float = 0.0,
    ) -> "MaterialSpectrum":
        centers = _float_list(center_frequencies_hz)
        return cls(
            center_frequencies_hz=centers,
            values=[float(value)] * len(centers),
            uncertainty_std=[float(uncertainty_std)] * len(centers),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialSpectrum":
        return cls(
            center_frequencies_hz=data["center_frequencies_hz"],
            values=data["values"],
            uncertainty_std=data.get("uncertainty_std", []),
        )


@dataclass(frozen=True)
class SurfaceMaterial:
    """Frequency-dependent surface properties with provenance.

    Optional complex impedance spectra are the real and imaginary parts of
    surface impedance in Pa*s/m.  They are kept separate because JSON has no
    complex scalar type.  Absorption does not imply impedance: both impedance
    spectra remain ``None`` unless phase-aware data or a declared prior exists.
    """

    material_id: str
    name: str
    family: str
    absorption: MaterialSpectrum
    scattering: MaterialSpectrum
    transmission: MaterialSpectrum
    provenance: str
    impedance_real: Optional[MaterialSpectrum] = None
    impedance_imag: Optional[MaterialSpectrum] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.material_id or not self.family or not self.provenance:
            raise ValueError("material_id, family, and provenance are required")
        _check_unit_interval("absorption", self.absorption.values)
        _check_unit_interval("scattering", self.scattering.values)
        _check_unit_interval("transmission", self.transmission.values)
        if (self.impedance_real is None) != (self.impedance_imag is None):
            raise ValueError("surface impedance requires both real and imaginary spectra")
        if self.impedance_real is not None and self.impedance_imag is not None:
            if any(value < 0.0 for value in self.impedance_real.values):
                raise ValueError(
                    "a passive surface impedance cannot have a negative real part"
                )
            if (
                self.impedance_real.center_frequencies_hz
                != self.impedance_imag.center_frequencies_hz
            ):
                raise ValueError(
                    "surface impedance real and imaginary spectra must share centers"
                )

    def impedance_at(self, frequency_hz: float) -> Optional[complex]:
        """Return complex surface impedance in Pa*s/m, or ``None`` if unknown."""
        if self.impedance_real is None or self.impedance_imag is None:
            return None
        return complex(
            self.impedance_real.at(frequency_hz),
            self.impedance_imag.at(frequency_hz),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SurfaceMaterial":
        return cls(
            material_id=str(data["material_id"]),
            name=str(data["name"]),
            family=str(data["family"]),
            absorption=MaterialSpectrum.from_dict(data["absorption"]),
            scattering=MaterialSpectrum.from_dict(data["scattering"]),
            transmission=MaterialSpectrum.from_dict(data["transmission"]),
            provenance=str(data["provenance"]),
            impedance_real=(
                MaterialSpectrum.from_dict(data["impedance_real"])
                if data.get("impedance_real") is not None
                else None
            ),
            impedance_imag=(
                MaterialSpectrum.from_dict(data["impedance_imag"])
                if data.get("impedance_imag") is not None
                else None
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class SurfacePatch:
    """A window, door, or other area-weighted patch on a shoebox boundary."""

    patch_id: str
    role: str
    area_fraction: float
    material_id: str

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.area_fraction) < 1.0:
            raise ValueError("surface patch area_fraction must be in [0, 1)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SurfacePatch":
        return cls(
            patch_id=str(data["patch_id"]),
            role=str(data["role"]),
            area_fraction=float(data["area_fraction"]),
            material_id=str(data["material_id"]),
        )


@dataclass(frozen=True)
class SceneSurface:
    """Named mesh associated with one shoebox boundary."""

    surface_id: str
    boundary: str
    vertices_m: list[list[float]]
    material_id: str
    patches: list[SurfacePatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.boundary not in SHOEBOX_BOUNDARIES:
            raise ValueError(f"unsupported shoebox boundary: {self.boundary}")
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 3:
            raise ValueError("surface vertices must have shape [N>=3, 3]")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("surface vertices must be finite")
        if sum(float(patch.area_fraction) for patch in self.patches) >= 1.0:
            raise ValueError("surface patch fractions must sum to less than one")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneSurface":
        return cls(
            surface_id=str(data["surface_id"]),
            boundary=str(data["boundary"]),
            vertices_m=[_float_list(vertex) for vertex in data["vertices_m"]],
            material_id=str(data["material_id"]),
            patches=[SurfacePatch.from_dict(item) for item in data.get("patches", [])],
        )


@dataclass(frozen=True)
class EnvironmentConfig:
    temperature_c: float = 20.0
    relative_humidity_percent: float = 50.0
    pressure_pa: float = 101325.0
    sound_speed_m_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not -50.0 <= float(self.temperature_c) <= 60.0:
            raise ValueError("temperature_c is outside the supported range")
        if not 0.0 <= float(self.relative_humidity_percent) <= 100.0:
            raise ValueError("relative humidity must be in [0, 100]")
        if float(self.pressure_pa) <= 0.0:
            raise ValueError("pressure_pa must be positive")
        speed = self.sound_speed_m_s
        if speed is None:
            # Compact engineering approximation for ordinary rooms.
            speed = (
                331.3
                + 0.606 * float(self.temperature_c)
                + 0.0124 * float(self.relative_humidity_percent)
            )
        if not 250.0 <= float(speed) <= 400.0:
            raise ValueError("sound_speed_m_s is outside the supported range")
        object.__setattr__(self, "sound_speed_m_s", float(speed))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentConfig":
        return cls(
            temperature_c=float(data.get("temperature_c", 20.0)),
            relative_humidity_percent=float(
                data.get("relative_humidity_percent", 50.0)
            ),
            pressure_pa=float(data.get("pressure_pa", 101325.0)),
            sound_speed_m_s=(
                float(data["sound_speed_m_s"])
                if data.get("sound_speed_m_s") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Pose:
    position_m: list[float]
    orientation_ypr_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or len(self.orientation_ypr_deg) != 3:
            raise ValueError("pose position and orientation must each have three values")
        values = _float_list([*self.position_m, *self.orientation_ypr_deg])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose values must be finite")
        object.__setattr__(self, "position_m", _float_list(self.position_m))
        object.__setattr__(
            self, "orientation_ypr_deg", _float_list(self.orientation_ypr_deg)
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pose":
        return cls(
            position_m=data["position_m"],
            orientation_ypr_deg=data.get("orientation_ypr_deg", [0.0, 0.0, 0.0]),
        )


@dataclass(frozen=True)
class TransducerConfig:
    """Source or receiver pose, pattern identity, and gain calibration."""

    transducer_id: str
    kind: str
    pose: Pose
    directivity_id: str = "omnidirectional"
    calibration_gain_db: float = 0.0
    power_db_spl_at_1m: Optional[float] = None
    array_id: Optional[str] = None
    channel_index: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"source", "receiver"}:
            raise ValueError("transducer kind must be 'source' or 'receiver'")
        if self.kind == "source" and self.power_db_spl_at_1m is None:
            raise ValueError("source transducers require power_db_spl_at_1m")
        if not math.isfinite(float(self.calibration_gain_db)):
            raise ValueError("transducer calibration gain must be finite")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransducerConfig":
        return cls(
            transducer_id=str(data["transducer_id"]),
            kind=str(data["kind"]),
            pose=Pose.from_dict(data["pose"]),
            directivity_id=str(data.get("directivity_id", "omnidirectional")),
            calibration_gain_db=float(data.get("calibration_gain_db", 0.0)),
            power_db_spl_at_1m=(
                float(data["power_db_spl_at_1m"])
                if data.get("power_db_spl_at_1m") is not None
                else None
            ),
            array_id=data.get("array_id"),
            channel_index=int(data.get("channel_index", 0)),
        )


@dataclass(frozen=True)
class SceneObject:
    """Interior object retained for metadata and the legacy occlusion model."""

    object_id: str
    family: str
    footprint: list[list[float]]
    z_min: float
    z_max: float
    material_id: str
    absorption: float
    scattering: float
    transmission: float = 0.0

    def __post_init__(self) -> None:
        if not self.object_id or not self.family or not self.material_id:
            raise ValueError(
                "scene object ID, family, and material ID are required"
            )
        polygon = np.asarray(self.footprint, dtype=np.float64)
        if (
            polygon.ndim != 2
            or polygon.shape[0] < 3
            or polygon.shape[1] != 2
            or not np.all(np.isfinite(polygon))
        ):
            raise ValueError(
                "scene object footprint must have shape [N>=3, 2]"
            )
        area_twice = float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
        if abs(area_twice) <= 1e-12:
            raise ValueError("scene object footprint area must be nonzero")
        if (
            not math.isfinite(float(self.z_min))
            or not math.isfinite(float(self.z_max))
            or float(self.z_min) < 0.0
            or float(self.z_max) <= float(self.z_min)
        ):
            raise ValueError(
                "scene object heights must satisfy 0 <= z_min < z_max"
            )
        _check_unit_interval("object absorption", [self.absorption])
        _check_unit_interval("object scattering", [self.scattering])
        _check_unit_interval("object transmission", [self.transmission])
        object.__setattr__(
            self,
            "footprint",
            polygon.astype(float).tolist(),
        )
        object.__setattr__(self, "z_min", float(self.z_min))
        object.__setattr__(self, "z_max", float(self.z_max))
        object.__setattr__(self, "absorption", float(self.absorption))
        object.__setattr__(self, "scattering", float(self.scattering))
        object.__setattr__(self, "transmission", float(self.transmission))

    @property
    def material(self) -> str:
        return self.material_id

    def contains_xy(self, point: Iterable[float]) -> bool:
        xy = np.asarray(list(point), dtype=np.float64)[:2]
        polygon = np.asarray(self.footprint, dtype=np.float64)
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = (yi > xy[1]) != (yj > xy[1]) and xy[0] < (
                (xj - xi) * (xy[1] - yi) / ((yj - yi) + 1e-15) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def contains_point(self, point: Iterable[float]) -> bool:
        xyz = np.asarray(list(point), dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("scene object point must contain three finite values")
        return bool(
            float(self.z_min) <= float(xyz[2]) <= float(self.z_max)
            and self.contains_xy(xyz[:2])
        )

    @property
    def center(self) -> np.ndarray:
        points = np.asarray(self.footprint, dtype=np.float64)
        return np.asarray(
            [
                float(points[:, 0].mean()),
                float(points[:, 1].mean()),
                0.5 * (float(self.z_min) + float(self.z_max)),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneObject":
        return cls(
            object_id=str(data["object_id"]),
            family=str(data["family"]),
            footprint=[_float_list(vertex) for vertex in data["footprint"]],
            z_min=float(data["z_min"]),
            z_max=float(data["z_max"]),
            material_id=str(data["material_id"]),
            absorption=float(data["absorption"]),
            scattering=float(data["scattering"]),
            transmission=float(data.get("transmission", 0.0)),
        )


def shoebox_surface_areas(dimensions_m: Iterable[float]) -> dict[str, float]:
    lx, ly, lz = _float_list(dimensions_m)
    return {
        "west": ly * lz,
        "east": ly * lz,
        "south": lx * lz,
        "north": lx * lz,
        "floor": lx * ly,
        "ceiling": lx * ly,
    }


def shoebox_surface_vertices(dimensions_m: Iterable[float]) -> dict[str, list[list[float]]]:
    lx, ly, lz = _float_list(dimensions_m)
    return {
        "west": [[0, 0, 0], [0, ly, 0], [0, ly, lz], [0, 0, lz]],
        "east": [[lx, 0, 0], [lx, 0, lz], [lx, ly, lz], [lx, ly, 0]],
        "south": [[0, 0, 0], [0, 0, lz], [lx, 0, lz], [lx, 0, 0]],
        "north": [[0, ly, 0], [lx, ly, 0], [lx, ly, lz], [0, ly, lz]],
        "floor": [[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0]],
        "ceiling": [[0, 0, lz], [0, ly, lz], [lx, ly, lz], [lx, 0, lz]],
    }


@dataclass(frozen=True)
class RoomSceneV2:
    """Material-first shoebox scene with lossless JSON round-tripping."""

    scene_id: str
    room_type: str
    dimensions_m: list[float]
    surfaces: list[SceneSurface]
    materials: dict[str, SurfaceMaterial]
    environment: EnvironmentConfig
    sources: list[TransducerConfig]
    receivers: list[TransducerConfig]
    objects: list[SceneObject] = field(default_factory=list)
    catalog_version: str = "puresound-materials.v1"
    schema_version: str = SCENE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_SCHEMA_VERSION:
            raise ValueError(f"unsupported scene schema: {self.schema_version}")
        if len(self.dimensions_m) != 3 or any(
            float(value) <= 0.0 for value in self.dimensions_m
        ):
            raise ValueError("room dimensions must contain three positive values")
        boundaries = [surface.boundary for surface in self.surfaces]
        if sorted(boundaries) != sorted(SHOEBOX_BOUNDARIES):
            raise ValueError("v2 shoebox scenes require exactly six named boundaries")
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("shoebox boundaries cannot be duplicated")
        dimensions = np.asarray(self.dimensions_m, dtype=np.float64)
        for obj in self.objects:
            footprint = np.asarray(obj.footprint, dtype=np.float64)
            if (
                np.any(footprint < 0.0)
                or np.any(footprint[:, 0] > dimensions[0])
                or np.any(footprint[:, 1] > dimensions[1])
                or obj.z_max > dimensions[2]
            ):
                raise ValueError(
                    f"scene object {obj.object_id} lies outside the room"
                )
        referenced = {
            surface.material_id for surface in self.surfaces
        } | {
            patch.material_id for surface in self.surfaces for patch in surface.patches
        } | {
            obj.material_id for obj in self.objects
        }
        missing = referenced.difference(self.materials)
        if missing:
            raise ValueError(f"scene references unknown materials: {sorted(missing)}")
        if not self.sources or any(source.kind != "source" for source in self.sources):
            raise ValueError("RoomSceneV2 requires source transducers")
        if not self.receivers or any(
            receiver.kind != "receiver" for receiver in self.receivers
        ):
            raise ValueError("RoomSceneV2 requires at least one receiver")
        for transducer in (*self.sources, *self.receivers):
            if any(
                obj.contains_point(transducer.pose.position_m)
                for obj in self.objects
            ):
                raise ValueError(
                    f"transducer {transducer.transducer_id} lies inside "
                    "a scene object"
                )

    @property
    def room_dim(self) -> list[float]:
        return _float_list(self.dimensions_m)

    @property
    def mic_pos(self) -> list[float]:
        return list(self.receivers[0].pose.position_m)

    @property
    def source_pos(self) -> list[list[float]]:
        return [list(source.pose.position_m) for source in self.sources]

    @property
    def source_labels(self) -> list[str]:
        return [source.transducer_id for source in self.sources]

    @property
    def obstacles(self) -> list[SceneObject]:
        return list(self.objects)

    def source_distances(self) -> list[float]:
        receiver = np.asarray(self.mic_pos, dtype=np.float64)
        sources = np.asarray(self.source_pos, dtype=np.float64)
        return np.linalg.norm(sources - receiver[None, :], axis=1).astype(float).tolist()

    def source_horizontal_distances(self) -> list[float]:
        receiver = np.asarray(self.mic_pos, dtype=np.float64)[:2]
        sources = np.asarray(self.source_pos, dtype=np.float64)[:, :2]
        return (
            np.linalg.norm(sources - receiver[None, :], axis=1).astype(float).tolist()
        )

    def effective_boundary_materials(self) -> dict[str, SurfaceMaterial]:
        """Return area-weighted materials for the six Pyroomacoustics walls."""
        result: dict[str, SurfaceMaterial] = {}
        for surface in self.surfaces:
            base = self.materials[surface.material_id]
            fractions = [(1.0 - sum(p.area_fraction for p in surface.patches), base)]
            fractions.extend(
                (patch.area_fraction, self.materials[patch.material_id])
                for patch in surface.patches
            )
            centers = base.absorption.center_frequencies_hz

            def mix(attr: str) -> MaterialSpectrum:
                values = [
                    sum(
                        float(weight) * getattr(material, attr).at(center)
                        for weight, material in fractions
                    )
                    for center in centers
                ]
                uncertainty = [
                    math.sqrt(
                        sum(
                            (
                                float(weight)
                                * (
                                    getattr(material, attr).uncertainty_std[index]
                                    if getattr(material, attr).uncertainty_std
                                    else 0.0
                                )
                            )
                            ** 2
                            for weight, material in fractions
                        )
                    )
                    for index in range(len(centers))
                ]
                return MaterialSpectrum(centers, values, uncertainty)

            impedance_real: Optional[MaterialSpectrum] = None
            impedance_imag: Optional[MaterialSpectrum] = None
            if all(
                material.impedance_at(centers[0]) is not None
                for _weight, material in fractions
            ):
                # Patches see approximately the same surface pressure, so their
                # normal admittances (not impedances) add in parallel by area.
                effective_impedance: list[complex] = []
                for center in centers:
                    components = [
                        (float(weight), complex(material.impedance_at(center)))
                        for weight, material in fractions
                    ]
                    if any(
                        weight > 0.0 and abs(impedance) <= 1e-15
                        for weight, impedance in components
                    ):
                        effective_impedance.append(complex(0.0, 0.0))
                        continue
                    admittance = sum(
                        weight / impedance
                        for weight, impedance in components
                        if weight > 0.0
                    )
                    if abs(admittance) <= 1e-15:
                        raise ValueError("effective boundary admittance cannot be zero")
                    effective_impedance.append(1.0 / admittance)
                impedance_real = MaterialSpectrum(
                    centers,
                    [max(0.0, value.real) for value in effective_impedance],
                )
                impedance_imag = MaterialSpectrum(
                    centers,
                    [value.imag for value in effective_impedance],
                )

            result[surface.boundary] = SurfaceMaterial(
                material_id=f"effective:{surface.surface_id}",
                name=f"Area-weighted {surface.surface_id}",
                family="composite",
                absorption=mix("absorption"),
                scattering=mix("scattering"),
                transmission=mix("transmission"),
                provenance="Area-weighted mixture of the serialized surface and patches.",
                impedance_real=impedance_real,
                impedance_imag=impedance_imag,
                notes="Used by the shoebox backend; component materials remain serialized.",
            )
        return result

    def predicted_octave_rt60_s(
        self, centers_hz: Optional[Iterable[float]] = None
    ) -> dict[str, float]:
        effective = self.effective_boundary_materials()
        centers = (
            _float_list(centers_hz)
            if centers_hz is not None
            else list(next(iter(effective.values())).absorption.center_frequencies_hz)
        )
        areas = shoebox_surface_areas(self.dimensions_m)
        volume = float(np.prod(np.asarray(self.dimensions_m, dtype=np.float64)))
        coefficient = (
            24.0 * math.log(10.0) / float(self.environment.sound_speed_m_s)
        )
        result: dict[str, float] = {}
        for center in centers:
            equivalent_absorption = sum(
                areas[boundary] * material.absorption.at(center)
                for boundary, material in effective.items()
            )
            result[f"{center:g}"] = float(
                np.clip(
                    coefficient * volume / max(equivalent_absorption, 1e-9),
                    0.02,
                    20.0,
                )
            )
        return result

    @property
    def rt60(self) -> float:
        """Legacy bridge: material-derived mid-band RT60, never a sampled input."""
        predicted = self.predicted_octave_rt60_s((500.0, 1000.0))
        return float(np.median(list(predicted.values())))

    def transducer_channel_gains(
        self, reference_source_spl_db: float = 94.0
    ) -> np.ndarray:
        receiver_gain = float(self.receivers[0].calibration_gain_db)
        gains = []
        for source in self.sources:
            source_level = float(source.power_db_spl_at_1m)
            gain_db = source_level - float(reference_source_spl_db) + receiver_gain
            gains.append(10.0 ** (gain_db / 20.0))
        return np.asarray(gains, dtype=np.float64)

    def to_dict(self, include_compatibility_fields: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "room_type": self.room_type,
            "dimensions_m": _float_list(self.dimensions_m),
            "surfaces": [asdict(surface) for surface in self.surfaces],
            "materials": {
                key: asdict(value) for key, value in sorted(self.materials.items())
            },
            "environment": asdict(self.environment),
            "sources": [asdict(source) for source in self.sources],
            "receivers": [asdict(receiver) for receiver in self.receivers],
            "objects": [asdict(obj) for obj in self.objects],
            "catalog_version": self.catalog_version,
            "predicted_octave_rt60_s": self.predicted_octave_rt60_s(),
        }
        if include_compatibility_fields:
            distances = self.source_distances()
            horizontal = self.source_horizontal_distances()
            data.update(
                {
                    "room_dim": self.room_dim,
                    "rt60": self.rt60,
                    "rt60_origin": "surface_material_sabine_prediction",
                    "mic_pos": self.mic_pos,
                    "source_pos": self.source_pos,
                    "source_labels": self.source_labels,
                    "source_distances": distances,
                    "source_horizontal_distances": horizontal,
                    "channel_map": [
                        {
                            "channel": index,
                            "label": source.transducer_id,
                            "source_pos": source.pose.position_m,
                            "distance_m": distances[index],
                            "horizontal_distance_m": horizontal[index],
                            "power_db_spl_at_1m": source.power_db_spl_at_1m,
                            "directivity_id": source.directivity_id,
                        }
                        for index, source in enumerate(self.sources)
                    ],
                }
            )
        return data

    def to_metadata(self) -> dict[str, Any]:
        return self.to_dict(include_compatibility_fields=True)

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoomSceneV2":
        if data.get("schema_version") != SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"expected schema_version={SCENE_SCHEMA_VERSION!r}, "
                f"got {data.get('schema_version')!r}"
            )
        return cls(
            scene_id=str(data["scene_id"]),
            room_type=str(data["room_type"]),
            dimensions_m=_float_list(data["dimensions_m"]),
            surfaces=[SceneSurface.from_dict(item) for item in data["surfaces"]],
            materials={
                key: SurfaceMaterial.from_dict(value)
                for key, value in data["materials"].items()
            },
            environment=EnvironmentConfig.from_dict(data["environment"]),
            sources=[TransducerConfig.from_dict(item) for item in data["sources"]],
            receivers=[
                TransducerConfig.from_dict(item) for item in data["receivers"]
            ],
            objects=[SceneObject.from_dict(item) for item in data.get("objects", [])],
            catalog_version=str(
                data.get("catalog_version", "puresound-materials.v1")
            ),
            schema_version=str(data["schema_version"]),
        )

    @classmethod
    def from_json(cls, payload_or_path: str | Path) -> "RoomSceneV2":
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


def load_room_scene(data: dict[str, Any]) -> RoomSceneV2 | dict[str, Any]:
    """Load v2 scenes while leaving legacy metadata available to old readers."""
    if data.get("schema_version") == SCENE_SCHEMA_VERSION:
        return RoomSceneV2.from_dict(data)
    return data


__all__ = [
    "EnvironmentConfig",
    "MaterialSpectrum",
    "Pose",
    "RoomSceneV2",
    "SCENE_SCHEMA_VERSION",
    "SHOEBOX_BOUNDARIES",
    "SceneObject",
    "SceneSurface",
    "SurfaceMaterial",
    "SurfacePatch",
    "TransducerConfig",
    "load_room_scene",
    "shoebox_surface_areas",
    "shoebox_surface_vertices",
]
