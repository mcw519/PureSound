"""Scene sampling: rooms, sources, obstacles, and the material-first upgrade.

Layer 2 of the RIR package.  Everything here is deterministic given a seeded
``numpy.random.Generator``; nothing renders audio.  ``hybrid_rir`` re-exports every one of them under their original private
names.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.geometry import (
    ArrayLike,
    distance_point_to_polygon,
    max_room_horizontal_distance_from_point,
    point_in_polygon,
    polygon_area,
    polygon_distance,
    polygons_overlap,
)
from puresound.audio.rir.scene.materials import (
    MATERIAL_CATALOG_VERSION,
    sample_materialized_shoebox,
)
from puresound.audio.rir.scene.schema import (
    EnvironmentConfig,
    Pose,
    RoomSceneV2,
    TransducerConfig,
)


@dataclass
class PolygonObstacle:
    footprint: list[list[float]]
    z_min: float
    z_max: float
    material: str
    absorption: float
    scattering: float

    def contains_xy(self, point: ArrayLike) -> bool:
        xy = np.asarray(point, dtype=np.float64)[:2]
        return point_in_polygon(xy, np.asarray(self.footprint, dtype=np.float64))

    @property
    def center(self) -> np.ndarray:
        pts = np.asarray(self.footprint, dtype=np.float64)
        return np.asarray(
            [float(pts[:, 0].mean()), float(pts[:, 1].mean()), self.z_max * 0.5],
            dtype=np.float64,
        )


@dataclass
class HybridRIRScene:
    room_dim: list[float]
    rt60: float
    mic_pos: list[float]
    source_pos: list[list[float]]
    source_labels: list[str]
    obstacles: list[PolygonObstacle] = field(default_factory=list)

    def source_distances(self) -> list[float]:
        mic = np.asarray(self.mic_pos, dtype=np.float64)
        src = np.asarray(self.source_pos, dtype=np.float64)
        return np.linalg.norm(src - mic[None, :], axis=1).astype(float).tolist()

    def source_horizontal_distances(self) -> list[float]:
        mic = np.asarray(self.mic_pos, dtype=np.float64)[:2]
        src = np.asarray(self.source_pos, dtype=np.float64)[:, :2]
        return np.linalg.norm(src - mic[None, :], axis=1).astype(float).tolist()

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_distances"] = self.source_distances()
        data["source_horizontal_distances"] = self.source_horizontal_distances()
        data["obstacle_floor_coverage_ratio"] = obstacle_floor_coverage(
            self.obstacles, np.asarray(self.room_dim, dtype=np.float64)
        )
        data["channel_map"] = [
            {
                "channel": idx,
                "label": label,
                "source_pos": self.source_pos[idx],
                "distance_m": data["source_distances"][idx],
                "horizontal_distance_m": data["source_horizontal_distances"][idx],
            }
            for idx, label in enumerate(self.source_labels)
        ]
        return data


def min_feasible_rt60(room_dim: np.ndarray, sound_speed: float = 343.0) -> float:
    """Shortest RT60 Sabine allows for this room (absorption capped at 0.99).

    Below this value ``pra.inverse_sabine`` needs an absorption coefficient
    above 1 and raises; the high band would silently fall back to a fixed
    absorption while the metadata kept the impossible request. Clamping at
    sampling time keeps the recorded rt60 equal to the realized one for both
    bands (the low band imposes whatever envelope it is told).
    """
    lx, ly, lz = (float(v) for v in room_dim)
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + lx * lz + ly * lz)
    sabine_coeff = 24.0 * np.log(10.0) / sound_speed
    return float(sabine_coeff * volume / (surface * 0.99))


def sample_hybrid_rir_scene(
    config: HybridRIRConfig,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> HybridRIRScene:
    rng = np.random.default_rng(seed) if rng is None else rng
    room_dim = np.asarray(
        [rng.uniform(low, high) for low, high in config.room_dim_range],
        dtype=np.float64,
    )
    rt60 = max(
        float(rng.uniform(*config.rt60_range)),
        min_feasible_rt60(room_dim, config.sound_speed),
    )
    mic_pos = sample_point(
        room_dim,
        config.mic_margin,
        rng,
        height_range=config.mic_height_range,
    )

    source_pos: list[np.ndarray] = []
    labels: list[str] = []
    for idx in range(config.num_near_sources):
        source_pos.append(
            sample_source_in_horizontal_shell(
                room_dim,
                mic_pos,
                config.near_distance_range[0],
                config.near_distance_range[1],
                config.source_margin,
                config.speech_source_height_range,
                rng,
            )
        )
        labels.append(f"near_{idx}")
    for idx in range(config.num_far_sources):
        max_far = min(
            float(config.far_distance_range[1]),
            max_room_horizontal_distance_from_point(
                room_dim, mic_pos, config.source_margin
            ),
        )
        min_far = min(float(config.far_distance_range[0]), max_far)
        source_pos.append(
            sample_source_in_horizontal_shell(
                room_dim,
                mic_pos,
                min_far,
                max_far,
                config.source_margin,
                config.speech_source_height_range,
                rng,
            )
        )
        labels.append(f"far_{idx}")

    obstacles = sample_polygon_obstacles(
        room_dim=room_dim,
        protected_points=[mic_pos, *source_pos],
        config=config,
        rng=rng,
    )
    return HybridRIRScene(
        room_dim=room_dim.astype(float).tolist(),
        rt60=rt60,
        mic_pos=mic_pos.astype(float).tolist(),
        source_pos=[src.astype(float).tolist() for src in source_pos],
        source_labels=labels,
        obstacles=obstacles,
    )


def upgrade_hybrid_scene_to_v2(
    scene: HybridRIRScene,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    room_type: Optional[str] = None,
    scene_id: str = "scene",
    material_variation_scale: float = 1.0,
) -> RoomSceneV2:
    """Materialize a legacy geometry as a v2 material-first scene.

    ``scene.rt60`` is deliberately not copied.  The v2 compatibility ``rt60``
    property is derived from the sampled surface absorption spectra.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    (
        sampled_room_type,
        surfaces,
        materials,
        objects,
    ) = sample_materialized_shoebox(
        dimensions_m=scene.room_dim,
        rng=rng,
        room_type=room_type,
        legacy_obstacles=scene.obstacles,
        variation_scale=material_variation_scale,
    )
    environment = EnvironmentConfig(
        temperature_c=float(rng.uniform(18.0, 25.0)),
        relative_humidity_percent=float(rng.uniform(30.0, 70.0)),
        pressure_pa=float(rng.uniform(98_000.0, 103_000.0)),
    )
    room_source_level = float(rng.normal(70.0, 2.0))
    sources = [
        TransducerConfig(
            transducer_id=label,
            kind="source",
            pose=Pose(
                position_m=[float(value) for value in position],
                orientation_ypr_deg=[
                    float(rng.uniform(-180.0, 180.0)),
                    float(rng.uniform(-15.0, 15.0)),
                    0.0,
                ],
            ),
            directivity_id="speech_cardioid",
            calibration_gain_db=0.0,
            power_db_spl_at_1m=float(room_source_level + rng.normal(0.0, 1.5)),
            channel_index=index,
        )
        for index, (label, position) in enumerate(
            zip(scene.source_labels, scene.source_pos)
        )
    ]
    receivers = [
        TransducerConfig(
            transducer_id="mic_0",
            kind="receiver",
            pose=Pose(position_m=[float(value) for value in scene.mic_pos]),
            directivity_id="omnidirectional",
            calibration_gain_db=0.0,
            array_id="mono_reference",
            channel_index=0,
        )
    ]
    return RoomSceneV2(
        scene_id=str(scene_id),
        room_type=sampled_room_type,
        dimensions_m=[float(value) for value in scene.room_dim],
        surfaces=surfaces,
        materials=materials,
        environment=environment,
        sources=sources,
        receivers=receivers,
        objects=objects,
        catalog_version=MATERIAL_CATALOG_VERSION,
    )


def sample_material_first_rir_scene(
    config: HybridRIRConfig,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    room_type: Optional[str] = None,
    scene_id: str = "scene",
    material_variation_scale: float = 1.0,
) -> RoomSceneV2:
    """Sample geometry and then physical materials without a requested RT60."""
    rng = np.random.default_rng(seed) if rng is None else rng
    legacy_geometry = sample_hybrid_rir_scene(config=config, rng=rng)
    return upgrade_hybrid_scene_to_v2(
        legacy_geometry,
        rng=rng,
        room_type=room_type,
        scene_id=scene_id,
        material_variation_scale=material_variation_scale,
    )


def sample_polygon_obstacles(
    room_dim: np.ndarray,
    protected_points: list[np.ndarray],
    config: HybridRIRConfig,
    rng: np.random.Generator,
) -> list[PolygonObstacle]:
    materials = _obstacle_material_profiles(room_dim)
    material_names = list(materials.keys())
    material_weights = np.asarray(
        [materials[name]["sample_weight"] for name in material_names], dtype=np.float64
    )
    material_weights = material_weights / material_weights.sum()
    num_obstacles = _sample_obstacle_count(room_dim, config, rng)
    obstacles: list[PolygonObstacle] = []
    floor_area = max(float(room_dim[0] * room_dim[1]), 1e-6)
    used_area = 0.0
    for _ in range(num_obstacles):
        for _attempt in range(128):
            material = str(rng.choice(material_names, p=material_weights))
            profile = materials[material]
            n_vertices = int(rng.integers(4, 8))
            radius = float(rng.uniform(*profile["radius_range"]))
            center = _sample_obstacle_center(
                room_dim=room_dim,
                radius=radius,
                margin=float(config.obstacle_margin),
                placement=str(profile["placement"]),
                rng=rng,
            )
            angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=n_vertices))
            radii = radius * rng.uniform(0.55, 1.0, size=n_vertices)
            footprint = np.column_stack(
                [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)]
            )
            if not _polygon_inside_room(footprint, room_dim, config.obstacle_margin * 0.5):
                continue
            if any(
                distance_point_to_polygon(point[:2], footprint)
                < float(config.obstacle_clearance)
                or point_in_polygon(point[:2], footprint)
                for point in protected_points
            ):
                continue
            if _obstacle_conflicts(
                footprint,
                obstacles,
                clearance=float(config.obstacle_obstacle_clearance),
            ):
                continue
            footprint_area = polygon_area(footprint)
            if (
                used_area + footprint_area
                > floor_area * float(config.max_obstacle_floor_coverage)
            ):
                continue
            z_min, z_max = _sample_obstacle_height(room_dim, profile, config, rng)
            obstacles.append(
                PolygonObstacle(
                    footprint=footprint.astype(float).tolist(),
                    z_min=float(z_min),
                    z_max=float(z_max),
                    material=material,
                    absorption=float(profile["absorption"]),
                    scattering=float(profile["scattering"]),
                )
            )
            used_area += footprint_area
            break
    return obstacles


def _obstacle_material_profiles(room_dim: np.ndarray) -> dict[str, dict[str, Any]]:
    room_height = float(room_dim[2])
    return {
        "table": {
            "absorption": 0.18,
            "scattering": 0.30,
            "height_range": (0.65, 0.90),
            "radius_range": (0.25, 0.75),
            "placement": "free",
            "sample_weight": 1.0,
        },
        "sofa": {
            "absorption": 0.65,
            "scattering": 0.65,
            "height_range": (0.60, 1.10),
            "radius_range": (0.45, 1.05),
            "placement": "wall",
            "sample_weight": 0.7,
        },
        "chair": {
            "absorption": 0.35,
            "scattering": 0.45,
            "height_range": (0.45, 1.10),
            "radius_range": (0.20, 0.45),
            "placement": "free",
            "sample_weight": 1.2,
        },
        "curtain": {
            "absorption": 0.75,
            "scattering": 0.55,
            "height_range": (max(1.8, room_height * 0.75), room_height),
            "radius_range": (0.20, 0.55),
            "placement": "wall",
            "sample_weight": 0.45,
        },
        "cabinet": {
            "absorption": 0.25,
            "scattering": 0.50,
            "height_range": (0.80, min(2.0, room_height)),
            "radius_range": (0.35, 0.85),
            "placement": "wall",
            "sample_weight": 0.65,
        },
    }


def _sample_obstacle_count(
    room_dim: np.ndarray,
    config: HybridRIRConfig,
    rng: np.random.Generator,
) -> int:
    n_min, n_max = config.num_obstacles_range
    floor_area = max(float(room_dim[0] * room_dim[1]), 1e-6)
    density = float(rng.uniform(*config.obstacle_density_per_m2))
    target = int(round(floor_area * density))
    return int(np.clip(target, int(n_min), int(n_max)))


def _sample_obstacle_center(
    room_dim: np.ndarray,
    radius: float,
    margin: float,
    placement: str,
    rng: np.random.Generator,
) -> np.ndarray:
    lower = np.asarray([margin + radius, margin + radius], dtype=np.float64)
    upper = np.maximum(room_dim[:2] - margin - radius, lower + 0.01)
    center = rng.uniform(lower, upper)
    if placement == "wall":
        axis = int(rng.integers(0, 2))
        side = int(rng.integers(0, 2))
        center[axis] = lower[axis] if side == 0 else upper[axis]
    return center


def _sample_obstacle_height(
    room_dim: np.ndarray,
    profile: dict[str, Any],
    config: HybridRIRConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    lo = max(float(profile["height_range"][0]), float(config.obstacle_height_range[0]))
    hi = min(
        float(profile["height_range"][1]),
        float(config.obstacle_height_range[1]),
        float(room_dim[2]),
    )
    if hi <= lo:
        hi = max(lo, min(float(room_dim[2]), lo + 0.01))
    return 0.0, float(rng.uniform(lo, hi))


def _polygon_inside_room(
    polygon: np.ndarray,
    room_dim: np.ndarray,
    margin: float,
) -> bool:
    return bool(
        np.all(polygon[:, 0] >= margin)
        and np.all(polygon[:, 1] >= margin)
        and np.all(polygon[:, 0] <= room_dim[0] - margin)
        and np.all(polygon[:, 1] <= room_dim[1] - margin)
    )


def _obstacle_conflicts(
    footprint: np.ndarray,
    obstacles: list[PolygonObstacle],
    clearance: float,
) -> bool:
    for obstacle in obstacles:
        existing = np.asarray(obstacle.footprint, dtype=np.float64)
        if polygons_overlap(footprint, existing):
            return True
        if polygon_distance(footprint, existing) < float(clearance):
            return True
    return False


def obstacle_floor_coverage(
    obstacles: list[PolygonObstacle],
    room_dim: np.ndarray,
) -> float:
    floor_area = max(float(room_dim[0] * room_dim[1]), 1e-6)
    area = sum(
        polygon_area(np.asarray(obstacle.footprint, dtype=np.float64))
        for obstacle in obstacles
    )
    return float(area / floor_area)


def sample_point(
    room_dim: np.ndarray,
    margin: float,
    rng: np.random.Generator,
    height_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    lower = np.full(3, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim - float(margin), lower + 0.01)
    point = rng.uniform(lower, upper)
    if height_range is not None:
        point[2] = _sample_height(room_dim, margin, height_range, rng)
    return point


def _sample_height(
    room_dim: np.ndarray,
    margin: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    lower = float(max(float(margin), min(height_range)))
    upper = float(min(float(room_dim[2]) - float(margin), max(height_range)))
    if upper <= lower:
        return 0.5 * (lower + upper)
    return float(rng.uniform(lower, upper))


def sample_source_in_horizontal_shell(
    room_dim: np.ndarray,
    mic_pos: np.ndarray,
    min_dist: float,
    max_dist: float,
    margin: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Place a source whose 3D mic distance falls in [min_dist, max_dist].

    The distance constrained here is the same quantity written to the bank's
    ``channel_map["distance_m"]`` and consumed by training labels and the
    real-bank d0 split, so near/far membership is decided on the true
    source-receiver distance. (Historically only the floor projection was
    constrained, which let a "near" source at 0.35 m horizontal sit > 1 m away
    in 3D once the height offset was counted.)

    Per attempt: draw the target 3D distance, then a height whose vertical
    offset does not exceed it, then derive the horizontal radius. When room
    margins or the height ranges make the shell unreachable, fall back to the
    closest achievable placement toward the farthest corner (mirroring the
    documented room-size clamping of the far shell).
    """
    lower = np.full(3, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim - float(margin), lower + 0.01)
    z_lo = max(float(margin), float(min(height_range)))
    z_hi = max(z_lo, min(float(room_dim[2]) - float(margin), float(max(height_range))))
    mic_z = float(mic_pos[2])
    for _ in range(512):
        target = float(rng.uniform(min_dist, max_dist))
        # height compatible with the target 3D distance
        zc_lo = max(z_lo, mic_z - target)
        zc_hi = min(z_hi, mic_z + target)
        if zc_lo > zc_hi:
            continue
        z = float(rng.uniform(zc_lo, zc_hi))
        dz = z - mic_z
        radius = float(np.sqrt(max(target * target - dz * dz, 0.0)))
        direction = rng.normal(size=2)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        point = np.asarray(mic_pos, dtype=np.float64).copy()
        point[:2] = mic_pos[:2] + direction / norm * radius
        point[2] = z
        if np.all(point >= lower) and np.all(point <= upper):
            return point

    # Fallback: closest achievable 3D distance, walking toward the farthest corner.
    z = float(np.clip(mic_z, z_lo, z_hi))
    dz = z - mic_z
    corners = np.array(
        [
            [x, y]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
        ],
        dtype=np.float64,
    )
    spans = np.linalg.norm(corners - mic_pos[None, :2], axis=1)
    corner_xy = corners[int(np.argmax(spans))]
    span = max(float(spans.max()), 1e-9)
    target = float(np.clip(rng.uniform(min_dist, max_dist), abs(dz), None))
    radius = float(np.clip(np.sqrt(max(target * target - dz * dz, 0.0)), 0.0, span))
    point = np.asarray(mic_pos, dtype=np.float64).copy()
    point[:2] = mic_pos[:2] + (corner_xy - mic_pos[:2]) / span * radius
    point[2] = z
    return point


def _sample_source_in_shell(
    room_dim: np.ndarray,
    mic_pos: np.ndarray,
    min_dist: float,
    max_dist: float,
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_source_in_horizontal_shell(
        room_dim,
        mic_pos,
        min_dist,
        max_dist,
        margin,
        (float(margin), float(room_dim[2]) - float(margin)),
        rng,
    )


__all__ = [
    "HybridRIRScene",
    "PolygonObstacle",
    "min_feasible_rt60",
    "obstacle_floor_coverage",
    "sample_hybrid_rir_scene",
    "sample_material_first_rir_scene",
    "sample_point",
    "sample_polygon_obstacles",
    "sample_source_in_horizontal_shell",
    "upgrade_hybrid_scene_to_v2",
]
