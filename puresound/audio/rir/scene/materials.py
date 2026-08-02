"""Documented material priors and correlated room-type sampling for RIR v2."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from puresound.audio.rir.scene.schema import (
    MaterialSpectrum,
    SceneObject,
    SceneSurface,
    SurfaceMaterial,
    SurfacePatch,
    shoebox_surface_vertices,
)


MATERIAL_CATALOG_VERSION = "puresound-materials.v1"
MATERIAL_BANDS_HZ = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
MATERIAL_PROVENANCE = (
    "Absorption prior adapted from the pyroomacoustics 0.10.1 materials "
    "database; uncertainty and scattering are explicit PureSound engineering "
    "priors, not certified measurements."
)


# Absorption coefficients are intentionally a compact, inspectable subset of
# the upstream database.  Six-band entries are extended at 8 kHz using their
# final measured/prior value rather than invented interpolation.
_CATALOG: dict[str, dict[str, Any]] = {
    "brickwork": {
        "name": "Rendered brickwork",
        "family": "wall_masonry",
        "absorption": [0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04],
        "scattering": [0.04, 0.05, 0.07, 0.10, 0.14, 0.18, 0.20],
        "transmission": [0.001] * 7,
    },
    "plasterboard": {
        "name": "Painted plasterboard on insulated steel frame",
        "family": "wall_lightweight",
        "absorption": [0.15, 0.10, 0.06, 0.04, 0.04, 0.05, 0.05],
        "scattering": [0.05, 0.06, 0.08, 0.11, 0.15, 0.20, 0.22],
        "transmission": [0.004, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001],
    },
    "wooden_lining": {
        "name": "12 mm wooden lining on frame",
        "family": "wall_wood",
        "absorption": [0.27, 0.23, 0.22, 0.15, 0.10, 0.07, 0.06],
        "scattering": [0.08, 0.10, 0.14, 0.20, 0.28, 0.34, 0.36],
        "transmission": [0.005, 0.004, 0.003, 0.002, 0.001, 0.001, 0.001],
    },
    "glass_window": {
        "name": "Ordinary glass window",
        "family": "window",
        "absorption": [0.10, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03],
        "scattering": [0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        "transmission": [0.15, 0.10, 0.06, 0.03, 0.02, 0.015, 0.01],
    },
    "wood_16mm": {
        "name": "16 mm wood panel",
        "family": "wood",
        "absorption": [0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07],
        "scattering": [0.08, 0.10, 0.15, 0.22, 0.30, 0.36, 0.38],
        "transmission": [0.01, 0.008, 0.005, 0.003, 0.002, 0.001, 0.001],
    },
    "wooden_door": {
        "name": "Wooden door",
        "family": "door",
        "absorption": [0.14, 0.10, 0.06, 0.08, 0.10, 0.10, 0.10],
        "scattering": [0.05, 0.07, 0.10, 0.14, 0.18, 0.22, 0.24],
        "transmission": [0.04, 0.025, 0.015, 0.008, 0.004, 0.003, 0.002],
    },
    "linoleum_on_concrete": {
        "name": "Linoleum on concrete",
        "family": "floor_hard",
        "absorption": [0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02],
        "scattering": [0.03, 0.04, 0.05, 0.07, 0.09, 0.11, 0.12],
        "transmission": [0.001] * 7,
    },
    "carpet_cotton": {
        "name": "Cotton carpet",
        "family": "floor_soft",
        "absorption": [0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48],
        "scattering": [0.10, 0.14, 0.20, 0.28, 0.36, 0.42, 0.45],
        "transmission": [0.001] * 7,
    },
    "carpet_tufted": {
        "name": "9.5 mm tufted carpet",
        "family": "floor_soft",
        "absorption": [0.10, 0.40, 0.62, 0.70, 0.63, 0.88, 0.88],
        "scattering": [0.11, 0.15, 0.22, 0.30, 0.39, 0.46, 0.49],
        "transmission": [0.001] * 7,
    },
    "curtain_cotton": {
        "name": "Draped 0.5 kg/m² cotton curtain",
        "family": "curtain",
        "absorption": [0.30, 0.45, 0.65, 0.56, 0.59, 0.71, 0.71],
        "scattering": [0.12, 0.18, 0.26, 0.36, 0.46, 0.55, 0.58],
        "transmission": [0.08, 0.06, 0.04, 0.025, 0.02, 0.015, 0.015],
    },
    "ceiling_plasterboard": {
        "name": "Plasterboard ceiling",
        "family": "ceiling_hard",
        "absorption": [0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.02],
        "scattering": [0.04, 0.05, 0.07, 0.10, 0.14, 0.18, 0.20],
        "transmission": [0.002] * 7,
    },
    "ceiling_fissured_tile": {
        "name": "Fissured acoustic ceiling tile",
        "family": "ceiling_absorber",
        "absorption": [0.49, 0.53, 0.53, 0.75, 0.92, 0.99, 0.99],
        "scattering": [0.08, 0.10, 0.14, 0.20, 0.28, 0.35, 0.38],
        "transmission": [0.002] * 7,
    },
    "ceiling_fibre_absorber": {
        "name": "Fibre absorber ceiling",
        "family": "ceiling_absorber",
        "absorption": [0.48, 0.97, 1.00, 0.97, 1.00, 1.00, 1.00],
        "scattering": [0.10, 0.13, 0.18, 0.25, 0.34, 0.42, 0.45],
        "transmission": [0.002] * 7,
    },
    "upholstered_furniture": {
        "name": "Medium upholstered furniture",
        "family": "furniture_soft",
        "absorption": [0.49, 0.66, 0.80, 0.88, 0.82, 0.70, 0.70],
        "scattering": [0.18, 0.26, 0.38, 0.52, 0.64, 0.72, 0.75],
        "transmission": [0.002] * 7,
    },
}


ROOM_TYPE_RECIPES: dict[str, dict[str, list[tuple[str, float]]]] = {
    "office": {
        "wall": [("plasterboard", 0.75), ("brickwork", 0.25)],
        "floor": [("carpet_tufted", 0.70), ("linoleum_on_concrete", 0.30)],
        "ceiling": [
            ("ceiling_fissured_tile", 0.75),
            ("ceiling_plasterboard", 0.25),
        ],
    },
    "meeting_room": {
        "wall": [("plasterboard", 0.65), ("wooden_lining", 0.35)],
        "floor": [("carpet_cotton", 0.75), ("linoleum_on_concrete", 0.25)],
        "ceiling": [
            ("ceiling_fissured_tile", 0.65),
            ("ceiling_fibre_absorber", 0.20),
            ("ceiling_plasterboard", 0.15),
        ],
    },
    "classroom": {
        "wall": [("brickwork", 0.55), ("plasterboard", 0.45)],
        "floor": [("linoleum_on_concrete", 0.80), ("carpet_cotton", 0.20)],
        "ceiling": [
            ("ceiling_fissured_tile", 0.55),
            ("ceiling_plasterboard", 0.45),
        ],
    },
    "living_room": {
        "wall": [("plasterboard", 0.55), ("brickwork", 0.20), ("wooden_lining", 0.25)],
        "floor": [("carpet_cotton", 0.65), ("wood_16mm", 0.35)],
        "ceiling": [("ceiling_plasterboard", 0.85), ("ceiling_fissured_tile", 0.15)],
    },
}


def _weighted_choice(
    choices: list[tuple[str, float]], rng: np.random.Generator
) -> str:
    names = [name for name, _ in choices]
    weights = np.asarray([weight for _, weight in choices], dtype=np.float64)
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def _vary_coefficients(
    values: Iterable[float],
    rng: np.random.Generator,
    room_latent: float,
    variation_scale: float,
) -> list[float]:
    """Perturb bounded coefficients coherently in logit space."""
    values = np.asarray(list(values), dtype=np.float64)
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    logits = np.log(clipped / (1.0 - clipped))
    frequency_axis = np.linspace(-1.0, 1.0, values.size)
    local_level = float(rng.normal(0.0, 0.18))
    local_slope = float(rng.normal(0.0, 0.12))
    shift = float(variation_scale) * (
        0.22 * float(room_latent) + local_level + local_slope * frequency_axis
    )
    varied = 1.0 / (1.0 + np.exp(-(logits + shift)))
    return np.clip(varied, 0.001, 0.999).astype(float).tolist()


def material_catalog() -> dict[str, SurfaceMaterial]:
    """Return the unperturbed, documented v1 material priors."""
    result: dict[str, SurfaceMaterial] = {}
    for material_id, item in _CATALOG.items():
        absorption = list(item["absorption"])
        uncertainty = [
            float(max(0.015, min(0.15, 0.20 * value))) for value in absorption
        ]
        result[material_id] = SurfaceMaterial(
            material_id=material_id,
            name=str(item["name"]),
            family=str(item["family"]),
            absorption=MaterialSpectrum(
                MATERIAL_BANDS_HZ, absorption, uncertainty
            ),
            scattering=MaterialSpectrum(
                MATERIAL_BANDS_HZ,
                item["scattering"],
                [0.08] * len(MATERIAL_BANDS_HZ),
            ),
            transmission=MaterialSpectrum(
                MATERIAL_BANDS_HZ,
                item["transmission"],
                [0.01] * len(MATERIAL_BANDS_HZ),
            ),
            provenance=MATERIAL_PROVENANCE,
        )
    return result


def _realize_material(
    material_id: str,
    rng: np.random.Generator,
    room_latent: float,
    variation_scale: float,
) -> SurfaceMaterial:
    base = material_catalog()[material_id]
    realized_id = f"{material_id}:sampled"
    return SurfaceMaterial(
        material_id=realized_id,
        name=base.name,
        family=base.family,
        absorption=MaterialSpectrum(
            MATERIAL_BANDS_HZ,
            _vary_coefficients(
                base.absorption.values, rng, room_latent, variation_scale
            ),
            base.absorption.uncertainty_std,
        ),
        scattering=MaterialSpectrum(
            MATERIAL_BANDS_HZ,
            _vary_coefficients(
                base.scattering.values, rng, 0.5 * room_latent, variation_scale
            ),
            base.scattering.uncertainty_std,
        ),
        transmission=base.transmission,
        provenance=base.provenance,
        notes=(
            f"Sampled from catalog entry {material_id!r}; coefficients share a "
            "room-level latent and a smaller material-specific perturbation."
        ),
    )


def sample_materialized_shoebox(
    dimensions_m: Iterable[float],
    rng: np.random.Generator,
    room_type: Optional[str] = None,
    legacy_obstacles: Optional[Iterable[Any]] = None,
    variation_scale: float = 1.0,
) -> tuple[str, list[SceneSurface], dict[str, SurfaceMaterial], list[SceneObject]]:
    """Sample correlated boundaries, patches, and furniture for one room."""
    if room_type in {None, "mixed"}:
        room_type = str(rng.choice(sorted(ROOM_TYPE_RECIPES)))
    if room_type not in ROOM_TYPE_RECIPES:
        raise ValueError(
            f"unknown room type {room_type!r}; choose from "
            f"{sorted(ROOM_TYPE_RECIPES)}"
        )
    if variation_scale < 0.0:
        raise ValueError("variation_scale cannot be negative")

    recipe = ROOM_TYPE_RECIPES[room_type]
    selected_wall = _weighted_choice(recipe["wall"], rng)
    selected_floor = _weighted_choice(recipe["floor"], rng)
    selected_ceiling = _weighted_choice(recipe["ceiling"], rng)
    # One room latent produces correlated "more/less absorptive" finishes.
    room_latent = float(rng.normal())
    base_names = {
        "west": selected_wall,
        "east": selected_wall,
        "south": selected_wall,
        "north": selected_wall,
        "floor": selected_floor,
        "ceiling": selected_ceiling,
    }

    required = set(base_names.values()) | {"glass_window", "wooden_door"}
    obstacle_material_names: list[str] = []
    for obstacle in legacy_obstacles or []:
        family = str(getattr(obstacle, "material", "chair"))
        obstacle_material_names.append(
            "upholstered_furniture"
            if family in {"sofa", "chair", "curtain"}
            else "wood_16mm"
        )
    required.update(obstacle_material_names)

    materials: dict[str, SurfaceMaterial] = {}
    for catalog_id in sorted(required):
        material = _realize_material(
            catalog_id, rng, room_latent, float(variation_scale)
        )
        materials[material.material_id] = material

    vertices = shoebox_surface_vertices(dimensions_m)
    window_boundary = str(rng.choice(["east", "north", "south", "west"]))
    door_boundary = str(
        rng.choice([name for name in ("east", "north", "south", "west") if name != window_boundary])
    )
    window_fraction = float(rng.uniform(0.06, 0.22))
    door_fraction = float(rng.uniform(0.025, 0.075))
    surfaces: list[SceneSurface] = []
    for boundary in ("west", "east", "south", "north", "floor", "ceiling"):
        patches: list[SurfacePatch] = []
        if boundary == window_boundary:
            patches.append(
                SurfacePatch(
                    patch_id=f"{boundary}:window",
                    role="window",
                    area_fraction=window_fraction,
                    material_id="glass_window:sampled",
                )
            )
        if boundary == door_boundary:
            patches.append(
                SurfacePatch(
                    patch_id=f"{boundary}:door",
                    role="door",
                    area_fraction=door_fraction,
                    material_id="wooden_door:sampled",
                )
            )
        surfaces.append(
            SceneSurface(
                surface_id=f"{boundary}_boundary",
                boundary=boundary,
                vertices_m=vertices[boundary],
                material_id=f"{base_names[boundary]}:sampled",
                patches=patches,
            )
        )

    objects: list[SceneObject] = []
    for index, obstacle in enumerate(legacy_obstacles or []):
        catalog_id = obstacle_material_names[index]
        material_id = f"{catalog_id}:sampled"
        material = materials[material_id]
        objects.append(
            SceneObject(
                object_id=f"object_{index:03d}",
                family=str(getattr(obstacle, "material", "furniture")),
                footprint=[
                    [float(value) for value in vertex]
                    for vertex in getattr(obstacle, "footprint")
                ],
                z_min=float(getattr(obstacle, "z_min")),
                z_max=float(getattr(obstacle, "z_max")),
                material_id=material_id,
                absorption=float(material.absorption.at(1000.0)),
                scattering=float(material.scattering.at(1000.0)),
                transmission=float(material.transmission.at(1000.0)),
            )
        )
    return room_type, surfaces, materials, objects


__all__ = [
    "MATERIAL_BANDS_HZ",
    "MATERIAL_CATALOG_VERSION",
    "MATERIAL_PROVENANCE",
    "ROOM_TYPE_RECIPES",
    "material_catalog",
    "sample_materialized_shoebox",
]
