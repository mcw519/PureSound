"""Scene layer: geometry, materials, and the versioned room-scene schema.

Layer 2 of the RIR package (see ``puresound.audio.rir``).  Everything here
describes the *acoustic cause* — room shape, surface materials, transducer
poses, environment — and must stay free of renderers, ``torch`` and the
filesystem so a scene can be built and inspected without a backend installed.
"""

from puresound.audio.rir.scene.materials import (
    MATERIAL_BANDS_HZ,
    MATERIAL_CATALOG_VERSION,
    MATERIAL_PROVENANCE,
    ROOM_TYPE_RECIPES,
    material_catalog,
    sample_materialized_shoebox,
)
from puresound.audio.rir.scene.schema import (
    SCENE_SCHEMA_VERSION,
    SHOEBOX_BOUNDARIES,
    EnvironmentConfig,
    MaterialSpectrum,
    Pose,
    RoomSceneV2,
    SceneObject,
    SceneSurface,
    SurfaceMaterial,
    SurfacePatch,
    TransducerConfig,
    load_room_scene,
    shoebox_surface_areas,
    shoebox_surface_vertices,
)

__all__ = [
    "MATERIAL_BANDS_HZ",
    "MATERIAL_CATALOG_VERSION",
    "MATERIAL_PROVENANCE",
    "ROOM_TYPE_RECIPES",
    "SCENE_SCHEMA_VERSION",
    "SHOEBOX_BOUNDARIES",
    "EnvironmentConfig",
    "MaterialSpectrum",
    "Pose",
    "RoomSceneV2",
    "SceneObject",
    "SceneSurface",
    "SurfaceMaterial",
    "SurfacePatch",
    "TransducerConfig",
    "load_room_scene",
    "material_catalog",
    "sample_materialized_shoebox",
    "shoebox_surface_areas",
    "shoebox_surface_vertices",
]
