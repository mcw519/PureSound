"""Compatibility shim for the room-scene schema.

The implementation moved to :mod:`puresound.audio.rir.scene.schema` during R1
of ``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged so
existing recipes, phase validators and tests keep working; prefer the new path
in new code.
"""

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
