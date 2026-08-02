"""Compatibility shim for the material catalog and room-type sampling.

The implementation moved to :mod:`puresound.audio.rir.scene.materials` during
R1 of ``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged;
prefer the new path in new code.
"""

from puresound.audio.rir.scene.materials import (
    MATERIAL_BANDS_HZ,
    MATERIAL_CATALOG_VERSION,
    MATERIAL_PROVENANCE,
    ROOM_TYPE_RECIPES,
    material_catalog,
    sample_materialized_shoebox,
)

__all__ = [
    "MATERIAL_BANDS_HZ",
    "MATERIAL_CATALOG_VERSION",
    "MATERIAL_PROVENANCE",
    "ROOM_TYPE_RECIPES",
    "material_catalog",
    "sample_materialized_shoebox",
]
