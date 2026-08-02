"""Compatibility shim for the hybrid RIR generator.

The implementation moved to :mod:`puresound.audio.rir.render.hybrid` during
R7 of ``RIR_MODULARIZATION_PLAN.md`` (the pieces it orchestrates moved in
R1-R6).  This module re-exports it unchanged; prefer the new path in new code.

The underscore names below are deliberate: ``test/test_rir_r0_api_inventory.py``
records them as helpers that recipes and tests reach for, so they stay
importable from this path until those callers move to the canonical modules.
"""

from puresound.audio.rir.render.hybrid import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    HybridRIRConfig,
    HybridRIRScene,
    ImpedanceModalLowFrequencyBackend,
    PathEventFDNHighFrequencyBackend,
    PathEventHighFrequencyBackend,
    PolygonObstacle,
    PyroomacousticsHighFrequencyBackend,
    PytARDWaveBackend,
    RIRBackend,
    apply_obstacle_high_frequency_effects,
    generate_hybrid_rir,
    hybrid_crossover,
    material_modal_damping_metadata,
    obstacle_effects_metadata,
    sample_hybrid_rir_scene,
    sample_material_first_rir_scene,
    sample_polygon_obstacles,
    upgrade_hybrid_scene_to_v2,
    write_hybrid_rir_dataset_item,
)
from puresound.audio.rir.render.hybrid import (  # noqa: F401  (legacy helpers)
    _align_high_band_direct,
    _apply_rt60_decay_envelope,
    _calibrate_pytard_signal,
    _clip_rir_before_physical_arrival,
    _distance_point_to_polygon,
    _hybrid_crossover_with_metadata,
    _max_room_horizontal_distance_from_point,
    _min_feasible_rt60,
    _obstacle_floor_coverage,
    _polygons_overlap,
    _pytard_green_delta_excitation,
    _sample_point,
    _sample_source_in_horizontal_shell,
    _solve_modal_ard,
)

__all__ = [
    "AnalyticModalLowFrequencyBackend",
    "GpuARDPytARDBackend",
    "GpuARDPytARDCuPyBackend",
    "HybridRIRConfig",
    "HybridRIRScene",
    "ImpedanceModalLowFrequencyBackend",
    "PathEventFDNHighFrequencyBackend",
    "PathEventHighFrequencyBackend",
    "PolygonObstacle",
    "PyroomacousticsHighFrequencyBackend",
    "PytARDWaveBackend",
    "RIRBackend",
    "apply_obstacle_high_frequency_effects",
    "generate_hybrid_rir",
    "hybrid_crossover",
    "material_modal_damping_metadata",
    "obstacle_effects_metadata",
    "sample_hybrid_rir_scene",
    "sample_material_first_rir_scene",
    "sample_polygon_obstacles",
    "upgrade_hybrid_scene_to_v2",
    "write_hybrid_rir_dataset_item",
]
