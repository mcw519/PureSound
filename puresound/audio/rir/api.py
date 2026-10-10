"""Convenience re-exports of the RIR symbols most callers reach for.

**This is not a stability boundary, and importing from here gives nothing a
direct import does not.** Callers import the layer module they mean, and much
of what they need (bank schemas, scene recipes, impedance solvers) is not
listed here.

What holds this package together is one layer below and is mechanical:
``contracts.py`` for the data contract, and the LAYER_RANK table in
``test/rir/test_rir_import_boundaries.py``, which fails the build on a
cross-layer import and keeps the schema/metrics modules importable without
torch. Depend on those.

Use this module only to get several common names at once; importing it loads
the whole renderer stack, torch included. If ``rir/`` ever ships as a
standalone package, design its public surface from what consumers need, not
from this list.
"""

from __future__ import annotations

from puresound.audio.rir.contracts import (
    RIR_AXIS_ORDER,
    RIR_COMPUTE_DTYPE,
    RIR_CONTRACT_VERSION,
    RIR_DELIVERY_DTYPE,
    RIR_WAV_SUBTYPE,
    BackendBand,
    BackendCapabilities,
    HybridRIRConfig,
    RenderContext,
    RIRArray,
    resolve_sound_speed,
    validate_rir_metadata,
)
from puresound.audio.rir.bank.loader import (
    PreGeneratedReleaseBank,
    PreGeneratedRoomBank,
)
from puresound.audio.rir.bank.storage import write_hybrid_rir_dataset_item
from puresound.audio.rir.metrics import analyze_rir, valid_octave_centers
from puresound.audio.rir.path_events import (
    PathEvent,
    PathEventSet,
    generate_scene_shoebox_path_events,
    render_path_events,
)
from puresound.audio.rir.render.backend import RIRBackend
from puresound.audio.rir.render.crossover import hybrid_crossover
from puresound.audio.rir.render.high_frequency import (
    PathEventFDNHighFrequencyBackend,
    PathEventHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
)
from puresound.audio.rir.render.hybrid import generate_hybrid_rir
from puresound.audio.rir.render.low_frequency import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    ImpedanceModalLowFrequencyBackend,
)
from puresound.audio.rir.scene.sampling import (
    HybridRIRScene,
    PolygonObstacle,
    sample_hybrid_rir_scene,
    sample_material_first_rir_scene,
    upgrade_hybrid_scene_to_v2,
)
from puresound.audio.rir.scene.schema import RoomSceneV2

__all__ = [
    # contracts
    "RIR_AXIS_ORDER",
    "RIR_COMPUTE_DTYPE",
    "RIR_CONTRACT_VERSION",
    "RIR_DELIVERY_DTYPE",
    "RIR_WAV_SUBTYPE",
    "BackendBand",
    "BackendCapabilities",
    "HybridRIRConfig",
    "RIRArray",
    "RenderContext",
    "resolve_sound_speed",
    "validate_rir_metadata",
    # scene
    "HybridRIRScene",
    "PolygonObstacle",
    "RoomSceneV2",
    "sample_hybrid_rir_scene",
    "sample_material_first_rir_scene",
    "upgrade_hybrid_scene_to_v2",
    # path events
    "PathEvent",
    "PathEventSet",
    "generate_scene_shoebox_path_events",
    "render_path_events",
    # render
    "AnalyticModalLowFrequencyBackend",
    "GpuARDPytARDBackend",
    "GpuARDPytARDCuPyBackend",
    "ImpedanceModalLowFrequencyBackend",
    "PathEventFDNHighFrequencyBackend",
    "PathEventHighFrequencyBackend",
    "PyroomacousticsHighFrequencyBackend",
    "RIRBackend",
    "generate_hybrid_rir",
    "hybrid_crossover",
    # metrics
    "analyze_rir",
    "valid_octave_centers",
    # bank
    "PreGeneratedReleaseBank",
    "PreGeneratedRoomBank",
    "write_hybrid_rir_dataset_item",
]
