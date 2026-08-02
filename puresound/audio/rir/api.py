"""Stable public entry point for the RIR domain.

Import from here unless you need something narrower.  The layer modules
(``scene``, ``physics``, ``path_events``, ``render``, ``metrics``,
``calibration``, ``bank``) remain importable directly and are the right choice
when you want to avoid pulling in a dependency you do not need — importing
this module loads the renderer stack, including ``torch``.

Anything not re-exported here is either an internal helper or a stage-specific
tool; reach into the layer module for those, and expect less stability.
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
