"""Convenience re-exports of the RIR symbols most callers reach for.

**This is not a stability boundary, and importing from here buys you nothing a
direct import does not.** It used to claim to be the package's stable public
entry point. Measured against how the package is actually used, that claim was
empty in both directions:

* Nothing imports it. Every caller -- the seven top-level scripts in
  ``egs/rir_generation``, its ``tools/`` and ``phases/``, and the tests --
  imports the layer module directly. A promise with no consumer has nothing
  enforcing it, so it would have rotted silently.
* It cannot serve them anyway. Of the 40 symbols the top-level scripts need,
  25 are absent here; for ``phases/`` it is 173 of 191. Those are the bank
  schemas, scene recipes and impedance solvers this module deliberately called
  "internal helpers or stage-specific tools" -- which is to say the line it drew
  and the line callers need are different lines.

What actually holds this package together is one layer below and is mechanical:
``contracts.py`` for the data contract, and the LAYER_RANK table in
``test/test_rir/test_rir_import_boundaries.py``, which fails the build on a
cross-layer import and keeps the schema/metrics modules importable without
torch. Depend on those.

So: import from the layer module you mean. Reach here only when you want
several common names at once and do not mind that importing it loads the whole
renderer stack, torch included. If ``rir/`` ever ships as a standalone package,
design that surface then -- from what the consumers turn out to need, not from
this list.
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
