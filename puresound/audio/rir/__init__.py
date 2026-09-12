"""PureSound RIR domain package.

Room impulse response generation, analysis and bank management.

Where this sits relative to training
------------------------------------

Almost all of this package is **generation-side**: it builds and audits RIR banks
offline, and ``egs/rir_generation`` is its only caller.  A training run touches exactly
one module -- ``bank.loader``, which reads a released bank and hands channels to the
augmentor.  Nothing under ``calibration``, ``physics``, ``path_events``, ``render`` or
``scene`` is imported on the training path.

It stays in the library because it is general acoustics code, not one recipe's private
machinery: any recipe that needs a room can build one with it.  But when you are reading
a training run, ``bank.loader`` is the whole surface, and when you change anything else
here, no training run is affected until a bank is rebuilt and re-released.

Layering
--------

Dependencies may only point downwards::

    api / CLI adapter
            v
    render, calibration, bank
            v
    path_events, scene, metrics
            v
    physics, contracts
            v
    numpy / scipy

``contracts`` is the bottom layer: it must never import a renderer, a bank
module, ``torch``/``torchaudio``, or touch the filesystem.  ``test/
test_rir_import_boundaries.py`` enforces this.
"""

from puresound.audio.rir.contracts import (
    RIR_AXIS_ORDER,
    RIR_COMPUTE_DTYPE,
    RIR_CONTRACT_VERSION,
    RIR_DELIVERY_DTYPE,
    RIR_METADATA_CHANNEL_MAP_KEYS,
    RIR_METADATA_REQUIRED_KEYS,
    RIR_WAV_SUBTYPE,
    BackendBand,
    BackendCapabilities,
    HybridRIRConfig,
    RenderContext,
    RIRArray,
    resolve_sound_speed,
    validate_rir_metadata,
)

__all__ = [
    "RIR_AXIS_ORDER",
    "RIR_COMPUTE_DTYPE",
    "RIR_CONTRACT_VERSION",
    "RIR_DELIVERY_DTYPE",
    "RIR_METADATA_CHANNEL_MAP_KEYS",
    "RIR_METADATA_REQUIRED_KEYS",
    "RIR_WAV_SUBTYPE",
    "BackendBand",
    "BackendCapabilities",
    "HybridRIRConfig",
    "RenderContext",
    "RIRArray",
    "resolve_sound_speed",
    "validate_rir_metadata",
]
