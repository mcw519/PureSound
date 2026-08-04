"""PureSound RIR domain package.

This package is the destination of the modularization described in
``RIR_EXP_LOG.md``.  It is being populated in stages; during the
migration the existing flat ``puresound.audio.*`` modules remain the canonical
implementations and keep working unchanged.

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
test_rir_r0_import_boundaries.py`` enforces this.

Stage status
------------

- **R0 (current)** — contracts frozen, no algorithm code moved.
- R1..R7 — see ``RIR_EXP_LOG.md``.
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
