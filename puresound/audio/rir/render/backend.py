"""The renderer backend protocol.

Lives at layer 3 rather than in ``contracts`` so its signature can name the
real scene types instead of weakening them to ``Any``: ``contracts`` sits below
``scene`` and may not import it.

``puresound.audio.rir.contracts.BackendCapabilities`` is the companion
declaration of what a concrete backend promises — notably whether it is
byte-reproducible for a fixed seed.
"""

from __future__ import annotations

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2

from typing import Protocol


class RIRBackend(Protocol):
    """Backend protocol returning RIRs with shape ``[num_sources, samples]``."""

    def simulate(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
    ) -> np.ndarray:
        ...


__all__ = ["RIRBackend"]
