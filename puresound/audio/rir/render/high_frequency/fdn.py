"""M4 backend: coherent PathEvent early field coupled to a multiband FDN tail.

Subclasses the M3 PathEvent backend and replaces the sparse late field with a
deterministic feedback-delay-network tail, crossfaded at the estimated mixing
time.  Moved out of ``puresound.audio.rir.render.hybrid`` in R2 of
``RIR_MODULARIZATION_PLAN.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.high_frequency.path_event import (
    PathEventHighFrequencyBackend,
)
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.physics.propagation import air_adjusted_rt60_s
from puresound.audio.rir.render.coupling import (
    PATH_EVENT_FDN_COUPLING_POLICY,
    couple_path_event_rir_with_fdn,
)
from puresound.audio.rir.metrics import valid_octave_centers


@dataclass
class PathEventFDNHighFrequencyBackend(PathEventHighFrequencyBackend):
    """Opt-in M4 PathEvent early / deterministic multiband-FDN late backend."""

    _LATE_PATH_AIR_ABSORPTION_POLICY = (
        "FDN_RT60_adds_sound_speed_times_atmospheric_loss"
    )

    mixing_time_s: float = 0.024
    transition_duration_s: float = 0.016
    delay_line_count: int = 16
    fdn_seed: int = 20260731
    fdn_filter_order: int = 4
    minimum_fdn_center_hz: float = 500.0
    last_late_field_metadata: Optional[dict[str, Any]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def _target_rt60_s_by_hz(
        self,
        scene: RoomSceneV2,
        config: HybridRIRConfig,
    ) -> dict[float, float]:
        predicted = {
            float(center): float(rt60_s)
            for center, rt60_s in scene.predicted_octave_rt60_s().items()
        }
        if self.air_absorption:
            predicted = {
                center: air_adjusted_rt60_s(
                    rt60_s,
                    center,
                    float(scene.environment.sound_speed_m_s),
                    temperature_c=float(scene.environment.temperature_c),
                    relative_humidity_percent=float(
                        scene.environment.relative_humidity_percent
                    ),
                    pressure_pa=float(scene.environment.pressure_pa),
                )
                for center, rt60_s in predicted.items()
            }
        valid = set(valid_octave_centers(config.sample_rate, predicted))
        lower = max(
            float(self.minimum_fdn_center_hz),
            0.5 * float(config.crossover_hz),
        )
        targets = {
            center: predicted[center]
            for center in sorted(valid)
            if center >= lower
        }
        if not targets:
            raise ValueError(
                "no material octave target remains below Nyquist and above "
                "the M4 FDN lower frequency"
            )
        return targets

    def _channel_seed(self, scene: RoomSceneV2, source_index: int) -> int:
        payload = (
            f"{int(self.fdn_seed)}\0{scene.scene_id}\0{int(source_index)}"
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF

    def simulate(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
    ) -> np.ndarray:
        if not isinstance(scene, RoomSceneV2):
            raise ValueError(
                "PathEventFDNHighFrequencyBackend requires a v2 material-first "
                "scene"
            )
        if not np.isfinite(self.mixing_time_s) or self.mixing_time_s <= 0.0:
            raise ValueError("FDN mixing_time_s must be finite and positive")
        if (
            not np.isfinite(self.transition_duration_s)
            or self.transition_duration_s <= 0.0
        ):
            raise ValueError(
                "FDN transition_duration_s must be finite and positive"
            )
        coherent = np.asarray(super().simulate(scene, config), dtype=np.float64)
        targets = self._target_rt60_s_by_hz(scene, config)
        distances = scene.source_distances()
        sound_speed = float(scene.environment.sound_speed_m_s)
        outputs: list[np.ndarray] = []
        channel_metadata: list[dict[str, Any]] = []
        for source_index, channel in enumerate(coherent):
            direct_sample = int(
                round(
                    distances[source_index]
                    / max(sound_speed, 1e-9)
                    * float(config.sample_rate)
                )
            )
            result = couple_path_event_rir_with_fdn(
                channel,
                sample_rate=int(config.sample_rate),
                direct_sample=direct_sample,
                target_rt60_s_by_hz=targets,
                mixing_time_s=float(self.mixing_time_s),
                transition_duration_s=float(self.transition_duration_s),
                delay_line_count=int(self.delay_line_count),
                seed=self._channel_seed(scene, source_index),
                filter_order=int(self.fdn_filter_order),
            )
            outputs.append(result.rir)
            channel_metadata.append(
                {
                    "channel": int(source_index),
                    "source_id": scene.sources[source_index].transducer_id,
                    **dict(result.metadata),
                }
            )
        self.last_late_field_metadata = {
            "policy": PATH_EVENT_FDN_COUPLING_POLICY,
            "renderer": "path_event_early_multiband_fdn_late",
            # Both fields describe the generate_hybrid_rir layer, whose default
            # the M4/M5 exit gates pin to pyroomacoustics. The M6 wrapper
            # (generate_m6_bank.py) selects this backend by default since
            # 2026-08-04, and does so by passing --high-backend explicitly.
            "opt_in": True,
            "production_default_changed": False,
            "target_rt60_origin": "scene_material_predicted_octave_rt60_s",
            "target_rt60_s_by_hz": {
                f"{center:g}": value for center, value in targets.items()
            },
            "mixing_time_s_after_direct": float(self.mixing_time_s),
            "transition_duration_s": float(self.transition_duration_s),
            "delay_line_count": int(self.delay_line_count),
            "base_seed": int(self.fdn_seed),
            "channels": channel_metadata,
        }
        return np.asarray(outputs, dtype=np.float32)


__all__ = ["PathEventFDNHighFrequencyBackend"]
