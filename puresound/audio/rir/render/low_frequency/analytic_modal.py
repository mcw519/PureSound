"""Analytic rectangular-eigenfunction low-frequency backend.

Reciprocal source/receiver modal coupling with causal onset, used as a fast
probe alongside the pytARD solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.low_frequency.modal_damping import (
    material_modal_decay_rates,
)
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2


@dataclass
class AnalyticModalLowFrequencyBackend:
    """Small deterministic fallback for tests and smoke runs.

    This is not a replacement for pytARD.  It creates low-frequency direct
    paths and damped axial/tangential room modes so the full pipeline can be
    exercised before connecting a wave solver.
    """

    num_modes_per_axis: int = 5
    max_modes: Optional[int] = 256
    material_modal_damping: bool = False
    material_modal_loss_scale: float = 1.0
    physical_mode_coupling: bool = True

    def simulate(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
    ) -> np.ndarray:
        if self.material_modal_damping and not isinstance(scene, RoomSceneV2):
            raise ValueError(
                "material_modal_damping requires a RoomSceneV2 with boundary materials"
            )
        fs = int(config.sample_rate)
        n = int(config.num_samples)
        t = np.arange(n, dtype=np.float64) / float(fs)
        room = np.asarray(scene.room_dim, dtype=np.float64)
        mic = np.asarray(scene.mic_pos, dtype=np.float64)
        srcs = np.asarray(scene.source_pos, dtype=np.float64)
        out = np.zeros((srcs.shape[0], n), dtype=np.float64)
        tau = max(float(scene.rt60) / 6.91, 1e-3)

        modes: list[tuple[float, int, int, int]] = []
        for nx in range(self.num_modes_per_axis + 1):
            for ny in range(self.num_modes_per_axis + 1):
                for nz in range(self.num_modes_per_axis + 1):
                    if nx == ny == nz == 0:
                        continue
                    f = config.sound_speed * 0.5 * math.sqrt(
                        (nx / room[0]) ** 2
                        + (ny / room[1]) ** 2
                        + (nz / room[2]) ** 2
                    )
                    if config.low_fmin_hz <= f <= config.low_fmax_hz:
                        modes.append((float(f), nx, ny, nz))
        modes = sorted(modes)
        if self.max_modes is not None:
            if self.max_modes < 1:
                raise ValueError("analytic modal max_modes must be positive")
            modes = modes[: int(self.max_modes)]
        if self.material_modal_damping:
            mode_frequencies = np.asarray([mode[0] for mode in modes])
            mode_gamma = material_modal_decay_rates(
                scene,
                nx=np.asarray([mode[1] for mode in modes]),
                ny=np.asarray([mode[2] for mode in modes]),
                nz=np.asarray([mode[3] for mode in modes]),
                omega_rad_s=2.0 * math.pi * mode_frequencies,
                sound_speed=float(config.sound_speed),
                loss_scale=float(self.material_modal_loss_scale),
            )
        else:
            mode_gamma = np.full(len(modes), 1.0 / tau, dtype=np.float64)
        first_mode_frequency = modes[0][0] if modes else 1.0
        receiver_mode_values = np.asarray(
            [
                math.cos(nx * math.pi * mic[0] / room[0])
                * math.cos(ny * math.pi * mic[1] / room[1])
                * math.cos(nz * math.pi * mic[2] / room[2])
                for _frequency, nx, ny, nz in modes
            ],
            dtype=np.float64,
        )

        for idx, src in enumerate(srcs):
            distance = float(np.linalg.norm(src - mic))
            direct = int(round(distance / float(config.sound_speed) * fs))
            if direct < n:
                out[idx, direct] += 1.0 / max(distance, 0.1)
            phase_seed = float(np.dot(src + mic, np.array([0.37, 0.61, 0.83])))
            direct_time = direct / float(fs)
            local_time = np.maximum(t - direct_time, 0.0)
            active = t >= direct_time
            for mode_idx, (freq, nx, ny, nz) in enumerate(modes):
                if self.physical_mode_coupling:
                    source_mode_value = (
                        math.cos(nx * math.pi * src[0] / room[0])
                        * math.cos(ny * math.pi * src[1] / room[1])
                        * math.cos(nz * math.pi * src[2] / room[2])
                    )
                    inverse_volume_norm = (
                        (2.0 if nx else 1.0)
                        * (2.0 if ny else 1.0)
                        * (2.0 if nz else 1.0)
                    )
                    modal_coupling = (
                        inverse_volume_norm
                        * source_mode_value
                        * receiver_mode_values[mode_idx]
                    )
                    amp = (
                        0.015
                        * modal_coupling
                        * first_mode_frequency
                        / max(freq, 1e-6)
                    )
                    wave = (
                        np.sin(2.0 * math.pi * freq * local_time)
                        * np.exp(-float(mode_gamma[mode_idx]) * local_time)
                        * active
                    )
                else:
                    phase = phase_seed * (mode_idx + 1)
                    amp = 0.015 / math.sqrt(mode_idx + 1)
                    if self.material_modal_damping:
                        wave = (
                            np.sin(
                                2.0 * math.pi * freq * local_time + phase
                            )
                            * np.exp(
                                -float(mode_gamma[mode_idx]) * local_time
                            )
                            * active
                        )
                    else:
                        wave = (
                            np.sin(2.0 * math.pi * freq * t + phase)
                            * np.exp(-t / tau)
                        )
                out[idx] += amp * wave
        return out.astype(np.float32)


__all__ = ["AnalyticModalLowFrequencyBackend"]
