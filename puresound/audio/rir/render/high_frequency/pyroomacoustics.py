"""Pyroomacoustics image-source / ray-tracing high-frequency backend.

The production default.  Moved out of ``puresound.audio.rir.render.hybrid`` in R2 of
``RIR_EXP_LOG.md``.

Determinism caveat: libroom keeps its own process-global RNG for ray tracing,
which the generator's per-task seeding does not reach, so this backend is not
byte-reproducible for a fixed seed.  See ``BackendCapabilities`` and the
review notes in ``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.arrays import pad_or_trim
from puresound.audio.rir.render.crossover import align_high_band_direct
from puresound.audio.rir.render.high_frequency.obstacles import (
    apply_obstacle_high_frequency_effects,
)
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.path_events import orientation_forward_unit


@dataclass
class PyroomacousticsHighFrequencyBackend:
    # ``absorption`` is only used as a fallback when the requested RT60 cannot be
    # satisfied by Sabine's formula for the given room (e.g. RT60 too short).
    absorption: float = 0.35
    max_order: int = 12
    ray_tracing: bool = True
    air_absorption: bool = True
    n_rays: int = 20000
    receiver_radius: float = 0.08
    rng_seed: Optional[int] = field(default=None, repr=False)

    def set_rng_seed(self, seed: int) -> None:
        """Pin libroom's process-global RNG immediately before one render."""

        value = int(seed)
        if not 0 <= value <= np.iinfo(np.uint32).max:
            raise ValueError("Pyroomacoustics RNG seed must fit uint32")
        self.rng_seed = value

    def _absorption_and_max_order(
        self, scene: HybridRIRScene, pra
    ) -> tuple[float, int]:
        """Derive wall absorption and ISM order from the requested scene RT60.

        The geometric backend models the full broadband room response, so its
        reverberation time must follow ``scene.rt60`` instead of a constant
        absorption. Image sources cover the early part and ray tracing fills the
        diffuse tail, so the ISM order is capped to keep generation tractable.
        """
        try:
            e_absorption, needed_order = pra.inverse_sabine(
                float(scene.rt60), list(scene.room_dim)
            )
        except Exception:
            # Scene rt60s are clamped to the Sabine-feasible minimum at sampling
            # time (_min_feasible_rt60), so this is a safety net for externally
            # constructed scenes. It changes the realized reverberation away
            # from scene.rt60 -- say so instead of diverging silently.
            warnings.warn(
                f"inverse_sabine failed for rt60={scene.rt60:.3f}s in room "
                f"{np.round(scene.room_dim, 2).tolist()}; falling back to "
                f"absorption={self.absorption} (metadata rt60 no longer matches)",
                RuntimeWarning,
                stacklevel=2,
            )
            return float(self.absorption), int(self.max_order)
        e_absorption = float(np.clip(e_absorption, 1e-3, 1.0))
        order = int(min(int(needed_order), int(self.max_order)))
        return e_absorption, max(0, order)

    def _v2_materials(self, scene: RoomSceneV2, pra) -> dict[str, Any]:
        materials: dict[str, Any] = {}
        for boundary, material in scene.effective_boundary_materials().items():
            materials[boundary] = pra.Material(
                material.absorption.to_pra_dict(),
                material.scattering.to_pra_dict(),
            )
        return materials

    def _v2_source_directivity(
        self,
        scene: RoomSceneV2,
        source_index: int,
        pra,
    ) -> Any:
        source = scene.sources[int(source_index)]
        if source.directivity_id == "omnidirectional":
            return None
        alpha_by_pattern = {
            "speech_cardioid": 0.5,
            "cardioid": 0.5,
            "hypercardioid": 0.25,
            "figure_eight": 0.0,
        }
        if source.directivity_id not in alpha_by_pattern:
            raise NotImplementedError(
                f"unsupported Pyroomacoustics source directivity: "
                f"{source.directivity_id}"
            )
        forward = orientation_forward_unit(
            source.pose.orientation_ypr_deg
        )
        return pra.directivities.CardioidFamily(
            orientation=forward,
            p=alpha_by_pattern[source.directivity_id],
        )

    def simulate(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
    ) -> np.ndarray:
        try:
            import pyroomacoustics as pra
        except ImportError as exc:
            raise ImportError(
                "Pyroomacoustics is required for high-frequency RIR generation. "
                "Install it with `pip install pyroomacoustics` or pass a custom "
                "high-frequency backend."
            ) from exc

        if self.rng_seed is not None:
            seed_api = getattr(getattr(pra, "random", None), "seed", None)
            if seed_api is None:
                raise RuntimeError(
                    "This Pyroomacoustics build cannot provide deterministic "
                    "ray tracing because pra.random.seed is unavailable"
                )
            # Ray tracing consumes both libroom's C++ generator (wall
            # scattering) and Pyroomacoustics' package-local NumPy Generator
            # (the stochastic noise realization used to synthesize the energy
            # histogram).  Seeding only one of them is still non-deterministic.
            seed_api(numpy=int(self.rng_seed), libroom=int(self.rng_seed))

        room_kwargs = {
            "fs": int(config.sample_rate),
        }
        if isinstance(scene, RoomSceneV2):
            room_kwargs.update(
                {
                    "max_order": int(self.max_order),
                    "materials": self._v2_materials(scene, pra),
                    "temperature": float(scene.environment.temperature_c),
                    "humidity": float(scene.environment.relative_humidity_percent),
                }
            )
        else:
            e_absorption, max_order = self._absorption_and_max_order(scene, pra)
            room_kwargs["max_order"] = int(max_order)
            if hasattr(pra, "Material"):
                room_kwargs["materials"] = pra.Material(e_absorption)
            else:
                room_kwargs["absorption"] = e_absorption
        try:
            room = pra.ShoeBox(
                scene.room_dim,
                air_absorption=bool(self.air_absorption),
                **room_kwargs,
            )
        except TypeError:
            room = pra.ShoeBox(scene.room_dim, **room_kwargs)
            if self.air_absorption and hasattr(room, "set_air_absorption"):
                room.set_air_absorption()
        if isinstance(scene, RoomSceneV2) and hasattr(room, "set_sound_speed"):
            room.set_sound_speed(float(scene.environment.sound_speed_m_s))
        if self.ray_tracing and hasattr(room, "set_ray_tracing"):
            try:
                room.set_ray_tracing(
                    receiver_radius=float(self.receiver_radius),
                    n_rays=int(self.n_rays),
                    energy_thres=1e-7,
                )
            except TypeError:
                room.set_ray_tracing(
                    receiver_radius=float(self.receiver_radius),
                    n_rays=int(self.n_rays),
                )

        for source_index, src in enumerate(scene.source_pos):
            directivity = (
                self._v2_source_directivity(scene, source_index, pra)
                if isinstance(scene, RoomSceneV2)
                else None
            )
            room.add_source(
                np.asarray(src, dtype=np.float64),
                directivity=directivity,
            )
        room.add_microphone_array(np.asarray(scene.mic_pos, dtype=np.float64).reshape(3, 1))
        room.compute_rir()

        rirs = []
        for source_idx in range(config.num_sources):
            raw = np.asarray(room.rir[0][source_idx], dtype=np.float64)
            rirs.append(raw)
        rir = pad_or_trim(rirs, config.num_samples)
        rir = align_high_band_direct(rir, scene, config)
        return apply_obstacle_high_frequency_effects(rir, scene, config)


__all__ = ["PyroomacousticsHighFrequencyBackend"]
