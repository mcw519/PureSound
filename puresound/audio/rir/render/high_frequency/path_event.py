"""Coherent M3 PathEvent high-frequency backend.

Renders ordered image-source path events with causal fractional delays and
angle-aware boundary gains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.physics.propagation import (
    AIR_ABSORPTION_POLICY,
    apply_air_absorption,
)
from puresound.audio.rir.path_events import (
    generate_scene_shoebox_path_events,
    material_absorption_relaxation_models,
    render_path_events,
)


@dataclass
class PathEventHighFrequencyBackend:
    """Opt-in inspectable M3 geometric backend for material-first scenes."""

    #: Late-path policy recorded in the air-absorption metadata.  A plain class
    #: attribute rather than an ``isinstance`` check against the FDN subclass:
    #: without it the base class would depend on its own subclass, which stops
    #: the two from living in separate modules.  Not annotated, so ``dataclass``
    #: does not turn it into a field.
    _LATE_PATH_AIR_ABSORPTION_POLICY = "direct_distance_lower_bound"

    max_order: int = 12
    edge_corner_policy: str = "exclude"
    include_scene_interactions: bool = True
    material_reference_frequency_hz: float = 1000.0
    fractional_delay_order: int = 3
    boundary_filter_tail_ms: float = 24.0
    air_absorption: bool = True
    air_absorption_filter_taps: int = 129
    last_boundary_metadata: Optional[dict[str, Any]] = field(
        default=None,
        init=False,
        repr=False,
    )
    last_air_absorption_metadata: Optional[dict[str, Any]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def _boundary_models(
        self,
        scene: RoomSceneV2,
    ) -> dict[str, Any]:
        models, metadata = material_absorption_relaxation_models(
            scene,
            reference_frequency_hz=(
                self.material_reference_frequency_hz
            ),
        )
        self.last_boundary_metadata = metadata
        return models

    def simulate(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
    ) -> np.ndarray:
        if not isinstance(scene, RoomSceneV2):
            raise ValueError(
                "PathEventHighFrequencyBackend requires a v2 material-first "
                "scene"
            )
        if not 0 <= int(self.max_order) <= 20:
            raise ValueError("PathEvent max_order must be in [0, 20]")
        boundary_models = self._boundary_models(scene)
        surface_models = {
            surface.surface_id: boundary_models[surface.boundary]
            for surface in scene.surfaces
        }
        outputs = []
        for source_index in range(len(scene.sources)):
            event_set = generate_scene_shoebox_path_events(
                scene,
                source_index=source_index,
                max_order=int(self.max_order),
                edge_corner_policy=self.edge_corner_policy,
                resolve_object_visibility=True,
                include_scene_interactions=(
                    self.include_scene_interactions
                ),
                boundary_admittance_models=boundary_models,
                reflection_frequencies_hz=(
                    60.0,
                    125.0,
                    250.0,
                    500.0,
                    1000.0,
                    2000.0,
                    4000.0,
                    8000.0,
                ),
            )
            outputs.append(
                render_path_events(
                    event_set,
                    sample_rate_hz=config.sample_rate,
                    num_samples=config.num_samples,
                    fractional_delay_order=(
                        self.fractional_delay_order
                    ),
                    surface_admittance_models=surface_models,
                    maximum_boundary_filter_tail_samples=max(
                        1,
                        int(
                            round(
                                float(self.boundary_filter_tail_ms)
                                * 1e-3
                                * float(config.sample_rate)
                            )
                        ),
                    ),
                )
            )
        if self.air_absorption:
            filtered_outputs = []
            channel_metadata = []
            distances = scene.source_distances()
            for source_index, output in enumerate(outputs):
                filtered, air_metadata = apply_air_absorption(
                    output,
                    int(config.sample_rate),
                    float(distances[source_index]),
                    temperature_c=float(scene.environment.temperature_c),
                    relative_humidity_percent=float(
                        scene.environment.relative_humidity_percent
                    ),
                    pressure_pa=float(scene.environment.pressure_pa),
                    num_taps=int(self.air_absorption_filter_taps),
                )
                filtered_outputs.append(filtered)
                channel_metadata.append(
                    {
                        "channel": int(source_index),
                        "source_id": scene.sources[
                            source_index
                        ].transducer_id,
                        **air_metadata,
                    }
                )
            outputs = filtered_outputs
            self.last_air_absorption_metadata = {
                "policy": AIR_ABSORPTION_POLICY,
                "early_path_distance_policy": (
                    "causal_minimum_phase_filter_at_direct_distance"
                ),
                "late_path_policy": self._LATE_PATH_AIR_ABSORPTION_POLICY,
                "channels": channel_metadata,
            }
        else:
            self.last_air_absorption_metadata = {
                "policy": "disabled",
            }
        return np.asarray(outputs, dtype=np.float32)


__all__ = ["PathEventHighFrequencyBackend"]
