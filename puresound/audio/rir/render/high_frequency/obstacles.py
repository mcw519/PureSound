"""Post-hoc obstacle occlusion and scatter for the geometric backend.

Applied by the Pyroomacoustics backend after rendering.  Moved out of
``puresound.audio.rir.render.hybrid`` in R2 of ``RIR_MODULARIZATION_PLAN.md``.

Scope note: this model scales the *whole* channel, reverberant tail included,
so it lowers level without lowering DRR.  The PathEvent backends instead treat
furniture as real visibility geometry.  The two are not equivalent; see the
review notes in ``egs/rir_generation/CLAUDE_REVIEW_ADVISE.md``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.geometry import (
    ArrayLike,
    point_in_polygon,
    segments_intersect,
)
from puresound.audio.rir.scene.sampling import (
    HybridRIRScene,
    obstacle_floor_coverage,
)


def segment_segment_t(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> Optional[float]:
    """Parameter ``t`` in ``[0, 1]`` where segment ``a->b`` crosses ``c->d``."""
    r = b - a
    s = d - c
    rxs = float(r[0] * s[1] - r[1] * s[0])
    if abs(rxs) < 1e-12:
        return None
    qp = c - a
    t = float(qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = float(qp[0] * r[1] - qp[1] * r[0]) / rxs
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return float(min(1.0, max(0.0, t)))
    return None


def segment_polygon_crossing_interval(
    start: ArrayLike,
    end: ArrayLike,
    polygon: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Range of ``t`` along ``start->end`` that lies inside ``polygon`` (XY)."""
    a = np.asarray(start, dtype=np.float64)[:2]
    b = np.asarray(end, dtype=np.float64)[:2]
    ts: list[float] = []
    if point_in_polygon(a, polygon):
        ts.append(0.0)
    if point_in_polygon(b, polygon):
        ts.append(1.0)
    for idx in range(polygon.shape[0]):
        t = segment_segment_t(a, b, polygon[idx], polygon[(idx + 1) % polygon.shape[0]])
        if t is not None:
            ts.append(t)
    if not ts:
        return None
    return (min(ts), max(ts))


def segment_intersects_polygon(
    start: ArrayLike,
    end: ArrayLike,
    polygon: np.ndarray,
) -> bool:
    start = np.asarray(start, dtype=np.float64)[:2]
    end = np.asarray(end, dtype=np.float64)[:2]
    if point_in_polygon(start, polygon) or point_in_polygon(end, polygon):
        return True
    for idx in range(polygon.shape[0]):
        if segments_intersect(start, end, polygon[idx], polygon[(idx + 1) % polygon.shape[0]]):
            return True
    return False


def obstacle_high_frequency_events(
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    mic = np.asarray(scene.mic_pos, dtype=np.float64)
    srcs = np.asarray(scene.source_pos, dtype=np.float64)
    fs = int(config.sample_rate)
    for source_idx, src in enumerate(srcs):
        for obstacle_idx, obstacle in enumerate(scene.obstacles):
            footprint = np.asarray(obstacle.footprint, dtype=np.float64)
            interval = segment_polygon_crossing_interval(src[:2], mic[:2], footprint)
            if interval is None:
                continue
            # The 2D footprint is crossed; only treat it as an occluder if the
            # source-to-mic line actually passes through the obstacle's vertical
            # extent. A low table does not occlude a path at head height.
            t_mid = 0.5 * (interval[0] + interval[1])
            z_line = float(src[2] + t_mid * (mic[2] - src[2]))
            if not (float(obstacle.z_min) <= z_line <= float(obstacle.z_max)):
                continue
            attenuation = max(
                0.2,
                1.0 - 0.5 * float(obstacle.absorption) - 0.3 * float(obstacle.scattering),
            )
            scatter_point = obstacle.center
            scatter_distance = float(
                np.linalg.norm(src - scatter_point) + np.linalg.norm(scatter_point - mic)
            )
            scatter_idx = int(round(scatter_distance / config.sound_speed * fs))
            direct_distance = max(float(np.linalg.norm(src - mic)), 0.1)
            direct_idx = int(
                math.floor(
                    direct_distance / float(config.sound_speed) * fs
                )
            )
            recovery_samples = max(
                1,
                int(
                    round(
                        float(config.obstacle_occlusion_recovery_ms)
                        * fs
                        / 1000.0
                    )
                ),
            )
            amp = (
                float(obstacle.scattering)
                * (1.0 - float(obstacle.absorption))
                * direct_distance
                / max(scatter_distance, 0.1)
            )
            events.append(
                {
                    "source_index": int(source_idx),
                    "source_label": (
                        scene.source_labels[source_idx]
                        if source_idx < len(scene.source_labels)
                        else str(source_idx)
                    ),
                    "obstacle_index": int(obstacle_idx),
                    "material": obstacle.material,
                    "z_line_m": z_line,
                    "attenuation": float(attenuation),
                    "attenuation_db": float(20.0 * math.log10(max(attenuation, 1e-12))),
                    "direct_index": direct_idx,
                    "occlusion_recovery_end_index": int(
                        direct_idx + recovery_samples
                    ),
                    "occlusion_recovery_ms": float(
                        config.obstacle_occlusion_recovery_ms
                    ),
                    "scatter_distance_m": scatter_distance,
                    "scatter_index": int(scatter_idx),
                    "scatter_delay_s": float(scatter_idx / max(fs, 1)),
                    "scatter_amplitude": float(0.08 * amp),
                }
            )
    return events


def apply_obstacle_high_frequency_effects(
    rir: np.ndarray,
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> np.ndarray:
    """Approximate furniture occlusion/scattering on high-frequency RIRs.

    Pyroomacoustics is used for the room response.  Internal furniture geometry
    is applied here with deterministic geometric attenuation and scatter taps so
    the behavior is stable across Pyroomacoustics versions.
    """

    out = np.asarray(rir, dtype=np.float64).copy()
    for event in obstacle_high_frequency_events(scene, config):
        source_idx = int(event["source_index"])
        direct_idx = int(event["direct_index"])
        recovery_end = min(
            out.shape[-1],
            int(event["occlusion_recovery_end_index"]) + 1,
        )
        if direct_idx < recovery_end:
            gain = np.linspace(
                float(event["attenuation"]),
                1.0,
                recovery_end - direct_idx,
            )
            out[source_idx, direct_idx:recovery_end] *= gain
        scatter_idx = int(event["scatter_index"])
        if 0 <= scatter_idx < out.shape[-1]:
            out[source_idx, scatter_idx] += float(event["scatter_amplitude"])
    return out.astype(np.float32)


def obstacle_effects_metadata(
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> dict[str, Any]:
    return {
        "obstacle_model": "direct_early_occlusion_with_diffuse_recovery",
        "low_band_obstacle_model": "none",
        "obstacle_count": len(scene.obstacles),
        "floor_coverage_ratio": obstacle_floor_coverage(
            scene.obstacles, np.asarray(scene.room_dim, dtype=np.float64)
        ),
        "events": obstacle_high_frequency_events(scene, config),
    }


__all__ = [
    "apply_obstacle_high_frequency_effects",
    "obstacle_effects_metadata",
    "obstacle_high_frequency_events",
    "segment_intersects_polygon",
    "segment_polygon_crossing_interval",
    "segment_segment_t",
]
