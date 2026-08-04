"""Per-mode damping derived from frequency-dependent surface materials.

Shared by the pytARD and analytic modal backends.  Moved out of
``puresound.audio.rir.render.hybrid`` in R2 of ``RIR_EXP_LOG.md``.

Note the M2 status recorded in ``RIR_EXP_LOG.md``: the default
surface-participation loss law has *not* passed its exit gate, so this module
is infrastructure, not an accepted production model.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.schema import RoomSceneV2


def _spectrum_values_at(
    spectrum: Any,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(spectrum.center_frequencies_hz, dtype=np.float64)
    values = np.asarray(spectrum.values, dtype=np.float64)
    safe_frequency = np.maximum(np.asarray(frequencies_hz, dtype=np.float64), centers[0])
    return np.interp(
        np.log2(safe_frequency),
        np.log2(centers),
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )


def material_modal_decay_rates(
    scene: RoomSceneV2,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    omega_rad_s: np.ndarray,
    sound_speed: float,
    loss_scale: float = 1.0,
) -> np.ndarray:
    """Return modal amplitude decay rates from boundary energy participation.

    For the rigid-wall cosine eigenfunctions, the volume norm along one axis is
    ``L`` for index zero and ``L/2`` otherwise. The surface-to-volume
    participation of each opposing wall pair is therefore
    ``(alpha_minus + alpha_plus) / norm``. Sabine's energy loss rate is
    ``c * participation / 4``; modal amplitude decays at half that rate.
    """
    if loss_scale < 0.0:
        raise ValueError("material modal loss scale cannot be negative")
    effective = scene.effective_boundary_materials()
    frequency_hz = np.asarray(omega_rad_s, dtype=np.float64) / (2.0 * math.pi)
    alpha = {
        boundary: _spectrum_values_at(material.absorption, frequency_hz)
        for boundary, material in effective.items()
    }
    lx, ly, lz = (float(value) for value in scene.room_dim)
    norm_x = np.where(np.asarray(nx) == 0, lx, 0.5 * lx)
    norm_y = np.where(np.asarray(ny) == 0, ly, 0.5 * ly)
    norm_z = np.where(np.asarray(nz) == 0, lz, 0.5 * lz)
    participation = (
        (alpha["west"] + alpha["east"]) / norm_x
        + (alpha["south"] + alpha["north"]) / norm_y
        + (alpha["floor"] + alpha["ceiling"]) / norm_z
    )
    gamma = (
        float(loss_scale)
        * float(sound_speed)
        * np.maximum(participation, 0.0)
        / 8.0
    )
    # Keep the current recurrence underdamped. Very low/DC modes are outside
    # the propagating acoustic range and receive no damping here.
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    gamma = np.minimum(gamma, 0.95 * np.maximum(omega, 0.0))
    gamma = np.where(omega <= 1e-7, 0.0, gamma)
    return np.asarray(gamma, dtype=np.float64)


def material_modal_damping_metadata(
    scene: RoomSceneV2,
    config: HybridRIRConfig,
    max_mode_index: int = 16,
    max_modes: int = 128,
    loss_scale: float = 1.0,
) -> dict[str, Any]:
    """Summarize the material-derived low-frequency modal decay distribution."""
    indices: list[tuple[int, int, int]] = []
    frequencies: list[float] = []
    room = np.asarray(scene.room_dim, dtype=np.float64)
    c = float(config.sound_speed)
    for nx in range(max_mode_index + 1):
        for ny in range(max_mode_index + 1):
            for nz in range(max_mode_index + 1):
                if nx == ny == nz == 0:
                    continue
                frequency = 0.5 * c * math.sqrt(
                    (nx / room[0]) ** 2
                    + (ny / room[1]) ** 2
                    + (nz / room[2]) ** 2
                )
                if config.low_fmin_hz <= frequency <= config.low_fmax_hz:
                    indices.append((nx, ny, nz))
                    frequencies.append(float(frequency))
    order = np.argsort(np.asarray(frequencies, dtype=np.float64))[:max_modes]
    indices = [indices[int(index)] for index in order]
    frequencies_array = np.asarray(
        [frequencies[int(index)] for index in order], dtype=np.float64
    )
    if not indices:
        return {
            "model": "surface_participation_sabine",
            "material_modal_loss_scale": float(loss_scale),
            "mode_count": 0,
            "modes": [],
        }
    nx = np.asarray([index[0] for index in indices], dtype=np.int64)
    ny = np.asarray([index[1] for index in indices], dtype=np.int64)
    nz = np.asarray([index[2] for index in indices], dtype=np.int64)
    omega = 2.0 * math.pi * frequencies_array
    gamma = material_modal_decay_rates(
        scene,
        nx,
        ny,
        nz,
        omega,
        c,
        loss_scale=loss_scale,
    )
    rt60 = math.log(1000.0) / np.maximum(gamma, 1e-12)
    quality_factor = omega / np.maximum(2.0 * gamma, 1e-12)
    modes = [
        {
            "indices": [int(nx[i]), int(ny[i]), int(nz[i])],
            "frequency_hz": float(frequencies_array[i]),
            "amplitude_decay_rate_per_s": float(gamma[i]),
            "rt60_s": float(rt60[i]),
            "quality_factor": float(quality_factor[i]),
        }
        for i in range(len(indices))
    ]
    return {
        "model": "surface_participation_sabine",
        "material_modal_loss_scale": float(loss_scale),
        "mode_count": len(modes),
        "global_rt60_envelope_applied": False,
        "rt60_s": {
            "minimum": float(np.min(rt60)),
            "median": float(np.median(rt60)),
            "maximum": float(np.max(rt60)),
        },
        "quality_factor": {
            "minimum": float(np.min(quality_factor)),
            "median": float(np.median(quality_factor)),
            "maximum": float(np.max(quality_factor)),
        },
        "modes": modes,
    }


__all__ = ["material_modal_damping_metadata", "material_modal_decay_rates"]
