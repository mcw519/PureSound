"""Source conventions shared by FDTD calibration and digital RIR rendering.

The validation FDTD solver adds a pressure increment directly to one grid
cell.  That is not the same input convention as a digital RIR whose free-field
direct path is a delayed impulse with amplitude ``1 / distance``.  This module
keeps the conversion explicit and versioned.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable

import numpy as np


FDTD_PRESSURE_CELL_SOURCE_CONVENTION = (
    "puresound.fdtd_pressure_cell_state_increment.v1"
)
FREE_FIELD_1_OVER_R_RIR_CONVENTION = (
    "puresound.free_field_pressure_1_over_r_discrete_rir.v1"
)
PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM = (
    "puresound.pressure_state_residue_to_free_field_1_over_r.v1"
)


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def fdtd_cell_center_position(
    cell_zyx: Iterable[int],
    spacing_xyz_m: Iterable[float],
) -> tuple[float, float, float]:
    """Return the physical center of an FDTD pressure cell."""
    indices = tuple(int(value) for value in cell_zyx)
    spacing = tuple(
        _positive_finite("grid spacing", value) for value in spacing_xyz_m
    )
    if len(indices) != 3 or len(spacing) != 3:
        raise ValueError("cell indices and grid spacing must each have length 3")
    if any(value < 0 for value in indices):
        raise ValueError("cell indices must be non-negative")
    iz, iy, ix = indices
    dx, dy, dz = spacing
    return (
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dy,
        (float(iz) + 0.5) * dz,
    )


def fdtd_pressure_cell_to_free_field_input(
    source_pressure_increment: np.ndarray,
    *,
    time_step_s: float,
    cell_volume_m3: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """Map a pressure-cell source sequence to the digital ``1/r`` input.

    If ``q[n]`` is the pressure increment added by the FDTD solver, the
    equivalent input to a digital RIR with direct path ``delta(t-r/c) / r`` is

    ``x(t) = cell_volume / (dt * 4*pi*c**2) * dq(t)/dt``.

    The derivative is evaluated on the FDTD sample grid.  This mapping is
    intended for band-limited validation sources, not discontinuous impulses.
    """
    source = np.asarray(source_pressure_increment, dtype=np.float64)
    if source.ndim != 1 or source.size < 3:
        raise ValueError("source pressure increment must be a 1D array of length >= 3")
    dt = _positive_finite("time step", time_step_s)
    volume = _positive_finite("cell volume", cell_volume_m3)
    sound_speed = _positive_finite("sound speed", sound_speed_m_s)
    derivative = np.gradient(source, dt, edge_order=2)
    return (
        volume
        / (dt * 4.0 * math.pi * sound_speed**2)
        * derivative
    )


def fdtd_pressure_cell_free_field_direct(
    source_pressure_increment: np.ndarray,
    *,
    time_step_s: float,
    cell_volume_m3: float,
    distance_m: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """Predict the free-field direct pressure produced by the FDTD source.

    For the pressure-cell update used by :mod:`puresound.audio.rir.physics.wave.fdtd`,
    the continuum limit is

    ``p(r,t) = cell_volume / (dt * 4*pi*c**2*r) * dq(t-r/c)/dt``.
    """
    source = np.asarray(source_pressure_increment, dtype=np.float64)
    dt = _positive_finite("time step", time_step_s)
    distance = _positive_finite("distance", distance_m)
    sound_speed = _positive_finite("sound speed", sound_speed_m_s)
    equivalent_input = fdtd_pressure_cell_to_free_field_input(
        source,
        time_step_s=dt,
        cell_volume_m3=cell_volume_m3,
        sound_speed_m_s=sound_speed,
    )
    time_s = np.arange(source.size, dtype=np.float64) * dt
    return (
        np.interp(
            time_s - distance / sound_speed,
            time_s,
            equivalent_input,
            left=0.0,
            right=0.0,
        )
        / distance
    )


def pressure_state_modal_residue_conversion_factor(
    complex_angular_frequency_rad_s: complex,
    *,
    sample_rate_hz: float,
    sound_speed_m_s: float,
) -> complex:
    """Return the residue factor for the digital ``1/r`` RIR convention.

    A fixed-pole residue fitted to the FDTD pressure-cell state convention is
    divided by its pole (time integration) and multiplied by
    ``4*pi*c**2/fs``.  The FDTD cell volume is absent here because the
    production eigenfunction coupling is continuous and volume-normalized; it
    cancels when the two source conventions are related.
    """
    pole = complex(complex_angular_frequency_rad_s)
    if not cmath.isfinite(pole) or abs(pole) <= 1e-15:
        raise ValueError("modal pole must be finite and non-zero")
    sample_rate = _positive_finite("sample rate", sample_rate_hz)
    sound_speed = _positive_finite("sound speed", sound_speed_m_s)
    return complex(
        4.0 * math.pi * sound_speed**2 / (sample_rate * pole)
    )


def convert_pressure_state_modal_residue(
    residue: complex,
    complex_angular_frequency_rad_s: complex,
    *,
    sample_rate_hz: float,
    sound_speed_m_s: float,
) -> complex:
    """Convert one fitted pressure-state pole residue to a digital RIR residue."""
    value = complex(residue)
    if not cmath.isfinite(value):
        raise ValueError("modal residue must be finite")
    return value * pressure_state_modal_residue_conversion_factor(
        complex_angular_frequency_rad_s,
        sample_rate_hz=sample_rate_hz,
        sound_speed_m_s=sound_speed_m_s,
    )


__all__ = [
    "FDTD_PRESSURE_CELL_SOURCE_CONVENTION",
    "FREE_FIELD_1_OVER_R_RIR_CONVENTION",
    "PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM",
    "convert_pressure_state_modal_residue",
    "fdtd_cell_center_position",
    "fdtd_pressure_cell_free_field_direct",
    "fdtd_pressure_cell_to_free_field_input",
    "pressure_state_modal_residue_conversion_factor",
]
