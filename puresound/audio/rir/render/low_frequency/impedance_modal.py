"""Separable 3D complex-impedance modal backend (M2.7 - M2.10).

Solves the nonlinear rectangular eigenproblem for passive rational wall
admittances and renders the FDTD-calibrated modal residues against the
free-field ``1/r`` convention.  Moved out of ``puresound.audio.rir.render.hybrid``
in R2 of ``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
    solve_rectangular_impedance_modes,
)
from puresound.audio.rir.physics.impedance.residues import ImpedanceModalResidueCalibration
from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.physics.wave.source_convention import (
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
    convert_pressure_state_modal_residue,
)


@dataclass
class ImpedanceModalLowFrequencyBackend:
    """Experimental separable 3D rational-impedance modal renderer.

    Boundary assignment is explicit and independent of the scene absorption
    catalog. The nonlinear eigenvalues and complex separable eigenfunctions are
    physical; the current modal residue scale remains an engineering bridge for
    RIR rendering and is recorded as such in metadata.
    """

    boundary_config: RectangularImpedanceBoundaryConfig
    num_modes_per_axis: int = 5
    max_modes: Optional[int] = 128
    amplitude_scale: float = 0.015
    residue_calibration: Optional[ImpedanceModalResidueCalibration] = None
    continuation_steps: int = 8
    search_margin: float = 0.25
    impedance_boundary_model: bool = field(default=True, init=False)
    physical_mode_coupling: bool = field(default=True, init=False)
    last_modal_metadata: Optional[dict[str, Any]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.num_modes_per_axis < 1:
            raise ValueError("impedance modal index limit must be positive")
        if self.max_modes is not None and self.max_modes < 1:
            raise ValueError("impedance modal max_modes must be positive")
        if not math.isfinite(float(self.amplitude_scale)) or self.amplitude_scale <= 0.0:
            raise ValueError("impedance modal amplitude scale must be positive")
        if (
            self.residue_calibration is not None
            and self.residue_calibration.boundary_reference_id
            != self.boundary_config.reference_id
        ):
            raise ValueError(
                "impedance residue calibration boundary reference does not "
                "match the selected boundary config"
            )
        if self.continuation_steps < 1:
            raise ValueError("impedance modal continuation steps must be positive")
        if (
            not math.isfinite(float(self.search_margin))
            or not 0.0 <= self.search_margin <= 1.0
        ):
            raise ValueError("impedance modal search margin must be in [0, 1]")

    def _solve_modes(
        self,
        scene: RoomSceneV2,
        config: HybridRIRConfig,
    ):
        room = np.asarray(scene.room_dim, dtype=np.float64)
        candidates: list[tuple[float, tuple[int, int, int]]] = []
        lower = max(
            0.0,
            float(config.low_fmin_hz) * (1.0 - float(self.search_margin)),
        )
        upper = float(config.low_fmax_hz) * (
            1.0 + float(self.search_margin)
        )
        for nx in range(self.num_modes_per_axis + 1):
            for ny in range(self.num_modes_per_axis + 1):
                for nz in range(self.num_modes_per_axis + 1):
                    if nx == ny == nz == 0:
                        continue
                    rigid_frequency = 0.5 * float(config.sound_speed) * math.sqrt(
                        (nx / room[0]) ** 2
                        + (ny / room[1]) ** 2
                        + (nz / room[2]) ** 2
                    )
                    if lower <= rigid_frequency <= upper:
                        candidates.append(
                            (float(rigid_frequency), (nx, ny, nz))
                        )
        candidates.sort()
        if self.max_modes is not None:
            candidates = candidates[: int(self.max_modes)]
        solved = solve_rectangular_impedance_modes(
            scene.room_dim,
            self.boundary_config,
            mode_indices=[indices for _frequency, indices in candidates],
            sound_speed_m_s=float(config.sound_speed),
            continuation_steps=int(self.continuation_steps),
        )
        return tuple(
            mode
            for mode in solved
            if float(config.low_fmin_hz)
            <= mode.frequency_hz
            <= float(config.low_fmax_hz)
        )

    def simulate(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
    ) -> np.ndarray:
        if not isinstance(scene, RoomSceneV2):
            raise ValueError(
                "impedance modal backend requires a RoomSceneV2; its explicit "
                "boundary config is not inferred from a legacy RT60 scene"
            )
        valid_range = self.boundary_config.applicability.get(
            "valid_frequency_range_hz"
        )
        if valid_range is not None:
            if (
                not isinstance(valid_range, (list, tuple))
                or len(valid_range) != 2
            ):
                raise ValueError(
                    "impedance boundary valid_frequency_range_hz must be [min, max]"
                )
            valid_minimum, valid_maximum = (
                float(value) for value in valid_range
            )
            if (
                float(config.low_fmin_hz) < valid_minimum
                or float(config.low_fmax_hz) > valid_maximum
            ):
                raise ValueError(
                    "requested impedance modal band lies outside boundary "
                    f"validity [{valid_minimum:g}, {valid_maximum:g}] Hz"
                )
        if self.residue_calibration is not None:
            residue_minimum, residue_maximum = (
                self.residue_calibration.valid_frequency_range_hz
            )
            if (
                float(config.low_fmin_hz) < residue_minimum
                or float(config.low_fmax_hz) > residue_maximum
            ):
                raise ValueError(
                    "requested impedance modal band lies outside residue "
                    f"calibration validity [{residue_minimum:g}, "
                    f"{residue_maximum:g}] Hz"
                )
        fs = int(config.sample_rate)
        n = int(config.num_samples)
        t = np.arange(n, dtype=np.float64) / float(fs)
        room = np.asarray(scene.room_dim, dtype=np.float64)
        mic = np.asarray(scene.mic_pos, dtype=np.float64)
        sources = np.asarray(scene.source_pos, dtype=np.float64)
        output = np.zeros((sources.shape[0], n), dtype=np.float64)
        modes = self._solve_modes(scene, config)
        first_frequency = modes[0].frequency_hz if modes else 1.0
        room_volume = float(np.prod(room))
        receiver_values = np.asarray(
            [mode.eigenfunction_at(mic) for mode in modes],
            dtype=np.complex128,
        )

        for source_index, source in enumerate(sources):
            distance = float(np.linalg.norm(source - mic))
            direct = int(
                round(
                    distance
                    / float(config.sound_speed)
                    * float(config.sample_rate)
                )
            )
            if direct < n:
                output[source_index, direct] += 1.0 / max(distance, 0.1)
            local_time = np.maximum(t - direct / float(fs), 0.0)
            active = t >= direct / float(fs)
            for mode_index, mode in enumerate(modes):
                source_value = mode.eigenfunction_at(source)
                if self.residue_calibration is not None:
                    coupling = source_value * receiver_values[mode_index]
                    weighted_residue = (
                        self.residue_calibration.complex_scale
                        * self.residue_calibration.frequency_weight(
                            mode.frequency_hz
                        )
                        * coupling
                    )
                    weighted_residue = convert_pressure_state_modal_residue(
                        weighted_residue,
                        mode.complex_angular_frequency_rad_s,
                        sample_rate_hz=fs,
                        sound_speed_m_s=float(config.sound_speed),
                    )
                    # The fixed-pole fit uses absolute source time. Clipping at
                    # the geometric arrival preserves causality without adding
                    # a position-dependent phase rotation to the residue.
                    response = np.real(
                        weighted_residue
                        * np.exp(
                            mode.complex_angular_frequency_rad_s * t
                        )
                    )
                    output[source_index] += response * active
                else:
                    coupling = (
                        room_volume
                        * source_value
                        * receiver_values[mode_index]
                    )
                    amplitude = (
                        float(self.amplitude_scale)
                        * first_frequency
                        / max(mode.frequency_hz, 1e-9)
                    )
                    response = np.imag(
                        coupling
                        * np.exp(
                            mode.complex_angular_frequency_rad_s * local_time
                        )
                    )
                    output[source_index] += amplitude * response * active

        calibrated_residue = self.residue_calibration is not None
        self.last_modal_metadata = {
            "model": "separable_3d_rational_impedance_eigenproblem",
            "reference": self.boundary_config.metadata(),
            "mode_count": len(modes),
            "global_rt60_envelope_applied": False,
            "production_material_mapping_enabled": False,
            "eigenvalue_formulation": (
                "three_axis_robin_characteristics_plus_3d_dispersion"
            ),
            "eigenfunction_normalization": (
                "separable_complex_bilinear_volume_norm"
            ),
            "modal_residue_model": (
                "fdtd_calibrated_complex_scale_power_law"
                if calibrated_residue
                else "engineering_scale_times_complex_eigenfunction_coupling"
            ),
            "modal_residue_fdtd_validated": calibrated_residue,
            "modal_residue_production_validated": False,
            "modal_residue_calibration": (
                self.residue_calibration.metadata()
                if self.residue_calibration is not None
                else None
            ),
            "rir_source_convention": FREE_FIELD_1_OVER_R_RIR_CONVENTION,
            "modal_residue_fitted_source_convention": (
                self.residue_calibration.fitted_source_convention
                if self.residue_calibration is not None
                else None
            ),
            "modal_residue_source_transform": (
                self.residue_calibration.residue_transform
                if self.residue_calibration is not None
                else None
            ),
            "modal_residue_causality_policy": (
                "absolute_modal_time_clipped_before_geometric_arrival"
                if calibrated_residue
                else "local_time_from_geometric_arrival"
            ),
            "direct_path_amplitude_calibrated_by_residue_fit": False,
            "direct_path_source_convention_matched": calibrated_residue,
            "modes": [mode.metadata() for mode in modes],
        }
        return output.astype(np.float32)


__all__ = ["ImpedanceModalLowFrequencyBackend"]
