import math

import numpy as np
import pytest

from egs.rir_generation.phases.m3_wave_path.scripts.validate_fourth_order_fdtd_3d import (
    _planned_time_step,
    _single_wall_boundaries,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_fourth_order_fdtd_holdouts import (
    _plane_mode_case,
)
from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir.physics.wave.fdtd import (
    FDTDReferenceConfig,
    simulate_fdtd_reference,
)


def _wall_model(normalized_admittance: float) -> FirstOrderRelaxationAdmittance:
    # A positive-real admittance whose largest value is `normalized_admittance`.
    return FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=normalized_admittance,
        normalized_admittance_relaxation=-0.95 * normalized_admittance,
        relaxation_frequency_hz=600.0,
    )


@pytest.mark.parametrize("normalized_admittance", [0.35, 1.5, 3.0])
def test_validator_time_step_is_the_one_the_solver_uses(normalized_admittance):
    """A hand-sized source signal must match the solver's sample count.

    A large wall admittance makes the solver's boundary Courant cap, not its
    interior CFL limit, set the step; sizing the signal from the interior
    limit alone then fails the solver's length check.
    """
    config = FDTDReferenceConfig(
        room_dim_m=(0.6, 0.36, 0.36),
        grid_spacing_m=0.06,
        duration_s=0.002,
        source_position_m=(0.45, 0.18, 0.18),
        receiver_position_m=(0.15, 0.18, 0.18),
        source_center_hz=270.0,
        source_delay_s=0.0,
        boundary_pressure_scheme="face_quadratic_time_quadratic",
        spatial_derivative_order=4,
        near_wall_closure="third_order_one_sided",
    )
    boundaries = _single_wall_boundaries("west", _wall_model(normalized_admittance))

    time_step = _planned_time_step(config, boundaries)
    num_samples = int(math.ceil(config.duration_s / time_step))
    result = simulate_fdtd_reference(
        config,
        boundary_admittance=boundaries,
        source_signal=np.zeros(num_samples),
    )

    assert result.time_step_s == pytest.approx(time_step, rel=1e-12)
    assert result.rir.size == num_samples


@pytest.mark.slow
def test_plane_mode_holdout_runs_with_a_boundary_limited_wall():
    case = _plane_mode_case(
        _wall_model(3.0),
        case_id="boundary_limited",
        normal_axis=0,
        dimensions_xyz_m=(3.0, 0.656, 0.36),
        mode_indices_xyz=(0, 1, 0),
        frequency_hz=285.0,
        duration_s=0.35,
    )

    assert np.isfinite(case["error"]["maximum_complex_error"])
    assert abs(complex(**case["measured_reflection"])) < 1.5
