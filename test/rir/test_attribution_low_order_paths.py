import argparse
from pathlib import Path

import numpy as np
import pytest

from egs.rir_generation.phases.m3_wave_path.scripts.validate_direct_early_later_attribution import (
    _case_report,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir.physics.wave.fdtd import (
    FDTDReferenceConfig,
    simulate_fdtd_reference,
)
from puresound.audio.rir.physics.wave.source_convention import (
    fdtd_cell_center_position,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_CONFIG = (
    REPO_ROOT
    / "egs/rir_generation/phases/m2_impedance/config"
    / "impedance_reference_glass_wool_14kgm3_100mm.json"
)


def test_attribution_report_tolerates_an_empty_later_bucket():
    """A small room at low image order has no path later than the early window."""
    boundary = RectangularImpedanceBoundaryConfig.from_json(BOUNDARY_CONFIG)
    room = (1.2, 1.0, 0.8)
    source = (0.37, 0.36, 0.30)
    receiver = (0.91, 0.62, 0.49)
    configuration = {
        "grid_spacing_m": 0.1,
        "duration_s": 0.12,
        "source_center_hz": 180.0,
        "source_delay_s": 0.025,
    }
    fdtd = simulate_fdtd_reference(
        FDTDReferenceConfig(
            room_dim_m=room,
            source_position_m=source,
            receiver_position_m=receiver,
            **configuration,
        ),
        boundary_admittance=boundary.boundaries,
    )
    source_case = {
        "case": {
            "case_id": "small_room",
            "split": "train",
            "room_dim_m": list(room),
            "source_position_m": list(source),
            "receiver_position_m": list(receiver),
        },
        "fdtd": {
            "source_cell_center_m": list(
                fdtd_cell_center_position(
                    fdtd.source_cell_zyx, fdtd.grid_spacing_xyz_m
                )
            ),
            "receiver_cell_center_m": list(
                fdtd_cell_center_position(
                    fdtd.receiver_cell_zyx, fdtd.grid_spacing_xyz_m
                )
            ),
        },
    }
    args = argparse.Namespace(
        path_event_max_order=2,
        analysis_sample_rate_hz=8000,
        early_window_s=0.05,
        transition_width_s=0.008,
    )

    report = _case_report(
        source_case,
        full_room={"configuration": configuration},
        boundary_config=boundary,
        args=args,
        frequencies_hz=np.linspace(168.0, 300.0, 16),
    )

    assert report["path_event_counts"]["later_reflections"] == 0
    assert report["path_event_counts"]["early_reflections"] > 0
    later = report["component_metrics_path_time_window_vs_fdtd"][
        "later_reflections"
    ]
    assert all(np.isfinite(value) for value in later.values())
    assert report["reconstruction"][
        "path_arrival_bucket_time_domain_nrmse"
    ] == pytest.approx(0.0, abs=1e-12)
