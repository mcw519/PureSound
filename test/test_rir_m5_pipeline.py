import json

import numpy as np
import pytest

from puresound.audio.rir.calibration.inverse_m5 import (
    analyze_local_identifiability,
    select_spatial_calibration_candidate,
)


def test_identifiability_keeps_priority_column_and_rejects_duplicate():
    jacobian = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )

    report = analyze_local_identifiability(
        jacobian,
        ("effective_reflection", "scattering", "late_decay"),
    )

    assert report.full_column_rank is False
    assert report.numerical_rank == 2
    assert report.accepted_parameters == (
        "effective_reflection",
        "late_decay",
    )
    assert report.rejected_parameters == {
        "scattering": "redundant_with:effective_reflection"
    }
    json.dumps(report.to_dict(), allow_nan=False)


def test_spatial_profile_requires_synchronized_array_and_selects_identity():
    rng = np.random.default_rng(10)
    target = rng.standard_normal((2, 1024)) * np.exp(-np.arange(1024) / 300.0)
    target[:, :20] = 0.0
    target[:, 20] = (1.0, 0.9)
    wrong = target.copy()
    wrong[1, 100:] *= -1.0

    selection = select_spatial_calibration_candidate(
        target,
        {"identity": target, "wrong": wrong},
        8000,
        (20, 20),
        physical_first_samples=(20, 20),
    )

    assert selection.best_candidate_id == "identity"
    assert selection.candidate_reports["identity"]["total"] == 0.0
    with pytest.raises(ValueError, match="synchronized receivers"):
        select_spatial_calibration_candidate(
            target[:1],
            {"identity": target[:1]},
            8000,
            (20,),
        )
