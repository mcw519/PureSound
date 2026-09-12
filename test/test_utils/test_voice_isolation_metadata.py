"""Voice-isolation scalar labels: NaN semantics and derived values.

The metadata is auxiliary supervision (DistHeadRegressionLoss) and eval
bucketing, so 'unknown' must be NaN — never a fabricated number — and the
derived quantities must come from the right sources."""
import math

import pytest
import torch

from puresound.task.voice_isolation import (
    MIX_MODE_CODES,
    VoiceIsolationDataset,
)


def _metadata(**kwargs):
    defaults = dict(
        foreground_metadata=None,
        interferer_metadata=[],
        target_absent=False,
        background_speech_reference=None,
    )
    defaults.update(kwargs)
    return VoiceIsolationDataset._build_voice_isolation_metadata(
        object.__new__(VoiceIsolationDataset), **defaults
    )


def test_unknowns_are_nan_not_numbers():
    md = _metadata()
    for key in ("foreground_distance", "foreground_drr",
                "nearest_interferer_distance", "strongest_interferer_drr",
                "drr_gap", "rt60"):
        assert math.isnan(float(md[key])), key
    assert float(md["n_interferers"]) == 0.0
    assert float(md["target_present"]) == 1.0
    assert float(md["has_background_speech"]) == 0.0


def test_derived_values_come_from_the_right_sources():
    md = _metadata(
        foreground_metadata={"source_receiver_distance": 0.5, "drr_db": 6.0, "rt60": 0.4},
        interferer_metadata=[
            {"source_receiver_distance": 3.0, "drr_db": -4.0},
            {"source_receiver_distance": 2.0, "drr_db": -9.0},
        ],
        target_absent=True,
        background_speech_reference=torch.ones(1, 10),
        mix_mode="physical",
        realized_speech_sir=4.5,
        turn_taking=1.0,
    )
    assert float(md["nearest_interferer_distance"]) == 2.0      # min distance
    assert float(md["strongest_interferer_drr"]) == -4.0        # max DRR
    assert float(md["drr_gap"]) == 6.0 - (-4.0)                 # fg - strongest itf
    assert float(md["n_interferers"]) == 2.0
    assert float(md["target_absent"]) == 1.0
    assert float(md["target_present"]) == 0.0
    assert float(md["has_background_speech"]) == 1.0
    assert float(md["mix_mode"]) == MIX_MODE_CODES["physical"]
    assert float(md["realized_speech_sir"]) == 4.5
    assert float(md["turn_taking"]) == 1.0


def test_real_recording_rows_mix_known_and_unknown():
    # real rows carry a distance but no DRR/rt60 -- exactly the NaN masking
    # DistHeadRegressionLoss depends on
    md = _metadata(
        foreground_metadata={"source_receiver_distance": 0.74, "origin": "real"},
        interferer_metadata=[{"source_receiver_distance": 3.02, "origin": "real"}],
    )
    assert float(md["foreground_distance"]) == pytest.approx(0.74)
    assert math.isnan(float(md["foreground_drr"]))
    assert float(md["nearest_interferer_distance"]) == pytest.approx(3.02)
    assert math.isnan(float(md["strongest_interferer_drr"]))
    assert math.isnan(float(md["drr_gap"]))
