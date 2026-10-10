"""The synthesis trace: what a tap keeps, and that recording leaves the dataset
exactly as it found it."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puresound.audio.augmentation import RirApplied, RirDetail
from puresound.task.ns import RowPlan
from puresound.task.trace import (
    STAGE_INDEX,
    STAGES,
    SynthesisTrace,
    bind_target,
    recording,
    summarize,
)


def test_stage_ids_are_unique_and_run_from_the_source_to_the_emitted_pair():
    ids = [spec.id for spec in STAGES]
    assert len(ids) == len(set(ids))
    assert ids[0] == "source.load" and ids[-1] == "row.emit"
    assert STAGE_INDEX["noise.floor"] < STAGE_INDEX["chain.src"] < STAGE_INDEX["chain.adc"] < STAGE_INDEX["chain.codec"]


def test_a_tap_keeps_a_detached_float32_copy_of_the_first_channel():
    trace = SynthesisTrace()
    noisy = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    target = torch.ones(1, 3)
    trace.tap("source.load", noisy, target, speaker="s1")
    noisy.add_(100.0)
    tap = trace.taps[0]
    assert tap.noisy.dtype == np.float32
    np.testing.assert_array_equal(tap.noisy, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(tap.target, [1.0, 1.0, 1.0])
    assert tap.params == {"speaker": "s1"}
    assert trace.stage_ids() == ["source.load"]


def test_a_tap_rejects_a_stage_the_registry_does_not_name():
    with pytest.raises(ValueError, match="unknown synthesis stage"):
        SynthesisTrace().tap("chain.eq", torch.zeros(1, 4), torch.zeros(1, 4))


def test_a_list_signal_is_summed_over_its_parts_and_padded_to_the_longest():
    trace = SynthesisTrace()
    parts = [torch.ones(1, 3), 2 * torch.ones(1, 5)]
    trace.tap(
        "interferers.sample",
        torch.zeros(1, 5),
        torch.zeros(1, 5),
        signals={"interferers": parts, "echo": None},
    )
    np.testing.assert_array_equal(trace.taps[0].signals["interferers"], [3, 3, 3, 2, 2])
    assert "echo" not in trace.taps[0].signals


def test_summarize_keeps_scalars_and_collapses_what_json_cannot_hold():
    summary = summarize(RowPlan(target_absent=True, speed_factor=1.1))
    assert summary["class"] == "RowPlan"
    assert summary["target_absent"] is True and summary["speed_factor"] == 1.1
    scene = {
        "_bank": True,
        "room_id": "r1",
        "rt60": 0.4,
        "room_dim": np.array([4.0, 3.0, 2.5]),
        "near": [{"channel": 0}],
        "used_channels": {1, 0},
    }
    assert summarize(scene) == {
        "room_id": "r1",
        "rt60": 0.4,
        "room_dim": [4.0, 3.0, 2.5],
        "near": "<list>",
        "used_channels": [0, 1],
    }
    assert summarize(torch.tensor(2.5)) == 2.5
    assert summarize(np.float32(0.5)) == 0.5
    nan = summarize(float("nan"))
    assert nan != nan  # NaN stays NaN; JSON sanitising is the report's job


class _Augmentor:
    def __init__(self):
        self.simulated_rir = OrderedDict()

    def apply_rir(
        self,
        wav,
        rir_mode="image",
        sr=16000,
        rir_id=None,
        room_scene=None,
        source_role="source",
        distance_range_override=None,
    ):
        rir_id = rir_id or f"bank-{len(self.simulated_rir)}"
        self.simulated_rir.setdefault(
            rir_id,
            {"impulse": torch.tensor([[1.0, 0.5, 0.25]]), "sample_rate": sr, "metadata": {}},
        )
        return RirApplied(
            wav * 2,
            RirDetail(rir_id, {"mode": rir_mode, "metadata": {"label": "near_0", "drr_db": 3.0}}),
        )


def test_recording_keeps_every_impulse_response_and_restores_the_augmentor():
    augmentor = _Augmentor()
    dataset = SimpleNamespace(augmentor=augmentor, trace=None)
    with recording(dataset) as trace:
        assert dataset.trace is trace
        trace.tap("source.load", torch.zeros(1, 3), torch.zeros(1, 3))
        out = augmentor.apply_rir(wav=torch.ones(1, 3), rir_mode="full", source_role="foreground")
        augmentor.apply_rir(torch.ones(1, 3), "early", 16000, out.detail.rir_id)
    torch.testing.assert_close(out.wav, 2 * torch.ones(1, 3))
    first, second = trace.rirs
    assert (first.position, first.role, first.mode) == (1, "foreground", "full")
    assert (second.mode, second.role, second.rir_id) == ("early", "source", first.rir_id)
    np.testing.assert_array_equal(first.impulse, [1.0, 0.5, 0.25])
    assert first.metadata == {"label": "near_0", "drr_db": 3.0}
    assert first.sample_rate == 16000
    assert dataset.trace is None and "apply_rir" not in vars(augmentor)


def test_recording_is_undone_on_error_and_refuses_to_nest():
    augmentor = _Augmentor()
    dataset = SimpleNamespace(augmentor=augmentor, trace=None)
    with pytest.raises(RuntimeError, match="boom"):
        with recording(dataset):
            raise RuntimeError("boom")
    assert dataset.trace is None and "apply_rir" not in vars(augmentor)
    with recording(dataset):
        with pytest.raises(RuntimeError, match="already being traced"):
            with recording(dataset):
                pass


def test_bind_target_taps_the_mixture_against_a_fixed_target():
    assert bind_target(None, torch.zeros(1, 2)) is None
    trace = SynthesisTrace()
    tap = bind_target(trace, torch.ones(1, 2))
    tap("noise.floor", torch.zeros(1, 2), level_dbfs=-45.0)
    np.testing.assert_array_equal(trace.taps[0].target, [1.0, 1.0])
    assert trace.taps[0].params == {"level_dbfs": -45.0}
