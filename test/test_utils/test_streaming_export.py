"""The ONNX export path, end to end, on both backbones.

This path was copied verbatim between `dpcrn.py` and `dparn.py` -- 83 lines,
byte for byte -- and had no test at all: the streaming suites cover config
validation, frame/state shapes and offline-vs-streaming parity, all of which
stop short of the export. Now that both backbones share one implementation, a
regression in it breaks every exported model at once, so it needs one.

Marked `slow` because each case builds a real ONNX graph and an onnxruntime
session (~7 s). `export_streaming_*_onnx` already asserts internally that the
ORT output matches PyTorch, so simply reaching the end is most of the check;
what these add is that the manifest describes the graph that was written, which
is the contract the runtime and the portable SDK read.
"""

from pathlib import Path

import numpy as np
import pytest

from puresound.streaming.base import StreamingOrt
import torch
import yaml

from test.test_utils.test_dparn_streaming import MINIMAL_DPARN_CONFIG

from puresound.system.postprocess import Postprocessor
from puresound.streaming import (
    StreamingOrt,
    export_streaming_dparn_onnx,
    export_streaming_dpcrn_onnx,
    load_streaming_dparn_model,
    load_streaming_dpcrn_model,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG = _REPO_ROOT / "egs/voice_isolate/config/exp"

VARIANTS = {
    # The lookahead recipe is the one that exercises `_extra_state_names`: its
    # skip caches, delayed noisy frame and warm-up counter are extra ONNX ports
    # that DPARN does not have.
    "dpcrn_lookahead": (
        _CFG / "train_dpcrn_wide_antisup.yaml",
        load_streaming_dpcrn_model,
        export_streaming_dpcrn_onnx,
    ),
    "dpcrn_causal": (
        _CFG / "train_dpcrn_wide_causal.yaml",
        load_streaming_dpcrn_model,
        export_streaming_dpcrn_onnx,
    ),
    # No shipped DPARN recipe is streaming-compliant (they all train the
    # frontend), so this reuses the minimal one `test_dparn_streaming` already
    # maintains rather than inventing a second copy that can drift from it.
    "dparn": (None, load_streaming_dparn_model, export_streaming_dparn_onnx),
}


@pytest.fixture
def minimal_dparn_recipe(tmp_path):
    path = tmp_path / "dparn.yaml"
    path.write_text(yaml.safe_dump(MINIMAL_DPARN_CONFIG, sort_keys=False))
    return path


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_manifest_describes_the_graph_that_was_written(
    variant, tmp_path, minimal_dparn_recipe
):
    config, load, export = VARIANTS[variant]
    config = minimal_dparn_recipe if config is None else config
    if not config.exists():
        pytest.skip(f"{config} is not in this checkout")

    # Random weights: the export path does not care what the numbers are, and
    # its own torch-vs-ORT check is a comparison against itself.
    torch.manual_seed(0)
    frame_model = load(str(config))
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": frame_model.system_model.state_dict()}, checkpoint)

    onnx_path = tmp_path / "model.onnx"
    manifest = export(str(config), checkpoint, onnx_path)

    assert onnx_path.exists() and onnx_path.stat().st_size > 0
    assert manifest["model_type"] == f"{variant.split('_')[0]}_streaming_frame"

    # The runtime feeds the session by name and sizes its buffers from these,
    # so a manifest that disagrees with the model is a runtime that silently
    # feeds the wrong port.
    assert manifest["state_input_names"] == frame_model.state_input_names
    assert manifest["state_output_names"] == frame_model.state_output_names
    assert manifest["input_names"] == ["noisy_frame"] + frame_model.state_input_names
    state = frame_model.initial_state_tensors(batch_size=1)
    assert len(state) == len(frame_model.state_input_names)
    assert manifest["state_shapes"] == {
        name: list(tensor.shape)
        for name, tensor in zip(frame_model.state_input_names, state)
    }


@pytest.mark.slow
def test_the_exported_graph_loads_in_the_shared_runtime(tmp_path):
    """One runtime drives both backbones off the manifest's port names.

    `StreamingDpcrnOrt` and `StreamingDparnOrt` are the same class; this is the
    check that the names it looks up are the names the export actually wrote.
    """
    config = _CFG / "train_dpcrn_wide_causal.yaml"
    torch.manual_seed(0)
    frame_model = load_streaming_dpcrn_model(str(config))
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": frame_model.system_model.state_dict()}, checkpoint)
    onnx_path = tmp_path / "model.onnx"
    export_streaming_dpcrn_onnx(str(config), checkpoint, onnx_path)

    runtime = StreamingOrt(onnx_path)
    enhanced = runtime.process_samples(torch.zeros(16000).numpy())
    assert enhanced.shape[0] <= 16000


@pytest.mark.slow
def test_the_export_records_the_postprocessing_the_runtime_must_apply(tmp_path):
    """The graph stops at the model, so `dry_blend` has to travel in the manifest.

    Both ends of one key, checked against a real export rather than a hand-built
    dict: drop it from the export and the runtime silently falls back to 1.0,
    which ships a system the scorecards never measured.
    """
    config = _CFG / "train_dpcrn_wide_causal.yaml"
    torch.manual_seed(0)
    frame_model = load_streaming_dpcrn_model(str(config))
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": frame_model.system_model.state_dict()}, checkpoint)
    onnx_path = tmp_path / "model.onnx"

    manifest = export_streaming_dpcrn_onnx(
        str(config), checkpoint, onnx_path, postprocess=Postprocessor(dry_blend=0.9)
    )
    recorded = manifest[Postprocessor.MANIFEST_KEY]
    assert recorded["dry_blend"] == pytest.approx(0.9)
    assert recorded["spec_floor"] == 0.0
    assert recorded["suppression_ceiling_db"] == pytest.approx(-20.0)
    # The note carries the latency the runtime has to compensate for, which is
    # what the shipped artefacts' own notes have always said.
    assert "latency-aligned" in recorded["note"]

    runtime = StreamingOrt(onnx_path)
    assert runtime.dry_blend == pytest.approx(0.9)

    # And it is actually applied, not merely parsed.
    quiet = StreamingOrt(onnx_path)
    quiet.dry_blend = 1.0
    samples = torch.zeros(8000).numpy()
    samples[::200] = 0.5
    blended = runtime.process_samples(samples)
    unblended = quiet.process_samples(samples)
    assert blended.shape == unblended.shape
    assert not (blended == unblended).all()


def test_state_outputs_are_located_from_the_manifest_not_assumed_at_index_one():
    """A graph with auxiliary heads puts their logits before the state outputs.

    Assuming the state starts at index 1 shifts every port by the number of
    extras -- here it fed a rank-4 conv cache into a rank-3 EMA slot. ORT
    rejected that one; a layout that happens to typecheck would corrupt state
    silently, so the offset is read from `extra_output_names`.
    """
    import numpy as np

    class FakeSession:
        def run(self, names, feeds):
            return [
                np.zeros((1, 4, 2), dtype=np.float32),   # enhanced_frame
                np.full((1, 1), 7.0, dtype=np.float32),  # aux_logit
                np.full((1, 3), 1.0, dtype=np.float32),  # next_s0
                np.full((1, 2), 2.0, dtype=np.float32),  # next_s1
            ]

    runtime = StreamingOrt.__new__(StreamingOrt)
    runtime.session = FakeSession()
    runtime.freq_bins = 4
    runtime.manifest = {
        "output_names": ["enhanced_frame", "aux_logit", "next_s0", "next_s1"],
        "extra_output_names": ["aux_logit"],
        "state_input_names": ["s0", "s1"],
    }
    runtime.state = {"s0": np.zeros((1, 3), np.float32), "s1": np.zeros((1, 2), np.float32)}
    runtime.extras = {}
    runtime.run_frame(np.zeros((1, 4, 2), dtype=np.float32))

    assert runtime.state["s0"].shape == (1, 3)
    assert float(runtime.state["s0"][0, 0]) == 1.0
    assert float(runtime.state["s1"][0, 0]) == 2.0
    assert float(runtime.extras["aux_logit"][0, 0]) == 7.0


def test_a_state_layout_mismatch_is_an_error_not_a_silent_truncation():
    import numpy as np

    class ShortSession:
        def run(self, names, feeds):
            return [np.zeros((1, 4, 2), np.float32), np.zeros((1, 3), np.float32)]

    runtime = StreamingOrt.__new__(StreamingOrt)
    runtime.session = ShortSession()
    runtime.freq_bins = 4
    runtime.manifest = {
        "output_names": ["enhanced_frame", "next_s0", "next_s1"],
        "extra_output_names": [],
        "state_input_names": ["s0", "s1"],
    }
    runtime.state = {}
    runtime.extras = {}
    with pytest.raises(RuntimeError, match="state tensors"):
        runtime.run_frame(np.zeros((1, 4, 2), dtype=np.float32))
