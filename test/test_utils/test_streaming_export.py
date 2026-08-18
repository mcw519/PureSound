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

import pytest
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
    assert manifest["postprocess"] == {
        "dry_blend": 0.9,
        "spec_floor": 0.0,
        "suppression_ceiling_db": pytest.approx(-20.0),
    }

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
