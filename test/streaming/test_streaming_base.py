"""`puresound.streaming.base`: the ONNX export shared by both backbones, and the
`StreamingOrt` runtime that drives an exported graph off its manifest.

`export_streaming_*_onnx` asserts internally that ORT matches PyTorch on one
frame from the initial state. What these add is that the manifest describes
the graph that was written -- the contract the runtime and the portable SDK
read -- that the graph keeps matching over many frames, and that the stages
the graph does not contain travel in the manifest and are applied.
"""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from puresound.streaming import (
    StreamingOrt,
    export_streaming_dparn_onnx,
    export_streaming_dpcrn_onnx,
    load_streaming_dparn_model,
    load_streaming_dpcrn_model,
)
from puresound.system.onset_guard import OnsetGuard
from puresound.system.postprocess import Postprocessor
from test.streaming.test_dparn_streaming import MINIMAL_DPARN_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES = REPO_ROOT / "test/fixtures/recipes"
SHIPPED = REPO_ROOT / "egs/voice_isolate/config"


def _dparn_config(root):
    # No shipped DPARN recipe is streaming-compliant (they all train the
    # frontend), so this reuses the minimal one the DPARN tests maintain.
    path = root / "dparn.yaml"
    path.write_text(yaml.safe_dump(MINIMAL_DPARN_CONFIG, sort_keys=False))
    return path


def _deep_filter_config(root):
    config = copy.deepcopy(yaml.safe_load((SHIPPED / "infer_dpcrn_heads.yaml").read_text()))
    config["model"]["backbone"]["backbone_args"].update(
        inter_type="lstm", delay=[1, 1, 1], df_head={"bins": 128, "order": 5, "hidden": 32}
    )
    path = root / "deep_filter.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _randomise_df_head(frame_model):
    """Its last layer starts at zero; a zero residual would export trivially."""
    with torch.no_grad():
        out = frame_model.backbone.df_head.out
        out.weight.normal_(0.0, 0.3)
        out.bias.normal_(0.0, 0.3)


VARIANTS = {
    "dparn": (_dparn_config, load_streaming_dparn_model, export_streaming_dparn_onnx, None),
    "dpcrn_causal": (lambda root: RECIPES / "dpcrn_causal.yaml",
                     load_streaming_dpcrn_model, export_streaming_dpcrn_onnx, None),
    # the look-ahead recipe exercises the extra state ports DPARN does not have:
    # skip caches, delayed noisy frame, warm-up counter
    "dpcrn_lookahead": (lambda root: SHIPPED / "infer_dpcrn.yaml",
                        load_streaming_dpcrn_model, export_streaming_dpcrn_onnx, None),
    "dpcrn_deep_filter": (_deep_filter_config, load_streaming_dpcrn_model,
                          export_streaming_dpcrn_onnx, _randomise_df_head),
    # auxiliary head logits are extra outputs ahead of the state outputs
    "dpcrn_heads": (lambda root: SHIPPED / "infer_dpcrn_heads.yaml",
                    load_streaming_dpcrn_model, export_streaming_dpcrn_onnx, None),
}

GUARD = OnsetGuard()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export each variant once, with random weights -- the export path does
    not care what the numbers are -- and, for DPCRN, the post-graph stages."""
    cache = {}

    def get(variant):
        if variant not in cache:
            make_config, load, export, prepare = VARIANTS[variant]
            root = tmp_path_factory.mktemp(variant)
            config = make_config(root)
            torch.manual_seed(0)
            frame_model = load(str(config)).eval()
            if prepare is not None:
                prepare(frame_model)
            checkpoint = root / "weights.ckpt"
            torch.save({"state_dict": frame_model.system_model.state_dict()}, checkpoint)
            onnx_path = root / "model.onnx"
            # DPARN is exported without relief, so the manifest records its absence
            kwargs = {}
            if export is export_streaming_dpcrn_onnx:
                kwargs = {"postprocess": Postprocessor(dry_blend=0.9), "onset_guard": GUARD}
            manifest = export(str(config), checkpoint, onnx_path, **kwargs)
            cache[variant] = (frame_model, onnx_path, manifest)
        return cache[variant]

    return get


@pytest.mark.slow  # each variant builds a real ONNX graph and session
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_exported_graph_matches_its_manifest_and_torch_over_many_frames(
    exported, variant
):
    """The runtime feeds the session by name and sizes its buffers from the
    manifest, so a manifest that disagrees with the model is a runtime that
    silently feeds the wrong port. And the export's own check runs ONE frame
    from the initial state, which is blind to anything that compounds -- a
    warm-up branch evaluated once at trace time, for one, freezes the head
    state from `warmup_frames` on while the first frame still matches."""
    import onnxruntime as ort

    frame_model, onnx_path, manifest = exported(variant)
    assert onnx_path.exists() and onnx_path.stat().st_size > 0
    assert manifest["model_type"] == f"{variant.split('_')[0]}_streaming_frame"
    assert manifest["state_input_names"] == frame_model.state_input_names
    assert manifest["state_output_names"] == frame_model.state_output_names
    assert manifest["input_names"] == ["noisy_frame"] + frame_model.state_input_names
    state = frame_model.initial_state_tensors(batch_size=1)
    assert len(state) == len(frame_model.state_input_names)
    assert manifest["state_shapes"] == {
        name: list(tensor.shape) for name, tensor in zip(frame_model.state_input_names, state)
    }
    assert manifest.get("extra_output_names", []) == frame_model.extra_output_names
    n_extras = len(frame_model.extra_output_names)
    if variant == "dpcrn_heads":
        assert n_extras == 2

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    length = 16000
    t = torch.arange(length, dtype=torch.float32) / 16000.0
    wav = (0.3 * torch.sin(2 * np.pi * 220 * t) + 0.1 * torch.randn(length)).unsqueeze(0)
    with torch.no_grad():
        tf = frame_model.system_model.encoder(wav)
        torch_state = frame_model.initial_state(batch_size=1)
        ort_state = [s.numpy() for s in state]
        drift = 0.0
        for i in range(tf.shape[2]):
            frame = tf[:, :, i, :]
            result = frame_model.forward_frame(frame, torch_state)
            torch_state = result[-1]
            expected = [result[0].numpy()]
            if n_extras:
                expected += [e.numpy() for e in result[1]]
            feeds = {"noisy_frame": frame.numpy(), **dict(zip(manifest["state_input_names"], ort_state))}
            outputs = session.run(manifest["output_names"], feeds)
            ort_state = outputs[1 + n_extras:]
            for got, want in zip(outputs[: 1 + n_extras], expected):
                drift = max(drift, float(np.abs(got.reshape(want.shape) - want).max()))
    assert tf.shape[2] > getattr(frame_model, "warmup_frames", 0) + 50
    assert drift < 1e-3, f"ORT drifted from torch by {drift}"

    # and the shared runtime drives it off the manifest's port names
    runtime = StreamingOrt(onnx_path, provider="cpu")
    enhanced = runtime.process_samples(np.zeros(16000, np.float32))
    assert enhanced.shape[0] <= 16000
    if variant == "dparn":
        assert "no post-graph relief" in manifest[Postprocessor.MANIFEST_KEY]["note"]
        assert runtime.dry_blend == 1.0


def _guard_bait(seconds=6.0, sr=16000, seed=5):
    """Modulated noise well over a floor: the guard arms on it and releases.

    Constant broadband energy would NOT arm it -- the floor tracker absorbs a
    stationary source by design. Two bursts with a `t_forget_s`-long gap
    between them, so the stream also re-protects.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for lo, hi in ((0.5, 2.0), (5.6, 6.0)):
        span = slice(int(lo * sr), int(hi * sr))
        env[span] = 0.5 * (1.0 + np.sin(2.0 * np.pi * 5.0 * t[span])) + 0.2
    return ((rng.standard_normal(n) * 1e-3) * (1.0 + env * 10.0 ** (30 / 20))).astype(np.float32)


def _stream(runtime, samples, chunk=1024):
    runtime.reset()
    parts = [runtime.process_samples(samples[i : i + chunk]) for i in range(0, samples.size, chunk)]
    parts.append(runtime.flush())
    return np.concatenate(parts)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["dpcrn_causal", "dpcrn_lookahead"])
def test_the_export_records_the_post_graph_stages_and_the_runtime_applies_them(
    exported, variant, tmp_path
):
    """The graph stops at the model, so `dry_blend` and the onset guard travel
    in the manifest: drop one from the export and the runtime silently runs a
    system the scorecards never measured.

    Both recipes, because they differ in the one thing the guard's alignment
    depends on: 0 and 3 frames of algorithmic latency. At 0 the frame the guard
    needs is supplied by the 512-sample analysis window alone (512//160 - 2 = 1
    spare hop), so the exact gain lands and no latency is added.
    """
    _, onnx_path, manifest = exported(variant)

    recorded = manifest[Postprocessor.MANIFEST_KEY]
    assert recorded["dry_blend"] == pytest.approx(0.9)
    assert recorded["spec_floor"] == 0.0
    assert recorded["suppression_ceiling_db"] == pytest.approx(-20.0)
    assert "latency-aligned" in recorded["note"]  # the latency to compensate

    guard_record = manifest[OnsetGuard.MANIFEST_KEY]
    assert OnsetGuard.from_manifest(guard_record) == GUARD
    # what a deployment cannot derive from the knobs
    assert "does NOT contain it" in guard_record["note"]
    assert "hop t+1" in guard_record["note"]

    on = StreamingOrt(onnx_path, provider="cpu")
    assert on.dry_blend == pytest.approx(0.9)
    assert on.onset_guard == GUARD
    assert on.onset_guard_lookahead_hops >= 0

    # An absent guard section means no guard ...
    bare_path = tmp_path / "bare.json"
    bare = dict(manifest)
    bare.pop(OnsetGuard.MANIFEST_KEY)
    bare_path.write_text(json.dumps(bare))
    off = StreamingOrt(onnx_path, bare_path, provider="cpu")
    assert off.onset_guard is None
    # ... and a request can refuse a recorded guard, landing where absence does.
    refused = StreamingOrt(onnx_path, provider="cpu", onset_guard_overrides={"enabled": False})
    assert refused.onset_guard is None

    samples = _guard_bait()
    guarded, unguarded = _stream(on, samples), _stream(off, samples)
    assert np.array_equal(_stream(refused, samples), unguarded)
    assert not np.array_equal(guarded, unguarded), "the guard must do something"

    # The blend is applied, not merely parsed.
    unblended = StreamingOrt(onnx_path, bare_path, provider="cpu",
                             postprocess_overrides={"dry_blend": 1.0})
    head = samples[:8000]
    assert not np.array_equal(_stream(unblended, head), _stream(off, head))

    # The streamed guard IS the offline guard, on the same audio, aligned by the
    # graph's own latency -- bit-identical, since both drive the same recursion
    # over the same float32 blend. Warm-up output has no input to restore.
    delay = int(manifest["streaming_delay_frames"]) * int(manifest["hop_length"])
    expected = GUARD.apply(
        torch.from_numpy(unguarded[delay:]), torch.from_numpy(samples),
        hop=int(manifest["hop_length"]), sr=float(manifest["sample_rate"]),
    ).numpy()
    assert np.array_equal(guarded[delay:], expected), float(np.abs(guarded[delay:] - expected).max())
    assert np.array_equal(guarded[:delay], unguarded[:delay])

    # The anchor is per-stream: a second stream starts protected again.
    assert np.array_equal(_stream(on, samples), guarded)


# --------------------------------------------------------------------------- #
# StreamingOrt without a real graph
# --------------------------------------------------------------------------- #


class _Session:
    def __init__(self, outputs):
        self.outputs = outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feeds):
        return [np.array(o, dtype=np.float32) for o in self.outputs]


def _runtime(tmp_path, outputs, output_names, extra_names, state_shapes, **kwargs):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"fake")
    onnx_path.with_suffix(".json").write_text(json.dumps({
        "sample_rate": 16000, "fft_length": 4, "win_length": 4, "hop_length": 2,
        "freq_bins": 2,
        "output_names": output_names,
        "extra_output_names": extra_names,
        "state_input_names": list(state_shapes),
        "state_shapes": state_shapes,
    }))
    return StreamingOrt(onnx_path, provider="cpu", session=_Session(outputs), **kwargs)


def test_state_outputs_are_located_from_the_manifest_not_assumed_at_index_one(tmp_path):
    """A graph with auxiliary heads puts their logits before the state outputs.

    Assuming the state starts at index 1 shifts every port by the number of
    extras; a layout that happens to typecheck would corrupt state silently.
    A layout that does not add up is an error, not a silent truncation.
    """
    frame = np.zeros((1, 2, 2), np.float32)
    runtime = _runtime(
        tmp_path,
        [np.zeros((1, 2, 2)), np.full((1, 1), 7.0), np.full((1, 3), 1.0), np.full((1, 2), 2.0)],
        ["enhanced_frame", "aux_logit", "next_s0", "next_s1"],
        ["aux_logit"],
        {"s0": [1, 3], "s1": [1, 2]},
    )
    runtime.run_frame(frame)
    assert runtime.state["s0"].shape == (1, 3)
    assert float(runtime.state["s0"][0, 0]) == 1.0
    assert float(runtime.state["s1"][0, 0]) == 2.0
    assert float(runtime.extras["aux_logit"][0, 0]) == 7.0

    short = _runtime(
        tmp_path,
        [np.zeros((1, 2, 2)), np.zeros((1, 3))],
        ["enhanced_frame", "next_s0", "next_s1"],
        [],
        {"s0": [1, 3], "s1": [1, 2]},
    )
    with pytest.raises(RuntimeError, match="state tensors"):
        short.run_frame(frame)


def test_side_information_collection_is_opt_in_and_drains(tmp_path):
    """Off by default, so a long stream cannot grow a history; `drain_extras`
    clears, so a caller that drains stays bounded."""
    def build(**kwargs):
        return _runtime(
            tmp_path,
            [np.zeros((1, 2, 2)), np.full((1, 1), 3.0), np.zeros((1, 1))],
            ["enhanced_frame", "aux_logit", "next_s0"],
            ["aux_logit"],
            {"s0": [1, 1]},
            **kwargs,
        )

    frame = np.zeros((1, 2, 2), np.float32)
    off = build()
    assert off.collect_extras is False and off.extra_names == ["aux_logit"]
    for _ in range(3):
        off.run_frame(frame)
    assert off.extra_history["aux_logit"] == []
    with pytest.raises(RuntimeError, match="collect_extras=True"):
        off.drain_extras()

    on = build(collect_extras=True)
    assert on.drain_extras()["aux_logit"].shape == (0,)
    for _ in range(3):
        on.run_frame(frame)
    drained = on.drain_extras()
    assert drained["aux_logit"].shape == (3,)
    assert float(drained["aux_logit"][0]) == 3.0
    assert on.drain_extras()["aux_logit"].shape == (0,), "drain must clear"


def test_the_runtime_asks_for_cuda_then_falls_back_to_cpu(tmp_path, monkeypatch):
    class FakeSession(_Session):
        def __init__(self, path, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=FakeSession,
    ))
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"fake")
    onnx_path.with_suffix(".json").write_text(json.dumps({
        "sample_rate": 16000, "fft_length": 512, "win_length": 512, "hop_length": 128,
        "freq_bins": 257, "state_input_names": ["state"], "state_output_names": ["next_state"],
        "output_names": ["enhanced_frame", "next_state"], "state_shapes": {"state": [1, 1, 1, 1]},
    }))

    runtime = StreamingOrt(onnx_path, provider="cuda")

    assert runtime.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
