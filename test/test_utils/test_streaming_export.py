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

import torch
import yaml

from test.test_utils.test_dparn_streaming import MINIMAL_DPARN_CONFIG


def _archivable(relative: str) -> Path:
    """A checkpoint or export that may have been archived out of the tree.

    The model zoo ships two voice-isolation versions; the rest moved to the
    gitignored `pretrained_ckpt/backup/` when the catalog was trimmed. They are
    still the right fixtures for these tests, so look for them in both places
    and let the caller skip if neither has the file.
    """

    root = Path(__file__).resolve().parents[2] / "egs/voice_isolate/pretrained_ckpt"
    in_tree = root / relative
    return in_tree if in_tree.is_file() else root / "backup" / relative

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
    runtime.collect_extras = False
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
    runtime.collect_extras = False
    with pytest.raises(RuntimeError, match="state tensors"):
        runtime.run_frame(np.zeros((1, 4, 2), dtype=np.float32))


def test_side_information_collection_is_opt_in_and_drains():
    """History collection is off by default so a long stream cannot grow one,
    and `drain_extras` clears so a caller that drains stays bounded."""
    import numpy as np

    def make(collect):
        r = StreamingOrt.__new__(StreamingOrt)
        r.session = None
        r.freq_bins = 2
        r.manifest = {
            "output_names": ["enhanced_frame", "aux_logit", "next_s0"],
            "extra_output_names": ["aux_logit"],
            "state_input_names": ["s0"],
        }
        r.collect_extras = collect
        r.extra_names = ["aux_logit"]
        r.state = {"s0": np.zeros((1, 1), np.float32)}
        r.extras = {}
        r.extra_history = {"aux_logit": []}

        class S:
            def run(self, names, feeds):
                return [np.zeros((1, 2, 2), np.float32),
                        np.full((1, 1), 3.0, np.float32),
                        np.zeros((1, 1), np.float32)]
        r.session = S()
        return r

    off = make(False)  # hand-built: exercises run_frame, not the constructor
    for _ in range(3):
        off.run_frame(np.zeros((1, 2, 2), np.float32))
    assert off.extra_history["aux_logit"] == []
    with pytest.raises(RuntimeError, match="collect_extras=True"):
        off.drain_extras()

    on = make(True)
    for _ in range(3):
        on.run_frame(np.zeros((1, 2, 2), np.float32))
    drained = on.drain_extras()
    assert drained["aux_logit"].shape == (3,)
    assert float(drained["aux_logit"][0]) == 3.0
    assert on.drain_extras()["aux_logit"].shape == (0,), "drain must clear"


@pytest.mark.skipif(
    not _archivable("streaming/dpcrn_v11_ep19_heads.onnx").is_file(),
    reason="needs the heads export",
)
def test_the_constructor_defaults_collection_off():
    """Through the REAL constructor: the default must be off.

    The hand-built fixtures above set the flag themselves, so they cannot see a
    changed default -- and an always-on default is exactly the regression that
    grows an unbounded history in a long-running stream.
    """
    onnx = _archivable("streaming/dpcrn_v11_ep19_heads.onnx")
    runtime = StreamingOrt(onnx_path=onnx, provider="cpu")
    assert runtime.collect_extras is False
    assert runtime.extra_names == ["vad_logit", "background_vad_logit"]
    with pytest.raises(RuntimeError, match="collect_extras=True"):
        runtime.drain_extras()

    opted_in = StreamingOrt(onnx_path=onnx, provider="cpu", collect_extras=True)
    assert opted_in.collect_extras is True
    assert opted_in.drain_extras()["vad_logit"].shape == (0,)


@pytest.mark.slow
@pytest.mark.skipif(
    not _archivable("dpcrn_v11_ep19.ckpt").is_file(),
    reason="needs the v11 checkpoint",
)
def test_exported_head_logits_track_torch_over_many_frames():
    """The export's own check runs ONE frame from the initial state.

    That is blind to anything that compounds -- and the warm-up gate did: written
    as `if float(counter) < warmup`, a Python branch, it was evaluated once at
    trace time and baked in, so the graph froze the head state forever. torch
    matched offline to 8e-06 while ORT drifted to 6.75, starting exactly at frame
    `warmup_frames`. Only a multi-frame ORT comparison sees it.
    """
    import json

    import numpy as np
    import onnxruntime as ort

    from puresound.audio.io import AudioIO
    from puresound.streaming import load_streaming_dpcrn_model

    recipe_dir = Path(__file__).resolve().parents[2] / "egs/voice_isolate"
    onnx_path = _archivable("streaming/dpcrn_v11_ep19_heads.onnx")
    if not onnx_path.is_file():
        pytest.skip("heads export not built")
    manifest = json.loads(onnx_path.with_suffix(".json").read_text())

    frame_model = load_streaming_dpcrn_model(
        recipe_dir / "config/infer_dpcrn_heads.yaml",
        _archivable("dpcrn_v11_ep19.ckpt"),
    ).eval()
    system = frame_model.system_model.eval()

    wav, _ = AudioIO.open(
        f_path=str(recipe_dir / "data_report/field_cases/test_vector_cases/90d_near1_raw.wav"),
        target_lvl=None, resample_to=16000)
    wav = wav.view(1, -1)[..., : 4 * 16000]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = manifest["output_names"]
    state_in = manifest["state_input_names"]

    with torch.no_grad():
        tf = system.encoder(wav)
        state = frame_model.initial_state(batch_size=1)
        torch_logits = []
        for i in range(tf.shape[2]):
            _, extras, state = frame_model.forward_frame(tf[:, :, i, :], state)
            torch_logits.append(float(extras[0].reshape(-1)[0]))

    ort_state = [t.numpy() for t in frame_model.initial_state_tensors(1)]
    ort_logits = []
    for i in range(tf.shape[2]):
        feeds = {"noisy_frame": tf[:, :, i, :].numpy()}
        feeds.update(dict(zip(state_in, ort_state)))
        outputs = session.run(names, feeds)
        ort_logits.append(float(outputs[1].reshape(-1)[0]))
        ort_state = outputs[1 + len(manifest["extra_output_names"]):]

    drift = np.abs(np.array(torch_logits) - np.array(ort_logits)).max()
    assert drift < 1e-3, f"ORT drifted from torch by {drift}"
    # And specifically past the warm-up boundary, where the baked branch bit.
    warm = frame_model.warmup_frames
    late = np.abs(np.array(torch_logits[warm + 1:]) - np.array(ort_logits[warm + 1:])).max()
    assert late < 1e-3, f"drift after warm-up: {late}"


# --------------------------------------------------------------------------- #
# The onset guard: recorded at export, applied by the runtime
# --------------------------------------------------------------------------- #


def _guard_bait(seconds=6.0, sr=16000, seed=5):
    """Modulated noise well over a floor: the guard arms on it and releases.

    Constant broadband energy would NOT arm it -- the floor tracker absorbs a
    stationary source by design -- so a fixture that is merely loud tests
    nothing. Two bursts with a `t_forget_s`-long gap between them, so the stream
    also re-protects.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for lo, hi in ((0.5, 2.0), (5.6, 6.0)):
        span = slice(int(lo * sr), int(hi * sr))
        env[span] = 0.5 * (1.0 + np.sin(2.0 * np.pi * 5.0 * t[span])) + 0.2
    return ((rng.standard_normal(n) * 1e-3) * (1.0 + env * 10.0 ** (30 / 20))).astype(
        np.float32
    )


def _stream(runtime, samples, chunk=1024):
    runtime.reset()
    parts = [
        runtime.process_samples(samples[i : i + chunk])
        for i in range(0, samples.size, chunk)
    ]
    parts.append(runtime.flush())
    return np.concatenate(parts)


@pytest.mark.slow
@pytest.mark.parametrize("recipe", ["train_dpcrn_wide_causal.yaml",
                                    "train_dpcrn_wide_antisup.yaml"])
def test_the_export_records_the_onset_guard_and_the_runtime_applies_it(
    tmp_path, recipe
):
    """The guard is inference-only and no graph contains it, so it travels in the
    manifest exactly as `dry_blend` does -- and the runtime has to reproduce the
    offline arithmetic, not approximate it.

    Both recipes, because the two differ in the one thing the guard's alignment
    depends on: `train_dpcrn_wide_causal` exports 0 frames of algorithmic
    latency and `train_dpcrn_wide_antisup` exports 3. At 0 the frame the guard
    needs is supplied by the 512-sample analysis window alone (512//160 - 2 = 1
    spare hop), so the exact gain still lands and NO latency is added -- the
    alternative would have been to hold a hop back or refuse, and neither is
    necessary at any shipped geometry.
    """
    import json

    from puresound.system.onset_guard import OnsetGuard

    config = _CFG / recipe
    if not config.exists():
        pytest.skip(f"{config} is not in this checkout")
    torch.manual_seed(0)
    frame_model = load_streaming_dpcrn_model(str(config))
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": frame_model.system_model.state_dict()}, checkpoint)
    onnx_path = tmp_path / "model.onnx"

    guard = OnsetGuard()
    manifest = export_streaming_dpcrn_onnx(
        str(config), checkpoint, onnx_path,
        postprocess=Postprocessor(dry_blend=0.9), onset_guard=guard,
    )

    recorded = manifest[OnsetGuard.MANIFEST_KEY]
    assert OnsetGuard.from_manifest(recorded) == guard
    # The note says what a deployment cannot derive from the knobs: that the
    # graph does not contain it, and how the one-hop analysis lag is paid for.
    assert "does NOT contain it" in recorded["note"]
    assert "hop t+1" in recorded["note"]

    delay = int(manifest["streaming_delay_frames"]) * int(manifest["hop_length"])
    hop = int(manifest["hop_length"])
    samples = _guard_bait()

    on = StreamingOrt(onnx_path, provider="cpu")
    assert on.onset_guard == guard
    assert on.onset_guard_lookahead_hops >= 0

    # (a) An absent section means no guard -- the documented convention, and what
    # every export written before the guard existed carries.
    bare_path = tmp_path / "bare.json"
    bare = dict(json.loads(onnx_path.with_suffix(".json").read_text()))
    bare.pop(OnsetGuard.MANIFEST_KEY)
    bare_path.write_text(json.dumps(bare))
    off = StreamingOrt(onnx_path, bare_path, provider="cpu")
    assert off.onset_guard is None

    # (c) ... and a request can refuse a recorded guard, landing exactly where
    # its absence does.
    refused = StreamingOrt(
        onnx_path, provider="cpu", onset_guard_overrides={"enabled": False}
    )
    assert refused.onset_guard is None

    guarded = _stream(on, samples)
    unguarded = _stream(off, samples)
    assert np.array_equal(_stream(refused, samples), unguarded)
    assert not np.array_equal(guarded, unguarded), "the guard must do something"

    # (b) The streamed guard IS the offline guard, on the same audio, aligned by
    # the graph's own latency. Bit-identical: `OnsetGuard.apply` and the runtime
    # drive the same `_advance` recursion over the same float32 blend, so
    # anything short of equality would be a real difference and not rounding.
    expected = guard.apply(
        torch.from_numpy(unguarded[delay:]), torch.from_numpy(samples),
        hop=hop, sr=float(manifest["sample_rate"]),
    ).numpy()
    assert np.array_equal(guarded[delay:], expected), float(
        np.abs(guarded[delay:] - expected).max()
    )
    # Warm-up output has no input to restore and passes through, as the blend's
    # does.
    assert np.array_equal(guarded[:delay], unguarded[:delay])

    # (d) The anchor is per-stream: a second stream through the same runtime must
    # start protected again, or its first talker is handed straight to the model.
    assert np.array_equal(_stream(on, samples), guarded)
