"""The SDK's dry blend, against the module's.

`dry_blend` lives in `puresound.system.postprocess.Postprocessor` for the torch
path and, unavoidably, a second time in numpy inside the SDK -- the SDK must not
import puresound or torch (see `test_portable_streaming_sdk`), so it cannot reuse
the class. Two implementations of one formula need a test that compares them,
which is what most of this file is.

The part that is not a straight copy is the alignment. Every current DPCRN export
reports `streaming_delay_frames = 3`, so the graph's output at a given index
carries the input from 480 samples earlier; blending index-for-index would mix in
a slice of the mixture 30 ms away from the speech it is meant to relieve.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puresound.system.postprocess import Postprocessor

SDK_PATH = "sdk/python"
HOP, WIN, FFT, BINS = 160, 512, 512, 257


def _manifest(*, dry_blend=1.0, spec_floor=0.0, delay_frames=0):
    manifest = {
        "model_type": "dpcrn_streaming_frame",
        "processor": "stft_frame_ort",
        "sample_rate": 16000,
        "fft_length": FFT,
        "win_length": WIN,
        "hop_length": HOP,
        "freq_bins": BINS,
        "streaming_delay_frames": delay_frames,
        "state_input_names": ["state"],
        "state_output_names": ["next_state"],
        "output_names": ["enhanced_frame", "next_state"],
        "state_shapes": {"state": [1, 1]},
        "postprocess": {"dry_blend": dry_blend, "spec_floor": spec_floor},
    }
    return manifest


@pytest.fixture
def identity_runtime(tmp_path, monkeypatch):
    """An identity graph: the OLA then reconstructs its input exactly, so any
    difference in the output is the blend and nothing else."""

    class IdentitySession:
        def __init__(self, path, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

        def run(self, names, inputs):
            return [inputs["noisy_frame"], inputs["state"]]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=IdentitySession,
        ),
    )
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / SDK_PATH))

    def build(**manifest_kwargs):
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(b"fake")
        manifest_path = tmp_path / "m.json"
        manifest_path.write_text(json.dumps(_manifest(**manifest_kwargs)))
        from puresound_streaming import PureSoundStreamingRuntime

        return PureSoundStreamingRuntime(onnx_path, manifest_path, provider="cpu")

    return build


def _run(runtime, samples):
    return np.concatenate([runtime.process_samples(samples), runtime.flush()])


def _expected(unblended, samples, dry_blend, delay):
    """What the blend must produce, spelled out independently of the SDK.

    Output index ``n`` is blended against input index ``n - delay``, because that
    is the input the graph's output at ``n`` carries. Indices with no
    corresponding input -- the first ``delay`` samples, and the tail past the end
    of the input -- pass through unblended: there is nothing to blend them with,
    and attenuating them by ``dry_blend`` would be inventing a dry signal.
    """
    out = unblended.astype(np.float32, copy=True)
    for n in range(unblended.size):
        source = n - delay
        if 0 <= source < samples.size:
            out[n] = dry_blend * unblended[n] + (1.0 - dry_blend) * samples[source]
    return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# The two implementations of one formula
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dry_blend", [0.8, 0.9, 0.95])
def test_the_sdk_blend_matches_the_module_blend(identity_runtime, dry_blend):
    """With an identity graph and zero delay, enhanced == input, so the SDK's
    output must equal `Postprocessor.blend_waveform(input, input)` -- which is
    just the input again. The value of the test is that it fails if either side's
    arithmetic drifts, including the clamp."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(3200) * 0.2).astype(np.float32)

    got = _run(identity_runtime(dry_blend=dry_blend), samples)
    unblended = _run(identity_runtime(dry_blend=1.0), samples)

    # Against the module's own implementation over the span where both have a
    # reference -- that is the half of this that must not drift.
    span = slice(0, min(unblended.size, samples.size))
    module = (
        Postprocessor(dry_blend=dry_blend)
        .blend_waveform(
            torch.from_numpy(unblended[span]), torch.from_numpy(samples[span])
        )
        .numpy()
    )
    assert np.allclose(got[span], module, atol=1e-5), float(
        np.abs(got[span] - module).max()
    )
    # And against the spelled-out reference over the whole output, which also
    # covers the pass-through tail the module never sees.
    expected = _expected(unblended, samples, dry_blend, delay=0)
    assert np.allclose(got, expected, atol=1e-5), float(np.abs(got - expected).max())


def test_a_disabled_blend_costs_nothing(identity_runtime):
    rng = np.random.default_rng(1)
    samples = (rng.standard_normal(2000) * 0.2).astype(np.float32)
    assert np.array_equal(
        _run(identity_runtime(dry_blend=1.0), samples),
        _run(identity_runtime(), samples),
    )


def test_the_blend_output_stays_inside_full_scale(identity_runtime):
    """Same bound the module clamps to; the identity graph makes it reachable."""
    samples = np.full(2000, 0.9, dtype=np.float32)
    assert float(np.abs(_run(identity_runtime(dry_blend=0.9), samples)).max()) <= 1.0


# --------------------------------------------------------------------------- #
# The part that is not a copy: alignment
# --------------------------------------------------------------------------- #


def test_the_reference_comes_from_the_graphs_own_latency_back(identity_runtime):
    """Every shipped DPCRN export reports 3 frames of look-ahead.

    Constructed so a misalignment cannot hide: the input is an impulse train
    every hop, so blending against the wrong hop lands the dry copy in a
    different sample than the enhanced one. The identity graph means enhanced ==
    input at the same index, so with the delay compensated the blend re-adds the
    dry copy exactly on top of itself and the output equals the unblended run.
    """
    samples = np.zeros(3200, dtype=np.float32)
    samples[::HOP] = 0.5

    unblended = _run(identity_runtime(delay_frames=3), samples)
    blended = _run(identity_runtime(dry_blend=0.9, delay_frames=3), samples)

    # The graph is identity, so its "3-frame latency" is a claim in the manifest
    # rather than real behaviour -- the SDK compensates for it, which shifts the
    # dry copy 480 samples early relative to the enhanced signal. That mismatch
    # is exactly what this test measures: the blend must NOT be index-aligned.
    naive = _expected(unblended, samples, 0.9, delay=0)
    assert not np.allclose(blended, naive, atol=1e-4), (
        "the SDK blended index-for-index; it must offset by streaming_delay_frames"
    )

    expected = _expected(unblended, samples, 0.9, delay=3 * HOP)
    assert np.allclose(blended, expected, atol=1e-5), float(
        np.abs(blended - expected).max()
    )


def test_chunk_size_does_not_change_the_blended_output(identity_runtime):
    """The blend keeps its own history and index counter, so feeding the same
    audio in different-sized chunks must still land the dry copy identically."""
    rng = np.random.default_rng(2)
    samples = (rng.standard_normal(3000) * 0.2).astype(np.float32)

    whole = _run(identity_runtime(dry_blend=0.9, delay_frames=3), samples)

    chunked = identity_runtime(dry_blend=0.9, delay_frames=3)
    parts = [
        chunked.process_samples(samples[:7]),
        chunked.process_samples(samples[7:913]),
        chunked.process_samples(samples[913:]),
        chunked.flush(),
    ]
    assert np.allclose(whole, np.concatenate(parts), atol=1e-6)


# --------------------------------------------------------------------------- #
# What the manifest is allowed to ask for
# --------------------------------------------------------------------------- #


def test_a_manifest_asking_for_spec_floor_is_refused(
    identity_runtime, tmp_path, monkeypatch
):
    """It is 0.0 in every shipped invocation and unimplemented in either runtime.
    Refusing beats the silent no-op the module used to have on four of its mask
    types -- and both copies have to refuse, or the two disagree."""
    from puresound.streaming.base import StreamingOrt

    with pytest.raises(ValueError, match="spec_floor"):
        identity_runtime(spec_floor=0.2)

    onnx_path = tmp_path / "floored.onnx"
    onnx_path.write_bytes(b"fake")
    manifest_path = tmp_path / "floored.json"
    manifest_path.write_text(json.dumps(_manifest(spec_floor=0.2)))
    with pytest.raises(ValueError, match="spec_floor"):
        StreamingOrt(onnx_path, manifest_path, provider="cpu")


@pytest.mark.parametrize("dry_blend", [0.0, -0.1, 1.5])
def test_a_manifest_with_an_unusable_blend_is_refused(
    identity_runtime, tmp_path, dry_blend
):
    from puresound.streaming.base import StreamingOrt

    with pytest.raises(ValueError, match="dry_blend"):
        identity_runtime(dry_blend=dry_blend)

    onnx_path = tmp_path / "bad.onnx"
    onnx_path.write_bytes(b"fake")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text(json.dumps(_manifest(dry_blend=dry_blend)))
    with pytest.raises(ValueError, match="dry_blend"):
        StreamingOrt(onnx_path, manifest_path, provider="cpu")


def test_an_export_records_what_the_runtime_will_apply():
    """The export writes the setting and the runtime reads that key: one name,
    checked from both ends, so a rename cannot leave the runtime silently at 1.0.
    """
    sys.path.insert(0, SDK_PATH)
    from puresound_streaming.runtime import StreamingRuntimeConfig

    recorded = Postprocessor(dry_blend=0.9).as_manifest()
    manifest = _manifest()
    manifest["postprocess"] = recorded
    config = StreamingRuntimeConfig.from_manifest(manifest)
    assert config.dry_blend == pytest.approx(0.9)
    assert recorded["suppression_ceiling_db"] == pytest.approx(-20.0)


# --------------------------------------------------------------------------- #
# The two runtimes are one loop written twice
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dry_blend,delay_frames", [(1.0, 0), (0.9, 0), (0.9, 3)])
def test_both_runtimes_produce_the_same_output(
    identity_runtime, tmp_path, monkeypatch, dry_blend, delay_frames
):
    """`puresound.streaming.base.StreamingOrt` and the SDK's
    `StftFrameOrtProcessor` are the same frame loop written twice.

    The duplication is deliberate -- the SDK must not import puresound or torch --
    but deliberate is not the same as free: `_add_ola_frame` is byte-identical
    between them and `flush` / `_process_frame` / `process_samples` are all above
    89%, so a fix to one silently misses the other. This is the only thing tying
    them together, and it is what makes the second copy safe rather than
    merely intentional.
    """
    from puresound.streaming.base import StreamingOrt

    onnx_path = tmp_path / "shared.onnx"
    onnx_path.write_bytes(b"fake")
    manifest_path = tmp_path / "shared.json"
    manifest_path.write_text(
        json.dumps(_manifest(dry_blend=dry_blend, delay_frames=delay_frames))
    )

    rng = np.random.default_rng(3)
    samples = (rng.standard_normal(2400) * 0.2).astype(np.float32)

    # `identity_runtime` already stubbed onnxruntime into sys.modules, so both
    # runtimes resolve the same identity session.
    sdk = identity_runtime(dry_blend=dry_blend, delay_frames=delay_frames)
    in_repo = StreamingOrt(onnx_path, manifest_path, provider="cpu")

    from_sdk = _run(sdk, samples)
    from_repo = np.concatenate(
        [in_repo.process_samples(samples), in_repo.flush()]
    )
    assert from_sdk.shape == from_repo.shape
    assert np.allclose(from_sdk, from_repo, atol=1e-6), float(
        np.abs(from_sdk - from_repo).max()
    )


@pytest.mark.parametrize("runtime_kind", ["sdk", "in_repo"])
def test_a_manifest_predating_the_field_says_so_out_loud(
    identity_runtime, tmp_path, monkeypatch, runtime_kind
):
    """An export from before `postprocess` existed looks exactly like one that
    deliberately asked for no relief, and the five checked-in DPCRN manifests are
    all of the first kind.

    The runtime cannot tell them apart, and guessing 0.9 would be inventing an
    intent the repo only records for two of the five versions. So it warns, at
    the point where it matters, and says how to make the artefact self-describing.
    """
    from puresound.streaming.base import StreamingOrt

    manifest = _manifest()
    del manifest["postprocess"]
    onnx_path = tmp_path / "old.onnx"
    onnx_path.write_bytes(b"fake")
    manifest_path = tmp_path / "old.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.warns(RuntimeWarning, match="postprocess"):
        if runtime_kind == "in_repo":
            built = StreamingOrt(onnx_path, manifest_path, provider="cpu")
            assert built.dry_blend == 1.0
        else:
            from puresound_streaming.runtime import StreamingRuntimeConfig

            assert StreamingRuntimeConfig.from_manifest(manifest).dry_blend == 1.0


def test_an_export_that_wants_no_relief_is_not_a_warning(identity_runtime):
    """`dry_blend: 1.0` written explicitly is a decision, not a missing field."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        identity_runtime(dry_blend=1.0)
