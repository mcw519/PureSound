"""The SDK's post-graph stages, against the module's.

`dry_blend` lives in `puresound.system.postprocess.Postprocessor` for the torch
path and, unavoidably, a second time in numpy inside the SDK -- the SDK must not
import puresound or torch (see `test_portable_streaming_sdk`), so it cannot reuse
the class. Two implementations of one formula need a test that compares them,
which is what most of this file is. `OnsetGuard` is the second such formula and
the bottom half of the file does the same job for it.

The part that is not a straight copy is the alignment. Every current DPCRN export
reports `streaming_delay_frames = 3`, so the graph's output at a given index
carries the input from 480 samples earlier; blending index-for-index would mix in
a slice of the mixture 30 ms away from the speech it is meant to relieve. The
guard reads the same alignment for the same reason, plus one of its own: its
frame energy spans two hops, so a frame is only decided once the following hop
has arrived.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puresound.system.onset_guard import OnsetGuard
from puresound.system.postprocess import Postprocessor

SDK_PATH = "sdk/python"
HOP, WIN, FFT, BINS = 160, 512, 512, 257


def _manifest(
    *,
    dry_blend=1.0,
    spec_floor=0.0,
    delay_frames=0,
    onset_guard=None,
    hop=HOP,
    win=WIN,
):
    manifest = {
        "model_type": "dpcrn_streaming_frame",
        "processor": "stft_frame_ort",
        "sample_rate": 16000,
        "fft_length": FFT,
        "win_length": win,
        "hop_length": hop,
        "freq_bins": BINS,
        "streaming_delay_frames": delay_frames,
        "state_input_names": ["state"],
        "state_output_names": ["next_state"],
        "output_names": ["enhanced_frame", "next_state"],
        "state_shapes": {"state": [1, 1]},
        "recommended_inference": {"dry_blend": dry_blend, "spec_floor": spec_floor},
    }
    if onset_guard is not None:
        manifest[OnsetGuard.MANIFEST_KEY] = dict(onset_guard)
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
    manifest[Postprocessor.MANIFEST_KEY] = recorded
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
def test_an_absent_section_means_no_relief(
    identity_runtime, tmp_path, monkeypatch, runtime_kind
):
    """The documented convention, and what `Postprocessor()` defaults to.

    Every export still in the catalog carries the section with `dry_blend: 0.9`,
    so the absent case is exercised here rather than by a shipped manifest.
    """
    from puresound.streaming.base import StreamingOrt

    manifest = _manifest()
    del manifest["recommended_inference"]
    onnx_path = tmp_path / "bare.onnx"
    onnx_path.write_bytes(b"fake")
    manifest_path = tmp_path / "bare.json"
    manifest_path.write_text(json.dumps(manifest))

    if runtime_kind == "in_repo":
        assert StreamingOrt(onnx_path, manifest_path, provider="cpu").dry_blend == 1.0
    else:
        from puresound_streaming.runtime import StreamingRuntimeConfig

        assert StreamingRuntimeConfig.from_manifest(manifest).dry_blend == 1.0


def test_both_runtimes_read_the_same_manifest_key():
    """The key name is duplicated -- the SDK cannot import puresound -- so a
    rename on one side has to fail here rather than silently leaving that runtime
    at 1.0."""
    sys.path.insert(0, SDK_PATH)
    import inspect

    from puresound_streaming.runtime import StreamingRuntimeConfig

    from puresound.system.postprocess import Postprocessor

    assert Postprocessor.MANIFEST_KEY == "recommended_inference"
    assert (
        f'manifest.get("{Postprocessor.MANIFEST_KEY}")'
        in inspect.getsource(StreamingRuntimeConfig.from_manifest)
    )


# --------------------------------------------------------------------------- #
# The onset guard: the second formula written twice
# --------------------------------------------------------------------------- #


class ScaleSession:
    """A graph that scales, so the guard's effect is visible.

    An identity graph cannot see the guard at all: it hands back the dry signal,
    and `g*dry + (1 - g)*dry == dry` for every gain. Scaling is the smallest
    graph that separates "the model's output" from "the input" -- 0.1 for a
    20 dB suppressor, and above 1.0 for a graph whose output leaves full scale,
    which is how a test can see whether the clamp ran.
    """

    def __init__(self, path=None, providers=None, scale=0.1):
        self.scale = scale

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, inputs):
        return [inputs["noisy_frame"] * self.scale, inputs["state"]]


#: Same operating point as the default guard except for the release, which is
#: shortened so the integrator reaches EXACTLY 0 inside a test-length burst (at
#: tau_dn 2.0 s it takes 2*ln(1/snap) = 13.8 s of speech to snap). The endpoints
#: are the part worth pinning: 1.0 is the input bit for bit, 0.0 is the model's
#: output untouched.
GUARD_SPEC = dict(OnsetGuard(tau_dn_s=0.2).as_manifest())


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    """Build either runtime over the same manifest and the same scaling graph."""
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / SDK_PATH))

    def build(kind="in_repo", overrides=None, scale=0.1, **manifest_kwargs):
        onnx_path = tmp_path / f"{kind}.onnx"
        onnx_path.write_bytes(b"fake")
        manifest_path = tmp_path / f"{kind}.json"
        manifest_path.write_text(json.dumps(_manifest(**manifest_kwargs)))
        session = ScaleSession(scale=scale)
        if kind == "in_repo":
            from puresound.streaming.base import StreamingOrt

            return StreamingOrt(
                onnx_path,
                manifest_path,
                provider="cpu",
                onset_guard_overrides=overrides,
                session=session,
            )
        assert overrides is None, "the SDK runtime takes no per-request overrides"
        from puresound_streaming import PureSoundStreamingRuntime

        return PureSoundStreamingRuntime(onnx_path, manifest_path, session=session)

    return build


def _inside(runtime):
    """The object that owns the buffers, either side.

    The SDK splits the loop into a runtime and a processor and forwards only the
    geometry; everything the guard keeps lives on the processor, so a test
    reading internals has to ask the right one.
    """
    return getattr(runtime, "processor", runtime)


def _speech_like(seconds=10.0, sr=16000, seed=11):
    """Modulated noise well over a floor, twice, with a long gap between.

    Modulated on purpose: the floor tracker absorbs stationary broadband energy
    by design (see `onset_guard`'s module docstring), so constant loud noise
    would stop counting as speech and the guard would never arm. The 5.5 s gap
    is longer than `t_forget_s`, so the guard arms, releases, re-protects and
    releases again inside one stream.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for lo, hi in ((1.0, 3.0), (8.5, 10.0)):
        span = slice(int(lo * sr), int(hi * sr))
        env[span] = 0.5 * (1.0 + np.sin(2.0 * np.pi * 5.0 * t[span])) + 0.2
    return ((rng.standard_normal(n) * 1e-3) * (1.0 + env * 10.0 ** (30 / 20))).astype(
        np.float32
    )


def _guard_gains(samples, spec=None):
    return OnsetGuard.from_manifest(spec or GUARD_SPEC).frame_gain(
        samples.astype(np.float64), hop=HOP, sr=16000.0
    )


def test_the_fixture_audio_actually_arms_and_re_arms_the_guard():
    """Without this the guard tests below could pass on a dead detector."""
    gains = _guard_gains(_speech_like())
    assert gains[0] == 1.0, "must start protected"
    assert (gains == 0.0).any(), "must hand over completely at least once"
    released = int(np.argmax(gains == 0.0))
    assert (gains[released:] == 1.0).any(), "must protect a later onset again"


@pytest.mark.parametrize("delay_frames", [0, 3])
def test_the_streamed_guard_is_the_offline_guard_on_the_streamed_output(
    guarded, delay_frames
):
    """The contract: streaming == offline, up to the graph's own latency.

    Compared against the module's own `apply` on the runtime's OWN unguarded
    output, which takes the graph out of the comparison entirely -- what is left
    is the guard's arithmetic and its alignment, and both are bit-exact rather
    than approximately right.
    """
    samples = _speech_like()
    delay = delay_frames * HOP

    unguarded = _run(guarded(delay_frames=delay_frames, dry_blend=0.9), samples)
    guarded_out = _run(
        guarded(delay_frames=delay_frames, dry_blend=0.9, onset_guard=GUARD_SPEC),
        samples,
    )
    assert guarded_out.shape == unguarded.shape

    # Output sample n carries input sample n - delay, so the offline face sees
    # the streamed output with those first `delay` warm-up samples dropped.
    expected = (
        OnsetGuard.from_manifest(GUARD_SPEC)
        .apply(
            torch.from_numpy(unguarded[delay:]),
            torch.from_numpy(samples),
            hop=HOP,
            sr=16000.0,
        )
        .numpy()
    )
    assert np.array_equal(guarded_out[delay:], expected), float(
        np.abs(guarded_out[delay:] - expected).max()
    )
    # And the warm-up samples, which have no input to restore, are untouched.
    assert np.array_equal(guarded_out[:delay], unguarded[:delay])


def test_the_guard_gain_comes_from_the_frame_the_input_sample_sits_in(guarded):
    """Alignment, spelled out independently of either runtime.

    A guard that ignored `streaming_delay_frames` would apply frame k's gain to
    output hop k instead of hop k + 3 -- 30 ms of the wrong decision on every
    onset and release, which is exactly where the gain is moving.
    """
    samples = _speech_like()
    unguarded = _run(guarded(delay_frames=3, dry_blend=0.9), samples)
    got = _run(guarded(delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC), samples)

    gains = _guard_gains(samples)
    expected = unguarded.astype(np.float32, copy=True)
    for n in range(unguarded.size):
        source = n - 3 * HOP
        if 0 <= source < samples.size:
            g = np.float32(gains[source // HOP])
            expected[n] = g * samples[source] + (1.0 - g) * unguarded[n]
    expected = np.clip(expected, -1.0, 1.0)
    assert np.allclose(got, expected, atol=1e-6), float(np.abs(got - expected).max())

    # The same expectation without the offset must NOT match, or the test would
    # pass on a runtime that blends index-for-index.
    naive = unguarded.astype(np.float32, copy=True)
    for n in range(min(unguarded.size, samples.size)):
        g = np.float32(gains[n // HOP])
        naive[n] = g * samples[n] + (1.0 - g) * unguarded[n]
    assert not np.allclose(got, np.clip(naive, -1.0, 1.0), atol=1e-4)


def test_both_runtimes_produce_the_same_guarded_output(guarded):
    """`OnsetGuard` is transcribed into the SDK, numpy-only, so this is the only
    thing keeping the two copies of the recursion together -- the same job
    `test_both_runtimes_produce_the_same_output` does for the blend, and the
    reason the second copy is safe rather than merely intentional."""
    samples = _speech_like()
    from_repo = _run(
        guarded("in_repo", delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC),
        samples,
    )
    from_sdk = _run(
        guarded("sdk", delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC), samples
    )
    assert from_sdk.shape == from_repo.shape
    assert np.array_equal(from_sdk, from_repo), float(
        np.abs(from_sdk - from_repo).max()
    )


@pytest.mark.parametrize("kind", ["in_repo", "sdk"])
def test_chunk_size_does_not_change_the_guarded_output(guarded, kind):
    """The guard consumes the dry stream in whole hops of its own, so arbitrary
    chunk boundaries must not move a single gain."""
    samples = _speech_like(seconds=4.0)
    whole = _run(
        guarded(kind, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC), samples
    )

    chunked = guarded(kind, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC)
    parts = [
        chunked.process_samples(samples[:13]),
        chunked.process_samples(samples[13:997]),
        chunked.process_samples(samples[997:]),
        chunked.flush(),
    ]
    assert np.array_equal(whole, np.concatenate(parts))


@pytest.mark.parametrize("kind", ["in_repo", "sdk"])
def test_the_guard_keeps_the_dry_stream_even_where_the_blend_would_not(guarded, kind):
    """`dry_blend >= 1.0` is the blend's no-op, and it used to skip the history.

    The guard needs the input regardless of what the blend does with it, so the
    early return now depends on BOTH stages being off -- and while both are off
    the graph's output must still come back untouched, `np.clip` included, which
    an amplifying graph makes visible.
    """
    samples = np.full(3200, 0.9, dtype=np.float32)

    unguarded = _inside(guarded(kind, dry_blend=1.0, scale=2.0))
    out = _run(unguarded, samples)
    assert unguarded.dry_history.size == 0, "no stage needs it; do not keep it"
    assert unguarded.guard_hops_fed == 0
    assert float(np.abs(out).max()) > 1.0, "the untouched path must not clamp"

    kept = _inside(guarded(kind, dry_blend=1.0, scale=2.0, onset_guard=GUARD_SPEC))
    head = kept.process_samples(samples[:1600])
    assert kept.dry_history.size > 0, "the guard reads the dry stream"
    assert kept.guard_hops_fed == 1600 // HOP
    guarded_out = np.concatenate(
        [head, kept.process_samples(samples[1600:]), kept.flush()]
    )
    # Constant broadband energy IS the floor, so the guard stays protected the
    # whole way through and hands back the input -- which is also in range.
    assert float(np.abs(guarded_out).max()) <= 1.0
    n = min(samples.size, guarded_out.size)
    assert np.array_equal(guarded_out[:n], samples[:n])


def test_a_recorded_guard_can_be_turned_off_per_request(guarded):
    """The operating point ships in the manifest; a request can still refuse it,
    exactly as `postprocess_overrides` can refuse the recorded blend."""
    samples = _speech_like(seconds=4.0)
    recorded = guarded(delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC)
    disabled = guarded(
        delay_frames=3,
        dry_blend=0.9,
        onset_guard=GUARD_SPEC,
        overrides={"enabled": False},
    )
    absent = guarded(delay_frames=3, dry_blend=0.9)

    assert disabled.onset_guard is None
    assert np.array_equal(_run(disabled, samples), _run(absent, samples))
    assert not np.array_equal(_run(recorded, samples), _run(absent, samples))


def test_knob_overrides_replace_the_recorded_values_and_arm_an_absent_guard(guarded):
    samples = _speech_like(seconds=4.0)
    # A 60 dB activity threshold: nothing in the fixture clears it, so the guard
    # never arms and the output is the input over the whole aligned span.
    deaf = guarded(
        delay_frames=3,
        dry_blend=0.9,
        onset_guard=GUARD_SPEC,
        overrides={"margin_db": 60.0},
    )
    assert deaf.onset_guard.margin_db == 60.0
    assert deaf.onset_guard.tau_dn_s == GUARD_SPEC["tau_dn_s"], "unnamed knobs stay"
    out = _run(deaf, samples)
    n = min(samples.size, out.size - 3 * HOP)
    assert np.array_equal(out[3 * HOP : 3 * HOP + n], samples[:n])

    # Knobs on a manifest that records no guard turn it on: a caller spelling an
    # operating point should not have to say "enabled" as well.
    armed = guarded(delay_frames=3, overrides={"margin_db": 8.0})
    assert armed.onset_guard is not None and armed.onset_guard.margin_db == 8.0
    # ... and an explicit flag still wins over the knobs.
    assert (
        guarded(
            delay_frames=3, overrides={"margin_db": 8.0, "enabled": False}
        ).onset_guard
        is None
    )


def test_an_unknown_override_is_refused(guarded):
    with pytest.raises(ValueError, match="onset_guard override"):
        guarded(delay_frames=3, overrides={"t_arm": 1.0})


@pytest.mark.parametrize("kind", ["in_repo", "sdk"])
def test_reset_starts_the_next_stream_protected_again(guarded, kind):
    """The anchor is per-stream. A guard state that survived `reset` would hand
    the next stream's first talker straight to the model -- the very deletion the
    guard exists to prevent.
    """
    samples = _speech_like(seconds=4.0)
    runtime = guarded(kind, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC)
    first = _run(runtime, samples)
    runtime.reset()
    second = _run(runtime, samples)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("kind", ["in_repo", "sdk"])
def test_an_absent_onset_guard_section_means_no_guard(guarded, kind):
    """The documented convention, the same one the postprocess section follows:
    every export written before the guard existed must stay unguarded."""
    runtime = _inside(guarded(kind, delay_frames=3, dry_blend=0.9))
    assert runtime.onset_guard is None
    assert runtime.guard_state is None


@pytest.mark.parametrize("kind", ["in_repo", "sdk"])
def test_a_geometry_that_cannot_supply_the_analysis_hop_is_refused(guarded, kind):
    """The guard's frame t needs dry hop t+1, and the runtime holds
    ``streaming_delay_frames + win_length//hop_length - 2`` spare hops when it
    emits an output hop.

    At the shipped 512/160 geometry that is 4 with the released look-ahead and
    still 1 for a causal export -- the analysis window alone covers the lag, so
    the exact gain is applied and NO latency is added. Only a window shorter
    than two hops on a zero-latency graph comes up short, and that is refused
    rather than served a gain from the wrong frame.
    """
    ok = _inside(guarded(kind, delay_frames=0, win=2 * HOP, onset_guard=GUARD_SPEC))
    assert ok.onset_guard_lookahead_hops == 0
    assert _inside(
        guarded(kind, delay_frames=3, onset_guard=GUARD_SPEC)
    ).onset_guard_lookahead_hops == 4

    with pytest.raises(ValueError, match="look-ahead"):
        guarded(kind, delay_frames=0, win=HOP, onset_guard=GUARD_SPEC)
    # One frame of graph latency buys back exactly the hop the window lost.
    assert _inside(
        guarded(kind, delay_frames=1, win=HOP, onset_guard=GUARD_SPEC)
    ).onset_guard_lookahead_hops == 0


def test_both_runtimes_read_the_same_onset_guard_manifest_key():
    """The key name is duplicated -- the SDK cannot import puresound -- so a
    rename on one side has to fail here rather than silently leaving that runtime
    unguarded."""
    sys.path.insert(0, SDK_PATH)
    import inspect

    from puresound_streaming.runtime import StreamingRuntimeConfig

    assert OnsetGuard.MANIFEST_KEY == "onset_guard"
    assert (
        f'manifest.get("{OnsetGuard.MANIFEST_KEY}")'
        in inspect.getsource(StreamingRuntimeConfig.from_manifest)
    )


def test_the_sdk_knob_list_matches_the_modules():
    """`from_manifest` filters by this list on both sides, so a knob added to the
    module and not to the SDK would be silently dropped there."""
    sys.path.insert(0, SDK_PATH)
    from puresound_streaming.runtime import _ONSET_GUARD_KNOBS

    assert set(_ONSET_GUARD_KNOBS) == set(OnsetGuard.__dataclass_fields__)
    assert OnsetGuard.from_manifest(GUARD_SPEC).as_manifest() == GUARD_SPEC
