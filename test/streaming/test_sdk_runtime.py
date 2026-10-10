"""The portable SDK runtime (`sdk/python/puresound_streaming`), against the
in-repo runtime and the modules it transcribes.

The SDK must not import puresound or torch, so the post-graph stages live in it
a second time, in numpy: `dry_blend` (`puresound.system.postprocess.Postprocessor`)
and the onset guard (`puresound.system.onset_guard.OnsetGuard`), inside a copy
of `puresound.streaming.base.StreamingOrt`'s frame loop. Two implementations of
one formula need a test that compares them, which is most of this file.

The part that is not a straight copy is the alignment. A look-ahead export
reports `streaming_delay_frames`, so the graph's output at a given index
carries the input from that many hops earlier; blending index-for-index would
mix in a slice of the mixture 30 ms away from the speech it is meant to
relieve. The guard reads the same alignment, plus one lag of its own: its frame
energy spans two hops, so a frame is only decided once the following hop has
arrived.
"""

import importlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puresound.system.onset_guard import OnsetGuard
from puresound.system.postprocess import Postprocessor

SDK_ROOT = Path(__file__).resolve().parents[2] / "sdk" / "python"
HOP, WIN, FFT, BINS = 160, 512, 512, 257
KINDS = ["in_repo", "sdk"]

#: The default operating point with a shortened release, so the integrator
#: reaches EXACTLY 0 inside a test-length burst. The endpoints are the part
#: worth pinning: 1.0 is the input bit for bit, 0.0 is the model's output.
GUARD_SPEC = dict(OnsetGuard(tau_dn_s=0.2).as_manifest())


def _manifest(
    *,
    dry_blend=1.0,
    spec_floor=0.0,
    mix_phase=False,
    delay_frames=0,
    onset_guard=None,
    hop=HOP,
    win=WIN,
    processor="stft_frame_ort",
):
    manifest = {
        "model_type": "dpcrn_streaming_frame",
        "processor": processor,
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
        Postprocessor.MANIFEST_KEY: {
            "dry_blend": dry_blend, "spec_floor": spec_floor, "mix_phase": mix_phase,
        },
    }
    if onset_guard is not None:
        manifest[OnsetGuard.MANIFEST_KEY] = dict(onset_guard)
    return manifest


class ScaleSession:
    """A graph that scales its input frame.

    1.0 is the identity: the overlap-add then reconstructs the input, so any
    difference in the output is a post-graph stage. The identity cannot see the
    guard at all (`g*dry + (1 - g)*dry == dry`), so guard tests use 0.1 -- a
    20 dB suppressor -- or 2.0, whose output leaves full scale and so shows
    whether the clamp ran.
    """

    def __init__(self, path=None, providers=None, scale=1.0):
        self.scale = scale
        self.providers = providers or ["CPUExecutionProvider"]

    def get_providers(self):
        return self.providers

    def run(self, names, inputs):
        return [inputs["noisy_frame"] * self.scale, inputs["state"]]


class LookaheadSession:
    """A look-ahead graph: emits the frame it was fed ``DELAY`` calls ago, times
    a fixed per-bin gain, so the first ``DELAY`` outputs are warm-up zeros.

    A per-bin gain is a filter, so unlike `ScaleSession` the output is not a
    windowed frame: it has energy where the synthesis window is near zero, as a
    real denoiser's does. That is what makes an under-covered overlap-add
    sample visible -- dividing it by the near-zero window sum blows it up.
    """

    DELAY = 3

    def __init__(self, gain):
        self.gain = np.asarray(gain, dtype=np.float32).reshape(1, BINS, 1)

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, inputs):
        history = inputs["history"]
        enhanced = history[0] * self.gain
        return [enhanced, np.concatenate([history[1:], inputs["noisy_frame"][None]])]


def _lookahead_manifest(**kwargs):
    manifest = _manifest(delay_frames=LookaheadSession.DELAY, **kwargs)
    manifest.update(
        state_input_names=["history"],
        state_output_names=["next_history"],
        output_names=["enhanced_frame", "next_history"],
        state_shapes={"history": [LookaheadSession.DELAY, 1, BINS, 2]},
    )
    return manifest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Build either runtime over the same manifest and the same scaling graph."""
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    count = iter(range(1000))

    def build(kind="in_repo", overrides=None, scale=1.0, manifest=None, session=None,
              **manifest_kwargs):
        stem = tmp_path / f"{kind}{next(count)}"
        onnx_path = stem.with_suffix(".onnx")
        onnx_path.write_bytes(b"fake")
        manifest_path = stem.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest or _manifest(**manifest_kwargs)))
        session = session or ScaleSession(scale=scale)
        if kind == "in_repo":
            from puresound.streaming.base import StreamingOrt

            return StreamingOrt(onnx_path, manifest_path, provider="cpu",
                                onset_guard_overrides=overrides, session=session)
        assert overrides is None, "the SDK runtime takes no per-request overrides"
        from puresound_streaming import PureSoundStreamingRuntime

        return PureSoundStreamingRuntime(onnx_path, manifest_path, session=session)

    return build


def _inside(runtime):
    """The object that owns the buffers: the SDK splits the loop into a runtime
    and a processor, and everything the stages keep lives on the processor."""
    return getattr(runtime, "processor", runtime)


def _run(runtime, samples):
    return np.concatenate([runtime.process_samples(samples), runtime.flush()])


def _noise(n, seed):
    return (np.random.default_rng(seed).standard_normal(n) * 0.2).astype(np.float32)


def _speech_like(seconds=10.0, sr=16000, seed=11):
    """Modulated noise well over a floor, twice, with a long gap between.

    Modulated on purpose: the floor tracker absorbs stationary broadband energy
    by design, so constant loud noise would stop counting as speech and the
    guard would never arm. The 5.5 s gap is longer than `t_forget_s`, so the
    guard arms, releases, re-protects and releases again inside one stream.
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


def _expected_blend(unblended, samples, dry_blend, delay):
    """What the blend must produce, spelled out independently of the SDK.

    Output index ``n`` is blended against input index ``n - delay``, because that
    is the input the graph's output at ``n`` carries. Indices with no
    corresponding input -- the first ``delay`` samples, which are warm-up -- pass
    through unblended: attenuating them by ``dry_blend`` would be inventing a dry
    signal.
    """
    out = unblended.astype(np.float32, copy=True)
    for n in range(unblended.size):
        source = n - delay
        if 0 <= source < samples.size:
            out[n] = dry_blend * unblended[n] + (1.0 - dry_blend) * samples[source]
    return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# The portable package itself
# --------------------------------------------------------------------------- #


def test_the_sdk_imports_without_puresound(monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    before = {name for name in sys.modules if name == "puresound" or name.startswith("puresound.")}

    module = importlib.import_module("puresound_streaming")

    after = {name for name in sys.modules if name == "puresound" or name.startswith("puresound.")}
    assert hasattr(module, "PureSoundStreamingRuntime")
    assert hasattr(module, "StftFrameOrtProcessor")
    assert not hasattr(module, "DparnStreamingRuntime")
    assert after == before


def test_the_sdk_opens_its_own_session_and_speaks_int16(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_ROOT))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=ScaleSession,
    ))
    from puresound_streaming import PureSoundStreamingRuntime

    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"fake")
    onnx_path.with_suffix(".json").write_text(json.dumps(_manifest()))
    sdk = PureSoundStreamingRuntime(onnx_path, provider="cpu")
    assert sdk.providers == ["CPUExecutionProvider"]

    assert sdk.process_int16(np.zeros(600, dtype=np.int16)).dtype == np.int16
    assert sdk.flush_int16().dtype == np.int16


@pytest.mark.parametrize(
    "manifest_kwargs,match,kinds",
    [
        # 0.0 in every shipped invocation and unimplemented in either runtime:
        # refusing beats a silent no-op, and both copies have to refuse
        (dict(spec_floor=0.2), "spec_floor", KINDS),
        # neither runtime re-analyses the finished output
        (dict(mix_phase=True), "mix_phase", KINDS),
        (dict(dry_blend=0.0), "dry_blend", KINDS),
        (dict(dry_blend=-0.1), "dry_blend", KINDS),
        (dict(dry_blend=1.5), "dry_blend", KINDS),
        (dict(processor="future_model"),
         "(?s)Unsupported streaming processor.*stft_frame_ort", ["sdk"]),
    ],
)
def test_a_manifest_the_runtime_cannot_honour_is_refused(
    runtime, manifest_kwargs, match, kinds
):
    for kind in kinds:
        with pytest.raises(ValueError, match=match):
            runtime(kind, **manifest_kwargs)


def test_the_sdk_reads_the_manifest_keys_and_knobs_the_module_writes():
    """The key names and the guard's knob list are duplicated -- the SDK cannot
    import puresound -- so a rename or an added knob on one side has to fail
    here rather than silently leave that runtime at dry_blend 1.0 or
    unguarded."""
    sys.path.insert(0, str(SDK_ROOT))
    from puresound_streaming.runtime import _ONSET_GUARD_KNOBS, StreamingRuntimeConfig

    source = inspect.getsource(StreamingRuntimeConfig.from_manifest)
    assert Postprocessor.MANIFEST_KEY == "recommended_inference"
    assert OnsetGuard.MANIFEST_KEY == "onset_guard"
    for key in (Postprocessor.MANIFEST_KEY, OnsetGuard.MANIFEST_KEY):
        assert f'manifest.get("{key}")' in source, key

    recorded = Postprocessor(dry_blend=0.9).as_manifest()
    assert recorded["suppression_ceiling_db"] == pytest.approx(-20.0)
    manifest = _manifest()
    manifest[Postprocessor.MANIFEST_KEY] = recorded
    assert StreamingRuntimeConfig.from_manifest(manifest).dry_blend == pytest.approx(0.9)

    assert set(_ONSET_GUARD_KNOBS) == set(OnsetGuard.__dataclass_fields__)
    assert OnsetGuard.from_manifest(GUARD_SPEC).as_manifest() == GUARD_SPEC


@pytest.mark.parametrize("kind", KINDS)
def test_an_absent_section_means_that_stage_is_off(runtime, kind):
    """The documented convention, and what `Postprocessor()` and no guard
    default to: every export written before a stage existed stays without it."""
    manifest = _manifest()
    del manifest[Postprocessor.MANIFEST_KEY]
    inside = _inside(runtime(kind, manifest=manifest))
    assert inside.dry_blend == 1.0
    assert inside.onset_guard is None
    assert inside.guard_state is None


# --------------------------------------------------------------------------- #
# dry_blend
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dry_blend", [0.8, 0.9, 0.95])
def test_the_sdk_blend_matches_the_module_blend(runtime, dry_blend):
    """With an identity graph and zero delay, enhanced == input. The value of
    the test is that it fails if either side's arithmetic drifts, the clamp
    included."""
    samples = _noise(3200, seed=0)

    got = _run(runtime("sdk", dry_blend=dry_blend), samples)
    unblended = _run(runtime("sdk"), samples)

    # Against the module over the span where both have a reference ...
    span = slice(0, min(unblended.size, samples.size))
    module = (
        Postprocessor(dry_blend=dry_blend)
        .blend_waveform(torch.from_numpy(unblended[span]), torch.from_numpy(samples[span]))
        .numpy()
    )
    assert np.allclose(got[span], module, atol=1e-5), float(np.abs(got[span] - module).max())
    # ... and against the spelled-out reference over the whole output.
    expected = _expected_blend(unblended, samples, dry_blend, delay=0)
    assert np.allclose(got, expected, atol=1e-5), float(np.abs(got - expected).max())


def test_the_blend_reference_comes_from_the_graphs_own_latency_back(runtime):
    """Constructed so a misalignment cannot hide: an impulse every hop, so
    blending against the wrong hop lands the dry copy in a different sample
    than the enhanced one."""
    samples = np.zeros(3200, dtype=np.float32)
    samples[::HOP] = 0.5

    unblended = _run(runtime("sdk", delay_frames=3), samples)
    blended = _run(runtime("sdk", dry_blend=0.9, delay_frames=3), samples)

    naive = _expected_blend(unblended, samples, 0.9, delay=0)
    assert not np.allclose(blended, naive, atol=1e-4), (
        "the SDK blended index-for-index; it must offset by streaming_delay_frames"
    )
    expected = _expected_blend(unblended, samples, 0.9, delay=3 * HOP)
    assert np.allclose(blended, expected, atol=1e-5), float(np.abs(blended - expected).max())


# --------------------------------------------------------------------------- #
# The two runtimes are one loop written twice
# --------------------------------------------------------------------------- #

STAGES = {
    "no-stage": dict(),
    "blend": dict(dry_blend=0.9),
    "blend-delayed": dict(dry_blend=0.9, delay_frames=3),
    "blend-delayed-guard": dict(dry_blend=0.9, delay_frames=3, onset_guard=GUARD_SPEC),
}


@pytest.mark.parametrize("stages", sorted(STAGES))
def test_both_runtimes_produce_the_same_output(runtime, stages):
    """The duplication is deliberate but not free: a fix to one frame loop
    silently misses the other, and this is the only thing tying them together."""
    samples = _speech_like(seconds=4.0)
    from_repo = _run(runtime("in_repo", scale=0.1, **STAGES[stages]), samples)
    from_sdk = _run(runtime("sdk", scale=0.1, **STAGES[stages]), samples)
    assert from_sdk.shape == from_repo.shape
    assert np.allclose(from_sdk, from_repo, atol=1e-6), float(np.abs(from_sdk - from_repo).max())
    if "guard" in stages:
        assert np.array_equal(from_sdk, from_repo)


@pytest.mark.parametrize("stages", sorted(STAGES))
@pytest.mark.parametrize("kind", KINDS)
def test_chunk_boundaries_do_not_change_the_output(runtime, kind, stages):
    """The blend keeps its own history and index counter and the guard consumes
    the dry stream in whole hops of its own, so arbitrary chunk boundaries must
    not move a sample."""
    samples = _speech_like(seconds=4.0)
    whole = _run(runtime(kind, scale=0.1, **STAGES[stages]), samples)

    chunked = runtime(kind, scale=0.1, **STAGES[stages])
    parts = [
        chunked.process_samples(samples[:13]),
        chunked.process_samples(samples[13:997]),
        chunked.process_samples(samples[997:]),
        chunked.flush(),
    ]
    assert np.allclose(whole, np.concatenate(parts), atol=1e-6)
    if "guard" in stages:
        assert np.array_equal(whole, np.concatenate(parts))


# --------------------------------------------------------------------------- #
# The end of the stream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [7, 4000, 4007])
@pytest.mark.parametrize("kind", KINDS)
def test_flush_drains_the_lookahead_and_emits_only_covered_samples(runtime, kind, n):
    """Output index ``i`` carries input index ``i - dry_delay``, so a stream of
    ``n`` samples owes ``n + dry_delay``. Those past the last full window only
    exist once flush has fed the graph enough zero frames to push its look-ahead
    out; emitting the overlap-add remainder instead divides by a window sum
    that is nearly zero at the very end, which spiked the real v3 export to
    126 at full scale 0.48."""
    gain = np.random.default_rng(3).uniform(0.0, 1.0, BINS)
    samples = _noise(n, seed=1)
    stream = runtime(kind, manifest=_lookahead_manifest(), session=LookaheadSession(gain))
    delay = LookaheadSession.DELAY * HOP

    out = _run(stream, samples)

    assert out.size == n + delay
    assert float(np.abs(out).max()) <= 2.0 * float(np.abs(samples).max())
    assert stream.flush().size == 0, "a finished stream owes nothing more"


@pytest.mark.parametrize("kind", KINDS)
def test_a_lookahead_stream_reconstructs_the_input_to_its_last_sample(runtime, kind):
    """Through an identity look-ahead graph the output aligned by ``dry_delay``
    is the input, up to and including its last sample. Only the first
    ``win - hop`` input samples differ: the warm-up frames that would have
    carried them are the graph's zeros, not input."""
    n = 4007
    samples = _noise(n, seed=2)
    stream = runtime(kind, manifest=_lookahead_manifest(),
                     session=LookaheadSession(np.ones(BINS)))
    delay = LookaheadSession.DELAY * HOP

    aligned = _run(stream, samples)[delay:]

    assert aligned.size == n
    head = WIN - HOP
    assert np.allclose(aligned[head:], samples[head:], atol=1e-5), float(
        np.abs(aligned[head:] - samples[head:]).max()
    )


@pytest.mark.parametrize("kind", KINDS)
def test_flush_on_an_empty_stream_emits_nothing(runtime, kind):
    stream = runtime(kind, manifest=_lookahead_manifest(), session=LookaheadSession(np.ones(BINS)))
    assert stream.flush().size == 0


# --------------------------------------------------------------------------- #
# The onset guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delay_frames", [0, 3])
def test_the_streamed_guard_is_the_offline_guard_on_the_streamed_output(
    runtime, delay_frames
):
    """Streaming == offline, up to the graph's own latency.

    Compared against the module's own `apply` on the runtime's OWN unguarded
    output, which takes the graph out of the comparison: what is left is the
    guard's arithmetic and its alignment, and both are bit-exact. A guard that
    ignored `streaming_delay_frames` would apply frame k's gain to output hop k.
    """
    samples = _speech_like()
    gains = OnsetGuard.from_manifest(GUARD_SPEC).frame_gain(
        samples.astype(np.float64), hop=HOP, sr=16000.0
    )
    # without these the comparison could pass on a dead detector
    assert gains[0] == 1.0, "must start protected"
    assert (gains == 0.0).any(), "must hand over completely at least once"
    assert (gains[int(np.argmax(gains == 0.0)):] == 1.0).any(), "must protect again"

    delay = delay_frames * HOP
    unguarded = _run(runtime(scale=0.1, delay_frames=delay_frames, dry_blend=0.9), samples)
    guarded = _run(
        runtime(scale=0.1, delay_frames=delay_frames, dry_blend=0.9, onset_guard=GUARD_SPEC),
        samples,
    )
    assert guarded.shape == unguarded.shape

    expected = (
        OnsetGuard.from_manifest(GUARD_SPEC)
        .apply(torch.from_numpy(unguarded[delay:]), torch.from_numpy(samples),
               hop=HOP, sr=16000.0)
        .numpy()
    )
    assert np.array_equal(guarded[delay:], expected), float(
        np.abs(guarded[delay:] - expected).max()
    )
    # warm-up samples have no input to restore and are untouched
    assert np.array_equal(guarded[:delay], unguarded[:delay])


@pytest.mark.parametrize("kind", KINDS)
def test_the_guard_keeps_the_dry_stream_even_where_the_blend_would_not(runtime, kind):
    """`dry_blend >= 1.0` is the blend's no-op, but the guard needs the input
    regardless, so the early return depends on BOTH stages being off -- and
    while both are off the graph's output comes back untouched, `np.clip`
    included, which an amplifying graph makes visible."""
    samples = np.full(3200, 0.9, dtype=np.float32)

    unguarded = _inside(runtime(kind, dry_blend=1.0, scale=2.0))
    out = _run(unguarded, samples)
    assert unguarded.dry_history.size == 0, "no stage needs it; do not keep it"
    assert unguarded.guard_hops_fed == 0
    assert float(np.abs(out).max()) > 1.0, "the untouched path must not clamp"

    kept = _inside(runtime(kind, dry_blend=1.0, scale=2.0, onset_guard=GUARD_SPEC))
    head = kept.process_samples(samples[:1600])
    assert kept.dry_history.size > 0, "the guard reads the dry stream"
    assert kept.guard_hops_fed == 1600 // HOP
    guarded = np.concatenate([head, kept.process_samples(samples[1600:]), kept.flush()])
    # Constant broadband energy IS the floor, so the guard stays protected the
    # whole way through and hands back the input -- which is also in range.
    assert float(np.abs(guarded).max()) <= 1.0
    n = min(samples.size, guarded.size)
    assert np.array_equal(guarded[:n], samples[:n])


def test_per_request_overrides_replace_or_refuse_the_recorded_guard(runtime):
    """The operating point ships in the manifest; a request can still refuse
    it, retune a knob, or arm a guard the manifest does not record -- exactly as
    `postprocess_overrides` does for the blend."""
    samples = _speech_like(seconds=4.0)
    recorded = runtime(scale=0.1, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC)
    disabled = runtime(scale=0.1, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC,
                       overrides={"enabled": False})
    absent = runtime(scale=0.1, delay_frames=3, dry_blend=0.9)
    assert disabled.onset_guard is None
    assert np.array_equal(_run(disabled, samples), _run(absent, samples))
    assert not np.array_equal(_run(recorded, samples), _run(absent, samples))

    # A 60 dB activity threshold: nothing clears it, so the guard never arms and
    # the output is the input over the whole aligned span.
    deaf = runtime(scale=0.1, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC,
                   overrides={"margin_db": 60.0})
    assert deaf.onset_guard.margin_db == 60.0
    assert deaf.onset_guard.tau_dn_s == GUARD_SPEC["tau_dn_s"], "unnamed knobs stay"
    out = _run(deaf, samples)
    n = min(samples.size, out.size - 3 * HOP)
    assert np.array_equal(out[3 * HOP : 3 * HOP + n], samples[:n])

    # knobs on a manifest without a guard turn it on; an explicit flag still wins
    armed = runtime(delay_frames=3, overrides={"margin_db": 8.0})
    assert armed.onset_guard is not None and armed.onset_guard.margin_db == 8.0
    assert runtime(delay_frames=3, overrides={"margin_db": 8.0, "enabled": False}).onset_guard is None

    with pytest.raises(ValueError, match="onset_guard override"):
        runtime(delay_frames=3, overrides={"t_arm": 1.0})


@pytest.mark.parametrize("kind", KINDS)
def test_reset_starts_the_next_stream_protected_again(runtime, kind):
    """The anchor is per-stream. A guard state that survived `reset` would hand
    the next stream's first talker straight to the model -- the very deletion
    the guard exists to prevent."""
    samples = _speech_like(seconds=4.0)
    guarded = runtime(kind, scale=0.1, delay_frames=3, dry_blend=0.9, onset_guard=GUARD_SPEC)
    first = _run(guarded, samples)
    guarded.reset()
    assert np.array_equal(first, _run(guarded, samples))


@pytest.mark.parametrize("kind", KINDS)
def test_a_geometry_that_cannot_supply_the_analysis_hop_is_refused(runtime, kind):
    """The guard's frame t needs dry hop t+1, and the runtime holds
    ``streaming_delay_frames + win_length//hop_length - 2`` spare hops when it
    emits an output hop.

    At the shipped 512/160 geometry that is 4 with a 3-frame look-ahead and
    still 1 for a causal export -- the analysis window alone covers the lag, so
    the exact gain is applied and NO latency is added. Only a window shorter
    than two hops on a zero-latency graph comes up short, and that is refused
    rather than served a gain from the wrong frame.
    """
    ok = _inside(runtime(kind, delay_frames=0, win=2 * HOP, onset_guard=GUARD_SPEC))
    assert ok.onset_guard_lookahead_hops == 0
    assert _inside(
        runtime(kind, delay_frames=3, onset_guard=GUARD_SPEC)
    ).onset_guard_lookahead_hops == 4

    with pytest.raises(ValueError, match="look-ahead"):
        runtime(kind, delay_frames=0, win=HOP, onset_guard=GUARD_SPEC)
    # one frame of graph latency buys back exactly the hop the window lost
    assert _inside(
        runtime(kind, delay_frames=1, win=HOP, onset_guard=GUARD_SPEC)
    ).onset_guard_lookahead_hops == 0


class HeadSession:
    """A graph with one auxiliary head: ``[enhanced, head_logit, next_state]``.

    The logit is side information and must never reach a state port; the state
    counts the frames it has been through, so a misrouted output shows up as a
    state that is not the frame count.
    """

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, inputs):
        logit = np.full((1, 1, 1), 7.0, dtype=np.float32)
        return [inputs["noisy_frame"], logit, inputs["state"] + 1.0]


@pytest.mark.parametrize("kind", KINDS)
def test_auxiliary_head_outputs_are_stepped_over_to_reach_the_state(runtime, kind):
    manifest = _manifest()
    manifest.update(
        extra_output_names=["vad_logit"],
        output_names=["enhanced_frame", "vad_logit", "next_state"],
    )
    stream = _inside(runtime(kind, manifest=manifest, session=HeadSession()))
    stream.process_samples(_noise(WIN + 3 * HOP, seed=4))
    assert stream.state["state"].shape == (1, 1)
    assert float(stream.state["state"][0, 0]) == 4.0


@pytest.mark.parametrize("kind", KINDS)
def test_a_non_finite_input_sample_is_silence_not_a_stream_wide_failure(runtime, kind):
    """The graph state is recurrent, so a NaN that reached it would turn every
    later output sample into NaN until `reset`."""
    samples = _noise(4000, seed=5)
    damaged = samples.copy()
    damaged[1000], damaged[2000] = np.nan, np.inf
    silenced = samples.copy()
    silenced[1000] = silenced[2000] = 0.0

    out = _run(runtime(kind, dry_blend=0.9), damaged)

    assert np.isfinite(out).all()
    assert np.array_equal(out, _run(runtime(kind, dry_blend=0.9), silenced))
