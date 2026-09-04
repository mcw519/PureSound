from __future__ import annotations

import numpy as np
import pytest

from puresound.inference import InferenceCancelled, load_model


@pytest.mark.parametrize(
    ("model_id", "variant"),
    [
        ("voice-isolate-dpcrn-v6", "default"),
        ("voice-isolate-dpcrn-v7", "default"),
        ("voice-isolate-dpcrn-v8", "default"),
        ("voice-isolate-dpcrn-v9", "default"),
        ("voice-isolate-dpcrn-v10", "default"),
        ("voice-isolate-dpcrn-v11-ep19", "default"),
        ("voice-isolate-dpcrn-v11-ep19", "heads"),
        ("voice-isolate-dpcrn-v16-ep19", "default"),
    ],
)
def test_every_voice_isolation_artifact_runs_on_cpu(model_id, variant):
    runtime = load_model(model_id, provider="cpu", variant=variant)
    result = runtime.infer({"audio": np.zeros(800, dtype=np.float32)})
    assert result.outputs["audio"].ndim == 1
    assert result.sample_rate == 16000


def test_speaker_verification_facade_returns_embeddings_and_verdict():
    runtime = load_model("speaker-verification-ps-spk-v1-1", provider="cpu")
    samples = np.zeros(16000, dtype=np.float32)
    result = runtime.infer({"enrollment": samples, "test": samples})
    assert result.task == "speaker_embedding"
    assert result.outputs["enrollment_embedding"].shape == (192,)
    assert result.outputs["test_embedding"].shape == (192,)
    assert result.scores["verdict"] is True
    assert result.sample_rate == 16000


@pytest.mark.parametrize("model_id", ["speaker-verification-ps-spk-v1", "speaker-verification-ps-spk-v1-1"])
def test_both_speaker_models_expose_192d_embeddings(model_id):
    runtime = load_model(model_id, provider="cpu")
    samples = np.zeros(16000, dtype=np.float32)
    result = runtime.infer({"enrollment": samples, "test": samples})
    assert result.outputs["enrollment_embedding"].shape == (192,)
    assert result.outputs["test_embedding"].shape == (192,)


def test_voice_isolation_default_and_request_override_are_applied_once():
    samples = np.random.default_rng(0).normal(0, 0.05, 1600).astype(np.float32)
    runtime = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    default_result = runtime.infer({"audio": samples})
    override_result = runtime.infer({"audio": samples}, {"dry_blend": 1.0})
    assert default_result.outputs["audio"].dtype == np.float32
    assert override_result.outputs["audio"].shape == default_result.outputs["audio"].shape
    # The override changes the graph post-processing, proving that it is
    # consumed by the streaming runtime rather than applied by a second facade
    # pass.  The exact numeric difference is model/input dependent, so compare
    # a non-trivial signal rather than a fixed tolerance.
    assert not np.array_equal(default_result.outputs["audio"], override_result.outputs["audio"])
    shared_session = runtime._session
    runtime.infer({"audio": samples})
    assert runtime._session is shared_session


def _speech_like(seconds: float = 3.0, sample_rate: int = 16_000) -> np.ndarray:
    """Modulated broadband energy, which the onset guard's detector reads as speech.

    A constant fixture would not work: the guard tracks a running-minimum floor,
    so sustained unmodulated energy IS the floor and never counts as a talker
    (`puresound/system/onset_guard.py`).
    """
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    envelope = 0.25 * (0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t))
    return (envelope * rng.normal(0, 1, t.size)).astype(np.float32)


def test_onset_guard_request_hands_back_the_input_until_it_releases():
    samples = _speech_like()
    runtime = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    plain = runtime.infer({"audio": samples}).outputs["audio"]
    # The lower end of the allowed release range keeps the whole protect ->
    # release -> handed over sequence inside a three-second clip.
    guarded = runtime.infer(
        {"audio": samples}, {"onset_guard": True, "onset_guard_tau_dn_s": 0.1}
    )
    output = guarded.outputs["audio"]
    assert output.shape == plain.shape
    assert not np.array_equal(output, plain)
    # The effective operating point travels back with the run: the requested
    # knob replaced the default, the rest are the guard's own defaults.
    assert guarded.metadata["onset_guard"]["tau_dn_s"] == 0.1
    assert guarded.metadata["onset_guard"]["t_arm_s"] == 1.0
    assert guarded.metadata["dry_blend"] == 0.9
    # While it protects, the output is the latency-aligned input bit for bit,
    # not `dry_blend` of it.
    delay = int(guarded.metadata["latency_samples"])
    assert delay > 0
    assert np.array_equal(output[delay:8000], samples[: 8000 - delay])
    # Afterwards the guard is out of the way and the two runs are identical.
    released = 2 * 16_000
    assert np.array_equal(output[released:], plain[released:])
    assert np.nonzero(output != plain)[0][-1] < released


def test_onset_guard_false_on_a_manifest_without_a_guard_is_bit_identical():
    samples = _speech_like(seconds=1.0)
    runtime = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    default_result = runtime.infer({"audio": samples})
    disabled = runtime.infer({"audio": samples}, {"onset_guard": False})
    assert np.array_equal(disabled.outputs["audio"], default_result.outputs["audio"])
    assert disabled.metadata["onset_guard"] is None
    assert default_result.metadata["onset_guard"] is None


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"onset_guard": True, "onset_guard_t_arm_s": 9.0}, "onset_guard_t_arm_s"),
        ({"onset_guard": True, "onset_guard_t_forget_s": 0.5}, "onset_guard_t_forget_s"),
        ({"onset_guard": True, "onset_guard_tau_dn_s": 25.0}, "onset_guard_tau_dn_s"),
        ({"onset_guard": True, "onset_guard_margin_db": 1.0}, "onset_guard_margin_db"),
        ({"onset_guard": True, "onset_guard_margin_db": "loud"}, "onset_guard_margin_db"),
        ({"onset_guard": "yes"}, "onset_guard must be a boolean"),
    ],
)
def test_onset_guard_rejects_values_outside_the_allowed_range(parameters, message):
    runtime = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    with pytest.raises(ValueError, match=message):
        runtime.infer({"audio": np.zeros(1600, dtype=np.float32)}, parameters)


def test_heads_variant_exposes_side_outputs_without_changing_waveform():
    samples = np.random.default_rng(1).normal(0, 0.05, 1600).astype(np.float32)
    plain = load_model(
        "voice-isolate-dpcrn-v11-ep19", provider="cpu", variant="default"
    ).infer({"audio": samples})
    heads = load_model(
        "voice-isolate-dpcrn-v11-ep19", provider="cpu", variant="heads"
    ).infer({"audio": samples}, {"collect_extras": True})
    assert np.allclose(plain.outputs["audio"], heads.outputs["audio"], rtol=1e-5, atol=1e-6)
    assert heads.outputs["vad_logit"].ndim == 1
    assert heads.outputs["background_vad_logit"].ndim == 1


def test_two_stream_runtimes_keep_independent_state():
    samples = np.random.default_rng(2).normal(0, 0.03, 2400).astype(np.float32)
    solo = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    solo_audio = np.concatenate([solo.process_samples(samples), solo.flush()])

    first = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    second = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    split = 800
    first_parts = [first.process_samples(samples[:split])]
    second_parts = [second.process_samples(samples[:split])]
    first_parts.append(first.process_samples(samples[split:]))
    second_parts.append(second.process_samples(samples[split:]))
    first_audio = np.concatenate(first_parts + [first.flush()])
    second_audio = np.concatenate(second_parts + [second.flush()])
    assert np.allclose(first_audio, solo_audio, rtol=1e-5, atol=1e-6)
    assert np.allclose(second_audio, solo_audio, rtol=1e-5, atol=1e-6)


def test_voice_isolation_reports_frame_progress_and_stops_between_frames():
    runtime = load_model("voice-isolate-dpcrn-v8", provider="cpu")
    samples = np.zeros(3200, dtype=np.float32)
    events: list[tuple[float, str]] = []

    def on_progress(value: float, phase: str) -> None:
        events.append((value, phase))

    def should_cancel() -> bool:
        return any(phase == "processing_frames" for _, phase in events)

    with pytest.raises(InferenceCancelled, match="cancelled"):
        runtime.infer(
            {"audio": samples},
            progress_callback=on_progress,
            cancel_check=should_cancel,
        )

    frame_events = [value for value, phase in events if phase == "processing_frames"]
    assert frame_events
    assert frame_events == sorted(frame_events)
    assert max(frame_events) < 1.0
