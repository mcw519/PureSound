from __future__ import annotations

import numpy as np
import pytest

from puresound.inference import load_model


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
