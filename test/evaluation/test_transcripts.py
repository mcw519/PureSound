"""Token alignment for per-word diagnosis, and the recogniser backends behind it."""

from __future__ import annotations

import random

import numpy as np
import pytest

from puresound.evaluation import transcribers, transcripts
from puresound.evaluation.tools import wer


def test_tokens_split_cjk_into_characters_and_keep_english_words():
    assert transcripts.tokens("Don't stop -- it's well-known!") == ["don't", "stop", "it's", "well", "known"]
    assert transcripts.tokens("我們在 Office 開會，OK？") == ["我", "們", "在", "office", "開", "會", "ok"]
    assert transcripts.unit(transcripts.tokens("我們在 Office 開會")) == "character"
    assert transcripts.unit(transcripts.tokens("a quiet room")) == "word"
    # English normalises exactly as the corpus WER tool does.
    for text in ("Don't stop -- it's well-known!", "He said, 'hello there.' Rock 'n' roll, 'twas O'Brien's."):
        assert transcripts.normalise(text) == wer.normalise(text)


def test_alignment_names_the_lost_word_and_counts_as_the_corpus_wer_tool_does():
    ops = transcripts.align(["the", "cat", "sat", "down"], ["the", "sat", "down", "now"])
    assert [op["op"] for op in ops] == ["hit", "del", "hit", "hit", "ins"]
    assert ops[1] == {"op": "del", "ref": 1, "hyp": None}
    assert ops[4] == {"op": "ins", "ref": None, "hyp": 3}

    rng = random.Random(0)
    vocabulary = "a b c d e f".split()
    for _ in range(300):
        reference = [rng.choice(vocabulary) for _ in range(rng.randint(0, 9))]
        hypothesis = [rng.choice(vocabulary) for _ in range(rng.randint(0, 9))]
        ours = transcripts.counts(transcripts.align(reference, hypothesis), len(reference))
        theirs = wer.edit_counts(" ".join(reference), " ".join(hypothesis))
        assert (ours["sub"], ours["del"], ours["ins"], ours["hit"]) == (theirs["sub"], theirs["del"], theirs["ins"], theirs["hit"])


def test_timed_tokens_share_a_word_span_between_its_characters():
    tokens, times = transcripts.timed_tokens([{"text": "開會", "start": 1.0, "end": 1.4}, {"text": "OK", "start": 2.0, "end": 2.2}])
    assert tokens == ["開", "會", "ok"]
    assert times[0] == pytest.approx((1.0, 1.2)) and times[1] == pytest.approx((1.2, 1.4)) and times[2] == (2.0, 2.2)


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_elevenlabs_reads_words_and_the_key_never_leaves_the_header():
    seen = {}

    def post(url, headers, data, files, timeout):
        seen.update(url=url, headers=headers, data=data, files=files)
        return _Response(200, {"text": "hello world", "language_code": "eng", "words": [
            {"text": "hello", "type": "word", "start": 0.1, "end": 0.4},
            {"text": " ", "type": "spacing", "start": 0.4, "end": 0.5},
            {"text": "world", "type": "word", "start": 0.5, "end": 0.9},
        ]})

    audio = np.zeros(16_000, dtype=np.float32)
    transcript = transcribers._elevenlabs(audio, "secret-key-123", None, "en-US", post=post)
    assert [word.text for word in transcript.words] == ["hello", "world"]
    assert transcript.model == transcribers.DEFAULT_ELEVENLABS_MODEL
    assert seen["headers"] == {"xi-api-key": "secret-key-123"}
    assert seen["data"]["language_code"] == "en" and seen["data"]["timestamps_granularity"] == "word"
    assert "secret-key-123" not in str(seen["data"]) and seen["files"]["file"][2] == "audio/wav"

    def rejected(url, headers, data, files, timeout):
        return _Response(401, {"detail": {"status": "invalid_api_key", "message": "Invalid API key: secret-key-123"}})

    with pytest.raises(transcribers.TranscriberError) as caught:
        transcribers._elevenlabs(audio, "secret-key-123", None, None, post=rejected)
    assert "secret-key-123" not in str(caught.value) and "***" in str(caught.value) and "401" in str(caught.value)


def test_transcribe_rejects_unknown_backends_and_too_short_audio():
    with pytest.raises(transcribers.TranscriberError, match="unknown recogniser"):
        transcribers.transcribe(np.zeros(16_000), 16_000, backend="nope")
    with pytest.raises(transcribers.TranscriberError, match="shorter"):
        transcribers.transcribe(np.zeros(100), 16_000, backend="whisper")


def test_whisper_only_loads_a_named_model_never_a_path_or_a_hub_repository(monkeypatch):
    """A model name arrives in a web request; faster-whisper would read anything
    that is not a size name as a directory to open or a repository to download."""
    faster_whisper = pytest.importorskip("faster_whisper")
    loaded = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda name, **options: loaded.append(name) or object())
    monkeypatch.setattr(transcribers, "_WHISPER_MODELS", {})

    for name in ("someone/else-model", "/tmp", "../models/large-v3"):
        with pytest.raises(transcribers.TranscriberError, match="unknown Whisper model"):
            transcribers._whisper_model(name)
    assert loaded == []

    transcribers._whisper_model("tiny")
    assert loaded == ["tiny"]


def test_integer_pcm_is_scaled_to_full_scale_one_not_clipped():
    pcm = (np.sin(np.linspace(0, 40, 4000)) * 16_000).astype(np.int16)
    audio = transcribers._to_16k(pcm, 16_000)
    assert audio.dtype == np.float32
    assert float(np.abs(audio).max()) == pytest.approx(16_000 / 32_767, rel=1e-3)
    assert np.allclose(audio, transcribers._to_16k(pcm / 32_767.0, 16_000), atol=1e-6)
