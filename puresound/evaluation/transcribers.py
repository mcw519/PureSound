"""Speech recognisers behind one call: local Whisper, Azure Speech, ElevenLabs.

Used by the web workspace's word check and usable from scripts.  Every backend
returns the same ``Transcript`` -- text plus words with their times on the
audio's own time axis -- so a transcript from one can be aligned against a
reference from another.

Credentials are arguments, never configuration: a caller passes a key for
one call and nothing here keeps, logs or writes it.  Error messages are
scrubbed of it before they leave.

A strong recogniser is the default on purpose: a weak one can hide an
over-suppressing model that a strong one exposes, and the point of transcribing
an enhanced signal is to find the words it lost, not to be fast.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

BACKENDS = ("whisper", "azure", "elevenlabs")
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_ELEVENLABS_MODEL = "scribe_v2"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
RATE = 16_000


class TranscriberError(RuntimeError):
    """A recogniser could not produce a transcript; the message is safe to show."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    words: tuple[Word, ...] = field(default_factory=tuple)
    language: str | None = None
    backend: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "words": [{"text": word.text, "start": word.start, "end": word.end} for word in self.words],
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "elapsed_seconds": self.elapsed_seconds,
            "notes": list(self.notes),
        }


def _scrub(message: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret and len(secret) >= 4:
            message = message.replace(secret, "***")
    return message


def _to_16k(samples: Any, sample_rate: int) -> np.ndarray:
    raw = np.asarray(samples)
    if np.issubdtype(raw.dtype, np.integer):
        # PCM integers span the dtype's range; clipping them as if they were
        # already floats in [-1, 1] would turn the signal into a square wave.
        raw = raw / float(np.iinfo(raw.dtype).max)
    audio = raw.astype(np.float32).reshape(-1)
    if int(sample_rate) != RATE:
        from puresound.inference.processors.base import load_audio

        audio, _ = load_audio((audio, int(sample_rate)), sample_rate=RATE)
    return np.clip(np.nan_to_num(audio), -1.0, 1.0).astype(np.float32)


def _wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(np.rint(audio * 32767.0).astype("<i2").tobytes())
    return buffer.getvalue()


def _base_language(language: str | None) -> str | None:
    return language.split("-")[0].lower() if language else None


# Whisper -----------------------------------------------------------------------

_WHISPER_MODELS: dict[str, Any] = {}
_WHISPER_LOCK = threading.Lock()


def _whisper_model(name: str):
    with _WHISPER_LOCK:
        if name not in _WHISPER_MODELS:
            try:
                from faster_whisper import WhisperModel
                from faster_whisper.utils import available_models
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise TranscriberError("Whisper needs faster-whisper: install the `asr` extra (uv sync --extra asr)") from exc
            # faster-whisper reads any other string as a local directory or a hub
            # repository to download, and the name can come from a web request.
            if name not in available_models():
                raise TranscriberError(f"unknown Whisper model {name!r}; choose one of {', '.join(available_models())}")
            try:
                model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=min(8, os.cpu_count() or 1))
            except Exception as exc:
                raise TranscriberError(f"could not load Whisper model {name!r}: {exc}") from exc
            _WHISPER_MODELS[name] = (model, threading.Lock())
        return _WHISPER_MODELS[name]


def cached_whisper_models() -> list[str]:
    """faster-whisper models already on this machine (no download on first use)."""

    names = []
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
    try:
        entries = os.listdir(hub)
    except OSError:
        return names
    prefix = "models--Systran--faster-"
    for entry in entries:
        if entry.startswith(prefix):
            # Systran/faster-whisper-large-v3 -> large-v3,
            # Systran/faster-distil-whisper-large-v3 -> distil-large-v3.
            names.append(entry[len(prefix):].replace("distil-whisper-", "distil-").removeprefix("whisper-"))
    return sorted(set(names))


def _whisper(audio: np.ndarray, model: str | None, language: str | None) -> Transcript:
    name = model or DEFAULT_WHISPER_MODEL
    whisper, lock = _whisper_model(name)
    started = time.perf_counter()
    with lock:
        segments, info = whisper.transcribe(
            audio,
            language=_base_language(language),
            beam_size=5,
            word_timestamps=True,
            # Previous-text conditioning is what lets Whisper loop on a segment
            # it cannot parse; a listening check is better off without it.
            condition_on_previous_text=False,
        )
        segments = list(segments)
    words = tuple(
        Word(str(word.word).strip(), float(word.start), float(word.end))
        for segment in segments
        for word in (segment.words or [])
        if str(word.word).strip()
    )
    text = " ".join(str(segment.text).strip() for segment in segments).strip()
    return Transcript(text, words, getattr(info, "language", None), "whisper", name, time.perf_counter() - started)


# Azure -------------------------------------------------------------------------

def _azure(audio: np.ndarray, key: str, region: str, language: str | None) -> Transcript:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TranscriberError("Azure needs the Speech SDK: pip install azure-cognitiveservices-speech") from exc
    if not key or not region:
        raise TranscriberError("Azure needs a key and a region")
    notes = []
    locale = language or "en-US"
    if not language:
        notes.append("Azure needs a language; Auto used en-US")
    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_recognition_language = locale
    config.output_format = speechsdk.OutputFormat.Detailed
    config.request_word_level_timestamps()
    # Continuous recognition over the whole clip, not recognize_once(): the
    # latter stops at the first pause and truncates the tail, which reads as
    # deletions for every system alike.
    config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "2000")
    config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
    stream = speechsdk.audio.PushAudioInputStream(speechsdk.audio.AudioStreamFormat(samples_per_second=RATE, bits_per_sample=16, channels=1))
    recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=speechsdk.audio.AudioConfig(stream=stream))
    texts: list[str] = []
    words: list[Word] = []
    failure: list[str] = []
    done = threading.Event()

    def recognized(event: Any) -> None:
        if event.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        texts.append(event.result.text)
        try:
            detail = json.loads(event.result.json)
            best = (detail.get("NBest") or [{}])[0]
            for word in best.get("Words") or []:
                start = word["Offset"] / 1e7
                words.append(Word(str(word["Word"]), start, start + word["Duration"] / 1e7))
        except (ValueError, KeyError, TypeError):
            pass

    def canceled(event: Any) -> None:
        details = event.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error:
            failure.append(f"{details.code}: {details.error_details}")
        done.set()

    recognizer.recognized.connect(recognized)
    recognizer.canceled.connect(canceled)
    recognizer.session_stopped.connect(lambda event: done.set())
    started = time.perf_counter()
    stream.write(np.rint(audio * 32767.0).astype("<i2").tobytes())
    stream.close()
    recognizer.start_continuous_recognition()
    done.wait(timeout=30.0 + 3.0 * audio.size / RATE)
    recognizer.stop_continuous_recognition()
    if failure:
        raise TranscriberError(_scrub(f"Azure: {failure[0]}", [key]))
    return Transcript(" ".join(text for text in texts if text).strip(), tuple(words), locale, "azure", f"azure/{region}", time.perf_counter() - started, tuple(notes))


# ElevenLabs --------------------------------------------------------------------

def _elevenlabs(audio: np.ndarray, key: str, model: str | None, language: str | None, post: Callable[..., Any] | None = None) -> Transcript:
    if not key:
        raise TranscriberError("ElevenLabs needs an API key")
    if post is None:
        import httpx

        post = httpx.post
    name = model or DEFAULT_ELEVENLABS_MODEL
    form = {"model_id": name, "timestamps_granularity": "word", "tag_audio_events": "false", "diarize": "false"}
    if language:
        form["language_code"] = _base_language(language)
    started = time.perf_counter()
    try:
        response = post(ELEVENLABS_URL, headers={"xi-api-key": key}, data=form, files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")}, timeout=300.0)
    except Exception as exc:
        raise TranscriberError(_scrub(f"ElevenLabs request failed: {exc}", [key])) from None
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
            message = detail.get("message") if isinstance(detail, Mapping) else detail
        except (ValueError, AttributeError):
            message = response.text[:300]
        raise TranscriberError(_scrub(f"ElevenLabs {response.status_code}: {message}", [key]))
    payload = response.json()
    words = tuple(
        Word(str(item.get("text", "")).strip(), item.get("start"), item.get("end"))
        for item in payload.get("words") or []
        if item.get("type", "word") == "word" and str(item.get("text", "")).strip()
    )
    return Transcript(str(payload.get("text", "")).strip(), words, payload.get("language_code"), "elevenlabs", name, time.perf_counter() - started)


def transcribe(
    samples: Any,
    sample_rate: int,
    *,
    backend: str,
    model: str | None = None,
    language: str | None = None,
    credentials: Mapping[str, str] | None = None,
) -> Transcript:
    """Transcribe mono audio.  ``language`` is BCP-47 (``en-US``, ``zh-TW``) or
    empty for auto-detection where the backend has it."""

    credentials = dict(credentials or {})
    secrets = [value for value in credentials.values() if isinstance(value, str)]
    audio = _to_16k(samples, sample_rate)
    if audio.size < RATE // 10:
        raise TranscriberError("the audio is shorter than a tenth of a second")
    try:
        if backend == "whisper":
            return _whisper(audio, model, language)
        if backend == "azure":
            return _azure(audio, credentials.get("key", ""), credentials.get("region", ""), language)
        if backend == "elevenlabs":
            return _elevenlabs(audio, credentials.get("key", ""), model, language)
    except TranscriberError:
        raise
    except Exception as exc:
        raise TranscriberError(_scrub(f"{backend}: {exc}", secrets)) from None
    raise TranscriberError(f"unknown recogniser {backend!r}; choose one of {', '.join(BACKENDS)}")


__all__ = [
    "BACKENDS",
    "DEFAULT_ELEVENLABS_MODEL",
    "DEFAULT_WHISPER_MODEL",
    "Transcript",
    "TranscriberError",
    "Word",
    "cached_whisper_models",
    "transcribe",
]
