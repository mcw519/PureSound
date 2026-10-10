"""Dependency-light HTTP API for the local PureSound inference workspace.

The service deliberately uses the Python standard library rather than adding
another web framework.  It is intended for a local or trusted network process
and delegates all model contracts to ``puresound.inference``.
"""

from __future__ import annotations

import atexit
import base64
import binascii
import hashlib
import io
import ipaddress
import struct
import json
import mimetypes
import os
import re
import shutil
import socket as socket_module
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
import wave
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from puresound.inference import (
    COREML_PROVIDER,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    InferenceCancelled,
    InferenceError,
    InferenceResult,
    ModelZoo,
    ModelZooError,
    available_providers,
    load_model,
    normalize_provider,
)
from puresound.evaluation import transcripts as transcript_tools
from puresound.evaluation.transcribers import (
    BACKENDS as ASR_BACKENDS,
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_WHISPER_MODEL,
    TranscriberError,
    cached_whisper_models,
    transcribe as run_transcriber,
)
from puresound.inference.processors.base import load_audio
from puresound.web.exports import build_report, build_zip
from puresound.web.pipeline import PipelineCatalog, PipelineRequestError
from puresound.web.measurements import (
    align_streaming_output,
    audio_metrics,
    audio_metrics_from_path,
    reference_free_metrics,
    reference_metrics,
)


#: Tasks whose models take one audio and return one enhanced audio. The
#: playground's enhancement tab, and the measurement comparison, serve all
#: of them through the same contract; speaker embedding is the other shape.
ENHANCEMENT_TASKS = ("voice_isolation", "noise_suppression")

#: Length of the warm-up pass a freshly loaded enhancement runtime runs before
#: any request is timed.  Creating the ORT session and its first memory arena
#: otherwise lands on the first request and inflates its RTF.
_WARMUP_SECONDS = 0.5

#: Timed repeats a measurement comparison may ask for; the median is reported.
_MAX_TIMING_REPEATS = 5

#: Enhancement models that may run before the requested one (e.g. noise
#: suppression ahead of voice isolation).
_MAX_STAGES = 3

#: Traces whose audio is kept for the Pipeline screen. A trace is a few dozen
#: float WAVs, so they get a small store of their own, in memory only, rather
#: than a share of the run history's bound and disk.
_PIPELINE_RUNS = 6

#: Clips one comparison may take; and the resamples behind its intervals.
_MAX_CLIPS = 50
_BOOTSTRAP_RESAMPLES = 2000
_MIN_RESOLVED_CLIPS = 5

#: Scores aggregated across clips, each as a paired change from the
#: unprocessed input of the same clip.
_AGGREGATE_METRICS = {
    "dnsmos_ovr": lambda report: (report.get("reference_free") or {}).get("dnsmos", {}).get("dnsmos_ovr"),
    "si_sdr_db": lambda report: (report.get("quality") or {}).get("si_sdr_db"),
    "stoi": lambda report: (report.get("quality") or {}).get("stoi"),
    "pesq_wb": lambda report: (report.get("quality") or {}).get("pesq_wb"),
}


def _paired_summary(deltas: list[float]) -> dict[str, Any]:
    """Mean paired change with a 95% percentile-bootstrap interval.

    ``resolved`` is true only when the interval excludes zero and there are at
    least ``_MIN_RESOLVED_CLIPS`` clips: a percentile bootstrap over two or
    three values is far narrower than the truth, so a small set never claims
    a difference.
    """

    values = np.asarray([value for value in deltas if value is not None and np.isfinite(value)], dtype=np.float64)
    summary: dict[str, Any] = {"n": int(values.size), "mean": None, "ci_low": None, "ci_high": None, "wins": None, "resolved": False}
    if values.size == 0:
        return summary
    summary["mean"] = float(values.mean())
    summary["wins"] = int(np.sum(values > 0))
    if values.size >= 2:
        rng = np.random.default_rng(0)
        means = values[rng.integers(0, values.size, size=(_BOOTSTRAP_RESAMPLES, values.size))].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        summary.update(
            ci_low=float(low),
            ci_high=float(high),
            resolved=bool(values.size >= _MIN_RESOLVED_CLIPS and (low > 0 or high < 0)),
        )
    return summary


def _aggregate_clip_reports(reports: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = []
    for index, candidate in enumerate(candidates):
        rows = [
            (report["baseline"], next((model for model in report["models"] if model.get("candidate_id") == index), None))
            for report in reports
        ]
        rows = [(baseline, model) for baseline, model in rows if model is not None and "error" not in model]
        metrics = {}
        for name, read in _AGGREGATE_METRICS.items():
            deltas = []
            for baseline, model in rows:
                after, before = read(model), read(baseline)
                if after is not None and before is not None:
                    deltas.append(float(after) - float(before))
            metrics[name] = _paired_summary(deltas)
        rtfs = [float(model["rtf"]) for _, model in rows if model.get("rtf") is not None]
        aggregate.append({
            "candidate_id": index,
            "model_id": candidate["model_id"],
            "label": candidate["label"],
            "clips_ok": len(rows),
            "clips_failed": len(reports) - len(rows),
            "rtf_median": float(np.median(rtfs)) if rtfs else None,
            "metrics": metrics,
        })
    return aggregate


class WebServiceError(RuntimeError):
    """Raised when an API request cannot be fulfilled."""


_MAX_BENCHMARK_TEXT = 200_000
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _markdown_cell(text: str) -> str:
    text = _MARKDOWN_LINK.sub(r"\1", text.strip())
    return re.sub(r"<br\s*/?>", " / ", text, flags=re.IGNORECASE)


def _markdown_rows_naming(text: str, names: set[str]) -> list[dict[str, Any]]:
    """Table rows whose first cell is one of ``names``, keyed by the header."""

    rows: list[dict[str, Any]] = []
    section = ""
    header: list[str] | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
            header = None
            continue
        if not stripped.startswith("|"):
            header = None
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if header is None:
            if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", following):
                header = [_markdown_cell(cell) for cell in cells]
            continue
        if re.fullmatch(r"[\s|:\-]+", stripped):
            continue
        first = re.sub(r"[*`]", "", cells[0]).strip()
        if first in names:
            rows.append({
                "section": section,
                "cells": [[header[position] if position < len(header) else "", _markdown_cell(cell)] for position, cell in enumerate(cells)],
            })
    return rows


def _gate_record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of a benchmark gate record a person reads: verdict, stages."""

    stages = []
    for stage in record.get("stages") or []:
        if not isinstance(stage, Mapping):
            continue
        difference = stage.get("difference") if isinstance(stage.get("difference"), Mapping) else None
        stages.append({
            key: stage.get(key)
            for key in ("name", "metric", "role", "n", "value", "baseline", "verdict", "direction", "notes")
        } | {"difference": difference})
    return {
        "tag": record.get("tag"),
        "verdict": record.get("verdict"),
        "unresolved_gates": list(record.get("unresolved_gates") or []),
        "inference": record.get("inference"),
        "recipe": record.get("recipe"),
        "stages": stages,
    }


def host_load() -> dict[str, Any]:
    """What else the machine was doing, so an RTF can be read in context."""

    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):  # not available on every platform
        load_1m = None
    return {"load_average_1m": load_1m, "cpu_count": os.cpu_count()}


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
#: A compressed recording of silence is a few KiB per hour, so the limit on an
#: upload's bytes does not bound what it decodes to: the audio itself, counted
#: as 16-bit samples, may be at most this many times that limit.
_MAX_DECODED_EXPANSION = 16

#: Environment variable listing host names, comma separated, that a server may
#: also be reached by (for example a reverse proxy's public name).
ALLOWED_HOSTS_ENV = "PURESOUND_ALLOWED_HOSTS"


def _authority_host(authority: str) -> str:
    """The lower-case host of a ``host[:port]`` authority, without brackets, port or root dot."""

    try:
        return (urllib.parse.urlsplit("//" + authority.strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_ip_literal(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _binds_loopback_only(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _configured_hosts(allowed_hosts: Any) -> frozenset[str]:
    """Host names given by the caller and by ``PURESOUND_ALLOWED_HOSTS``."""

    names = [*(allowed_hosts or ()), *os.environ.get(ALLOWED_HOSTS_ENV, "").split(",")]
    return frozenset(name for name in (_authority_host(str(item)) for item in names if str(item).strip()) if name)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":")).encode(
        "utf-8"
    )


def _float_wav(samples: Any, sample_rate: int) -> bytes:
    """Mono 32-bit float WAV: pipeline stages before the converter may exceed
    full scale, and an integer WAV would clip exactly what the page shows."""
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32).reshape(-1), int(sample_rate), format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _safe_filename(value: str | None, default: str = "upload.wav") -> str:
    name = Path(str(value or default)).name
    name = _SAFE_NAME.sub("_", name).strip("._")
    return name or default


def _refuse_oversized_audio(path: Path, limit: int) -> None:
    """Raise when the audio in ``path`` decodes to more than ``limit`` bytes of 16-bit samples."""

    try:
        import soundfile as sf

        info = sf.info(str(path))
    except Exception:  # a format libsndfile cannot probe is left to the runtime's decoder
        return
    decoded = int(info.frames) * int(info.channels) * 2
    if decoded > limit:
        raise WebServiceError(
            f"the audio decodes to {decoded // (1024 * 1024)} MiB; at most {limit // (1024 * 1024)} MiB is accepted"
        )


def _decode_data_url(value: str) -> tuple[bytes, str | None]:
    """Decode a browser data URL and return bytes plus an optional MIME type."""

    if not value.startswith("data:"):
        raise WebServiceError("audio data must be a data URL")
    try:
        header, encoded = value.split(",", 1)
    except ValueError as exc:
        raise WebServiceError("malformed audio data URL") from exc
    if ";base64" not in header.lower():
        raise WebServiceError("audio data URL must use base64 encoding")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WebServiceError("malformed base64 audio payload") from exc
    mime = header[5:].split(";", 1)[0] or None
    return payload, mime


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WebServiceError("malformed base64 audio payload") from exc


def _audio_wav_bytes(samples: Any, sample_rate: int) -> bytes:
    """Encode a mono float waveform as a browser-compatible PCM16 WAV."""

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise WebServiceError("runtime returned an empty audio output")
    if not np.all(np.isfinite(array)):
        raise WebServiceError("runtime returned non-finite audio output")
    pcm = np.clip(array, -1.0, 1.0)
    pcm = np.rint(pcm * 32767.0).astype("<i2", copy=False)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


#: Side-information curves are thinned to at most this many points each, and
#: a head with several columns shows its first few.
_MAX_CURVE_POINTS = 4000
_MAX_CURVE_SERIES = 4


def _extras_curves(result: InferenceResult) -> dict[str, Any]:
    """Per-frame auxiliary outputs (VAD heads and the like) as plottable curves.

    The runtime collects one value per processed frame, frame t describing
    input frame t, so on the input's time axis (the one the aligned output
    uses) frame t sits at ``t * hop / sample_rate`` seconds.
    """

    rate = int(result.sample_rate or 16_000)
    hop = int(result.metadata.get("hop_length", 0) or 0)
    if hop <= 0:
        return {}
    curves: dict[str, Any] = {}
    for name, value in result.outputs.items():
        if name == "audio":
            continue
        array = np.asarray(value)
        if array.size == 0 or array.dtype.kind not in "fiu" or array.ndim == 0:
            continue
        frames = array.reshape(array.shape[0], -1).astype(np.float64)
        if not np.all(np.isfinite(frames)):
            frames = np.nan_to_num(frames)
        step = max(1, int(np.ceil(frames.shape[0] / _MAX_CURVE_POINTS)))
        thinned = frames[::step, :_MAX_CURVE_SERIES]
        curves[str(name)] = {
            "hop_seconds": hop * step / rate,
            "offset_seconds": 0.0,
            "frames": int(frames.shape[0]),
            "range": [float(min(0.0, frames.min())), float(max(1.0, frames.max()))],
            "series": [np.round(thinned[:, column], 4).tolist() for column in range(thinned.shape[1])],
        }
    return curves


@dataclass(frozen=True)
class StoredOutput:
    data: bytes
    content_type: str
    filename: str
    created_at: float


_TOKEN = re.compile(r"^[0-9a-f]{32}$")


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a sibling temp file so a crash never leaves half a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


class RunStore:
    """Bounded store for downloadable inference outputs.

    In memory by default.  Given a ``directory`` the outputs live on disk
    instead, one folder per run, so a restarted server can still serve the
    audio its run history points at; the same bound applies to the folders.
    """

    def __init__(self, max_runs: int = 32, directory: str | Path | None = None):
        self.max_runs = max(1, int(max_runs))
        self.directory = Path(directory) if directory else None
        self._runs: dict[str, dict[str, StoredOutput]] = {}
        self._disk: dict[str, float] = {}
        self._lock = threading.Lock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for folder in self.directory.iterdir():
                index = folder / "index.json"
                if folder.is_dir() and _TOKEN.match(folder.name) and index.is_file():
                    try:
                        self._disk[folder.name] = float(json.loads(index.read_text())["created_at"])
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
            with self._lock:
                self._evict_locked()

    def put(self, outputs: Mapping[str, StoredOutput]) -> str:
        token = uuid.uuid4().hex
        now = time.time()
        stamped = {
            name: StoredOutput(item.data, item.content_type, item.filename, now)
            for name, item in outputs.items()
        }
        with self._lock:
            if self.directory is None:
                self._runs[token] = stamped
            else:
                folder = self.directory / token
                index = {"created_at": now, "outputs": {}}
                for position, (name, item) in enumerate(stamped.items()):
                    stored_name = f"{position:02d}{Path(item.filename).suffix or '.bin'}"
                    _write_atomic(folder / stored_name, item.data)
                    index["outputs"][name] = {
                        "file": stored_name,
                        "content_type": item.content_type,
                        "filename": item.filename,
                    }
                _write_atomic(folder / "index.json", _json_bytes(index))
                self._disk[token] = now
            self._evict_locked()
        return token

    def _evict_locked(self) -> None:
        while len(self._runs) > self.max_runs:
            oldest = min(self._runs, key=lambda key: min(item.created_at for item in self._runs[key].values()))
            self._runs.pop(oldest, None)
        while len(self._disk) > self.max_runs:
            oldest = min(self._disk, key=self._disk.__getitem__)
            self._disk.pop(oldest, None)
            folder = self.directory / oldest
            for child in folder.glob("*"):
                child.unlink(missing_ok=True)
            try:
                folder.rmdir()
            except OSError:
                pass

    def get(self, token: str, output_name: str) -> StoredOutput | None:
        with self._lock:
            if token in self._runs:
                return self._runs[token].get(output_name)
            if self.directory is None or token not in self._disk:
                return None
            folder = self.directory / token
            try:
                entry = json.loads((folder / "index.json").read_text())["outputs"][output_name]
                data = (folder / entry["file"]).read_bytes()
            except (OSError, ValueError, KeyError, TypeError):
                return None
            return StoredOutput(data, entry["content_type"], entry["filename"], self._disk[token])


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_TEXT, _WS_BINARY, _WS_CLOSE, _WS_PING, _WS_PONG = 0x1, 0x2, 0x8, 0x9, 0xA
_WS_MAX_MESSAGE = 4 * 1024 * 1024
#: A live session's frames: at most this many samples per message, and it
#: ends after this long so a forgotten tab does not hold a thread forever.
_LIVE_MAX_CHUNK = 16_000
_LIVE_MAX_SECONDS = 30 * 60
#: How much of a live session the run history keeps.
_LIVE_KEEP_SECONDS = 10 * 60


class WebSocketClosed(Exception):
    """The peer closed the WebSocket or the connection dropped."""


class WebSocket:
    """The server side of RFC 6455 over an accepted HTTP connection.

    Just what a live audio session needs: text and binary messages
    (fragmented or not), ping/pong and close.  Client frames are masked,
    server frames are not.
    """

    def __init__(self, rfile: Any, wfile: Any):
        self.rfile = rfile
        self.wfile = wfile
        self._send_lock = threading.Lock()

    @staticmethod
    def accept_key(key: str) -> str:
        return base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")

    def _read_exact(self, count: int) -> bytes:
        data = self.rfile.read(count)
        if data is None or len(data) < count:
            raise WebSocketClosed("connection closed")
        return data

    def _read_frame(self) -> tuple[bool, int, bytes]:
        first, second = self._read_exact(2)
        fin, opcode = bool(first & 0x80), first & 0x0F
        masked, length = bool(second & 0x80), second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        if length > _WS_MAX_MESSAGE:
            raise WebSocketClosed("message too large")
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload)) if length < 64 else (
                np.frombuffer(payload, dtype=np.uint8) ^ np.resize(np.frombuffer(mask, dtype=np.uint8), length)
            ).tobytes()
        return fin, opcode, payload

    def receive(self) -> tuple[int, bytes]:
        """The next text or binary message; control frames are answered here."""

        parts: list[bytes] = []
        message_opcode = None
        while True:
            fin, opcode, payload = self._read_frame()
            if opcode == _WS_PING:
                self.send(_WS_PONG, payload)
                continue
            if opcode == _WS_PONG:
                continue
            if opcode == _WS_CLOSE:
                try:
                    self.send(_WS_CLOSE, payload[:2])
                except OSError:
                    pass
                raise WebSocketClosed("closed by peer")
            if opcode in (_WS_TEXT, _WS_BINARY):
                message_opcode = opcode
                parts = [payload]
            elif opcode == 0x0 and message_opcode is not None:
                parts.append(payload)
            else:
                raise WebSocketClosed(f"unexpected opcode {opcode}")
            if sum(len(part) for part in parts) > _WS_MAX_MESSAGE:
                raise WebSocketClosed("message too large")
            if fin:
                return message_opcode, b"".join(parts)

    def send(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header += struct.pack("!H", length)
        else:
            header.append(127)
            header += struct.pack("!Q", length)
        with self._send_lock:
            self.wfile.write(bytes(header) + payload)
            self.wfile.flush()

    def send_json(self, value: Any) -> None:
        self.send(_WS_TEXT, _json_bytes(value))


class UploadStore:
    """Bounded store of raw uploaded files, so one recording is sent once.

    The browser uploads a file's bytes as they are (no base64) and refers to
    it by id afterwards; re-running with other settings, or comparing more
    models, then costs no transfer.  The oldest uploads go first once either
    bound is reached; a request naming an evicted id is told to upload again.
    """

    def __init__(self, max_files: int = 64, max_bytes: int = 1024 * 1024 * 1024, max_decoded_bytes: int | None = None):
        self.max_files = max(1, int(max_files))
        self.max_bytes = max(1, int(max_bytes))
        self.max_decoded_bytes = max_decoded_bytes
        self.directory = Path(tempfile.mkdtemp(prefix="puresound-uploads-"))
        atexit.register(shutil.rmtree, self.directory, ignore_errors=True)
        self._files: dict[str, tuple[Path, int, float, str]] = {}
        self._lock = threading.Lock()

    def put(self, data: bytes, filename: str) -> dict[str, Any]:
        upload_id = uuid.uuid4().hex
        name = _safe_filename(filename)
        suffix = Path(name).suffix.lower()
        if not 0 < len(suffix) <= 16:
            suffix = ".wav"
        path = self.directory / f"{upload_id}{suffix}"
        _write_atomic(path, data)
        if self.max_decoded_bytes is not None:
            try:
                _refuse_oversized_audio(path, self.max_decoded_bytes)
            except WebServiceError:
                path.unlink(missing_ok=True)
                raise
        with self._lock:
            self._files[upload_id] = (path, len(data), time.time(), name)
            while len(self._files) > self.max_files or sum(item[1] for item in self._files.values()) > self.max_bytes:
                oldest = min(self._files, key=lambda key: self._files[key][2])
                if oldest == upload_id:
                    break
                old_path = self._files.pop(oldest)[0]
                old_path.unlink(missing_ok=True)
        return {"upload_id": upload_id, "filename": name, "bytes": len(data)}

    def get(self, upload_id: str) -> tuple[Path, str] | None:
        with self._lock:
            entry = self._files.get(str(upload_id))
        if entry is None or not entry[0].is_file():
            return None
        return entry[0], entry[3]

    def describe(self, upload_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._files.get(str(upload_id))
        if entry is None:
            return None
        return {"upload_id": str(upload_id), "filename": entry[3], "bytes": entry[1]}


@dataclass
class InferenceJob:
    """Small in-memory record for inference and measurement work."""

    job_id: str
    model_id: str
    created_at: float
    kind: str = "inference"
    status: str = "queued"
    phase: str = "queued"
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False


class InferenceJobStore:
    """Thread-safe bounded history for asynchronous browser work."""

    def __init__(self, max_jobs: int = 64, directory: str | Path | None = None):
        self.max_jobs = max(1, int(max_jobs))
        self.directory = Path(directory) if directory else None
        self._jobs: dict[str, InferenceJob] = {}
        self._lock = threading.RLock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._load_finished()

    # Only finished jobs are written: one still running when the server
    # stopped has no result to show, and would otherwise read as running.
    def _load_finished(self) -> None:
        for path in self.directory.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                job = InferenceJob(
                    job_id=str(record["job_id"]),
                    model_id=str(record["model_id"]),
                    created_at=float(record["created_at"]),
                    kind=str(record.get("kind", "inference")),
                    status=str(record["status"]),
                    phase=str(record.get("phase", record["status"])),
                    progress=float(record.get("progress", 1.0)),
                    started_at=record.get("started_at"),
                    finished_at=record.get("finished_at"),
                    result=record.get("result"),
                    error=record.get("error"),
                    cancel_requested=bool(record.get("cancel_requested", False)),
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if job.finished_at is not None and _TOKEN.match(job.job_id):
                self._jobs[job.job_id] = job
        with self._lock:
            self._trim_locked()

    def _persist_locked(self, job: InferenceJob) -> None:
        if self.directory is None or job.finished_at is None:
            return
        try:
            _write_atomic(self.directory / f"{job.job_id}.json", _json_bytes(self._snapshot_locked(job)))
        except OSError:
            # History is a convenience; a full disk must not fail the job.
            pass

    def create(self, model_id: str, *, kind: str = "inference") -> InferenceJob:
        job = InferenceJob(uuid.uuid4().hex, model_id, time.time(), kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
            self._trim_locked()
        return job

    def _trim_locked(self) -> None:
        if len(self._jobs) <= self.max_jobs:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job.finished_at is not None),
            key=lambda job: job.finished_at or job.created_at,
        )
        for job in finished[: max(0, len(self._jobs) - self.max_jobs)]:
            self._jobs.pop(job.job_id, None)
            if self.directory is not None:
                (self.directory / f"{job.job_id}.json").unlink(missing_ok=True)

    def get(self, job_id: str) -> InferenceJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"job not found: {job_id}") from exc

    def start(self, job_id: str, *, phase: str, progress: float) -> bool:
        with self._lock:
            job = self.get(job_id)
            if job.cancel_requested or job.status == "cancelled":
                return False
            job.status = "running"
            job.phase = phase
            job.progress = float(progress)
            job.started_at = time.time()
            return True

    def update_progress(self, job_id: str, *, phase: str, progress: float) -> bool:
        with self._lock:
            job = self.get(job_id)
            if job.cancel_requested or job.status != "running":
                return False
            job.phase = phase
            job.progress = max(job.progress, min(0.999, float(progress)))
            return True

    def update_result(self, job_id: str, result: dict[str, Any]) -> None:
        """Keep completed and pending sweep cells available during cancellation."""
        with self._lock:
            self.get(job_id).result = {**result, "cells": [dict(c) for c in result["cells"]]} if "cells" in result else result

    def succeed(self, job_id: str, result: dict[str, Any]) -> bool:
        with self._lock:
            job = self.get(job_id)
            if job.cancel_requested or job.status != "running":
                return False
            job.status = "succeeded"
            job.phase = "complete"
            job.progress = 1.0
            job.result = result
            job.finished_at = time.time()
            self._persist_locked(job)
            return True

    def fail(self, job_id: str, error: str) -> bool:
        with self._lock:
            job = self.get(job_id)
            if job.cancel_requested or job.status == "cancelled":
                return False
            job.status = "failed"
            job.phase = "failed"
            job.error = str(error)
            job.finished_at = time.time()
            self._persist_locked(job)
            return True

    def finish_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self.get(job_id)
            if job.status in {"succeeded", "failed"}:
                return
            job.cancel_requested = True
            job.status = "cancelled"
            job.phase = "cancelled"
            job.finished_at = time.time()
            self._persist_locked(job)

    def cancel(self, job_id: str) -> InferenceJob:
        with self._lock:
            job = self.get(job_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.phase = "cancelled"
                job.finished_at = time.time()
            else:
                job.phase = "cancelling"
            return job

    @staticmethod
    def _snapshot_locked(job: InferenceJob) -> dict[str, Any]:
        elapsed = None
        if job.started_at is not None:
            elapsed = (job.finished_at or time.time()) - job.started_at
        return {
            "job_id": job.job_id,
            "kind": job.kind,
            "model_id": job.model_id,
            "status": job.status,
            "phase": job.phase,
            "progress": round(float(job.progress), 3),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "elapsed_seconds": elapsed,
            "cancel_requested": job.cancel_requested,
            "result": job.result,
            "error": job.error,
        }

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(self.get(job_id))

    def list_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )[: max(1, min(int(limit), self.max_jobs))]
            return [self._snapshot_locked(job) for job in jobs]


class WebService:
    """Application object shared by the HTTP handler and tests."""

    def __init__(
        self,
        *,
        zoo: ModelZoo | None = None,
        static_dir: str | Path | None = None,
        max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
        max_runs: int = 32,
        allow_local_paths: bool = False,
        history_dir: str | Path | None = None,
        pipeline_root: str | Path | None = None,
        allowed_hosts: Iterable[str] | None = None,
    ):
        self.zoo = zoo or ModelZoo.default()
        self.allowed_hosts = _configured_hosts(allowed_hosts)
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).with_name("static")
        self.max_upload_bytes = max(1, int(max_upload_bytes))
        self.allow_local_paths = bool(allow_local_paths)
        self.history_dir = Path(history_dir).expanduser() if history_dir else None
        self.runs = RunStore(
            max_runs=max_runs,
            directory=self.history_dir / "runs" if self.history_dir else None,
        )
        self.jobs = InferenceJobStore(
            max_jobs=max(16, max_runs * 2),
            directory=self.history_dir / "jobs" if self.history_dir else None,
        )
        self.uploads = UploadStore(max_decoded_bytes=self.max_upload_bytes * _MAX_DECODED_EXPANSION)
        # Word checks are derived from runs that already exist: kept in memory
        # only, never in the run history, so they cannot crowd it out.
        self.transcriptions = InferenceJobStore(max_jobs=32)
        self._runtime_cache: dict[tuple[str, str, str | None], Any] = {}
        self._runtime_lock = threading.Lock()
        self.pipeline = PipelineCatalog(pipeline_root, self.static_dir / "samples" / "pipeline")
        # Pipeline traces and their audio are kept in memory only, apart from
        # the playground history: neither may push the other out.
        self.pipeline_jobs = InferenceJobStore(max_jobs=16)
        self.pipeline_runs = RunStore(max_runs=_PIPELINE_RUNS)
        #: The tracing function; None loads puresound.evaluation.pipeline_trace
        #: on first use. Tests substitute their own.
        self.trace_row: Callable[..., dict[str, Any]] | None = None

    # Catalog -----------------------------------------------------------------
    def _model_payload(self, model: Any) -> dict[str, Any]:
        artifacts: list[dict[str, Any]] = []
        for artifact in model.artifacts:
            artifact_path = self.zoo.path_for(artifact.path)
            manifest_path = self.zoo.path_for(artifact.manifest) if artifact.manifest else None
            artifacts.append(
                {
                    "format": artifact.format,
                    "variant": artifact.variant,
                    "path": artifact.path,
                    "filename": artifact_path.name,
                    "manifest": artifact.manifest,
                    "manifest_filename": manifest_path.name if manifest_path else None,
                    "sha256": artifact.sha256,
                    "processor": artifact.processor,
                    "input_names": list(artifact.input_names),
                    "output_names": list(artifact.output_names),
                    "description": artifact.description,
                    "available": artifact_path.is_file(),
                }
            )
        audio = model.audio.model_dump(mode="json") if model.audio else None
        parameters = {
            name: spec.model_dump(mode="json") for name, spec in model.parameters.items()
        }
        default_variant = None
        if model.artifacts:
            default_variant = next(
                (artifact.variant for artifact in model.artifacts if artifact.variant == "default"),
                model.artifacts[0].variant,
            )
        return {
            "id": model.id,
            "display_name": model.display_name,
            "task": model.task,
            "lifecycle": model.lifecycle,
            "roles": list(model.roles),
            "description": model.description,
            "inputs": list(model.inputs),
            "outputs": list(model.outputs),
            "audio": audio,
            "sample_rate": model.sample_rate,
            "channels": model.channels,
            "capabilities": dict(model.capabilities),
            "preprocessing": dict(model.preprocessing),
            "postprocessing": dict(model.postprocessing),
            "recommended_inference": dict(model.recommended_inference),
            "parameters": parameters,
            "default_variant": default_variant,
            "artifacts": artifacts,
            "source_checkpoint": model.source_checkpoint,
            "source_config": model.source_config,
            "benchmark_references": list(model.benchmark_references),
            "runnable": bool(
                model.artifacts
                and any(self.zoo.path_for(artifact.path).is_file() for artifact in model.artifacts)
            ),
        }

    def list_models(self, task: str | None = None, include_empty: bool = False) -> list[dict[str, Any]]:
        models = self.zoo.list(task=task, runnable_only=not include_empty)
        # Keep the release default at the top while retaining stable id order.
        task_order = {"voice_isolation": 0, "noise_suppression": 1, "speaker_embedding": 2}
        lifecycle_order = {
            "released": 0,
            "candidate": 1,
            "reference": 2,
            "experimental": 3,
            "historical": 4,
        }
        models.sort(
            key=lambda model: (
                task_order.get(model.task, 9),
                "default" not in model.roles,
                lifecycle_order.get(model.lifecycle, 9),
                model.id,
            )
        )
        return [self._model_payload(model) for model in models]

    def inspect_model(self, model_id: str) -> dict[str, Any]:
        return self._model_payload(self.zoo.get(model_id))

    def model_benchmarks(self, model_id: str) -> dict[str, Any]:
        """The catalog's benchmark references for one model, made readable.

        A gate record (JSON) is summarised stage by stage; a README contributes
        the table rows whose first cell names this model's checkpoint, plus the
        full text.  Only paths the catalog lists, inside the repository, are read.
        """

        model = self.zoo.get(model_id)
        checkpoint = Path(model.source_checkpoint).name if model.source_checkpoint else None
        names = {name for name in (checkpoint, Path(checkpoint).stem if checkpoint else None) if name}
        root = Path(self.zoo.root).resolve()
        references = []
        for reference in model.benchmark_references:
            entry: dict[str, Any] = {"path": reference}
            path = Path(self.zoo.path_for(reference)).resolve()
            if root not in path.parents or not path.is_file():
                entry["kind"] = "missing"
            elif path.suffix.lower() == ".json":
                try:
                    entry.update(kind="record", record=_gate_record_summary(json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError) as exc:
                    entry.update(kind="unreadable", error=str(exc))
            elif path.suffix.lower() in {".md", ".markdown"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                entry.update(kind="markdown", rows=_markdown_rows_naming(text, names), text=text[:_MAX_BENCHMARK_TEXT])
            else:
                entry["kind"] = "file"
            references.append(entry)
        return {"model_id": model.id, "checkpoint": checkpoint, "references": references}

    def validate(self, *, verify_hash: bool = True, check_graph: bool = True) -> dict[str, Any]:
        report = self.zoo.validate(
            verify_hash=verify_hash,
            check_graph=check_graph,
            raise_on_error=False,
        )
        return {
            "ok": report.ok,
            "models": report.models,
            "artifacts": report.artifacts,
            "errors": list(report.errors),
        }

    # Inference ---------------------------------------------------------------
    def _runtime(self, model_id: str, provider: str, variant: str | None) -> Any:
        key = (model_id, provider, variant)
        with self._runtime_lock:
            runtime = self._runtime_cache.get(key)
            if runtime is None:
                runtime = load_model(
                    model_id,
                    provider=provider,
                    zoo=self.zoo,
                    variant=variant,
                )
                self._warm_up(runtime)
                self._runtime_cache[key] = runtime
        return runtime

    @staticmethod
    def _warm_up(runtime: Any) -> None:
        """Run a short untimed pass so the first real request is not a cold one."""

        model = getattr(runtime, "model", None)
        if getattr(model, "task", None) not in ENHANCEMENT_TASKS:
            return
        rate = int(getattr(runtime, "sample_rate", 16_000) or 16_000)
        noise = np.random.default_rng(0).standard_normal(int(rate * _WARMUP_SECONDS))
        try:
            runtime.infer(inputs={"audio": (0.01 * noise).astype(np.float32)}, parameters={})
        except Exception:
            # A runtime that cannot process noise fails the real request with
            # its own error and context; the warm-up only exists for timing.
            pass

    def _materialize_input(self, value: Any, temp_paths: list[Path]) -> Any:
        """Turn a browser input descriptor into a temporary local file path."""

        if isinstance(value, Mapping):
            filename = _safe_filename(value.get("filename") or value.get("name"))
            if "upload_id" in value:
                stored = self.uploads.get(str(value["upload_id"]))
                if stored is None:
                    raise WebServiceError("upload not found: it expired or the server restarted; upload the file again")
                return stored[0]
            if "data" in value:
                raw = value["data"]
                if not isinstance(raw, str):
                    raise WebServiceError("input data must be a base64 string")
                payload, mime = _decode_data_url(raw) if raw.startswith("data:") else (_decode_base64(raw), None)
                if mime and filename == "upload.wav":
                    extension = mimetypes.guess_extension(mime) or ".wav"
                    filename = f"upload{extension}"
            elif "base64" in value:
                raw = value["base64"]
                if not isinstance(raw, str):
                    raise WebServiceError("input base64 must be a string")
                payload = _decode_base64(raw)
            elif "path" in value:
                if not self.allow_local_paths:
                    raise WebServiceError("local input paths are disabled for this web service")
                path = Path(str(value["path"])).expanduser().resolve()
                if not path.is_file():
                    raise WebServiceError(f"input file not found: {path}")
                return path
            else:
                raise WebServiceError("input must include data, base64, or path")
        elif isinstance(value, str):
            if value.startswith("data:"):
                payload, mime = _decode_data_url(value)
                extension = mimetypes.guess_extension(mime or "audio/wav") or ".wav"
                filename = f"upload{extension}"
            elif self.allow_local_paths:
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    raise WebServiceError(f"input file not found: {path}")
                return path
            else:
                raise WebServiceError(
                    "local input paths are disabled; provide an uploaded data URL or descriptor"
                )
        else:
            raise WebServiceError("input must be an uploaded file descriptor")

        if not payload:
            raise WebServiceError("uploaded audio is empty")
        if len(payload) > self.max_upload_bytes:
            raise WebServiceError(
                f"uploaded audio exceeds the {self.max_upload_bytes // (1024 * 1024)} MiB limit"
            )
        suffix = Path(filename).suffix.lower() or ".wav"
        fd, raw_path = tempfile.mkstemp(prefix="puresound-web-", suffix=suffix)
        path = Path(raw_path)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        temp_paths.append(path)
        _refuse_oversized_audio(path, self.max_upload_bytes * _MAX_DECODED_EXPANSION)
        return path

    def infer(
        self,
        payload: Mapping[str, Any],
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        def emit_progress(value: float, phase: str) -> None:
            if progress_callback is not None:
                progress_callback(max(0.0, min(1.0, float(value))), phase)

        def ensure_active() -> None:
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("inference cancelled")

        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise WebServiceError("model_id is required")
        try:
            provider = normalize_provider(payload.get("provider") or "auto")
        except ValueError as exc:
            raise WebServiceError(str(exc)) from exc
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise WebServiceError("parameters must be an object")
        measurement_options = payload.get("measurements")
        if measurement_options is not None and not isinstance(measurement_options, Mapping):
            raise WebServiceError("measurements must be an object")
        variant = payload.get("variant")
        if variant is not None and not isinstance(variant, str):
            raise WebServiceError("variant must be a string")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise WebServiceError("inputs must be a non-empty object")
        stages = self._parse_stages(payload.get("stages"))

        temp_paths: list[Path] = []
        try:
            ensure_active()
            emit_progress(0.02, "preparing_inputs")
            materialized = {
                str(name): self._materialize_input(value, temp_paths)
                for name, value in inputs.items()
            }
            ensure_active()
            emit_progress(0.08, "loading_model")
            runtime = self._runtime(model_id, provider, variant)
            ensure_active()
            model_inputs: dict[str, Any] = dict(materialized)
            stage_span = 0.82 * len(stages) / (len(stages) + 1)
            stage_reports: list[dict[str, Any]] = []
            stage_outputs: list[np.ndarray] = []
            if stages:
                if "audio" not in materialized:
                    raise WebServiceError("stages need an audio input")
                chained, stage_reports, stage_outputs = self._run_stages(
                    stages,
                    materialized["audio"],
                    provider=provider,
                    sample_rate=int(getattr(runtime, "sample_rate", 16_000) or 16_000),
                    progress=lambda value, phase: (ensure_active(), emit_progress(0.12 + stage_span * value, phase)),
                    cancel_check=cancel_check,
                )
                model_inputs["audio"] = chained

            def processor_progress(value: float, phase: str) -> None:
                ensure_active()
                emit_progress(0.12 + stage_span + (0.82 - stage_span) * value, phase)

            result: InferenceResult = runtime.infer(
                inputs=model_inputs,
                parameters=dict(parameters),
                progress_callback=processor_progress,
                cancel_check=cancel_check,
            )
            ensure_active()
            emit_progress(0.96, "writing_outputs")
            response = result.as_dict()
            downloadable: dict[str, StoredOutput] = {}
            output_urls: dict[str, str] = {}
            output_rate = result.sample_rate or 16_000
            measured_output = None
            if "audio" in result.outputs:
                raw_output = np.asarray(result.outputs["audio"], dtype=np.float32).reshape(-1)
                measured_output = raw_output
                downloadable["audio"] = StoredOutput(
                    _audio_wav_bytes(raw_output, output_rate), "audio/wav", "enhanced_stream.wav", time.time()
                )
                source_path = materialized.get("audio")
                if result.task in ENHANCEMENT_TASKS and isinstance(source_path, Path):
                    # A streaming model emits its output ``latency_samples`` late
                    # and a flush tail past the end.  The browser compares input
                    # and output sample by sample, so it gets both at the model
                    # rate and on one time axis, plus what the model took out.
                    source, _ = load_audio(source_path, sample_rate=output_rate)
                    latency = int(result.metadata.get("latency_samples", 0) or 0)
                    aligned = align_streaming_output(
                        raw_output, latency_samples=latency, target_samples=source.size
                    )
                    source = source[: aligned.size]
                    measured_output = aligned
                    now = time.time()
                    downloadable["aligned"] = StoredOutput(
                        _audio_wav_bytes(aligned, output_rate), "audio/wav", "enhanced.wav", now
                    )
                    downloadable["input"] = StoredOutput(
                        _audio_wav_bytes(source, output_rate), "audio/wav", "input.wav", now
                    )
                    downloadable["removed"] = StoredOutput(
                        _audio_wav_bytes(source - aligned, output_rate), "audio/wav", "removed.wav", now
                    )
                    for position, stage_output in enumerate(stage_outputs, start=1):
                        downloadable[f"stage-{position}"] = StoredOutput(
                            _audio_wav_bytes(stage_output[: source.size], output_rate), "audio/wav", f"stage_{position}.wav", now
                        )
                    response["alignment"] = {
                        "latency_samples": latency,
                        "latency_ms": 1000.0 * latency / output_rate,
                        "samples": int(aligned.size),
                        "sample_rate": int(output_rate),
                    }
            if downloadable:
                token = self.runs.put(downloadable)
                response["run_id"] = token
                output_urls = {
                    name: f"/api/runs/{token}/{urllib.parse.quote(name, safe='')}"
                    for name in downloadable
                }
            response["output_urls"] = output_urls
            response["input_names"] = list(materialized)
            response["input_files"] = {
                str(name): _safe_filename(value.get("filename") or value.get("name"))
                for name, value in inputs.items()
                if isinstance(value, Mapping) and (value.get("filename") or value.get("name"))
            }
            curves = _extras_curves(result)
            if curves:
                response["extras"] = curves
            response["host"] = host_load()
            if stage_reports:
                duration = float(result.metadata.get("duration_seconds") or 0.0)
                total_elapsed = sum(stage["elapsed_seconds"] for stage in stage_reports) + float(result.elapsed_seconds)
                response["pipeline"] = {
                    "stages": stage_reports,
                    "elapsed_seconds": total_elapsed,
                    "rtf": total_elapsed / duration if duration > 0 else None,
                    "latency_ms": sum(stage["latency_ms"] for stage in stage_reports) + float(result.metadata.get("latency_ms", 0.0) or 0.0),
                }
            if measurement_options is not None and measured_output is not None:
                output_samples = measured_output
                include_dnsmos = bool(
                    measurement_options.get(
                        "include_dnsmos",
                        measurement_options.get("dnsmos", False),
                    )
                )
                response["measurements"] = {
                    "output": audio_metrics(output_samples, output_rate),
                    "reference_free": reference_free_metrics(
                        output_samples,
                        output_rate,
                        include_dnsmos=include_dnsmos,
                    ),
                }
            emit_progress(1.0, "complete")
            return response
        except (InferenceError, ModelZooError, OSError, ValueError, TypeError) as exc:
            raise WebServiceError(str(exc)) from exc
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    # Asynchronous inference --------------------------------------------------
    def _job_cancelled(self, job_id: str) -> bool:
        return self.jobs.get(job_id).cancel_requested

    def _update_job_progress(self, job_id: str, value: float, phase: str) -> None:
        if not self.jobs.update_progress(job_id, phase=phase, progress=value):
            raise InferenceCancelled("inference cancelled")

    def _finish_cancelled_job(self, job_id: str) -> None:
        self.jobs.finish_cancelled(job_id)

    def _run_job(self, job_id: str, payload: Mapping[str, Any]) -> None:
        if not self.jobs.start(job_id, phase="preparing", progress=0.01):
            return
        try:
            result = self.infer(
                payload,
                progress_callback=lambda value, phase: self._update_job_progress(
                    job_id, value, phase
                ),
                cancel_check=lambda: self._job_cancelled(job_id),
            )
            result["job_id"] = job_id
            if not self.jobs.succeed(job_id, result):
                self._finish_cancelled_job(job_id)
        except InferenceCancelled:
            self._finish_cancelled_job(job_id)
        except Exception as exc:  # surfaced through the job status endpoint
            if not self.jobs.fail(job_id, str(exc)):
                self._finish_cancelled_job(job_id)

    def _start_job(
        self,
        job: InferenceJob,
        target: Callable[[str, Mapping[str, Any]], None],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        thread = threading.Thread(
            target=target,
            args=(job.job_id, dict(payload)),
            name=f"puresound-{job.kind}-{job.job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return self.jobs.snapshot(job.job_id)

    def submit_inference(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise WebServiceError("model_id is required")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise WebServiceError("inputs must be a non-empty object")
        job = self.jobs.create(model_id)
        return self._start_job(job, self._run_job, payload)

    def _run_measurement_job(self, job_id: str, payload: Mapping[str, Any]) -> None:
        if not self.jobs.start(
            job_id,
            phase="preparing_measurement",
            progress=0.01,
        ):
            return
        try:
            result = self.measure(
                payload,
                progress_callback=lambda value, phase: self._update_job_progress(
                    job_id, value, phase
                ),
                cancel_check=lambda: self._job_cancelled(job_id),
            )
            result["job_id"] = job_id
            if not self.jobs.succeed(job_id, result):
                self._finish_cancelled_job(job_id)
        except InferenceCancelled:
            self._finish_cancelled_job(job_id)
        except Exception as exc:
            if not self.jobs.fail(job_id, str(exc)):
                self._finish_cancelled_job(job_id)

    def submit_measurement(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping) or ("audio" not in inputs and not inputs.get("clips")):
            raise WebServiceError("inputs.audio or inputs.clips is required")
        job = self.jobs.create("model-comparison", kind="measurement")
        return self._start_job(job, self._run_measurement_job, payload)

    # Dynamic acoustic worlds --------------------------------------------------
    def world_overview(self):
        from puresound.web.world import overview
        return overview()

    def validate_world(self, payload):
        from puresound.web.world import validate_request
        try:
            return {"scene": validate_request(payload).to_dict()}
        except (ValueError, TypeError, KeyError) as exc:
            raise WebServiceError(str(exc)) from exc

    def world_materials(self, payload):
        from puresound.web.world import with_materials
        try:
            return {"scene": with_materials(payload)}
        except (ValueError, TypeError, KeyError) as exc:
            raise WebServiceError(str(exc)) from exc

    def submit_world(self, payload):
        from puresound.web.world import validate_request, sweep_scenes
        from puresound.web.aicoustics import options, capabilities
        try:
            validate_request(payload)
            comparison = options(payload.get("aicoustics"))
            if comparison and not capabilities()["available"]:
                raise ValueError("Install aic-sdk==3.3.0 to enable ai-coustics comparison.")
            if payload.get("kind") == "world_sweep":
                sweep_scenes(payload)
            model_id = payload.get("model_id")
            if model_id and self.zoo.get(model_id).task not in ENHANCEMENT_TASKS:
                raise ValueError("world jobs require an enhancement model")
        except (ValueError, TypeError, KeyError) as exc:
            raise WebServiceError(str(exc)) from exc
        job = self.jobs.create(model_id or "acoustic-world", kind=payload.get("kind", "world_render"))
        return self._start_job(job, self._run_world_job, payload)

    def _world_result(self, payload, spec, assets, progress):
        from puresound.web.world import render_report, world_zip
        from puresound.web.aicoustics import options, process
        enhance, model = self._pipeline_enhancer(payload.get("model_id"), str(payload.get("provider") or "cpu"))
        if payload.get("model_id") and enhance is None:
            raise ValueError("selected model cannot process a 16 kHz world")
        comparison_config = options(payload.get("aicoustics"))
        comparison = (lambda samples, rate, notify: process(samples, rate, comparison_config, progress=notify)) if comparison_config else None
        report, audio, sources = render_report(spec, assets, enhance, model, progress, payload.get("assets"), comparison=comparison)
        if comparison_config:
            report["comparisons"]["aicoustics"].update({k: v for k, v in comparison_config.items() if k != "api_key"})
        stored = {name: StoredOutput(_float_wav(samples, 16000), "audio/wav", name+".wav", 0.) for name, samples in audio.items()}
        package = world_zip(report, audio, sources, _float_wav)
        stored["scene-package"] = StoredOutput(package, "application/zip", "scene.zip", 0.)
        token = self.runs.put(stored)
        report["run_id"] = token
        report["output_urls"] = {name: f"/api/runs/{token}/{name}" for name in stored}
        return report

    def _run_world_job(self, job_id, payload):
        from puresound.web.world import validate_request, load_assets, sweep_scenes
        from puresound.web.aicoustics import public_request, error_message
        if not self.jobs.start(job_id, phase="preparing", progress=.01):
            return
        def progress(value, phase):
            self._update_job_progress(job_id, value, phase)
        try:
            spec = validate_request(payload)
            assets = load_assets(spec, payload.get("assets", {}), self.static_dir, self._upload_path)
            if payload.get("kind") == "world_sweep":
                cells = sweep_scenes(payload)
                reports = [{**{k: v for k, v in cell.items() if k != "scene"}, "status": "queued"} for cell in cells]
                partial = {"axes": payload["axes"], "cells": reports, "scene": spec.to_dict(), "job_id": job_id, "request": public_request(payload)}
                self.jobs.update_result(job_id, partial)
                for i, cell in enumerate(cells):
                    row = reports[i]
                    try:
                        progress(i/max(1, len(cells)), "sweep")
                    except InferenceCancelled:
                        for pending in reports[i:]:
                            pending["status"] = "cancelled"
                        self.jobs.update_result(job_id, partial)
                        raise
                    if "error" in cell:
                        row["status"] = "failed"
                    else:
                        try:
                            row["result"] = self._world_result(payload, cell["scene"], assets, lambda v, p: progress((i+v)/len(cells), p))
                            row["status"] = "succeeded"
                        except InferenceCancelled:
                            for pending in reports[i:]:
                                pending["status"] = "cancelled"
                            self.jobs.update_result(job_id, partial)
                            raise
                        except Exception as exc:
                            row.update(status="failed", error=error_message(exc, payload))
                    self.jobs.update_result(job_id, {**partial, "cells": [dict(r) for r in reports]})
                result = {**partial, "cells": reports}
            else:
                result = self._world_result(payload, spec, assets, progress)
            result["job_id"] = job_id
            if not self.jobs.succeed(job_id, result):
                self.jobs.finish_cancelled(job_id)
        except InferenceCancelled:
            self.jobs.finish_cancelled(job_id)
        except Exception as exc:
            if not self.jobs.fail(job_id, error_message(exc, payload)):
                self.jobs.finish_cancelled(job_id)

    # Pipeline inspector -----------------------------------------------------------
    def run_output(self, token: str, output_name: str) -> StoredOutput | None:
        """A stored output of a playground run or of a pipeline trace."""
        return self.runs.get(token, output_name) or self.pipeline_runs.get(token, output_name)

    def pipeline_overview(self) -> dict[str, Any]:
        return {**self.pipeline.status(), "recipes": self.pipeline.recipes(), "samples": self.pipeline.samples()}

    def _upload_path(self, upload_id: str) -> Path | None:
        stored = self.uploads.get(upload_id)
        return Path(stored[0]) if stored else None

    def submit_pipeline_trace(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            request, options = self.pipeline.build_request(payload, self._upload_path)
        except PipelineRequestError as exc:
            raise WebServiceError(str(exc)) from exc
        model_id = options["model_id"]
        if model_id:
            try:
                task = self.zoo.get(model_id).task
            except (ModelZooError, KeyError) as exc:
                raise WebServiceError(f"model {model_id!r} is not in the model zoo") from exc
            if task not in ENHANCEMENT_TASKS:
                raise WebServiceError(f"model {model_id!r} is a {task} model; a trace is scored by an enhancement model")
        job = self.pipeline_jobs.create(model_id or "pipeline", kind="pipeline")
        thread = threading.Thread(
            target=self._run_pipeline_job,
            args=(job.job_id, request, options),
            name=f"puresound-pipeline-{job.job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return self.pipeline_jobs.snapshot(job.job_id)

    def _run_pipeline_job(self, job_id: str, request: Any, options: Mapping[str, Any]) -> None:
        store = self.pipeline_jobs
        if not store.start(job_id, phase="preparing", progress=0.01):
            return

        def progress(value: float, phase: str) -> None:
            if not store.update_progress(job_id, phase=phase, progress=value):
                raise InferenceCancelled("pipeline trace cancelled")

        try:
            result = self._pipeline_trace(request, options, progress)
            result["job_id"] = job_id
            if not store.succeed(job_id, result):
                store.finish_cancelled(job_id)
        except InferenceCancelled:
            store.finish_cancelled(job_id)
        except Exception as exc:  # surfaced through the job status endpoint
            if not store.fail(job_id, str(exc)):
                store.finish_cancelled(job_id)

    def _pipeline_enhancer(self, model_id: str | None, provider: str) -> tuple[Callable | None, dict[str, Any] | None]:
        """The model a trace scores every stage with, or why it cannot."""
        if not model_id:
            return None, None
        runtime = self._runtime(model_id, normalize_provider(provider), None)
        rate = int(getattr(runtime, "sample_rate", 0) or 0)
        model = getattr(runtime, "model", None)
        info: dict[str, Any] = {
            "id": model_id,
            "display_name": getattr(model, "display_name", model_id),
            "task": getattr(model, "task", None),
            "sample_rate": rate,
        }
        if rate != 16_000:
            info["skipped"] = f"the model runs at {rate} Hz; rows are built at 16 kHz"
            return None, info

        def enhance(noisy: np.ndarray, sample_rate: int) -> np.ndarray:
            result = runtime.infer(inputs={"audio": np.asarray(noisy, dtype=np.float32)}, parameters={})
            latency = int(result.metadata.get("latency_samples", 0) or 0)
            return align_streaming_output(
                np.asarray(result.outputs["audio"], dtype=np.float32).reshape(-1),
                latency_samples=latency,
                target_samples=int(np.asarray(noisy).size),
            )

        return enhance, info

    def _pipeline_trace(self, request: Any, options: Mapping[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        trace_row = self.trace_row
        if trace_row is None:
            from puresound.evaluation.pipeline_trace import trace_row
        enhance, model = self._pipeline_enhancer(options.get("model_id"), str(options.get("provider") or "cpu"))
        with tempfile.TemporaryDirectory(prefix="puresound-pipeline-") as workspace:
            traced = trace_row(request, workspace_root=Path(workspace), enhance=enhance, model=model, progress=progress)
        outputs = {
            name: StoredOutput(_float_wav(samples, 16_000), "audio/wav", f"{name}.wav", 0.0)
            for name, samples in traced["audio"].items()
        }
        token = self.pipeline_runs.put(outputs)
        report = dict(traced["report"])
        report["recipe"] = options["recipe"]
        report["run_id"] = token
        report["audio_urls"] = {name: f"/api/runs/{token}/{urllib.parse.quote(name, safe='')}" for name in outputs}
        return report

    def _store_for(self, job_id: str) -> InferenceJobStore:
        for store in (self.jobs, self.transcriptions, self.pipeline_jobs):
            try:
                store.get(job_id)
                return store
            except KeyError:
                continue
        raise KeyError(f"job not found: {job_id}")

    def job(self, job_id: str) -> dict[str, Any]:
        from puresound.web.world import current_result

        job = self._store_for(job_id).snapshot(job_id)
        return {**job, "result": current_result(job["kind"], job.get("result"))}

    def job_export(self, job_id: str, kind: str, *, tz: str | None = None) -> tuple[bytes, str, str]:
        """``(body, content type, file name)`` for a finished job's ZIP or report."""

        job = self.job(job_id)
        partial_sweep = job["kind"] == "world_sweep" and job["status"] == "cancelled" and job.get("result")
        if job["status"] != "succeeded" and not partial_sweep:
            raise WebServiceError(f"job {job_id} has not succeeded")
        stamp = time.strftime("%Y%m%d-%H%M", time.gmtime(job.get("finished_at") or time.time()))
        base = f"puresound-{job['kind']}-{_safe_filename(job['model_id'], 'run')}-{stamp}"
        if kind == "zip":
            data, _ = build_zip(job, self.runs.get)
            return data, "application/zip", f"{base}.zip"
        if kind == "report":
            return build_report(job, self.runs.get, tz=tz).encode("utf-8"), "text/html; charset=utf-8", f"{base}.html"
        raise WebServiceError("export must be zip or report")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        store = self._store_for(job_id)
        store.cancel(job_id)
        return store.snapshot(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        from puresound.web.world import current_result

        return [{**job, "result": current_result(job["kind"], job.get("result"))} for job in self.jobs.list_snapshots(limit)]

    # Transcription -----------------------------------------------------------
    def asr_capabilities(self) -> dict[str, Any]:
        return {
            "backends": list(ASR_BACKENDS),
            "whisper": {"default": DEFAULT_WHISPER_MODEL, "cached": cached_whisper_models()},
            "elevenlabs": {"default": DEFAULT_ELEVENLABS_MODEL},
        }

    def _audio_for(self, source: Any, temp_paths: list[Path]) -> tuple[np.ndarray, int]:
        """Mono audio for a track: a stored run output or an upload."""

        if not isinstance(source, Mapping):
            raise WebServiceError("each track needs a source: {url} of a run output or {upload_id}")
        if source.get("url"):
            match = re.fullmatch(r"/api/runs/([0-9a-f]{32})/([^/]+)", str(source["url"]))
            stored = self.runs.get(match.group(1), urllib.parse.unquote(match.group(2))) if match else None
            if stored is None:
                raise WebServiceError("that output is no longer stored; run it again")
            import soundfile

            audio, rate = soundfile.read(io.BytesIO(stored.data), dtype="float32", always_2d=True)
            return audio.mean(axis=1), int(rate)
        path = self._materialize_input(source, temp_paths)
        audio, rate = load_audio(path, sample_rate=16_000)
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate)

    def transcribe(
        self,
        payload: Mapping[str, Any],
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Transcribe tracks and align each against a reference.

        With ``reference_text`` (what was actually said) every track is scored
        against it: a real error rate.  Without it, each track is compared
        with one of the tracks' own transcripts (``reference_track``, by
        default the first -- the input): a difference, not an error rate,
        because a voice-isolation model is meant to remove some words.
        ``credentials`` go to the recogniser for this call only.
        """

        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        backend = str(payload.get("backend") or "whisper")
        if backend not in ASR_BACKENDS:
            raise WebServiceError(f"backend must be one of {', '.join(ASR_BACKENDS)}")
        tracks = payload.get("tracks")
        if not isinstance(tracks, (list, tuple)) or not tracks:
            raise WebServiceError("tracks must be a non-empty array")
        if len(tracks) > 8:
            raise WebServiceError("at most 8 tracks per transcription")
        ids = []
        for index, track in enumerate(tracks):
            if not isinstance(track, Mapping) or not str(track.get("id") or "").strip():
                raise WebServiceError(f"tracks[{index}] needs an id")
            ids.append(str(track["id"]))
        reference_text = str(payload.get("reference_text") or "").strip()
        reference_track = str(payload.get("reference_track") or ids[0])
        if not reference_text and reference_track not in ids:
            raise WebServiceError("reference_track must be one of the tracks")
        credentials = payload.get("credentials") or {}
        if not isinstance(credentials, Mapping):
            raise WebServiceError("credentials must be an object")
        credentials = {str(key): str(value) for key, value in credentials.items() if value}
        model = str(payload.get("model") or "") or None
        language = str(payload.get("language") or "") or None
        started = time.time()
        temp_paths: list[Path] = []
        results: list[dict[str, Any]] = []
        try:
            for index, track in enumerate(tracks):
                if cancel_check is not None and cancel_check():
                    raise InferenceCancelled("transcription cancelled")
                if progress_callback is not None:
                    progress_callback(index / len(tracks), f"track_{index + 1}_of_{len(tracks)}:transcribing")
                audio, rate = self._audio_for(track.get("source"), temp_paths)
                try:
                    transcript = run_transcriber(audio, rate, backend=backend, model=model, language=language, credentials=credentials)
                except TranscriberError as exc:
                    raise WebServiceError(f"{track.get('label') or track['id']}: {exc}") from None
                if transcript.words:
                    tokens, times = transcript_tools.timed_tokens(transcript.words)
                else:
                    tokens = transcript_tools.tokens(transcript.text)
                    times = [(None, None)] * len(tokens)
                results.append({
                    "id": str(track["id"]),
                    "label": str(track.get("label") or track["id"]),
                    **transcript.as_dict(),
                    "tokens": tokens,
                    "times": times,
                })
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)
        if reference_text:
            reference = {"kind": "text", "track_id": None, "text": reference_text, "tokens": transcript_tools.tokens(reference_text), "times": None}
        else:
            source = next(result for result in results if result["id"] == reference_track)
            reference = {"kind": "track", "track_id": reference_track, "text": source["text"], "tokens": source["tokens"], "times": source["times"]}
        for result in results:
            if reference["kind"] == "track" and result["id"] == reference_track:
                result.update(alignment=None, counts=None, looped=False, is_reference=True)
                continue
            ops = transcript_tools.align(reference["tokens"], result["tokens"])
            result.update(
                alignment=ops,
                counts=transcript_tools.counts(ops, len(reference["tokens"])),
                looped=transcript_tools.looped(len(reference["tokens"]), len(result["tokens"])),
                is_reference=False,
            )
        if progress_callback is not None:
            progress_callback(1.0, "complete")
        return {
            "backend": backend,
            "model": results[0]["model"] if results else model,
            "language": language,
            "mode": "reference" if reference_text else "track",
            "unit": transcript_tools.unit(reference["tokens"] or (results[0]["tokens"] if results else [])),
            "reference": reference,
            "tracks": results,
            "elapsed_seconds": time.time() - started,
        }

    def submit_transcription(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        backend = str(payload.get("backend") or "whisper")
        if backend not in ASR_BACKENDS:
            raise WebServiceError(f"backend must be one of {', '.join(ASR_BACKENDS)}")
        job = self.transcriptions.create(f"asr:{backend}", kind="transcription")
        thread = threading.Thread(target=self._run_transcription_job, args=(job.job_id, dict(payload)), name=f"puresound-asr-{job.job_id[:8]}", daemon=True)
        thread.start()
        return self.transcriptions.snapshot(job.job_id)

    def _run_transcription_job(self, job_id: str, payload: dict[str, Any]) -> None:
        store = self.transcriptions
        if not store.start(job_id, phase="preparing", progress=0.01):
            return
        secrets = [str(value) for value in (payload.get("credentials") or {}).values() if value]
        try:
            result = self.transcribe(
                payload,
                progress_callback=lambda value, phase: store.update_progress(job_id, phase=phase, progress=value),
                cancel_check=lambda: store.get(job_id).cancel_requested,
            )
            result["job_id"] = job_id
            if not store.succeed(job_id, result):
                store.finish_cancelled(job_id)
        except InferenceCancelled:
            store.finish_cancelled(job_id)
        except Exception as exc:
            message = str(exc)
            for secret in secrets:
                if len(secret) >= 4:
                    message = message.replace(secret, "***")
            if not store.fail(job_id, message):
                store.finish_cancelled(job_id)
        finally:
            # The payload -- and the key in it -- dies with this thread.
            payload.clear()

    # Measurements ------------------------------------------------------------
    def measure(
        self,
        payload: Mapping[str, Any],
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Run candidates over one clip, or over several and aggregate.

        One clip (``inputs.audio``, optional ``inputs.reference``) returns that
        clip's report.  ``inputs.clips`` -- a list of such pairs -- returns
        every clip's report plus, per candidate, the paired change from the
        unprocessed input across clips with a bootstrap interval: one clip is
        one draw, and a ranking needs several.
        """
        try:
            return self._measure(payload, progress_callback=progress_callback, cancel_check=cancel_check)
        except (InferenceError, ModelZooError, OSError, ValueError, TypeError) as exc:
            raise WebServiceError(str(exc)) from exc

    def _measure(
        self,
        payload: Mapping[str, Any],
        *,
        progress_callback: Callable[[float, str], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        def emit_progress(value: float, phase: str) -> None:
            if progress_callback is not None:
                progress_callback(max(0.0, min(1.0, float(value))), phase)

        def ensure_active() -> None:
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("measurement cancelled")

        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping):
            raise WebServiceError("inputs.audio is required")
        if inputs.get("clips") is not None:
            clips = inputs["clips"]
            if not isinstance(clips, (list, tuple)) or not clips:
                raise WebServiceError("inputs.clips must be a non-empty array")
            if len(clips) > _MAX_CLIPS:
                raise WebServiceError(f"at most {_MAX_CLIPS} clips per comparison")
            for index, clip in enumerate(clips):
                if not isinstance(clip, Mapping) or clip.get("audio") is None:
                    raise WebServiceError(f"inputs.clips[{index}].audio is required")
        elif "audio" in inputs:
            clips = [inputs]
        else:
            raise WebServiceError("inputs.audio is required")
        try:
            provider = normalize_provider(payload.get("provider") or "auto")
        except ValueError as exc:
            raise WebServiceError(str(exc)) from exc
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise WebServiceError("parameters must be an object")
        candidates = self._measurement_candidates(payload, parameters)
        model_ids = [candidate["model_id"] for candidate in candidates]
        measurement_options = payload.get("measurements") or {}
        if not isinstance(measurement_options, Mapping):
            raise WebServiceError("measurements must be an object")
        include_dnsmos = bool(
            measurement_options.get(
                "include_dnsmos",
                measurement_options.get("dnsmos", False),
            )
        )
        try:
            timing_repeats = int(measurement_options.get("timing_repeats", 1))
        except (TypeError, ValueError) as exc:
            raise WebServiceError("measurements.timing_repeats must be an integer") from exc
        if not 1 <= timing_repeats <= _MAX_TIMING_REPEATS:
            raise WebServiceError(
                f"measurements.timing_repeats must be in [1, {_MAX_TIMING_REPEATS}]"
            )
        sample_rate = int(payload.get("sample_rate") or 16_000)
        started = time.time()
        request = {
            "models": model_ids,
            "provider": provider,
            "parameters": dict(parameters),
            "measurements": {
                "include_dnsmos": include_dnsmos,
                "timing_repeats": timing_repeats,
            },
        }
        reports = []
        for clip_index, clip in enumerate(clips):
            ensure_active()
            prefix = "" if len(clips) == 1 else f"clip_{clip_index + 1}_of_{len(clips)}:"

            def clip_progress(value: float, phase: str, clip_index: int = clip_index, prefix: str = prefix) -> None:
                emit_progress((clip_index + value) / len(clips) * 0.98, f"{prefix}{phase}")

            report = self._measure_clip(
                clip,
                candidates,
                provider=provider,
                sample_rate=sample_rate,
                include_dnsmos=include_dnsmos,
                timing_repeats=timing_repeats,
                progress=clip_progress,
                ensure_active=ensure_active,
                cancel_check=cancel_check,
                # With several clips one bad clip must not sink the rest.
                require_success=len(clips) == 1,
            )
            report["request"] = request
            reports.append(report)
        emit_progress(0.99, "finalizing_report")
        if len(clips) == 1:
            result = reports[0]
            result["host"] = host_load()
            result["elapsed_seconds"] = time.time() - started
            emit_progress(1.0, "complete")
            return result
        succeeded = sum(report["summary"]["succeeded"] for report in reports)
        if succeeded == 0:
            raise WebServiceError("every candidate failed on every clip")
        result = {
            "sample_rate": sample_rate,
            "request": request,
            "host": host_load(),
            "clips": reports,
            "aggregate": _aggregate_clip_reports(reports, candidates),
            "summary": {
                "clips": len(reports),
                "succeeded": succeeded,
                "failed": sum(report["summary"]["failed"] for report in reports),
            },
            "elapsed_seconds": time.time() - started,
        }
        emit_progress(1.0, "complete")
        return result

    def _measure_clip(
        self,
        clip: Mapping[str, Any],
        candidates: list[dict[str, Any]],
        *,
        provider: str,
        sample_rate: int,
        include_dnsmos: bool,
        timing_repeats: int,
        progress: Callable[[float, str], None],
        ensure_active: Callable[[], None],
        cancel_check: Callable[[], bool] | None,
        require_success: bool = True,
    ) -> dict[str, Any]:
        temp_paths: list[Path] = []
        clip_started = time.time()
        try:
            ensure_active()
            progress(0.02, "preparing_probe")
            audio_path = self._materialize_input(clip["audio"], temp_paths)
            ensure_active()
            progress(0.05, "analyzing_probe")
            input_values, input_stats = audio_metrics_from_path(audio_path, sample_rate)
            reference_values = None
            reference_stats = None
            if clip.get("reference") is not None:
                ensure_active()
                progress(0.08, "analyzing_reference")
                reference_path = self._materialize_input(clip["reference"], temp_paths)
                reference_values, reference_stats = audio_metrics_from_path(reference_path, sample_rate)
            # The unprocessed input goes through the same scoring as every
            # model: a score only means something next to what leaving the
            # audio alone would have scored.
            ensure_active()
            progress(0.09, "scoring_unprocessed_input")
            baseline = {
                "label": "Unprocessed input",
                "output": input_stats,
                "quality": reference_metrics(reference_values, input_values, sample_rate)
                if reference_values is not None
                else {},
                "reference_free": reference_free_metrics(
                    input_values, sample_rate, include_dnsmos=include_dnsmos
                ),
            }
            reports: list[dict[str, Any]] = []
            downloadable: dict[str, StoredOutput] = {
                "input": StoredOutput(_audio_wav_bytes(input_values, sample_rate), "audio/wav", "input.wav", time.time()),
            }
            if reference_values is not None:
                downloadable["reference"] = StoredOutput(
                    _audio_wav_bytes(reference_values, sample_rate), "audio/wav", "reference.wav", time.time()
                )
            model_count = max(1, len(candidates))
            for index, candidate in enumerate(candidates):
                model_id = candidate["model_id"]
                candidate_parameters = candidate["parameters"]
                ensure_active()
                model_start = 0.1 + 0.86 * index / model_count
                model_end = 0.1 + 0.86 * (index + 1) / model_count
                model_report: dict[str, Any] = {
                    "model_id": model_id,
                    "candidate_id": index,
                    "label": candidate["label"],
                    "requested_parameters": dict(candidate_parameters),
                }
                try:
                    model = self.zoo.get(model_id)
                    if model.task not in ENHANCEMENT_TASKS:
                        raise WebServiceError(
                            "measurement comparison only supports audio-in/audio-out enhancement models"
                        )

                    progress(model_start, f"model_{index + 1}_of_{model_count}:loading_model")
                    runtime = self._runtime(model_id, provider, None)
                    model_input: Any = audio_path
                    stage_reports: list[dict[str, Any]] = []
                    stage_elapsed = 0.0
                    if candidate.get("stages"):
                        stage_share = 0.5 * len(candidate["stages"]) / (len(candidate["stages"]) + 1)

                        def stage_progress(value: float, phase: str) -> None:
                            ensure_active()
                            progress(model_start + (model_end - model_start) * stage_share * value, f"model_{index + 1}_of_{model_count}:{phase}")

                        model_input, stage_reports, _ = self._run_stages(
                            candidate["stages"],
                            audio_path,
                            provider=provider,
                            sample_rate=int(getattr(runtime, "sample_rate", sample_rate) or sample_rate),
                            progress=stage_progress,
                            cancel_check=cancel_check,
                        )
                        stage_elapsed = sum(stage["elapsed_seconds"] for stage in stage_reports)
                    else:
                        stage_share = 0.0
                    # One RTF is one draw: identical runs on a shared host
                    # spread widely, so a comparison can ask for several and
                    # read the median.  The first run's output is the one scored.
                    rtf_runs: list[float] = []
                    result: InferenceResult | None = None
                    for repeat in range(timing_repeats):

                        def model_progress(value: float, phase: str, repeat: int = repeat) -> None:
                            fraction = stage_share + (1 - stage_share) * (repeat + value) / timing_repeats
                            label = phase if timing_repeats == 1 else f"timing_{repeat + 1}_of_{timing_repeats}"
                            progress(
                                model_start + (model_end - model_start) * fraction,
                                f"model_{index + 1}_of_{model_count}:{label}",
                            )

                        run = runtime.infer(
                            inputs={"audio": model_input},
                            parameters=dict(candidate_parameters),
                            progress_callback=model_progress,
                            cancel_check=cancel_check,
                        )
                        ensure_active()
                        if run.rtf is not None:
                            rtf_runs.append(float(run.rtf))
                        if result is None:
                            result = run
                    assert result is not None
                    output = np.asarray(result.outputs.get("audio"), dtype=np.float32).reshape(-1)
                    latency_samples = int(result.metadata.get("latency_samples", 0))
                    aligned_output = align_streaming_output(
                        output,
                        latency_samples=latency_samples,
                        target_samples=input_values.size,
                    )
                    progress(
                        model_end - (model_end - model_start) * 0.04,
                        f"model_{index + 1}_of_{model_count}:scoring_output",
                    )
                    reference_free = reference_free_metrics(
                        aligned_output,
                        result.sample_rate or sample_rate,
                        include_dnsmos=include_dnsmos,
                    )
                    duration = input_values.size / float(sample_rate)
                    model_rtf = float(np.median(rtf_runs)) if rtf_runs else result.rtf
                    model_report.update(
                        {
                            "display_name": model.display_name,
                            "provider": result.provider,
                            "elapsed_seconds": result.elapsed_seconds + stage_elapsed,
                            # A chain's RTF is every stage's time plus the
                            # model's, over the clip's duration.
                            "rtf": (model_rtf + stage_elapsed / duration) if model_rtf is not None and duration > 0 else model_rtf,
                            "rtf_runs": rtf_runs,
                            "latency_samples": latency_samples + sum(stage["latency_samples"] for stage in stage_reports),
                            "stages": stage_reports,
                            # The post-graph stages as the runtime applied them,
                            # so the report says what ran rather than what was
                            # requested.  ``None`` means no onset guard.
                            "onset_guard": result.metadata.get("onset_guard"),
                            "dry_blend": result.metadata.get("dry_blend"),
                            "output": audio_metrics(aligned_output, result.sample_rate or sample_rate),
                            "quality": reference_metrics(reference_values, aligned_output, result.sample_rate or sample_rate) if reference_values is not None else {},
                            "reference_free": reference_free,
                        }
                    )
                    output_key = f"model-{index}"
                    downloadable[output_key] = StoredOutput(
                        _audio_wav_bytes(aligned_output, result.sample_rate or sample_rate),
                        "audio/wav",
                        f"{_safe_filename(model_id, 'model')}.wav",
                        time.time(),
                    )
                    model_report["_output_key"] = output_key
                except InferenceCancelled:
                    raise
                except Exception as exc:
                    model_report["error"] = str(exc)
                reports.append(model_report)
                progress(model_end, f"model_{index + 1}_of_{model_count}:complete")
            successes = sum("error" not in report for report in reports)
            if successes == 0 and require_success:
                errors = "; ".join(
                    f"{report['model_id']}: {report.get('error', 'unknown error')}"
                    for report in reports
                )
                raise WebServiceError(f"all selected models failed: {errors}")
            token = self.runs.put(downloadable)
            run_url = lambda key: f"/api/runs/{token}/{urllib.parse.quote(key, safe='')}"  # noqa: E731
            for report in reports:
                output_key = report.pop("_output_key", None)
                if output_key is not None:
                    report["output_url"] = run_url(output_key)
            baseline["output_url"] = run_url("input")
            ensure_active()
            return {
                "sample_rate": sample_rate,
                "input": input_stats,
                "reference": reference_stats,
                "input_files": {
                    str(name): _safe_filename(value.get("filename") or value.get("name"))
                    for name, value in clip.items()
                    if isinstance(value, Mapping) and (value.get("filename") or value.get("name"))
                },
                "baseline": baseline,
                "reference_url": run_url("reference") if reference_values is not None else None,
                "models": reports,
                "summary": {
                    "succeeded": successes,
                    "failed": len(reports) - successes,
                },
                "elapsed_seconds": time.time() - clip_started,
            }
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    # Live -------------------------------------------------------------------
    def live_session(self, socket: WebSocket) -> None:
        """Stream microphone audio through a model and back, frame by frame.

        The first message is a JSON configuration (``model_id``, optional
        ``variant``, ``provider``, ``parameters``).  After the ``ready`` reply
        every binary message is ``uint32 sequence`` + float32 samples at the
        model rate; each is answered with ``uint32 sequence`` + ``float32
        processing milliseconds`` + the samples the stream emitted.  The reply
        stream carries the model's output ``latency_samples`` late, exactly as
        the offline runtime's does.  ``{"type": "stop"}`` ends the session.
        """

        opcode, message = socket.receive()
        if opcode != _WS_TEXT:
            socket.send_json({"type": "error", "message": "the first message must be the JSON configuration"})
            return
        try:
            config = json.loads(message.decode("utf-8"))
            if not isinstance(config, Mapping):
                raise ValueError("configuration must be an object")
            model_id = str(config.get("model_id") or "")
            model = self.zoo.get(model_id)
            if model.task not in ENHANCEMENT_TASKS:
                raise WebServiceError("live mode runs audio-in/audio-out enhancement models")
            provider = normalize_provider(config.get("provider") or "auto")
            parameters = config.get("parameters") or {}
            if not isinstance(parameters, Mapping):
                raise WebServiceError("parameters must be an object")
            runtime = self._runtime(model_id, provider, config.get("variant") or None)
            if not hasattr(runtime, "open_stream"):
                raise WebServiceError(f"{model_id} has no streaming runtime")
            stream = runtime.open_stream(dict(parameters))
        except (KeyError, ValueError, TypeError, WebServiceError, InferenceError, ModelZooError) as exc:
            socket.send_json({"type": "error", "message": str(exc)})
            return
        rate = int(stream.sample_rate)
        latency = int(getattr(stream, "dry_delay", 0))
        socket.send_json({
            "type": "ready",
            "model_id": model.id,
            "display_name": model.display_name,
            "sample_rate": rate,
            "hop_length": int(stream.hop_length),
            "win_length": int(getattr(stream, "win_length", stream.hop_length)),
            "latency_samples": latency,
            "latency_ms": 1000.0 * latency / rate,
            "provider": (getattr(stream, "providers", None) or [""])[0],
            "dry_blend": float(getattr(stream, "dry_blend", 1.0)),
            "onset_guard": stream.onset_guard.as_manifest() if getattr(stream, "onset_guard", None) is not None else None,
            "max_seconds": _LIVE_MAX_SECONDS,
        })
        job = self.jobs.create(model.id, kind="live")
        self.jobs.start(job.job_id, phase="streaming", progress=0.0)
        kept_inputs: list[np.ndarray] = []
        kept_outputs: list[np.ndarray] = []
        kept = 0
        processed = 0
        busy = 0.0
        window_samples = 0
        window_busy = 0.0
        ended = "disconnected"
        kept_already = False

        def keep(reason: str) -> None:
            # Before "stopped" goes out, so the run it names already exists.
            nonlocal kept_already, ended
            if not kept_already:
                kept_already = True
                ended = reason
                self._keep_live_session(job.job_id, model, stream, kept_inputs, kept_outputs, processed=processed, busy=busy, ended=reason)

        try:
            while True:
                opcode, message = socket.receive()
                if opcode == _WS_TEXT:
                    try:
                        command = json.loads(message.decode("utf-8"))
                    except ValueError:
                        command = {}
                    if isinstance(command, Mapping) and command.get("type") == "stop":
                        keep("stopped")
                        socket.send_json({"type": "stopped", "processed_seconds": processed / rate, "rtf": busy / max(processed / rate, 1e-9), "job_id": job.job_id})
                        return
                    continue
                if len(message) < 4 or (len(message) - 4) % 4:
                    socket.send_json({"type": "error", "message": "binary frames are uint32 sequence + float32 samples"})
                    return
                sequence = struct.unpack("<I", message[:4])[0]
                samples = np.frombuffer(message, dtype="<f4", offset=4)
                if samples.size > _LIVE_MAX_CHUNK:
                    socket.send_json({"type": "error", "message": f"at most {_LIVE_MAX_CHUNK} samples per message"})
                    return
                started = time.perf_counter()
                output = np.asarray(stream.process_samples(samples.astype(np.float32)), dtype="<f4")
                elapsed = time.perf_counter() - started
                processed += samples.size
                busy += elapsed
                window_samples += samples.size
                window_busy += elapsed
                if kept < _LIVE_KEEP_SECONDS * rate:
                    kept_inputs.append(samples.astype(np.float32))
                    kept_outputs.append(output.astype(np.float32))
                    kept += samples.size
                socket.send(_WS_BINARY, struct.pack("<If", sequence, 1000.0 * elapsed) + output.tobytes())
                if window_samples >= rate:
                    socket.send_json({"type": "stats", "rtf": window_busy / (window_samples / rate), "processed_seconds": processed / rate})
                    window_samples = 0
                    window_busy = 0.0
                if processed >= _LIVE_MAX_SECONDS * rate:
                    keep("time limit")
                    socket.send_json({"type": "stopped", "reason": "time limit", "processed_seconds": processed / rate, "job_id": job.job_id})
                    return
        finally:
            keep(ended)

    def _keep_live_session(
        self,
        job_id: str,
        model: Any,
        stream: Any,
        inputs: list[np.ndarray],
        outputs: list[np.ndarray],
        *,
        processed: int,
        busy: float,
        ended: str,
    ) -> None:
        """Put a finished live session in the run history, like any other run.

        The first ``_LIVE_KEEP_SECONDS`` are kept, on the input's time axis: the
        reply stream's look-ahead is dropped, and the input is cut to what has
        a matching output (the last window never came back).
        """

        rate = int(stream.sample_rate)
        latency = int(getattr(stream, "dry_delay", 0))
        source = np.concatenate(inputs) if inputs else np.zeros(0, dtype=np.float32)
        emitted = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
        aligned = emitted[latency:]
        length = min(source.size, aligned.size)
        if length < rate // 2:
            self.jobs.fail(job_id, "live session shorter than half a second; nothing kept")
            return
        source, aligned = source[:length], aligned[:length]
        now = time.time()
        try:
            token = self.runs.put({
                "input": StoredOutput(_audio_wav_bytes(source, rate), "audio/wav", "live_input.wav", now),
                "aligned": StoredOutput(_audio_wav_bytes(aligned, rate), "audio/wav", "live_output.wav", now),
                "removed": StoredOutput(_audio_wav_bytes(source - aligned, rate), "audio/wav", "live_removed.wav", now),
            })
        except (WebServiceError, OSError) as exc:
            self.jobs.fail(job_id, str(exc))
            return
        url = lambda name: f"/api/runs/{token}/{name}"  # noqa: E731
        seconds = processed / rate
        self.jobs.succeed(job_id, {
            "model_id": model.id,
            "task": model.task,
            "sample_rate": rate,
            "rtf": busy / seconds if seconds > 0 else None,
            "provider": (getattr(stream, "providers", None) or [""])[0],
            "run_id": token,
            "output_urls": {"aligned": url("aligned"), "input": url("input"), "removed": url("removed")},
            "alignment": {"latency_samples": latency, "latency_ms": 1000.0 * latency / rate, "samples": int(length), "sample_rate": rate},
            "input_files": {"audio": "live session"},
            "metadata": {
                "duration_seconds": length / rate,
                "latency_samples": latency,
                "latency_ms": 1000.0 * latency / rate,
                "hop_length": int(stream.hop_length),
                "win_length": int(getattr(stream, "win_length", stream.hop_length)),
                "dry_blend": float(getattr(stream, "dry_blend", 1.0)),
                "onset_guard": stream.onset_guard.as_manifest() if getattr(stream, "onset_guard", None) is not None else None,
            },
            "live": {"streamed_seconds": seconds, "kept_seconds": length / rate, "ended": ended},
            "job_id": job_id,
            "host": host_load(),
        })

    def _parse_stages(self, raw: Any) -> list[dict[str, Any]]:
        """``stages``: enhancement models to run, in order, before the requested one."""

        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            raise WebServiceError("stages must be an array")
        if len(raw) > _MAX_STAGES:
            raise WebServiceError(f"at most {_MAX_STAGES} stages may run before a model")
        stages = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping) or not str(item.get("model_id") or "").strip():
                raise WebServiceError(f"stages[{index}] needs a model_id")
            stage_parameters = item.get("parameters") or {}
            if not isinstance(stage_parameters, Mapping):
                raise WebServiceError(f"stages[{index}].parameters must be an object")
            model_id = str(item["model_id"])
            try:
                task = self.zoo.get(model_id).task
            except KeyError as exc:
                raise WebServiceError(f"stages[{index}]: unknown model {model_id}") from exc
            if task not in ENHANCEMENT_TASKS:
                raise WebServiceError(f"stages[{index}]: only audio-in/audio-out enhancement models can be chained")
            variant = item.get("variant")
            stages.append({"model_id": model_id, "variant": str(variant) if variant else None, "parameters": dict(stage_parameters)})
        return stages

    def _run_stages(
        self,
        stages: list[dict[str, Any]],
        source: Any,
        *,
        provider: str,
        sample_rate: int,
        progress: Callable[[float, str], Any],
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[np.ndarray]]:
        """Run each stage on the previous one's aligned output.

        Every stage output is cut back onto the input's time axis (streaming
        latency removed, input length kept), so the next model -- and the
        final comparison -- see a signal sample-aligned with the input.
        """

        original, _ = load_audio(source, sample_rate=sample_rate)
        current: Any = source
        reports: list[dict[str, Any]] = []
        outputs: list[np.ndarray] = []
        for index, stage in enumerate(stages):
            stage_runtime = self._runtime(stage["model_id"], provider, stage["variant"])
            stage_rate = int(getattr(stage_runtime, "sample_rate", sample_rate) or sample_rate)
            if stage_rate != sample_rate:
                raise WebServiceError(
                    f"stage {stage['model_id']} runs at {stage_rate} Hz but the model after it at {sample_rate} Hz"
                )

            def stage_progress(value: float, phase: str, index: int = index) -> None:
                progress((index + value) / len(stages), f"stage_{index + 1}_of_{len(stages)}:{phase}")

            run = stage_runtime.infer(
                inputs={"audio": current},
                parameters=dict(stage["parameters"]),
                progress_callback=stage_progress,
                cancel_check=cancel_check,
            )
            latency = int(run.metadata.get("latency_samples", 0) or 0)
            aligned = align_streaming_output(
                np.asarray(run.outputs["audio"], dtype=np.float32).reshape(-1),
                latency_samples=latency,
                target_samples=original.size,
            )
            if aligned.size < original.size:
                aligned = np.pad(aligned, (0, original.size - aligned.size))
            model = self.zoo.get(stage["model_id"])
            reports.append({
                "model_id": stage["model_id"],
                "display_name": model.display_name,
                "elapsed_seconds": float(run.elapsed_seconds),
                "rtf": run.rtf,
                "latency_samples": latency,
                "latency_ms": 1000.0 * latency / sample_rate,
                "dry_blend": run.metadata.get("dry_blend"),
            })
            outputs.append(aligned)
            current = aligned
        return current, reports, outputs

    def _measurement_candidates(
        self, payload: Mapping[str, Any], shared: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """What a comparison runs: ``candidates`` (a model and its own
        parameters each), else ``models`` with the shared parameters, else the
        release defaults.  One model may appear several times with different
        parameters -- a dry-blend or onset-guard sweep is a comparison too."""

        raw = payload.get("candidates")
        if raw is not None:
            if not isinstance(raw, (list, tuple)) or not raw:
                raise WebServiceError("candidates must be a non-empty array")
            candidates = []
            for index, item in enumerate(raw):
                if not isinstance(item, Mapping) or not str(item.get("model_id") or "").strip():
                    raise WebServiceError(f"candidates[{index}] needs a model_id")
                own = item.get("parameters") or {}
                if not isinstance(own, Mapping):
                    raise WebServiceError(f"candidates[{index}].parameters must be an object")
                model_id = str(item["model_id"])
                label = str(item.get("label") or "").strip() or model_id
                candidates.append({
                    "model_id": model_id,
                    "parameters": {**shared, **own},
                    "label": label,
                    "stages": self._parse_stages(item.get("stages")),
                })
            return candidates
        model_ids = payload.get("models") or []
        if not isinstance(model_ids, (list, tuple)):
            raise WebServiceError("models must be an array of model ids")
        model_ids = [str(model_id) for model_id in model_ids if str(model_id).strip()]
        if not model_ids:
            model_ids = [
                model.id
                for task in ENHANCEMENT_TASKS
                for model in self.zoo.list(task=task, runnable_only=True)
                if "default" in model.roles
            ]
        return [{"model_id": model_id, "parameters": dict(shared), "label": model_id, "stages": []} for model_id in model_ids]

    # HTTP --------------------------------------------------------------------
    def create_server(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        *,
        tls: tuple[str | Path, str | Path] | None = None,
    ) -> ThreadingHTTPServer:
        service = self

        class Handler(_RequestHandler):
            app = service
            trusted_hosts = service.allowed_hosts
            # A server only the local machine can reach is still reachable by
            # a web page the user opens, through DNS rebinding; such a page
            # addresses it by a name that is not its own.
            allowed_hosts = (
                frozenset({"localhost", "127.0.0.1", "::1", _authority_host(host), *service.allowed_hosts})
                if _binds_loopback_only(host)
                else None
            )

        server = _Server((host, int(port)), Handler)
        if tls is not None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(tls[0]), str(tls[1]))
            # The handshake runs in each request's own thread (see
            # _RequestHandler.setup), so a slow or plain-HTTP client cannot
            # stall the accept loop.
            server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        server.tls = tls is not None
        return server


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    tls = False

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A browser that gave up on a self-signed certificate, or plain HTTP
        # sent to the HTTPS port: not worth a traceback.
        if isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def self_signed_certificate(directory: str | Path, host: str) -> tuple[Path, Path]:
    """A self-signed certificate for ``host``, made once with the openssl CLI.

    Browsers only give a page the microphone over HTTPS (or on localhost), so
    serving the UI to another machine needs a certificate; this one makes the
    browser warn once and then work.
    """

    directory = Path(directory).expanduser()
    cert, key = directory / "cert.pem", directory / "key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    if shutil.which("openssl") is None:
        raise WebServiceError("--https needs the openssl command to make a certificate; or pass --tls-cert and --tls-key")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    names = ["DNS:localhost", "IP:127.0.0.1"]
    if host not in {"", "0.0.0.0", "::", "localhost", "127.0.0.1"}:
        names.append(f"IP:{host}" if _is_ip_literal(host) else f"DNS:{host}")
    try:
        names += [f"IP:{address}" for address in {info[4][0] for info in socket_module.getaddrinfo(socket_module.gethostname(), None, socket_module.AF_INET)}]
    except OSError:
        pass
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "825",
            "-keyout", str(key), "-out", str(cert), "-subj", "/CN=PureSound local",
            "-addext", "subjectAltName=" + ",".join(dict.fromkeys(names)),
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    return cert, key


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP adapter kept private so the application object remains testable."""

    app: WebService
    server_version = "PureSoundWeb/0.1"
    #: Host names a request may address; ``None`` accepts any.
    allowed_hosts: frozenset[str] | None = None
    #: Host names whose origin may write although its host differs from the
    #: request's own (a reverse proxy's public name).
    trusted_hosts: frozenset[str] = frozenset()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # The CLI can be used in notebooks and tests; keep access logs quiet.
        return

    def setup(self) -> None:
        if isinstance(self.request, ssl.SSLSocket):
            self.request.settimeout(10)
            self.request.do_handshake()
            self.request.settimeout(None)
        super().setup()

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json; charset=utf-8",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        # The whole app is cross-origin isolated, so WebAssembly in the
        # Playground's on-device runs can use threads.  Everything the page
        # loads comes from this server, which require-corp allows; a worker's
        # script needs the headers as well as the page.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: Any) -> None:
        self._send(status, _json_bytes(body))

    def _error(self, status: int, message: str, code: str = "request_error") -> None:
        self._json(status, {"error": {"code": code, "message": message}})

    def _refuse_request(self, *, writes: bool) -> bool:
        """Answer 403 and return True for a request that is not meant for this server.

        The Host header must name this server; a request that changes state or
        opens a live session must also come from this server's own pages, so
        another site cannot drive it through the user's browser.
        """

        host = (self.headers.get("Host") or "").strip()
        name = _authority_host(host)
        if self.allowed_hosts is not None and host and name not in self.allowed_hosts and not _is_ip_literal(name):
            self._error(
                HTTPStatus.FORBIDDEN,
                f"this server does not answer to the host name {name!r}; pass --allow-host or add it to {ALLOWED_HOSTS_ENV}",
                "forbidden_host",
            )
            return True
        origin = self.headers.get("Origin")
        if writes and origin is not None:
            site = urllib.parse.urlsplit(origin).netloc.lower()
            if site != host.lower() and _authority_host(site) not in self.trusted_hosts:
                self._error(HTTPStatus.FORBIDDEN, "cross-origin requests are not accepted", "forbidden_origin")
                return True
        return False

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self._refuse_request(writes=False):
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/health":
                models = self.app.list_models(include_empty=True)
                ort_providers = available_providers()
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "puresound",
                        "schema_version": self.app.zoo.schema_version,
                        "models": len(models),
                        "runnable_models": sum(bool(item["runnable"]) for item in models),
                        "available_providers": ort_providers,
                        "provider_capabilities": {
                            "cpu": CPU_PROVIDER in ort_providers,
                            "cuda": CUDA_PROVIDER in ort_providers,
                            "coreml": COREML_PROVIDER in ort_providers,
                        },
                        "host": host_load(),
                        "pipeline": self.app.pipeline.status(),
                    },
                )
                return
            if path == "/api/world":
                self._json(HTTPStatus.OK, self.app.world_overview())
                return
            if path == "/api/pipeline":
                self._json(HTTPStatus.OK, self.app.pipeline_overview())
                return
            if path == "/api/models":
                task = query.get("task", [None])[0]
                include_empty = query.get("include_empty", ["0"])[0].lower() in {"1", "true", "yes"}
                self._json(HTTPStatus.OK, {"models": self.app.list_models(task, include_empty)})
                return
            if path.startswith("/api/models/") and path.endswith("/benchmarks"):
                model_id = urllib.parse.unquote(path[len("/api/models/") : -len("/benchmarks")])
                self._json(HTTPStatus.OK, self.app.model_benchmarks(model_id))
                return
            if path.startswith("/api/models/"):
                model_id = urllib.parse.unquote(path[len("/api/models/") :])
                self._json(HTTPStatus.OK, self.app.inspect_model(model_id))
                return
            if path == "/api/asr":
                self._json(HTTPStatus.OK, self.app.asr_capabilities())
                return
            if path == "/api/live":
                self._upgrade_live()
                return
            if path == "/api/validate":
                verify_hash = query.get("hash", ["1"])[0].lower() not in {"0", "false", "no"}
                check_graph = query.get("graph", ["1"])[0].lower() not in {"0", "false", "no"}
                self._json(
                    HTTPStatus.OK,
                    self.app.validate(verify_hash=verify_hash, check_graph=check_graph),
                )
                return
            if path == "/api/jobs":
                try:
                    limit = int(query.get("limit", ["20"])[0])
                except ValueError:
                    limit = 20
                self._json(HTTPStatus.OK, {"jobs": self.app.list_jobs(limit)})
                return
            export = re.fullmatch(r"/api/jobs/([^/]+)/(outputs\.zip|report\.html)", path)
            if export:
                body, content_type, filename = self.app.job_export(
                    urllib.parse.unquote(export.group(1)),
                    "zip" if export.group(2).endswith(".zip") else "report",
                    tz=query.get("tz", [None])[0],
                )
                self._send(HTTPStatus.OK, body, content_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})
                return
            if path.startswith("/api/jobs/"):
                job_id = urllib.parse.unquote(path[len("/api/jobs/") :])
                state = self.app.job(job_id)
                if query.get("summary", ["0"])[0] == "1" and state["status"] in {"queued", "running"}:
                    from puresound.web.world import sweep_progress
                    state = {**state, "result": sweep_progress(state.get("result"))}
                self._json(HTTPStatus.OK, state)
                return
            if path.startswith("/api/uploads/"):
                described = self.app.uploads.describe(urllib.parse.unquote(path[len("/api/uploads/") :]))
                if described is None:
                    self._error(HTTPStatus.NOT_FOUND, "upload not found", "not_found")
                else:
                    self._json(HTTPStatus.OK, described)
                return
            if path.startswith("/api/runs/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[1:3] != ["api", "runs"]:
                    self._error(HTTPStatus.NOT_FOUND, "output not found", "not_found")
                    return
                token, output_name = parts[3], urllib.parse.unquote(parts[4])
                stored = self.app.run_output(token, output_name)
                if stored is None:
                    self._error(HTTPStatus.NOT_FOUND, "output not found", "not_found")
                    return
                self._send(
                    HTTPStatus.OK,
                    stored.data,
                    stored.content_type,
                )
                return
            self._serve_static(path)
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc), "not_found")
        except (ModelZooError, WebServiceError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), "internal_error")

    def do_POST(self) -> None:  # noqa: N802
        if self._refuse_request(writes=True):
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = urllib.parse.unquote(path[len("/api/jobs/") : -len("/cancel")])
            try:
                self._json(HTTPStatus.OK, self.app.cancel_job(job_id))
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), "not_found")
            return
        if path == "/api/uploads":
            self._receive_upload()
            return
        if path not in {"/api/infer", "/api/jobs", "/api/measure", "/api/world/validate", "/api/world/materials"}:
            self._error(HTTPStatus.NOT_FOUND, "endpoint not found", "not_found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length", "invalid_request")
            return
        if content_length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "request body is empty", "invalid_request")
            return
        # JSON/base64 has a modest expansion over the original audio.  Keep a
        # separate body ceiling so a malicious request cannot allocate freely.
        if content_length > self.app.max_upload_bytes * 2:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large", "payload_too_large")
            return
        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            if path == "/api/jobs":
                kind = payload.get("kind", "inference") if isinstance(payload, Mapping) else "inference"
                if kind == "measurement":
                    response = self.app.submit_measurement(payload)
                elif kind == "transcription":
                    response = self.app.submit_transcription(payload)
                elif kind == "inference":
                    response = self.app.submit_inference(payload)
                elif kind in {"world_render", "world_sweep"}:
                    response = self.app.submit_world(payload)
                elif kind == "pipeline_trace":
                    response = self.app.submit_pipeline_trace(payload)
                else:
                    raise WebServiceError("job kind must be one of: inference, measurement, transcription, pipeline_trace, world_render, world_sweep")
                self._json(HTTPStatus.ACCEPTED, response)
            elif path == "/api/world/validate":
                self._json(HTTPStatus.OK, self.app.validate_world(payload))
            elif path == "/api/world/materials":
                self._json(HTTPStatus.OK, self.app.world_materials(payload))
            elif path == "/api/measure":
                response = self.app.measure(payload)
                self._json(HTTPStatus.OK, response)
            else:
                response = self.app.infer(payload)
                self._json(HTTPStatus.OK, response)
        except json.JSONDecodeError as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc.msg}", "invalid_json")
        except WebServiceError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), "internal_error")

    def _upgrade_live(self) -> None:
        if self._refuse_request(writes=True):
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if "websocket" not in (self.headers.get("Upgrade") or "").lower() or not key:
            self._error(HTTPStatus.BAD_REQUEST, "live mode needs a WebSocket upgrade", "upgrade_required")
            return
        # Written by hand: the handler speaks HTTP/1.0, and a WebSocket
        # handshake must answer with an HTTP/1.1 status line.
        self.wfile.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {WebSocket.accept_key(key)}\r\n\r\n"
            ).encode("ascii")
        )
        self.wfile.flush()
        self.close_connection = True
        try:
            # 20 ms frames each way: without this, Nagle holds small replies back.
            self.connection.setsockopt(socket_module.IPPROTO_TCP, socket_module.TCP_NODELAY, 1)
        except OSError:
            pass
        socket = WebSocket(self.rfile, self.wfile)
        try:
            self.app.live_session(socket)
        except (WebSocketClosed, ConnectionError, OSError):
            return
        except Exception as exc:  # pragma: no cover - defensive session boundary
            try:
                socket.send_json({"type": "error", "message": str(exc)})
            except OSError:
                pass
        try:
            socket.send(_WS_CLOSE, struct.pack("!H", 1000))
        except OSError:
            pass

    def _receive_upload(self) -> None:
        """POST /api/uploads: the file's bytes as the body, its name in X-Filename."""

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length", "invalid_request")
            return
        if content_length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "upload is empty", "invalid_request")
            return
        if content_length > self.app.max_upload_bytes:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"uploaded audio exceeds the {self.app.max_upload_bytes // (1024 * 1024)} MiB limit",
                "payload_too_large",
            )
            return
        filename = urllib.parse.unquote(self.headers.get("X-Filename") or "upload.wav")
        data = self.rfile.read(content_length)
        if len(data) != content_length:
            self._error(HTTPStatus.BAD_REQUEST, "the upload ended before its Content-Length", "invalid_request")
            return
        try:
            stored = self.app.uploads.put(data, filename)
        except WebServiceError as exc:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc), "payload_too_large")
            return
        self._json(HTTPStatus.CREATED, stored)

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        root = self.app.static_dir.resolve()
        try:
            candidate = (root / relative).resolve()
            found = root in candidate.parents and candidate.is_file()
        except (OSError, ValueError):  # a name too long for the file system, a NUL byte
            found = False
        if not found:
            self._error(HTTPStatus.NOT_FOUND, "asset not found", "not_found")
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, candidate.read_bytes(), content_type)


def create_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    zoo: ModelZoo | None = None,
    static_dir: str | Path | None = None,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
    max_runs: int = 32,
    allow_local_paths: bool = False,
    history_dir: str | Path | None = None,
    tls: tuple[str | Path, str | Path] | None = None,
    pipeline_root: str | Path | None = None,
    allowed_hosts: Iterable[str] | None = None,
) -> ThreadingHTTPServer:
    """Create (but do not start) a threaded PureSound web server.

    A server bound to a loopback address answers only to loopback host names
    and its own bind address, and every server refuses cross-origin writes;
    ``allowed_hosts`` (and ``PURESOUND_ALLOWED_HOSTS``) add the names it may
    also be reached by, such as a reverse proxy's public name.
    """

    service = WebService(
        zoo=zoo,
        static_dir=static_dir,
        max_upload_bytes=max_upload_bytes,
        max_runs=max_runs,
        allow_local_paths=allow_local_paths,
        history_dir=history_dir,
        pipeline_root=pipeline_root,
        allowed_hosts=allowed_hosts,
    )
    return service.create_server(host, port, tls=tls)


def run(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    zoo: ModelZoo | None = None,
    static_dir: str | Path | None = None,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
    max_runs: int = 32,
    allow_local_paths: bool = False,
    history_dir: str | Path | None = None,
    tls: tuple[str | Path, str | Path] | None = None,
    pipeline_root: str | Path | None = None,
    allowed_hosts: Iterable[str] | None = None,
) -> None:
    """Run the local web service until interrupted."""

    server = create_server(
        host,
        port,
        zoo=zoo,
        static_dir=static_dir,
        max_upload_bytes=max_upload_bytes,
        max_runs=max_runs,
        allow_local_paths=allow_local_paths,
        history_dir=history_dir,
        tls=tls,
        pipeline_root=pipeline_root,
        allowed_hosts=allowed_hosts,
    )
    print(f"PureSound web UI: {'https' if tls else 'http'}://{host}:{server.server_port}")
    if tls:
        print("HTTPS: a self-signed certificate makes the browser warn once; accept it to continue.")
    if history_dir:
        print(f"Run history: {Path(history_dir).expanduser()} (--no-history keeps it in memory)")
    if pipeline_root:
        print(f"Pipeline inspector recipes: {Path(pipeline_root).expanduser()}/egs/*/config/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["WebService", "WebServiceError", "create_server", "run", "self_signed_certificate"]
