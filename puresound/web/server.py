"""Dependency-light HTTP API for the local PureSound inference workspace.

The service deliberately uses the Python standard library rather than adding
another web framework.  It is intended for a local or trusted network process
and delegates all model contracts to ``puresound.inference``.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import mimetypes
import os
import re
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
from typing import Any, Mapping

import numpy as np

from puresound.inference import InferenceError, InferenceResult, ModelZoo, ModelZooError, load_model
from puresound.web.measurements import audio_metrics, audio_metrics_from_path, reference_metrics


class WebServiceError(RuntimeError):
    """Raised when an API request cannot be fulfilled."""


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024 * 1024


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


def _safe_filename(value: str | None, default: str = "upload.wav") -> str:
    name = Path(str(value or default)).name
    name = _SAFE_NAME.sub("_", name).strip("._")
    return name or default


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


@dataclass(frozen=True)
class StoredOutput:
    data: bytes
    content_type: str
    filename: str
    created_at: float


class RunStore:
    """Bounded in-memory store for downloadable inference outputs."""

    def __init__(self, max_runs: int = 32):
        self.max_runs = max(1, int(max_runs))
        self._runs: dict[str, dict[str, StoredOutput]] = {}
        self._lock = threading.Lock()

    def put(self, outputs: Mapping[str, StoredOutput]) -> str:
        token = uuid.uuid4().hex
        now = time.time()
        stamped = {
            name: StoredOutput(item.data, item.content_type, item.filename, now)
            for name, item in outputs.items()
        }
        with self._lock:
            self._runs[token] = stamped
            while len(self._runs) > self.max_runs:
                oldest = min(self._runs, key=lambda key: min(item.created_at for item in self._runs[key].values()))
                self._runs.pop(oldest, None)
        return token

    def get(self, token: str, output_name: str) -> StoredOutput | None:
        with self._lock:
            return self._runs.get(token, {}).get(output_name)


@dataclass
class InferenceJob:
    """Small in-memory job record used by the browser progress workflow."""

    job_id: str
    model_id: str
    created_at: float
    status: str = "queued"
    phase: str = "queued"
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False


class InferenceJobStore:
    """Thread-safe bounded history for asynchronous inference requests."""

    def __init__(self, max_jobs: int = 64):
        self.max_jobs = max(1, int(max_jobs))
        self._jobs: dict[str, InferenceJob] = {}
        self._lock = threading.RLock()

    def create(self, model_id: str) -> InferenceJob:
        job = InferenceJob(uuid.uuid4().hex, model_id, time.time())
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

    def get(self, job_id: str) -> InferenceJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"job not found: {job_id}") from exc

    def update(self, job_id: str, **values: Any) -> InferenceJob:
        with self._lock:
            job = self.get(job_id)
            for name, value in values.items():
                setattr(job, name, value)
            return job

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

    def list(self, limit: int = 20) -> list[InferenceJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)[: max(1, min(int(limit), self.max_jobs))]

    @staticmethod
    def snapshot(job: InferenceJob) -> dict[str, Any]:
        elapsed = None
        if job.started_at is not None:
            elapsed = (job.finished_at or time.time()) - job.started_at
        return {
            "job_id": job.job_id,
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
    ):
        self.zoo = zoo or ModelZoo.default()
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).with_name("static")
        self.max_upload_bytes = max(1, int(max_upload_bytes))
        self.allow_local_paths = bool(allow_local_paths)
        self.runs = RunStore(max_runs=max_runs)
        self.jobs = InferenceJobStore(max_jobs=max(16, max_runs * 2))
        self._runtime_cache: dict[tuple[str, str, str | None], Any] = {}
        self._runtime_lock = threading.Lock()

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
        task_order = {"voice_isolation": 0, "speaker_embedding": 1}
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
                self._runtime_cache[key] = runtime
        return runtime

    def _materialize_input(self, value: Any, temp_paths: list[Path]) -> Any:
        """Turn a browser input descriptor into a temporary local file path."""

        if isinstance(value, Mapping):
            filename = _safe_filename(value.get("filename") or value.get("name"))
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
        return path

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise WebServiceError("model_id is required")
        provider = str(payload.get("provider") or "auto").lower()
        if provider not in {"auto", "cpu", "cuda"}:
            raise WebServiceError("provider must be one of: auto, cpu, cuda")
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise WebServiceError("parameters must be an object")
        variant = payload.get("variant")
        if variant is not None and not isinstance(variant, str):
            raise WebServiceError("variant must be a string")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise WebServiceError("inputs must be a non-empty object")

        temp_paths: list[Path] = []
        try:
            materialized = {
                str(name): self._materialize_input(value, temp_paths)
                for name, value in inputs.items()
            }
            runtime = self._runtime(model_id, provider, variant)
            result: InferenceResult = runtime.infer(
                inputs=materialized,
                parameters=dict(parameters),
            )
            response = result.as_dict()
            downloadable: dict[str, StoredOutput] = {}
            output_urls: dict[str, str] = {}
            for name, value in result.outputs.items():
                if name == "audio":
                    audio_data = _audio_wav_bytes(value, result.sample_rate or 16000)
                    downloadable[name] = StoredOutput(audio_data, "audio/wav", "enhanced.wav", time.time())
            if downloadable:
                token = self.runs.put(downloadable)
                response["run_id"] = token
                output_urls = {
                    name: f"/api/runs/{token}/{urllib.parse.quote(name, safe='')}"
                    for name in downloadable
                }
            response["output_urls"] = output_urls
            response["input_names"] = list(materialized)
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
    def _run_job(self, job_id: str, payload: Mapping[str, Any]) -> None:
        job = self.jobs.get(job_id)
        if job.status == "cancelled":
            return
        started = time.time()
        self.jobs.update(
            job_id,
            status="running",
            phase="preparing",
            progress=0.08,
            started_at=started,
        )
        try:
            if self.jobs.get(job_id).cancel_requested:
                self.jobs.update(job_id, status="cancelled", phase="cancelled", finished_at=time.time())
                return
            self.jobs.update(job_id, phase="running", progress=0.25)
            result = self.infer(payload)
            job = self.jobs.get(job_id)
            if job.cancel_requested:
                self.jobs.update(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    progress=1.0,
                    finished_at=time.time(),
                )
                return
            self.jobs.update(
                job_id,
                status="succeeded",
                phase="complete",
                progress=1.0,
                result=result,
                finished_at=time.time(),
            )
        except Exception as exc:  # surfaced through the job status endpoint
            job = self.jobs.get(job_id)
            if job.cancel_requested:
                self.jobs.update(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    progress=1.0,
                    finished_at=time.time(),
                )
            else:
                self.jobs.update(
                    job_id,
                    status="failed",
                    phase="failed",
                    error=str(exc),
                    finished_at=time.time(),
                )

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
        thread = threading.Thread(
            target=self._run_job,
            args=(job.job_id, dict(payload)),
            name=f"puresound-infer-{job.job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return InferenceJobStore.snapshot(job)

    def job(self, job_id: str) -> dict[str, Any]:
        return InferenceJobStore.snapshot(self.jobs.get(job_id))

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return InferenceJobStore.snapshot(self.jobs.cancel(job_id))

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return [InferenceJobStore.snapshot(job) for job in self.jobs.list(limit)]

    # Measurements ------------------------------------------------------------
    def measure(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise WebServiceError("request body must be a JSON object")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping) or "audio" not in inputs:
            raise WebServiceError("inputs.audio is required")
        model_ids = payload.get("models") or []
        if not isinstance(model_ids, (list, tuple)):
            raise WebServiceError("models must be an array of model ids")
        model_ids = [str(model_id) for model_id in model_ids if str(model_id).strip()]
        if not model_ids:
            model_ids = [model.id for model in self.zoo.list(task="voice_isolation", runnable_only=True) if "default" in model.roles]
        provider = str(payload.get("provider") or "cpu").lower()
        if provider not in {"auto", "cpu", "cuda"}:
            raise WebServiceError("provider must be one of: auto, cpu, cuda")
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise WebServiceError("parameters must be an object")
        sample_rate = int(payload.get("sample_rate") or 16_000)
        temp_paths: list[Path] = []
        started = time.time()
        try:
            audio_path = self._materialize_input(inputs["audio"], temp_paths)
            _, input_stats = audio_metrics_from_path(audio_path, sample_rate)
            reference_values = None
            reference_stats = None
            if inputs.get("reference") is not None:
                reference_path = self._materialize_input(inputs["reference"], temp_paths)
                reference_values, reference_stats = audio_metrics_from_path(reference_path, sample_rate)
            reports: list[dict[str, Any]] = []
            for model_id in model_ids:
                report: dict[str, Any] = {"model_id": model_id}
                try:
                    model = self.zoo.get(model_id)
                    if model.task != "voice_isolation":
                        raise WebServiceError("measurement comparison only supports voice isolation models")
                    runtime = self._runtime(model_id, provider, None)
                    result: InferenceResult = runtime.infer(inputs={"audio": audio_path}, parameters=dict(parameters))
                    output = np.asarray(result.outputs.get("audio"), dtype=np.float32).reshape(-1)
                    report.update(
                        {
                            "display_name": model.display_name,
                            "provider": result.provider,
                            "elapsed_seconds": result.elapsed_seconds,
                            "rtf": result.rtf,
                            "output": audio_metrics(output, result.sample_rate or sample_rate),
                            "quality": reference_metrics(reference_values, output, result.sample_rate or sample_rate) if reference_values is not None else {},
                        }
                    )
                except Exception as exc:
                    report["error"] = str(exc)
                reports.append(report)
            return {
                "sample_rate": sample_rate,
                "input": input_stats,
                "reference": reference_stats,
                "models": reports,
                "elapsed_seconds": time.time() - started,
            }
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    # HTTP --------------------------------------------------------------------
    def create_server(self, host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
        service = self

        class Handler(_RequestHandler):
            app = service

        return ThreadingHTTPServer((host, int(port)), Handler)


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP adapter kept private so the application object remains testable."""

    app: WebService
    server_version = "PureSoundWeb/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # The CLI can be used in notebooks and tests; keep access logs quiet.
        return

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: Any) -> None:
        self._send(status, _json_bytes(body))

    def _error(self, status: int, message: str, code: str = "request_error") -> None:
        self._json(status, {"error": {"code": code, "message": message}})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/health":
                models = self.app.list_models(include_empty=True)
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "puresound",
                        "schema_version": self.app.zoo.schema_version,
                        "models": len(models),
                        "runnable_models": sum(bool(item["runnable"]) for item in models),
                    },
                )
                return
            if path == "/api/models":
                task = query.get("task", [None])[0]
                include_empty = query.get("include_empty", ["0"])[0].lower() in {"1", "true", "yes"}
                self._json(HTTPStatus.OK, {"models": self.app.list_models(task, include_empty)})
                return
            if path.startswith("/api/models/"):
                model_id = urllib.parse.unquote(path[len("/api/models/") :])
                self._json(HTTPStatus.OK, self.app.inspect_model(model_id))
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
            if path.startswith("/api/jobs/"):
                job_id = urllib.parse.unquote(path[len("/api/jobs/") :])
                self._json(HTTPStatus.OK, self.app.job(job_id))
                return
            if path.startswith("/api/runs/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[1:3] != ["api", "runs"]:
                    self._error(HTTPStatus.NOT_FOUND, "output not found", "not_found")
                    return
                token, output_name = parts[3], urllib.parse.unquote(parts[4])
                stored = self.app.runs.get(token, output_name)
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
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = urllib.parse.unquote(path[len("/api/jobs/") : -len("/cancel")])
            try:
                self._json(HTTPStatus.OK, self.app.cancel_job(job_id))
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), "not_found")
            return
        if path not in {"/api/infer", "/api/jobs", "/api/measure"}:
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
                response = self.app.submit_inference(payload)
                self._json(HTTPStatus.ACCEPTED, response)
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

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (self.app.static_dir / relative).resolve()
        root = self.app.static_dir.resolve()
        if root not in candidate.parents and candidate != root:
            self._error(HTTPStatus.NOT_FOUND, "asset not found", "not_found")
            return
        if not candidate.is_file():
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
) -> ThreadingHTTPServer:
    """Create (but do not start) a threaded PureSound web server."""

    service = WebService(
        zoo=zoo,
        static_dir=static_dir,
        max_upload_bytes=max_upload_bytes,
        max_runs=max_runs,
        allow_local_paths=allow_local_paths,
    )
    return service.create_server(host, port)


def run(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    zoo: ModelZoo | None = None,
    static_dir: str | Path | None = None,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
    max_runs: int = 32,
    allow_local_paths: bool = False,
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
    )
    print(f"PureSound web UI: http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["WebService", "WebServiceError", "create_server", "run"]
