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
        variant = payload.get("variant")
        if variant is not None and not isinstance(variant, str):
            raise WebServiceError("variant must be a string")
        inputs = payload.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise WebServiceError("inputs must be a non-empty object")
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise WebServiceError("parameters must be an object")

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
        if parsed.path.rstrip("/") != "/api/infer":
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
