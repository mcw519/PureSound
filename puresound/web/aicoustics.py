"""Optional ai-coustics SDK adapter for application comparisons only.

No training or renderer imports. Weights may be shared; a processor and its
credential belong to one call. Returned metadata never contains credentials.
API: https://docs.ai-coustics.com/reference/sdk/language-bindings/python
"""

from dataclasses import dataclass
import copy
import hashlib
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

SDK_VERSION = "3.3.0"
MODELS = {
    "quail-vf-2.2-l-16khz": ("Quail Voice Focus · L", "voice_isolation"),
    "quail-vf-2.2-s-16khz": ("Quail Voice Focus · S", "voice_isolation"),
    "quail-ms-l-16khz": ("Quail Multi Speaker · L", "noise_suppression"),
    "quail-ms-s-16khz": ("Quail Multi Speaker · S", "noise_suppression"),
    "rook-ms-l-16khz": ("Rook Multi Speaker · L", "noise_suppression"),
    "rook-ms-s-16khz": ("Rook Multi Speaker · S", "noise_suppression"),
}
DEFAULT_MODEL = "quail-vf-2.2-l-16khz"
_models = {}
_model_lock = threading.Lock()


class AicousticsError(ValueError):
    """A safe, credential-free error suitable for a job or report."""


@dataclass
class AicousticsResult:
    output: np.ndarray
    metadata: dict


def capabilities():
    try:
        version = importlib.metadata.version("aic-sdk")
        installed = importlib.util.find_spec("aic_sdk") is not None
    except (importlib.metadata.PackageNotFoundError, ImportError, ValueError):
        version, installed = None, False
    return {
        "available": installed and version == SDK_VERSION,
        "package_version": version,
        "required_version": SDK_VERSION,
        "install": 'uv pip install "aic-sdk==3.3.0"',
        "default_model": DEFAULT_MODEL,
        "models": [
            {"id": mid, "label": label, "task": task}
            for mid, (label, task) in MODELS.items()
        ],
    }


def options(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"enabled", "api_key", "model_id", "enhancement_level"}:
        raise AicousticsError("Invalid ai-coustics comparison settings.")
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise AicousticsError("ai-coustics enabled must be a boolean.")
    if not enabled:
        return None
    model = value.get("model_id", DEFAULT_MODEL)
    if not isinstance(model, str) or model not in MODELS:
        raise AicousticsError("Unknown ai-coustics enhancement model.")
    key = value.get("api_key", "")
    if not isinstance(key, str) or not key.strip() or len(key) > 8192 or "\0" in key:
        raise AicousticsError("Enter the complete ai-coustics SDK key to compare.")
    level = value.get("enhancement_level", 1.0)
    if type(level) not in (int, float) or not math.isfinite(level) or not 0 <= level <= 1:
        raise AicousticsError("ai-coustics enhancement level must be between 0 and 1.")
    return {"model_id": model, "api_key": key.strip(), "enhancement_level": float(level)}


def public_request(payload):
    """A replayable request that requires re-entering its credential."""
    saved = copy.deepcopy(dict(payload))
    if isinstance(saved.get("aicoustics"), dict):
        saved["aicoustics"].pop("api_key", None)
    return saved


def error_message(error, payload):
    message = str(error)
    key = (payload.get("aicoustics") or {}).get("api_key")
    if isinstance(key, str) and key.strip():
        message = message.replace(key, "***").replace(key.strip(), "***")
    return message


def _sdk():
    if not capabilities()["available"]:
        raise AicousticsError('Install aic-sdk==3.3.0 to enable ai-coustics comparison.')
    # Disable optional SDK error telemetry before native initialization.
    # Licensing and usage reporting still follow the account entitlement.
    os.environ.setdefault("DO_NOT_TRACK", "1")
    try:
        import aic_sdk
    except (ImportError, OSError):
        raise AicousticsError("ai-coustics SDK could not be loaded on this server.") from None
    return aic_sdk


def _model(sdk, model_id):
    cache = Path(os.environ.get("PURESOUND_AIC_CACHE", str(Path.home() / ".cache/puresound/ai-coustics")))
    key = (model_id, str(cache))
    with _model_lock:
        if key not in _models:
            cache.mkdir(parents=True, exist_ok=True)
            path = Path(sdk.Model.download(model_id, cache))
            model = sdk.Model.from_file(path)
            _models[key] = (model, hashlib.sha256(path.read_bytes()).hexdigest())
        return _models[key]


def _sdk_error(exc):
    # Do not forward SDK/network diagnostic bodies: credentials can appear
    # in them. Report a useful category instead, including on cleanup paths.
    name = type(exc).__name__
    if name.startswith("License") or name == "TokenUnsupportedError":
        return AicousticsError("ai-coustics rejected the SDK key. Check its format, validity and account access.")
    if name == "ProcessingNotAllowedError":
        return AicousticsError("ai-coustics processing was not authorized. Check the SDK key, account access and network connection.")
    if name in {"ModelDownloadError", "FileSystemError", "OSError"}:
        return AicousticsError("ai-coustics model could not be downloaded or loaded. Check connectivity and the model cache.")
    return AicousticsError("ai-coustics SDK could not process this audio. Check SDK installation, model compatibility and authorization.")


def process(audio, sample_rate, config, *, progress=None):
    """Fixed-block local inference, padded flush and SDK delay compensation.

    Progress exceptions propagate unchanged for the existing job cancellation
    lifecycle. SDK exceptions are translated into credential-free errors.
    """
    config = options({"enabled": True, **config})
    samples = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000 or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise AicousticsError("ai-coustics comparison requires finite 16 kHz mono audio.")
    notify = progress or (lambda value, phase: None)
    notify(0, "aicoustics_model")
    sdk = _sdk()
    setup_start = time.perf_counter()
    try:
        model, checksum = _model(sdk, config["model_id"])
    except Exception as exc:
        raise _sdk_error(exc) from None
    notify(0.02, "aicoustics_inference")
    processor = None
    try:
        try:
            settings = sdk.ProcessorConfig.optimal(model, sample_rate=sample_rate)
            processor = sdk.Processor(model, config["api_key"])
            processor.initialize(settings)
            context = processor.get_context()
            context.set_parameter(sdk.ProcessorParameter.EnhancementLevel, config["enhancement_level"])
            delay = int(context.get_audio_delay())
            block = int(settings.block_size)
            if block < 1 or not 0 <= delay <= 2 * sample_rate:
                raise ValueError("invalid SDK audio geometry")
        except Exception as exc:
            raise _sdk_error(exc) from None
        setup_seconds = time.perf_counter() - setup_start
        length = ((len(samples) + delay + block - 1) // block) * block
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(samples)] = samples
        output = np.empty_like(padded)
        processing_seconds = 0.0
        for start in range(0, length, block):
            # Checking outside SDK exception handlers preserves cancellation.
            if start % (block * 16) == 0:
                notify(0.02 + 0.98 * start / length, "aicoustics_inference")
            began = time.perf_counter()
            try:
                frame = np.asarray(processor.process(padded[start:start + block]), dtype=np.float32)
                if frame.shape != (block,) or not np.isfinite(frame).all():
                    raise ValueError("invalid SDK output")
                output[start:start + block] = frame
            except Exception as exc:
                raise _sdk_error(exc) from None
            processing_seconds += time.perf_counter() - began
        aligned = output[delay:delay + len(samples)].copy()
        info = {
            "provider": "ai-coustics", "model_id": config["model_id"],
            "resolved_model_id": model.get_id(), "display_name": MODELS[config["model_id"]][0],
            "task": MODELS[config["model_id"]][1], "enhancement_level": config["enhancement_level"],
            "package_version": SDK_VERSION, "sdk_version": sdk.get_sdk_version(),
            "model_sha256": checksum, "sample_rate": sample_rate, "block_size": block,
            "audio_delay_samples": delay, "audio_delay_ms": delay * 1000 / sample_rate,
            "alignment": "sdk_delay_compensated", "setup_seconds": setup_seconds,
            "processing_seconds": processing_seconds, "rtf": processing_seconds / (len(samples) / sample_rate),
            "processed_samples": length, "output_samples": len(aligned), "execution": "local_cpu",
        }
        notify(1, "aicoustics_inference")
        return AicousticsResult(aligned, info)
    finally:
        if processor is not None:
            try:
                processor.terminate_session()
            except Exception:
                pass
