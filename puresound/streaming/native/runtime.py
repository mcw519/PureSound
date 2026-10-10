"""Select an optional CPU companion graph, with a standard-ONNX fallback."""

from dataclasses import dataclass
import os
import hashlib
from pathlib import Path
from typing import Any
import warnings


@dataclass(frozen=True)
class SessionSelection:
    """An ORT session with the graph it actually runs.

    Pass this, not the bare session, to share one session between streams:
    the bare session cannot say whether it is the native companion.
    """

    session: Any
    execution_path: Path
    native_ssm_enabled: bool = False
    native_fallback_reason: str | None = None


def create_session(onnx_path, manifest, providers, *, native_ssm="auto",
                   native_library=None, intra_op_num_threads=None) -> SessionSelection:
    import onnxruntime as ort

    if native_ssm not in {"auto", "off", "required"}:
        raise ValueError("native_ssm must be auto, off, or required")
    if intra_op_num_threads is not None and intra_op_num_threads < 1:
        raise ValueError("intra_op_num_threads must be positive")
    onnx_path = Path(onnx_path)
    config = manifest.get("cpu_optimization") or {}
    threads = intra_op_num_threads
    if threads is None and providers == ["CPUExecutionProvider"]:
        threads = config.get("intra_op_num_threads")
    library = native_library or os.environ.get("PURESOUND_ORT_SSM_LIBRARY")
    graph = config.get("native_graph")
    reason = None
    if native_ssm != "off":
        if providers != ["CPUExecutionProvider"]:
            reason = "native SSM requires the CPU provider"
        elif not graph:
            reason = "manifest has no native companion graph"
        elif not library:
            reason = "no native library specified"
        else:
            # The companion is a sibling of the primary ONNX, even when the
            # manifest is stored elsewhere. Do not follow manifest path traversal.
            if not isinstance(graph, str) or Path(graph).name != graph:
                raise ValueError("native_graph must be a sibling filename")
            native_path = onnx_path.with_name(graph)
            try:
                expected_hash = config.get("native_sha256")
                actual_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
                if not expected_hash or actual_hash != str(expected_hash).lower():
                    raise ValueError("native companion SHA256 mismatch or missing digest")
                options = ort.SessionOptions()
                if threads is not None:
                    options.intra_op_num_threads = int(threads)
                options.register_custom_ops_library(str(Path(library).resolve()))
                session = ort.InferenceSession(str(native_path), sess_options=options,
                                               providers=providers)
                return SessionSelection(session, native_path, True)
            except Exception as exc:
                reason = f"native SSM could not be loaded: {exc}"
                if native_ssm == "auto":
                    warnings.warn(reason + "; using portable ONNX", RuntimeWarning, stacklevel=2)
        if native_ssm == "required":
            raise RuntimeError(reason)
    if threads is None:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
    return SessionSelection(session, onnx_path, False, reason)
