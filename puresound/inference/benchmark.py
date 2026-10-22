"""Real-time factor of the catalog's streaming artifacts on this machine.

Each artifact runs the way an application runs it: ``load_model(...).infer()``
on one stream, with the provider asked for and the thread count its manifest
recommends. The figure is the facade's own RTF -- STFT, graph, overlap-add and
flush over the clip, divided by the clip's length -- so creating the ONNX
Runtime session is reported on its own, as the cold-start time.

Only streaming (``stft_frame_ort``) artifacts are measured: they are the ones
with a per-frame budget. A shared CI machine or a laptop on battery moves these
numbers by tens of percent, so compare runs from one machine, or read a trend.
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

STREAMING_PROCESSOR = "stft_frame_ort"
SCHEMA_VERSION = 1
#: A packaged speech-in-noise clip, so a run needs no data of its own.
SAMPLE_CLIP = Path(__file__).resolve().parents[1] / "web/static/samples/noisy-speech.wav"


def streaming_artifacts(
    zoo,
    models: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
) -> list[tuple[Any, Any]]:
    """``(model, artifact)`` pairs to measure, by model id.

    A model asked for by id without a streaming artifact, or a variant no
    artifact has, raises instead of selecting nothing: an empty run would
    publish no numbers and still look like a success.
    """
    chosen = [zoo.get(model_id) for model_id in models] if models else zoo.list()
    pairs = [
        (model, artifact)
        for model in chosen
        for artifact in model.artifacts
        if artifact.processor == STREAMING_PROCESSOR
        and (not variants or artifact.variant in variants)
    ]
    for model in chosen if models else ():
        if not any(artifact.processor == STREAMING_PROCESSOR for artifact in model.artifacts):
            raise ValueError(f"model {model.id!r} has no streaming artifact to benchmark")
    known = {artifact.variant for _, artifact in pairs}
    missing = sorted(set(variants or ()) - known)
    if missing:
        raise ValueError("no streaming artifact has variant(s): " + ", ".join(missing))
    return pairs


def benchmark_clip(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """The sample clip repeated to ``seconds``, at ``sample_rate``."""
    from puresound.inference.processors.base import load_audio

    clip, _ = load_audio(SAMPLE_CLIP, sample_rate=sample_rate)
    count = max(1, int(round(seconds * sample_rate)))
    return np.resize(clip, count).astype(np.float32)


def run_benchmark(
    zoo,
    *,
    models: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    provider: str = "cpu",
    seconds: float = 10.0,
    repeats: int = 3,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure every selected artifact and return a JSON-ready record."""
    from puresound.inference.runtime import load_model

    if seconds <= 0 or repeats < 1:
        raise ValueError("seconds must be positive and repeats at least one")
    pairs = streaming_artifacts(zoo, models, variants)
    # Import the runtime up front, so the first artifact's cold start is session
    # creation alone, as every later one's is, and not the process's imports.
    import onnxruntime  # noqa: F401
    import puresound.streaming  # noqa: F401
    clips: dict[int, np.ndarray] = {}
    results = []
    for model, artifact in pairs:
        if progress is not None:
            progress(f"{model.id} [{artifact.variant}]")
        runtime = load_model(model.id, provider=provider, zoo=zoo, variant=artifact.variant)
        rate = int(runtime.sample_rate)
        samples = clips.setdefault(rate, benchmark_clip(seconds, rate))
        # The first call creates the session; it is the cold start, not a repeat.
        first = runtime.infer({"audio": samples})
        rtfs = [float(runtime.infer({"audio": samples}).rtf) for _ in range(repeats)]
        metadata = first.metadata
        results.append({
            "model_id": model.id,
            "task": model.task,
            "variant": artifact.variant,
            "rtf": statistics.median(rtfs),
            "rtf_repeats": rtfs,
            "cold_start_seconds": float(metadata.get("setup_seconds", 0.0)),
            "provider": first.provider,
            "execution_graph": metadata.get("execution_graph"),
            "native_ssm": metadata.get("native_ssm"),
            "intra_op_threads": metadata.get("intra_op_threads"),
            "latency_ms": metadata.get("latency_ms"),
        })
    measured = {model.id: model for model, _ in pairs}
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision": _revision(),
        "repository": _repository(),
        "machine": machine_info(),
        "settings": {"provider": provider, "seconds": seconds, "repeats": repeats,
                     "clip": "puresound/web/static/samples/noisy-speech.wav, repeated"},
        "models": {model_id: model_card(zoo, model) for model_id, model in measured.items()},
        "results": results,
    }


def model_card(zoo, model) -> dict[str, Any]:
    """The catalog's account of a model, as of this revision, beside its timings.

    A record then reads on its own: which model each number belongs to, what it
    is for, and how big each streaming artifact is to download.
    """
    return {
        "display_name": model.display_name,
        "task": model.task,
        "lifecycle": model.lifecycle,
        "roles": list(model.roles),
        "description": model.description,
        "recommended_inference": dict(model.recommended_inference),
        # The first README among the references is the model card.
        "card": next((ref for ref in model.benchmark_references if ref.endswith("README.md")), None),
        "artifacts": {
            artifact.variant: {
                "file": Path(artifact.path).name,
                "description": artifact.description,
                "size_bytes": zoo.path_for(artifact.path).stat().st_size,
            }
            for artifact in model.artifacts
            if artifact.processor == STREAMING_PROCESSOR
        },
    }


def _repository() -> str | None:
    """The repository's web address under GitHub Actions, for links to the cards."""
    server, name = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY")
    return f"{server}/{name}" if server and name else None


def machine_info() -> dict[str, Any]:
    """What a benchmark ran on: enough to tell two runs' machines apart."""
    import onnxruntime

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
        # GitHub-hosted runners name their image; elsewhere this is None.
        "ci_image": " ".join(
            filter(None, (os.environ.get("ImageOS"), os.environ.get("ImageVersion")))
        ) or None,
    }


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        elif sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return platform.processor() or "unknown"


def _revision() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def format_table(record: dict[str, Any]) -> str:
    """The results as an aligned text table, one row per artifact."""
    rows = [("model", "variant", "RTF", "cold start", "graph")]
    for item in record["results"]:
        graph = "native" if item.get("native_ssm") else "portable"
        rows.append((item["model_id"], item["variant"], f"{item['rtf']:.4f}",
                     f"{item['cold_start_seconds']:.2f} s", graph))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    machine = record["machine"]
    lines.append(
        f"{machine['cpu']} ({machine['system']} {machine['machine']}), "
        f"onnxruntime {machine['onnxruntime']}, {record['settings']['seconds']:g} s clip, "
        f"median of {record['settings']['repeats']}"
    )
    return "\n".join(lines)


__all__ = [
    "SAMPLE_CLIP",
    "STREAMING_PROCESSOR",
    "benchmark_clip",
    "format_table",
    "machine_info",
    "run_benchmark",
    "streaming_artifacts",
]
