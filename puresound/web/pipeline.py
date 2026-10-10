"""The pipeline inspector's service side: which recipes it offers, what its
samples are, and what a request may name.

Nothing here imports Torch. The synthesis stack is loaded by the trace job the
first time a row is built (``puresound.evaluation.pipeline_trace``), the way the
playground loads audio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from puresound.evaluation.pipeline_trace import ROW_SECONDS_RANGE, TRACE_TASKS

MAX_TALKERS = 8
MAX_NOISES = 16
MAX_RIRS = 8
MAX_SEED = 2**31 - 1
MAX_EPOCH = 10_000


class PipelineRequestError(ValueError):
    """A request naming something the inspector does not offer."""


#: Parsed recipe summaries by path, with the (mtime, size) they were read at:
#: the list is asked for on every health check and page load, and a recipe
#: rarely changes between two of them.
_RECIPE_CACHE: dict[Path, tuple[tuple[int, int], Optional[dict[str, Any]]]] = {}


def _recipe_summary(root: Path, path: Path) -> Optional[dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _RECIPE_CACHE.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        data = None
    summary = None
    if isinstance(data, dict) and data.get("purpose") == "train" and data.get("task") in TRACE_TASKS:
        curriculum = data.get("curriculum")
        summary = {
            "id": path.relative_to(root).as_posix(),
            "task": data["task"],
            "name": path.stem,
            "recipe_dir": path.parent.parent.name,
            "row_seconds": (data.get("dataset") or {}).get("training_length_seconds"),
            "curriculum": bool(isinstance(curriculum, dict) and curriculum.get("used")),
        }
    _RECIPE_CACHE[path] = (stamp, summary)
    return summary


def list_recipes(root: Optional[Path]) -> list[dict[str, Any]]:
    """Released training recipes under ``root/egs/*/config/``.

    Experiment and evaluation recipes live in subfolders of ``config/`` and are
    not offered; neither is a file that is not a noise-suppression or
    voice-isolation training recipe.
    """
    if root is None:
        return []
    root = Path(root)
    summaries = (_recipe_summary(root, path) for path in sorted(root.glob("egs/*/config/*.yaml")))
    return [dict(summary) for summary in summaries if summary is not None]


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise PipelineRequestError(f"{name} must be an integer from {low} to {high}")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= float(value) <= high:
        raise PipelineRequestError(f"{name} must be a number from {low:g} to {high:g}")
    return float(value)


def _items(value: Any, name: str, limit: int) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PipelineRequestError(f"{name} must be a list")
    if len(value) > limit:
        raise PipelineRequestError(f"{name}: at most {limit}")
    return value


class PipelineCatalog:
    """The recipes under a repository root and the bundled samples."""

    def __init__(self, root: Optional[str | Path], samples_dir: Path):
        self.root = Path(root).expanduser().resolve() if root else None
        self.samples_dir = Path(samples_dir)
        self._samples: Optional[dict[str, Any]] = None

    def recipes(self) -> list[dict[str, Any]]:
        return list_recipes(self.root)

    def samples(self) -> dict[str, Any]:
        if self._samples is None:
            try:
                self._samples = json.loads((self.samples_dir / "index.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._samples = {"talkers": [], "noises": [], "rooms": []}
        return self._samples

    def status(self) -> dict[str, Any]:
        if self.root is None:
            return {"available": False, "root": None,
                    "reason": "no repository root: start the server from a PureSound checkout or pass --pipeline-root"}
        if not self.recipes():
            return {"available": False, "root": str(self.root),
                    "reason": f"no training recipes under {self.root}/egs/*/config/"}
        return {"available": True, "root": str(self.root), "reason": None}

    def _sample_files(self, kind: str, sample_id: Any, what: str) -> list[Path]:
        entry = next((item for item in self.samples().get(kind, []) if item.get("id") == sample_id), None)
        if entry is None:
            raise PipelineRequestError(f"{what}: no sample {sample_id!r}")
        files = entry.get("files") or [entry["file"]]
        allowed = self.samples_dir.parent.resolve()
        paths = []
        for file in files:
            path = (self.samples_dir / file).resolve()
            if allowed not in path.parents or not path.is_file():
                raise PipelineRequestError(f"{what}: sample {sample_id!r} is missing")
            paths.append(path)
        return paths

    def _source(self, spec: Any, what: str, kind: str, resolve_upload: Callable[[str], Optional[Path]]) -> list[Path]:
        if not isinstance(spec, Mapping):
            raise PipelineRequestError(f"{what} must be an upload or a sample")
        if "upload_id" in spec:
            path = resolve_upload(str(spec["upload_id"]))
            if path is None:
                raise PipelineRequestError(f"{what}: the upload was not found (it expired or the server restarted); upload the file again")
            return [Path(path)]
        if "sample" in spec:
            return self._sample_files(kind, spec["sample"], what)
        raise PipelineRequestError(f"{what} must be an upload or a sample")

    def build_request(self, payload: Any, resolve_upload: Callable[[str], Optional[Path]]):
        """Validate a trace request and resolve every file it names.

        Returns the library's ``TraceRequest`` and the options the service needs
        (``model_id``, ``provider``, the offered ``recipe`` id).
        """
        from puresound.evaluation.pipeline_trace import RirChoice, TraceRequest

        if not isinstance(payload, Mapping):
            raise PipelineRequestError("request body must be a JSON object")
        offered = {item["id"] for item in self.recipes()}
        recipe_id = payload.get("recipe")
        if self.root is None or not isinstance(recipe_id, str) or recipe_id not in offered:
            raise PipelineRequestError("recipe must be one of the offered training recipes")
        role = payload.get("role", "train")
        if role not in ("train", "validation"):
            raise PipelineRequestError("role must be train or validation")
        seed = _integer(payload.get("seed", 0), "seed", 0, MAX_SEED)
        epoch = None if payload.get("epoch") is None else _integer(payload["epoch"], "epoch", 0, MAX_EPOCH)
        seconds = None if payload.get("seconds") is None else _number(payload["seconds"], "seconds", *ROW_SECONDS_RANGE)
        foreground = self._source(payload.get("foreground"), "foreground", "talkers", resolve_upload)[0]
        talkers = tuple(
            tuple(self._source(item, f"talker {index + 1}", "talkers", resolve_upload))
            for index, item in enumerate(_items(payload.get("talkers"), "talkers", MAX_TALKERS))
        )
        noises = tuple(
            path
            for index, item in enumerate(_items(payload.get("noises"), "noises", MAX_NOISES))
            for path in self._source(item, f"noise {index + 1}", "noises", resolve_upload)
        )
        rir_spec = payload.get("rir") or {"kind": "none"}
        kind = rir_spec.get("kind") if isinstance(rir_spec, Mapping) else None
        if kind == "samples":
            rooms = _items(rir_spec.get("rooms"), "rir.rooms", MAX_RIRS)
            rir = RirChoice("samples", tuple(self._sample_files("rooms", room, "room")[0] for room in rooms))
        elif kind == "upload":
            files = _items(rir_spec.get("files"), "rir.files", MAX_RIRS)
            rir = RirChoice("upload", tuple(self._source(item, "impulse response", "rooms", resolve_upload)[0] for item in files))
        elif kind == "none":
            rir = RirChoice("none")
        else:
            raise PipelineRequestError("rir.kind must be samples, upload or none")
        model_id = payload.get("model_id") or None
        if model_id is not None and not isinstance(model_id, str):
            raise PipelineRequestError("model_id must be a string")
        request = TraceRequest(
            recipe=self.root / recipe_id,
            foreground=foreground,
            talkers=talkers,
            noises=noises,
            rir=rir,
            role=role,
            seed=seed,
            epoch=epoch,
            seconds=seconds,
        )
        return request, {"model_id": model_id, "provider": str(payload.get("provider") or "cpu"), "recipe": recipe_id}
