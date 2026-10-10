"""Build one training row from chosen audio and report it stage by stage.

The web pipeline inspector's back end. A released recipe is used for its
augmentation knobs only; the audio is what the user chose -- a foreground
talker, other talkers, noise clips, sample rooms or uploaded impulse responses
-- written into a throwaway workspace, with the recipe's corpus paths pointed
at it. Blocks that need a corpus the workspace cannot stand in for are switched
off and reported, never silently dropped.

One row at a time, process-wide: synthesis reseeds the global RNGs, so two
traces running together would draw from each other's streams. Torch and the
synthesis stack are imported on first use, so importing this module is cheap.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np
import soundfile as sf
import yaml

from puresound.evaluation.pipeline_report import (
    gain_staged,
    pair_metrics,
    rir_summary,
    rms_dbfs,
    room_geometry,
    sanitize,
    score_pair,
)

SCHEMA = "puresound.pipeline-trace/1"
SAMPLE_RATE = 16_000
#: Longer files are cut once decoded: a row is at most ROW_SECONDS_RANGE[1] long,
#: and an hour of noise in the workspace would only slow its noise draws down.
MAX_INPUT_SECONDS = 60.0
ROW_SECONDS_RANGE = (1.0, 20.0)
MIN_UTTERANCE_SECONDS = 0.5
#: Below this RMS a clip is treated as silence.
SILENT_RMS = 1e-5
FOREGROUND = "foreground"
TRACE_TASKS = ("noise_suppression", "voice_isolation")

#: Blocks that read a corpus the workspace has no stand-in for.
UNAVAILABLE_BLOCKS = {
    "augmentation_realfar": "needs a pool of real far-field recordings",
    "augmentation_realnear": "needs a pool of real close-mic recordings",
    "augmentation_session_rows": "needs conversation-length rows drawn from a many-talker corpus",
}

Enhancer = Callable[[np.ndarray, int], np.ndarray]
Progress = Callable[[float, str], None]

_LOCK = threading.Lock()


class PipelineInputError(ValueError):
    """A request the inspector cannot build a row from; the message says why."""


@dataclass(frozen=True)
class RirChoice:
    """Where the row's impulse responses come from.

    ``samples``: bank rooms, each path a room's WAV with its sidecar JSON beside
    it; ``upload``: impulse-response files, applied to the whole mixture;
    ``none``: no reverberation.
    """

    kind: Literal["samples", "upload", "none"] = "none"
    paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TraceRequest:
    recipe: Path
    foreground: Path
    talkers: tuple[tuple[Path, ...], ...] = ()
    noises: tuple[Path, ...] = ()
    rir: RirChoice = RirChoice()
    role: Literal["train", "validation"] = "train"
    seed: int = 0
    epoch: Optional[int] = None
    seconds: Optional[float] = None


@dataclass
class Workspace:
    """The one-row corpus a request is turned into."""

    root: Path
    metafile: Path
    noise_dir: Path
    rooms_dir: Path
    rir_dir: Path
    noise_files: list[Path] = field(default_factory=list)
    room_files: list[Path] = field(default_factory=list)
    rir_files: list[Path] = field(default_factory=list)
    talker_count: int = 0
    notes: list[dict[str, str]] = field(default_factory=list)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0


def _decode(path: Path, workspace: Workspace, what: str, *, first_channel: bool = False) -> np.ndarray:
    from puresound.audio.io import AudioIO

    try:
        wav, _ = AudioIO.open(str(path), resample_to=SAMPLE_RATE)
    except Exception as exc:
        raise PipelineInputError(f"could not read the {what} file: {exc}") from exc
    mono = wav[0] if first_channel else wav.mean(dim=0)
    samples = mono.detach().cpu().numpy().astype(np.float32)
    limit = int(MAX_INPUT_SECONDS * SAMPLE_RATE)
    if samples.size > limit:
        workspace.notes.append({"subject": what, "reason": f"cut to its first {MAX_INPUT_SECONDS:g} s"})
        samples = samples[:limit]
    return samples


def _save(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="FLOAT")
    return path


def prepare_workspace(request: TraceRequest, root: Path) -> Workspace:
    """Decode every chosen file to 16 kHz mono under ``root`` and lay out the
    one-row corpus: a metafile in which the foreground and each other talker is
    a speaker of its own, a noise folder, and a room bank or an RIR folder."""
    root = Path(root)
    workspace = Workspace(
        root=root,
        metafile=root / "corpus.csv",
        noise_dir=root / "noise",
        rooms_dir=root / "rooms",
        rir_dir=root / "rir",
    )
    rows = ["uttid, spkid, gender, path, length, sample rate, channels"]

    def utterance(speaker: str, index: int, samples: np.ndarray) -> None:
        path = _save(root / "speech" / speaker / f"{speaker}_{index}.wav", samples)
        rows.append(f"{speaker}_{index}, {speaker}, f, {path}, {samples.size}, {SAMPLE_RATE}, 1")

    foreground = _decode(request.foreground, workspace, "foreground")
    if foreground.size < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
        raise PipelineInputError(f"the foreground clip is shorter than {MIN_UTTERANCE_SECONDS:g} s")
    if _rms(foreground) < SILENT_RMS:
        raise PipelineInputError("the foreground clip is silent")
    utterance(FOREGROUND, 0, foreground)

    for number, files in enumerate(request.talkers, start=1):
        speaker = f"talker_{number}"
        kept = 0
        for index, path in enumerate(files):
            samples = _decode(path, workspace, f"talker {number}")
            if samples.size < MIN_UTTERANCE_SECONDS * SAMPLE_RATE or _rms(samples) < SILENT_RMS:
                workspace.notes.append({"subject": f"talker {number}", "reason": "a clip was silent or shorter than 0.5 s and was left out"})
                continue
            utterance(speaker, index, samples)
            kept += 1
        workspace.talker_count += int(kept > 0)
    workspace.metafile.parent.mkdir(parents=True, exist_ok=True)
    workspace.metafile.write_text("\n".join(rows) + "\n", encoding="utf-8")

    for index, path in enumerate(request.noises):
        samples = _decode(path, workspace, f"noise {index + 1}")
        if _rms(samples) < SILENT_RMS:
            workspace.notes.append({"subject": f"noise {index + 1}", "reason": "silent, left out"})
            continue
        workspace.noise_files.append(_save(workspace.noise_dir / f"noise_{index}.wav", samples))
    if len(workspace.noise_files) == 1:
        # A dynamic-noise row splices two different files of the pool; with one
        # clip that draw has nothing to pick, so the pool holds it twice.
        only = workspace.noise_files[0]
        workspace.noise_files.append(Path(shutil.copyfile(only, only.with_name(f"{only.stem}_copy.wav"))))
        workspace.notes.append({"subject": "noise", "reason": "one clip chosen: rows that splice two noises splice it with itself"})

    if request.rir.kind == "samples":
        for path in request.rir.paths:
            path = Path(path)
            sidecar = path.with_suffix(".json")
            if not path.is_file() or not sidecar.is_file():
                raise PipelineInputError(f"sample room {path.stem!r} is missing its audio or geometry")
            items = workspace.rooms_dir / "items"
            items.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, items / path.name)
            shutil.copyfile(sidecar, items / sidecar.name)
            workspace.room_files.append(items / path.name)
    elif request.rir.kind == "upload":
        for index, path in enumerate(request.rir.paths):
            samples = _decode(path, workspace, f"impulse response {index + 1}", first_channel=True)
            if not np.any(samples):
                workspace.notes.append({"subject": f"impulse response {index + 1}", "reason": "silent, left out"})
                continue
            workspace.rir_files.append(_save(workspace.rir_dir / f"rir_{index}.wav", samples))
    return workspace


def _enabled(block: Any) -> bool:
    return isinstance(block, dict) and bool(block.get("used") or block.get("enabled"))


def _first_bank_member(pregenerated: Any) -> Mapping[str, Any]:
    if not isinstance(pregenerated, dict):
        return {}
    banks = pregenerated.get("banks")
    if isinstance(banks, list) and banks and isinstance(banks[0], dict):
        return banks[0]
    return pregenerated


def explorer_recipe(
    template: Mapping[str, Any], workspace: Workspace, rir: RirChoice
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """``template`` with every corpus path pointed at ``workspace``.

    Knobs stay as the recipe states them; what changes is where the audio comes
    from, and blocks with nothing to read are switched off. Every change that
    alters what the recipe would synthesise is returned as a note.
    """
    recipe = copy.deepcopy(dict(template))
    notes: list[dict[str, str]] = []

    def note(subject: str, reason: str) -> None:
        notes.append({"subject": subject, "reason": reason})

    corpus = recipe.setdefault("dataset", {})
    if corpus.get("target_sample_rate") != SAMPLE_RATE:
        note("dataset.target_sample_rate", f"rows are built at {SAMPLE_RATE // 1000} kHz")
    unused = str(workspace.root / "unused")
    corpus.update(
        train_metafile=str(workspace.metafile),
        valid_metafile=str(workspace.metafile),
        test_folder=unused,
        proc_output_folder=unused,
        target_sample_rate=SAMPLE_RATE,
        filter_min_utterance_length=MIN_UTTERANCE_SECONDS,
        filter_min_utterance_per_speaker=1,
    )

    for name, reason in UNAVAILABLE_BLOCKS.items():
        block = recipe.get(name)
        if _enabled(block):
            block["enabled" if name == "augmentation_session_rows" else "used"] = False
            note(name, reason)

    noise = recipe.get("augmentation_noise")
    if _enabled(noise):
        if workspace.noise_files:
            noise["noise_folder"] = str(workspace.noise_dir)
            noise.pop("noise_sources", None)
        else:
            noise["used"] = False
            note("augmentation_noise", "no noise clip was chosen; the capture floor is off with it")
    elif workspace.noise_files:
        note("augmentation_noise", "this recipe adds no recorded noise, so the chosen clips are not used")

    reverb = recipe.get("augmentation_reverb")
    rooms_replace_banks = False
    if _enabled(reverb):
        if rir.kind == "samples" and workspace.room_files:
            rooms_replace_banks = True
            simulator = dict(reverb.get("simulator") or {})
            member = _first_bank_member(simulator.get("pregenerated"))
            simulator["used"] = True
            simulator["source_level"] = bool(simulator.get("source_level", False))
            simulator["pregenerated"] = {
                "used": True,
                "banks": [{
                    "name": "samples",
                    "weight": 1.0,
                    "bank_type": "room",
                    "folder": str(workspace.rooms_dir),
                    "near_labels": list(member.get("near_labels") or ["near_0", "near_1"]),
                    "far_labels": list(member.get("far_labels") or ["far_0", "far_1", "far_2"]),
                    "drr_window_ms": float(member.get("drr_window_ms", 2.5)),
                    "cache_size": 8,
                }],
            }
            reverb["simulator"] = simulator
            reverb["rir_folder"] = None
        elif rir.kind == "upload" and workspace.rir_files:
            reverb["rir_folder"] = str(workspace.rir_dir)
            reverb["simulator"] = None
            note(
                "augmentation_reverb",
                "uploaded impulse responses reverberate the whole mixture; placing each source needs a room",
            )
        elif rir.kind == "upload" and rir.paths:
            reverb["used"] = False
            note("augmentation_reverb", "every uploaded impulse response was silent, so the row has no reverberation")
        else:
            reverb["used"] = False
            note("augmentation_reverb", "no impulse response was chosen")
    elif rir.kind != "none":
        note("augmentation_reverb", "this recipe adds no reverberation, so the chosen room is not used")

    vad = recipe.get("vad_label")
    if _enabled(vad) and vad.get("backend") == "silero":
        vad["backend"] = "energy"
        vad.pop("args", None)
        note("vad_label", "labels use the energy detector here; Silero needs its model")

    curriculum = recipe.get("curriculum")
    if isinstance(curriculum, dict) and curriculum.get("used"):
        kept = []
        for track in curriculum.get("tracks") or []:
            path = str(track.get("path", ""))
            if path.startswith("bank:"):
                note(
                    f"curriculum {path}",
                    "schedules a bank the sample rooms replace" if rooms_replace_banks else "schedules a bank this row does not use",
                )
            elif path.startswith("aug:") and not _enabled(recipe.get(path[4:].split(".")[0])):
                note(f"curriculum {path}", "schedules a block that is off here")
            else:
                kept.append(track)
        curriculum["tracks"] = kept
        if not kept:
            curriculum["used"] = False
    return recipe, notes


def build_dataset(recipe, role: str):
    """The task's dataset for ``recipe`` -- the same construction the training
    runner uses, with the curriculum on training rows only."""
    from puresound.task.ns import NoiseSuppressionDataset
    from puresound.task.voice_isolation import VoiceIsolationDataset

    classes = {"noise_suppression": NoiseSuppressionDataset, "voice_isolation": VoiceIsolationDataset}
    corpus = recipe.dataset
    train = role == "train"
    curriculum = recipe.curriculum if train and recipe.curriculum is not None and recipe.curriculum.used else None
    return classes[recipe.task](
        metafile_path=corpus.train_metafile,
        dataset_role="train" if train else "validation",
        pipeline_role=corpus.train_pipeline_role if train else corpus.validation_pipeline_role,
        curriculum=curriculum,
        **recipe.dataset_kwargs(),
    )


def recipe_state(dataset, spec) -> dict[str, Any]:
    """Whether the recipe has the stage's block on, and its firing probability
    as the row saw it (after any curriculum step)."""
    if spec.block is None:
        return {"configured": True, "enabled": True, "prob": None}
    name, *rest = spec.block.split(".")
    block = getattr(dataset, name, None)
    for part in rest:
        block = getattr(block, part, None) if block is not None else None
    if block is None:
        return {"configured": False, "enabled": False, "prob": None}
    prob = getattr(block, spec.prob_field, None) if spec.prob_field else None
    return {
        "configured": True,
        "enabled": bool(getattr(block, "used", True)),
        "prob": float(prob) if isinstance(prob, (int, float)) and not isinstance(prob, bool) else None,
    }


def _read_template(path: Path) -> dict[str, Any]:
    try:
        template = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineInputError(f"could not read the recipe: {exc}") from exc
    if not isinstance(template, dict) or template.get("task") not in TRACE_TASKS:
        raise PipelineInputError("the recipe is not a noise-suppression or voice-isolation training recipe")
    return template


def _check_request(request: TraceRequest, recipe: Mapping[str, Any], workspace: Workspace) -> None:
    if request.seconds is not None and not ROW_SECONDS_RANGE[0] <= float(request.seconds) <= ROW_SECONDS_RANGE[1]:
        low, high = ROW_SECONDS_RANGE
        raise PipelineInputError(f"the row length must be between {low:g} and {high:g} s")
    if _enabled(recipe.get("augmentation_speech")) and workspace.talker_count == 0:
        if request.talkers:
            raise PipelineInputError(
                "every other-talker clip was silent or shorter than 0.5 s; this recipe mixes in "
                "other talkers (interferers or echo) and needs at least one"
            )
        raise PipelineInputError(
            "this recipe mixes in other talkers (interferers or echo); add at least one other talker"
        )


def _emitted(sample: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    scalars: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.numel() == 1:
            scalars[key] = float(value.item())
        elif isinstance(value, (bool, int, float)) or (isinstance(value, str) and value):
            scalars[key] = value
    return scalars


class _AudioStore:
    """Named float32 arrays for the report, one copy per distinct signal."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self._names: dict[str, str] = {}

    def put(self, name: str, samples: np.ndarray) -> str:
        data = np.ascontiguousarray(samples, dtype=np.float32)
        digest = hashlib.sha1(data.tobytes()).hexdigest()
        if digest not in self._names:
            self._names[digest] = name
            self.arrays[name] = data
        return self._names[digest]


def _assemble(trace, dataset, store: _AudioStore) -> tuple[list[dict], list[dict]]:
    from puresound.task.trace import STAGES

    stage_at = [tap.stage for tap in trace.taps]
    rirs: list[dict[str, Any]] = []
    # One entry per response and stage: a far channel can serve an interferer and
    # also colour the noise through the room, and each stage has to find it.
    by_key: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
    for record in trace.rirs:
        stage = stage_at[record.position] if record.position < len(stage_at) else None
        entry = by_key.get((record.rir_id, stage))
        if entry is None:
            entry = {
                "index": len(rirs),
                "rir_id": record.rir_id,
                "stage": stage,
                "role": record.role,
                "modes": [record.mode],
                "metadata": record.metadata,
                "summary": rir_summary(record.impulse, record.sample_rate) if record.impulse is not None else None,
            }
            by_key[(record.rir_id, stage)] = entry
            rirs.append(entry)
        elif record.mode not in entry["modes"]:
            entry["modes"].append(record.mode)

    taps = {tap.stage: tap for tap in trace.taps}
    stages: list[dict[str, Any]] = []
    previous = None
    for index, spec in enumerate(STAGES):
        tap = taps.get(spec.id)
        row: dict[str, Any] = {
            "id": spec.id,
            "group": spec.group,
            "block": spec.block,
            "fired": tap is not None,
            "recipe": recipe_state(dataset, spec),
            "params": {},
            "audio": None,
            "metrics": None,
            "model": None,
            "rirs": [],
            "changed": None,
        }
        if tap is not None:
            prefix = f"{index:02d}-{spec.id}"
            row["params"] = tap.params
            row["audio"] = {
                "noisy": store.put(f"{prefix}-mixture", tap.noisy),
                "target": store.put(f"{prefix}-target", tap.target),
            }
            for name, signal in tap.signals.items():
                row["audio"][name] = store.put(f"{prefix}-{name}", signal)
            row["metrics"] = pair_metrics(tap.noisy, tap.target, SAMPLE_RATE)
            row["changed"] = previous is None or not (
                np.array_equal(previous.noisy, tap.noisy) and np.array_equal(previous.target, tap.target)
            )
            row["rirs"] = [entry["index"] for entry in rirs if entry["stage"] == spec.id]
            previous = tap
        stages.append(row)
    return stages, rirs


def _score(stages, trace, enhance: Enhancer, store: _AudioStore, progress: Progress) -> None:
    from puresound.task.trace import STAGE_INDEX

    taps = {tap.stage: tap for tap in trace.taps}
    fired = [row for row in stages if row["fired"]]
    last = None
    for number, row in enumerate(fired):
        progress(0.4 + 0.55 * number / max(1, len(fired)), f"scoring {row['id']}")
        tap = taps[row["id"]]
        if last is not None and np.array_equal(last[0].noisy, tap.noisy) and np.array_equal(last[0].target, tap.target):
            row["model"] = {**last[1], "same_input_as": last[2]}
            continue
        noisy, target, gain = gain_staged(tap.noisy, tap.target)
        output = np.asarray(enhance(noisy, SAMPLE_RATE), dtype=np.float32).reshape(-1)[: noisy.size]
        before, after = rms_dbfs(noisy[: output.size]), rms_dbfs(output)
        result = {
            "gain": gain,
            "input": score_pair(target, noisy, SAMPLE_RATE),
            "output": score_pair(target, output, SAMPLE_RATE),
            "level_change_db": after - before if before is not None and after is not None else None,
            "audio": store.put(f"{STAGE_INDEX[row['id']]:02d}-{row['id']}-model", output),
        }
        row["model"] = result
        last = (tap, result, row["id"])


def trace_row(
    request: TraceRequest,
    *,
    workspace_root: Path,
    enhance: Optional[Enhancer] = None,
    model: Optional[Mapping[str, Any]] = None,
    progress: Optional[Progress] = None,
) -> dict[str, Any]:
    """Build the requested row, trace it and, given ``enhance``, score every stage.

    Returns ``{"report": <strict-JSON dict>, "audio": {name: float32 array}}``;
    every audio name in the report is a key of ``audio``. ``enhance`` maps a
    mixture to the model's aligned output at the same rate; the pair it sees is
    gain-staged the way the converter would deliver it. ``progress`` is called
    between steps and may raise to cancel.
    """
    report_progress = progress or (lambda value, phase: None)
    # Synthesis reseeds the global RNGs, so rows are built one at a time; scoring
    # draws nothing and runs outside the lock.
    if not _LOCK.acquire(blocking=False):
        report_progress(0.02, "waiting for the row being built")
        _LOCK.acquire()
    try:
        started = time.perf_counter()
        report_progress(0.05, "preparing the inputs")
        workspace = prepare_workspace(request, Path(workspace_root))
        template = _read_template(request.recipe)
        recipe_dict, notes = explorer_recipe(template, workspace, request.rir)
        _check_request(request, recipe_dict, workspace)
        recipe_path = workspace.root / "recipe.yaml"
        recipe_path.write_text(yaml.safe_dump(recipe_dict, sort_keys=False), encoding="utf-8")

        from puresound.config import load_recipe
        from puresound.task.trace import recording

        recipe = load_recipe(recipe_path, expected_task=recipe_dict["task"], expected_purpose="train")
        report_progress(0.15, "building the dataset")
        dataset = build_dataset(recipe, request.role)
        built = time.perf_counter()

        curriculum = dataset.curriculum is not None and request.role == "train"
        epoch = request.epoch if curriculum else None
        if request.epoch is not None and epoch is None:
            notes.append({"subject": "epoch", "reason": "no curriculum applies to these rows, so the epoch changes nothing"})
        report_progress(0.3, "synthesising the row")
        with recording(dataset) as trace:
            sample = dataset[(FOREGROUND, None, int(request.seed), request.seconds, epoch)]
        synthesised = time.perf_counter()

        store = _AudioStore()
        stages, rirs = _assemble(trace, dataset, store)
    finally:
        _LOCK.release()
    foreground = next((tap for tap in trace.taps if tap.stage == "foreground.channel"), None)
    room = room_geometry(foreground.params.get("room_scene") if foreground else None, rirs)
    if enhance is not None:
        _score(stages, trace, enhance, store, report_progress)
    report_progress(0.98, "writing the report")
    report = {
        "schema": SCHEMA,
        "recipe": Path(request.recipe).name,
        "task": recipe.task,
        "role": request.role,
        "seed": int(request.seed),
        "epoch": epoch,
        "curriculum": curriculum,
        "sample_rate": SAMPLE_RATE,
        "row_seconds": sample["noisy_speech"].shape[-1] / SAMPLE_RATE,
        "talkers": workspace.talker_count,
        "stages": stages,
        "rirs": rirs,
        "room": room,
        "notes": notes + workspace.notes,
        "emitted": _emitted(sample),
        "model": dict(model) if model else None,
        "timing": {
            "build_seconds": built - started,
            "synthesis_seconds": synthesised - built,
            "total_seconds": time.perf_counter() - started,
        },
    }
    return {"report": sanitize(report), "audio": store.arrays}
