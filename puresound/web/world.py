"""Adapters for acoustic world jobs; renderer and metrics stay in the library."""

from __future__ import annotations
import copy
import io
import itertools
import json
from pathlib import Path
import zipfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from puresound.audio.rir.scene.dynamic import DynamicSceneSpec, scene_limits
from puresound.audio.rir.scene.materials import ROOM_TYPE_RECIPES
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.scene.world_presets import (
    PRESETS,
    replace_room_materials,
    world_preset,
)
from puresound.audio.rir.render.dynamic import render_dynamic_scene
from puresound.evaluation.world import world_metrics

ASSETS = {
    "speaker-a": "samples/speaker-a-full.wav",
    # Retain the existing scene/sample ID, but never use the verification
    # half-utterance as a world source. This is the same reader as speaker-a.
    "speaker-a-2": "samples/speaker-a-full.wav",
    "speaker-b": "samples/speaker-b.wav",
    **{
        name: f"samples/pipeline/noise/{name}.wav"
        for name in ("fan", "babble", "clatter", "hum", "pink", "rumble")
    },
}
# Sweep parameters change every source of one role, relative to its own
# setting, so several sources of a kind keep their differences.
SWEEP_ROLES = {
    "noise_gain_change_db": "noise",
    "interferer_gain_change_db": "interferer",
    "target_distance_change_m": "target",
}
SWEEP_PARAMETERS = tuple(SWEEP_ROLES)
_ROLE_NAMES = {"noise": "noise source", "interferer": "other talker", "target": "target talker"}
SWEEP_CELLS = 25
# The reference a model of each task is scored against unless the user picks.
POLICY_DEFAULTS = {"noise_suppression": "speech", "voice_isolation": "near"}


def overview():
    from puresound.web.aicoustics import capabilities
    return {
        "available": True,
        "presets": {name: world_preset(name).to_dict() for name in PRESETS},
        "assets": [
            {"id": k, "url": "/" + v,
             "duration_s": sf.info(Path(__file__).parent / "static" / v).duration}
            for k, v in ASSETS.items()
        ],
        "limits": {**scene_limits(), "sweep_cells": SWEEP_CELLS},
        "sweep_parameters": list(SWEEP_PARAMETERS),
        "sweep_roles": dict(SWEEP_ROLES),
        "policy_defaults": dict(POLICY_DEFAULTS),
        "materials": sorted(ROOM_TYPE_RECIPES),
        "aicoustics": capabilities(),
    }


def _parse(parser, data):
    """Parse a request's scene or room, reporting a malformed one as ``ValueError``.

    The library parsers assume the JSON has the right shape, so a list or a
    number where an object belongs surfaces as ``AttributeError``, and an
    infinite index as ``OverflowError``; both are the request's fault.
    """
    try:
        return parser(data)
    except (AttributeError, OverflowError) as exc:
        raise ValueError(f"malformed scene: {exc}") from exc


def with_materials(payload):
    """The scene with its room's materials re-drawn for a room type and seed.

    Only the room is checked, so a scene that is still being edited (a
    keyframe too close to the microphone, say) can change its room type.
    """
    scene = payload["scene"]
    room = _parse(RoomSceneV2.from_dict, scene["room"])
    room_type = payload.get("room_type", room.room_type)
    seed = payload.get("seed", scene.get("seed", 0))
    if room_type not in ROOM_TYPE_RECIPES:
        raise ValueError("unknown room type")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    room = replace_room_materials(room, room_type, seed)
    return {**scene, "seed": seed, "room": room.to_dict()}


def asset_refs(spec, inputs):
    """The sample or upload behind each of the scene's assets; an asset the
    request does not name is the sample of the same ID."""
    return {
        source.asset_id: inputs.get(source.asset_id, {"sample": source.asset_id})
        for source in spec.sources
    }


def validate_request(payload):
    spec = _parse(DynamicSceneSpec.from_dict, payload["scene"])
    inputs = payload.get("assets", {})
    if not isinstance(inputs, dict):
        raise ValueError("assets must be an object")
    for asset in asset_refs(spec, inputs).values():
        if not isinstance(asset, dict) or not (
            set(asset) <= {"sample", "upload_id", "filename"}
        ):
            raise ValueError("assets must reference a sample or upload")
        if bool(asset.get("sample")) == bool(asset.get("upload_id")):
            raise ValueError("each asset must reference exactly one sample or upload")
        if asset.get("sample") and asset["sample"] not in ASSETS:
            raise ValueError("unknown sample asset")
    return spec


def load_assets(spec, refs, static_dir, upload_path):
    loaded = {}
    for asset_id, asset in asset_refs(spec, refs).items():
        path = (
            Path(static_dir) / ASSETS[asset["sample"]]
            if asset.get("sample")
            else upload_path(asset["upload_id"])
        )
        if path is None:
            raise ValueError("uploaded asset has expired; upload it again")
        info = sf.info(path)
        if info.duration > 120 or info.channels > 2 or info.frames < 1:
            raise ValueError(
                "assets must be nonempty mono/stereo audio under two minutes"
            )
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != spec.sample_rate:
            divisor = np.gcd(rate, spec.sample_rate)
            audio = resample_poly(audio, spec.sample_rate // divisor, rate // divisor)
        if not np.isfinite(audio).all():
            raise ValueError("nonfinite audio asset")
        loaded[asset_id] = audio
    return loaded


def render_report(spec, assets, enhance, model, progress, refs=None, comparison=None):
    result = render_dynamic_scene(
        spec, assets, progress=lambda v, p: progress(v * 0.8, p)
    )
    progress(0.81, "inference")
    began = time.perf_counter()
    output = (
        enhance(result.mixture.copy(), spec.sample_rate) if enhance else result.mixture.copy()
    )
    processing_seconds = time.perf_counter() - began
    # An enhancer that comes back a little short is padded: that end of the
    # render is reverberant tail. Any larger mismatch is refused below.
    shortfall = len(result.mixture) - len(output)
    if 0 < shortfall <= spec.sample_rate // 10:
        output = np.pad(output, (0, shortfall))
    if len(output) != len(result.mixture) or not np.isfinite(output).all():
        raise ValueError("model output has invalid length or values")
    comparisons = {}
    audio = {**result.audio(), "aligned": output, "removed": result.mixture - output}
    if comparison:
        from puresound.web.aicoustics import AicousticsError
        try:
            other = comparison(result.mixture.copy(), spec.sample_rate, lambda value, phase: progress(0.85 + value * 0.08, phase))
            if other.output.shape != result.mixture.shape or not np.isfinite(other.output).all():
                raise AicousticsError("ai-coustics returned invalid output length or samples.")
            audio["aicoustics"] = other.output
            audio["aicoustics-removed"] = result.mixture - other.output
            comparisons["aicoustics"] = {"status": "succeeded", **other.metadata, "output_rms_dbfs": float(10 * np.log10(max(float(np.mean(other.output.astype(float) ** 2)), 1e-12))), "metrics": world_metrics(result, other.output)}
        except AicousticsError as exc:
            # Preserve the local comparison when the optional provider fails;
            # no bypass track is labelled as enhanced audio.
            comparisons["aicoustics"] = {"status": "failed", "error": str(exc)}
    progress(0.94, "measurements")
    report = {
        "metadata": result.metadata,
        "scene": spec.to_dict(),
        "timeline": result.timeline,
        "metrics": world_metrics(result, output),
        "model": model,
        "processing": {"seconds": processing_seconds, "rtf": processing_seconds / (len(result.mixture) / spec.sample_rate)} if enhance else None,
        "comparisons": comparisons,
        "output_rms_dbfs": float(10 * np.log10(max(float(np.mean(output.astype(float) ** 2)), 1e-12))),
        "assets": asset_refs(spec, refs or {}),
        "scene_assets": {
            asset: {
                "sha256": result.metadata["asset_hashes"][asset],
                "file": f"asset-{i}.wav",
            }
            for i, asset in enumerate(result.source_audio)
        },
    }
    return report, audio, result.source_audio


def sweep_scenes(payload):
    base = validate_request(payload)
    axes = payload.get("axes")
    if (
        not isinstance(axes, list)
        or len(axes) != 2
        or any(not isinstance(axis, dict) for axis in axes)
    ):
        raise ValueError("sweep requires two axes")
    if axes[0].get("parameter") == axes[1].get("parameter"):
        raise ValueError("sweep axes must differ")
    for axis in axes:
        vals = axis.get("values")
        if (
            axis.get("parameter") not in SWEEP_PARAMETERS
            or not isinstance(vals, list)
            or not 1 <= len(vals) <= SWEEP_CELLS
        ):
            raise ValueError("invalid sweep axis")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in vals
        ):
            raise ValueError("sweep values must be finite numbers")
    if len(axes[0]["values"]) * len(axes[1]["values"]) > SWEEP_CELLS:
        raise ValueError(f"sweep is limited to {SWEEP_CELLS} cells")
    for axis in axes:
        role = SWEEP_ROLES[axis["parameter"]]
        if not any(source.role == role for source in base.sources):
            raise ValueError(f"the scene has no {_ROLE_NAMES[role]}")
    requested = payload.get("cell_indices")
    if requested is not None and (
        not isinstance(requested, list)
        or not requested
        or len(set(requested)) != len(requested)
        or any(
            type(i) is not int
            or i < 0
            or i >= len(axes[0]["values"]) * len(axes[1]["values"])
            for i in requested
        )
    ):
        raise ValueError("invalid cell indices")
    cells = []
    for index, (x, y) in enumerate(
        itertools.product(axes[0]["values"], axes[1]["values"])
    ):
        if requested is not None and index not in requested:
            continue
        data = copy.deepcopy(base.to_dict())
        try:
            for axis, value in zip(axes, [x, y]):
                _apply_sweep(data, axis["parameter"], value, np.array(base.room.mic_pos))
            cell = {
                "index": index,
                "x": x,
                "y": y,
                "scene": _parse(DynamicSceneSpec.from_dict, data),
            }
        except ValueError as exc:
            cell = {"index": index, "x": x, "y": y, "error": str(exc)}
        cells.append(cell)
    return cells


def _apply_sweep(data, parameter, value, mic):
    """Change every source of the parameter's role in a scene dict."""
    for source in data["sources"]:
        if source["role"] != SWEEP_ROLES[parameter]:
            continue
        if parameter == "target_distance_change_m":
            for key in source["keyframes"]:
                point = np.array(key["position_m"])
                distance = np.linalg.norm(point - mic)
                if distance + value <= 0:
                    raise ValueError("the distance change carries a target talker past the microphone")
                key["position_m"] = (point + value * (point - mic) / distance).tolist()
        else:
            source["gain_db"] += value


def sweep_progress(result):
    """What a running sweep has finished, without its audio: per cell the
    status and the output SI-SDR of each target policy.  None for other jobs."""
    if not result or "cells" not in result:
        return None
    cells = []
    for cell in result["cells"]:
        metrics = (cell.get("result") or {}).get("metrics") or {}
        comparison = (cell.get("result") or {}).get("comparisons", {}).get("aicoustics", {})
        cells.append(
            {
                **{k: cell.get(k) for k in ("index", "x", "y", "status", "error")},
                "si_sdr_db": {
                    policy: (scores.get("output") or {}).get("si_sdr_db")
                    for policy, scores in metrics.items()
                },
                "aicoustics_status": comparison.get("status"),
                "aicoustics_error": comparison.get("error"),
                "aicoustics_si_sdr_db": {
                    policy: (scores.get("output") or {}).get("si_sdr_db")
                    for policy, scores in comparison.get("metrics", {}).items()
                },
            }
        )
    return {"axes": result["axes"], "cells": cells, "partial": True}


# Renders stored before scene v2 call the target policy "fixed" and their
# references "target-*"; History reopens them under today's names.
_RENAMED = {"fixed": "target", "target-fixed": "reference-target", "target-near": "reference-near"}


def current_result(kind, result):
    """A stored world result under today's names; others pass through."""
    if not result or kind not in {"world_render", "world_sweep"}:
        return result
    if kind == "world_render":
        return _current_report(result)
    cells = [
        {**cell, "result": _current_report(cell["result"])} if cell.get("result") else cell
        for cell in result.get("cells", [])
    ]
    return {**result, "cells": cells}


def _current_report(report):
    scene = report.get("scene") or {}
    if scene.get("schema_version") != "puresound.dynamic_scene.v1":
        return report
    rename = lambda table: {_RENAMED.get(k, k): v for k, v in (table or {}).items()}
    return {
        **report,
        "scene": DynamicSceneSpec.from_dict(scene).to_dict(),
        "metrics": rename(report.get("metrics")),
        "output_urls": rename(report.get("output_urls")),
    }


def world_zip(report, audio, assets, wav_bytes):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "scene.json",
            json.dumps(
                {"scene": report["scene"], "assets": report["scene_assets"]},
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        archive.writestr(
            "report.json", json.dumps(report, ensure_ascii=False, allow_nan=False)
        )
        for name, samples in audio.items():
            archive.writestr(f"{name}.wav", wav_bytes(samples, 16000))
        for asset, samples in assets.items():
            archive.writestr(
                report["scene_assets"][asset]["file"], wav_bytes(samples, 16000)
            )
    return buffer.getvalue()
