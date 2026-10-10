"""What the pipeline inspector reports about one traced row.

NumPy only, plain arrays in: the part of the inspector that turns a stage's
pair, an impulse response or a room into what the page draws, kept apart from
the synthesis side so it can be tested on hand-made signals.

The quantity the page follows through the pipeline is the *effective SNR*.
After any stage, (mixture, target) is a training example as it would stand if
synthesis stopped there, and ``target`` against ``mixture - target`` is how much
the model would be asked to remove at that point: noise, interfering speech
that is not target, late reverberation past an ``early`` target, transmission
damage. Stages that add nothing can still move it -- a reverberant channel
with an early target is the usual case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Frame length of the per-frame effective-SNR lane; the deck's level lane uses 20 ms too.
FRAME_SECONDS = 0.02
#: Frames whose target is below this are silence; a ratio against them means nothing.
SILENT_FRAME_DBFS = -70.0
#: A residual this far below the target is the target itself; ratios are capped here.
IDENTICAL_DB = 100.0
#: Per-frame values are clamped to this band so one near-silent frame cannot set the axis.
FRAME_RANGE_DB = (-30.0, 60.0)
_FLOOR_DB = -90.0
_TINY = 1e-20


def sanitize(value: Any) -> Any:
    """``value`` with every non-finite float replaced by None, recursively.

    JSON has no NaN or Infinity, and a browser's ``JSON.parse`` rejects the
    tokens Python writes for them.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [sanitize(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def rms_dbfs(values) -> Optional[float]:
    """RMS level in dBFS; None for an empty or all-zero signal."""
    x = _as_array(values)
    if x.size == 0:
        return None
    power = float(np.mean(np.square(x)))
    return 10.0 * math.log10(power) if power > _TINY else None


def peak_dbfs(values) -> Optional[float]:
    x = _as_array(values)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * math.log10(peak) if peak > 0.0 else None


def si_sdr_db(reference, estimate) -> Optional[float]:
    """Scale-invariant SDR of ``estimate`` against ``reference``, capped at
    ``IDENTICAL_DB``; None when the reference is silent."""
    target, guess = _trimmed(reference, estimate)
    target = target - target.mean()
    guess = guess - guess.mean()
    energy = float(np.dot(target, target))
    if energy <= _TINY:
        return None
    projection = float(np.dot(guess, target)) / energy * target
    residual = guess - projection
    ratio = float(np.dot(projection, projection)) / max(float(np.dot(residual, residual)), _TINY)
    if ratio <= 0.0:
        return None
    return min(10.0 * math.log10(ratio), IDENTICAL_DB)


def _trimmed(first, second) -> tuple[np.ndarray, np.ndarray]:
    a, b = _as_array(first), _as_array(second)
    n = min(a.size, b.size)
    return a[:n], b[:n]


def pair_metrics(noisy, target, sample_rate: int) -> dict[str, Any]:
    """Levels and effective SNR of a (mixture, target) pair after one stage."""
    mixture, clean = _trimmed(noisy, target)
    residual = mixture - clean
    target_energy = float(np.dot(clean, clean))
    residual_energy = float(np.dot(residual, residual))
    if target_energy <= _TINY:
        state, esnr = "no_target", None
    elif residual_energy <= target_energy * 10.0 ** (-IDENTICAL_DB / 10.0):
        state, esnr = "identical", None
    else:
        state, esnr = "finite", 10.0 * math.log10(target_energy / residual_energy)
    return {
        "samples": int(mixture.size),
        "seconds": mixture.size / float(sample_rate),
        "noisy_rms_dbfs": rms_dbfs(mixture),
        "noisy_peak_dbfs": peak_dbfs(mixture),
        "target_rms_dbfs": rms_dbfs(clean),
        "target_peak_dbfs": peak_dbfs(clean),
        "residual_rms_dbfs": rms_dbfs(residual),
        "esnr_db": esnr,
        "esnr_state": state,
        "input_si_sdr_db": si_sdr_db(clean, mixture) if state == "finite" else None,
        "frame_esnr_db": frame_esnr_db(mixture, clean, sample_rate),
        "frame_hop_seconds": FRAME_SECONDS,
    }


def frame_esnr_db(noisy, target, sample_rate: int) -> list[Optional[float]]:
    """Effective SNR per non-overlapping 20 ms frame, None where the target is
    silent, clamped to ``FRAME_RANGE_DB`` and rounded to 0.1 dB."""
    mixture, clean = _trimmed(noisy, target)
    hop = max(1, int(round(FRAME_SECONDS * sample_rate)))
    count = mixture.size // hop
    if count == 0:
        return []
    clean_frames = clean[: count * hop].reshape(count, hop)
    residual_frames = (mixture - clean)[: count * hop].reshape(count, hop)
    target_power = np.mean(np.square(clean_frames), axis=1)
    residual_power = np.mean(np.square(residual_frames), axis=1)
    low, high = FRAME_RANGE_DB
    silent = 10.0 ** (SILENT_FRAME_DBFS / 10.0)
    values: list[Optional[float]] = []
    for power, rest in zip(target_power, residual_power):
        if power < silent:
            values.append(None)
            continue
        ratio = high if rest <= _TINY else 10.0 * math.log10(power / rest)
        values.append(round(min(max(ratio, low), high), 1))
    return values


def gain_staged(noisy, target) -> tuple[np.ndarray, np.ndarray, float]:
    """The pair as a converter would deliver it: both divided by their shared
    peak when it exceeds full scale (the device chain's A/D step), else as is.
    Returns float32 copies and the gain applied."""
    mixture = np.asarray(noisy, dtype=np.float32).reshape(-1)
    clean = np.asarray(target, dtype=np.float32).reshape(-1)
    peak = max(float(np.max(np.abs(mixture), initial=0.0)), float(np.max(np.abs(clean), initial=0.0)))
    if peak > 1.0:
        return mixture / peak, clean / peak, 1.0 / peak
    return mixture.copy(), clean.copy(), 1.0


def score_pair(target, estimate, sample_rate: int) -> dict[str, Optional[float]]:
    """SI-SDR, STOI and wideband PESQ of ``estimate`` against ``target``.

    Each is None where it is undefined (a silent target) or its optional scorer
    cannot run (not installed, a clip too short for PESQ, not 16 kHz).
    """
    result: dict[str, Optional[float]] = {"si_sdr_db": si_sdr_db(target, estimate), "stoi": None, "pesq_wb": None}
    if result["si_sdr_db"] is None:
        return result
    clean, guess = (part.astype(np.float32) for part in _trimmed(target, estimate))
    try:
        from pystoi import stoi

        result["stoi"] = float(stoi(clean, guess, sample_rate))
    except Exception:
        pass
    if sample_rate == 16000:
        try:
            from pesq import pesq

            result["pesq_wb"] = float(pesq(16000, clean, guess, "wb"))
        except Exception:
            pass
    return sanitize(result)


def rir_summary(impulse, sample_rate: int) -> dict[str, Any]:
    """What the page draws for one impulse response: a 1 ms peak envelope, the
    Schroeder decay from the direct path, a T20 decay-time estimate, and where
    the direct (6 ms) and early (50 ms) target windows end -- the windows
    ``wav_apply_rir`` cuts after the peak."""
    x = _as_array(impulse)
    if x.size == 0:
        return {"sample_rate": int(sample_rate), "length_ms": 0.0, "peak_ms": 0.0, "bin_ms": 1.0,
                "envelope_db": [], "edc_db": [], "t20_s": None, "direct_end_ms": None, "early_end_ms": None}
    hop = max(1, int(round(sample_rate / 1000.0)))
    bin_ms = 1000.0 * hop / sample_rate
    peak_index = int(np.argmax(np.abs(x)))
    count = int(math.ceil(x.size / hop))
    padded = np.zeros(count * hop)
    padded[: x.size] = np.abs(x)
    envelope = padded.reshape(count, hop).max(axis=1)
    top = max(float(envelope.max()), _TINY)
    envelope_db = [round(max(20.0 * math.log10(max(value, _TINY) / top), _FLOOR_DB), 1) for value in envelope]

    energy = np.square(x[peak_index:])
    decay = np.cumsum(energy[::-1])[::-1]
    total = max(float(decay[0]), _TINY)
    decay_db = 10.0 * np.log10(np.maximum(decay / total, 10.0 ** (_FLOOR_DB / 10.0)))
    edc_db = [round(float(value), 1) for value in decay_db[::hop]]
    peak_ms = 1000.0 * peak_index / sample_rate
    return {
        "sample_rate": int(sample_rate),
        "length_ms": 1000.0 * x.size / sample_rate,
        "peak_ms": peak_ms,
        "bin_ms": bin_ms,
        "envelope_db": envelope_db,
        "edc_db": edc_db,
        "t20_s": _t20(decay_db, sample_rate),
        "direct_end_ms": peak_ms + 6.0,
        "early_end_ms": peak_ms + 50.0,
    }


def _t20(decay_db: np.ndarray, sample_rate: int) -> Optional[float]:
    """Decay time from the -5 to -25 dB span of the Schroeder curve, x3."""
    below_5 = np.nonzero(decay_db <= -5.0)[0]
    below_25 = np.nonzero(decay_db <= -25.0)[0]
    if below_5.size == 0 or below_25.size == 0 or below_25[0] <= below_5[0]:
        return None
    span = decay_db[below_5[0] : below_25[0] + 1]
    times = np.arange(span.size) / float(sample_rate)
    slope = float(np.polyfit(times, span, 1)[0])
    return -60.0 / slope if slope < 0.0 else None


def room_geometry(scene: Optional[Mapping[str, Any]], rirs: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """The room of a bank row, in the shape the page draws.

    ``scene`` is the summarised bank scene a trace recorded; it names the room's
    ``wav_path``, and the geometry is read from the sidecar JSON the bank reads
    its channel map from. ``rirs`` are the row's impulse-response records; each
    source they used is marked with the role it played. None when the row has
    no room or the room has no geometry.
    """
    if not scene or not scene.get("wav_path"):
        return None
    wav = Path(str(scene["wav_path"]))
    meta = None
    for candidate in (wav.with_suffix(".json"), wav.parent / "metadata.json"):
        if candidate.is_file():
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            break
    room = (meta or {}).get("scene") or {}
    if not room.get("room_dim") or not room.get("mic_pos"):
        return None
    roles: dict[str, list[str]] = {}
    for record in rirs:
        label = (record.get("metadata") or {}).get("label")
        role = record.get("role")
        if label and role and role not in roles.setdefault(str(label), []):
            roles[str(label)].append(str(role))
    sources = [
        {
            "label": str(entry.get("label", "")),
            "channel": entry.get("channel"),
            "position": entry.get("source_pos"),
            "distance_m": entry.get("distance_m"),
            "roles": roles.get(str(entry.get("label", "")), []),
        }
        for entry in room.get("channel_map") or []
    ]
    obstacles = [
        {
            "footprint": item.get("footprint") or [],
            "z_min": item.get("z_min", 0.0),
            "z_max": item.get("z_max", 0.0),
            "material": item.get("material"),
        }
        for item in room.get("obstacles") or []
    ]
    return sanitize({
        "kind": "box",
        "room_id": scene.get("room_id") or wav.stem,
        "room_dim": list(room["room_dim"]),
        "receiver": list(room["mic_pos"]),
        "rt60": room.get("rt60"),
        "sources": sources,
        "obstacles": obstacles,
    })
