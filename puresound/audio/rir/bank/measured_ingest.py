"""Ingest published measured RIRs as a QC-passing M6 ``measured`` variant.

M6's ``real_native`` release recipe is blocked until a QC-passed measured variant
exists, and measured corpora do not pass M6 item QC as published: sampling eight
items from each of five corpora quarantined all forty, every one of them on
``prearrival_energy`` (23 to 34 channels per corpus). The cause is a time-origin
mismatch, not corrupt data. M6 defines ``t = 0`` as the emission instant, so samples
before ``floor(distance / c * fs)`` must be silent. Published RIRs are referenced
to their own direct arrival instead — measured pre-arrival energy sits 44 to 74 dB
*above* the measurement noise floor, which is the signature of the direct path
falling inside the geometric window rather than of noise leaking into it.

So the ingest re-inserts the propagation delay that the corpora removed. The room
response itself is never modified: each channel is shifted so its ISO 3382-1
impulse-response start lands on the geometric arrival sample implied by the
published source-receiver distance, and the (now pre-arrival) region ahead of it
is zeroed. That is a pure delay plus a leading-window mute, both recorded per
channel.

Two properties keep this from becoming a way to force bad data through a gate.

*It is falsifiable.* A corpus that already publishes a shared emission origin
needs no shift, so its measured shift must come out near zero. BRUDEX is such a
corpus — its onset tracks distance with slope 1.00 — and it is the cross-check
that the estimator is finding the direct path rather than an arbitrary landmark.
It is also how the onset threshold was chosen: ISO 3382-1's 20 dB below peak puts
BRUDEX within one sample (21 mm), while every "first sample above the noise floor"
variant lands 13 to 66 samples early, latching onto the acausal pre-ringing that
sweep deconvolution leaves ahead of the direct path.

*It rejects rather than repairs.* A channel whose direct path cannot be located
is dropped, not aligned on a guess. ``earlier_arrival`` catches the case the ISO
threshold is blind to — a direct path more than 20 dB below a later reflection,
which the forward scan steps over — by looking for a loud pre-onset window
*separated* from the onset by a gap. Level alone cannot decide this: the
pre-onset region legitimately holds the direct path's own rising edge, 40 dB or
more above the noise floor. Alignment also refuses to discard signal
(``removed_energy``) or to apply a shift the record cannot justify
(``implausible_shift``).

What this module deliberately does not do is relax M6 QC. Items that stay broken
after alignment — DIFFRIR's faded tails, for instance — are quarantined by the
same gates that quarantine synthetic items, which is the correct outcome.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..metrics import estimate_noise_floor_lundeby
from ..scene.schema import EnvironmentConfig
from .qc import (
    RIRBankQCPolicy,
    audit_rir_bank_qc_release,
    run_rir_bank_qc,
)
from .schema import (
    BANK_SPLITS,
    BankGeneratorProvenance,
    BankRendererProfile,
    BankSplitPolicy,
    RIRBankItem,
    RIRBankManifest,
    canonical_json_sha256,
    sha256_file,
    task_plan_rows,
    write_split_indexes,
)

#: Bumped whenever the alignment changes what samples a released variant holds.
MEASURED_TIME_ORIGIN_POLICY = (
    "puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1"
)

MEASURED_INGEST_SCHEMA_VERSION = "puresound.measured_m6_ingest.v1"

#: Environment assumed for every measured item. None of the five reference
#: corpora publishes air temperature or humidity, so the sound speed used to
#: place the geometric arrival is an assumption, not a measurement. It is
#: recorded in each item's scene so QC recomputes the arrival from the same
#: number the alignment used, and so the assumption travels with the data.
ASSUMED_MEASURED_ENVIRONMENT = EnvironmentConfig(
    temperature_c=20.0, relative_humidity_percent=50.0
)


@dataclass(frozen=True)
class MeasuredAlignmentPolicy:
    """Versioned bounds for locating and relocating the measured direct path."""

    #: ISO 3382-1 takes the impulse response to start where the broadband signal
    #: *first* rises above 20 dB below its peak, scanning forward from the start
    #: of the record. Searching backward from the peak instead would find the last
    #: quiet moment before it, leaving every earlier arrival outside the response.
    onset_threshold_db_below_peak: float = -20.0
    #: ISO 3382-1 also requires the start threshold to clear the noise floor.
    #: Without this a channel whose peak sits only 20 dB above its own noise
    #: would take its first noise sample as the start of the response.
    onset_margin_above_noise_db: float = 20.0
    #: Direct path is taken to be the strongest arrival within this window from
    #: the start of the record; beyond it, a late reflection could outrank the
    #: direct path in a strongly reverberant far-field measurement.
    anchor_search_ms: float | None = None
    #: Excluded from the prominence window so the direct path's own decay does
    #: not count as reverberation.
    direct_window_ms: float = 2.5
    prominence_window_ms: float = 5.0
    #: Reported for every channel; only pathologically undetectable direct paths
    #: are rejected, because a weak direct path is exactly what a genuine
    #: far-field measurement looks like and is the reason this corpus is useful.
    minimum_direct_prominence_db: float = 0.0
    #: An arrival ahead of the detected onset means the anchor is a reflection and
    #: the true direct path is more than 20 dB weaker, so the ISO threshold
    #: stepped over it. What separates that from the direct path's own rising
    #: edge is a *gap*: a rising edge is contiguous with the onset, a distinct
    #: earlier arrival is not. Level alone cannot decide it — the pre-onset region
    #: legitimately holds the rising edge at up to 20 dB below the peak, which is
    #: already 40 dB or more above the noise floor.
    earlier_arrival_window_ms: float = 1.0
    earlier_arrival_margin_db: float = 10.0
    earlier_arrival_separation_ms: float = 1.0
    #: Muting the pre-arrival window may not cost more than this fraction of the
    #: channel's energy; more means the mute is eating the response.
    maximum_removed_energy_fraction: float = 1e-2
    #: Prepending propagation delay is bounded by physical source distance;
    #: removing a corpus's leading pad is bounded only by what the file holds,
    #: and REVERB publishes a 131 ms one, so the two directions differ.
    maximum_delay_ms: float = 100.0
    maximum_advance_ms: float = 250.0
    #: Raised-cosine ramp that eases the leading mute in. It occupies the samples
    #: *before* the impulse-response start, never the direct path: the onset is
    #: placed this many samples after the geometric arrival so the ramp only ever
    #: touches sub-threshold pre-onset signal. Attenuating the arrival itself
    #: would silently discard most of a near-field channel's energy.
    fade_in_samples: int = 4
    policy_id: str = MEASURED_TIME_ORIGIN_POLICY

    def __post_init__(self) -> None:
        if self.policy_id != MEASURED_TIME_ORIGIN_POLICY:
            raise ValueError("unsupported measured time-origin policy")
        if self.onset_threshold_db_below_peak >= 0.0:
            raise ValueError("onset threshold must sit below the channel peak")
        if self.direct_window_ms <= 0.0 or self.prominence_window_ms <= 0.0:
            raise ValueError("direct and prominence windows must be positive")
        if not 0.0 < self.maximum_removed_energy_fraction < 1.0:
            raise ValueError("maximum removed energy fraction must lie in (0, 1)")
        if self.maximum_delay_ms <= 0.0 or self.maximum_advance_ms <= 0.0:
            raise ValueError("shift bounds must be positive")
        if self.earlier_arrival_window_ms <= 0.0:
            raise ValueError("earlier-arrival window must be positive")
        if self.fade_in_samples < 0:
            raise ValueError("fade length cannot be negative")
        if self.anchor_search_ms is not None and self.anchor_search_ms <= 0.0:
            raise ValueError("anchor search window must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "onset_threshold_db_below_peak": self.onset_threshold_db_below_peak,
            "onset_margin_above_noise_db": self.onset_margin_above_noise_db,
            "anchor_search_ms": self.anchor_search_ms,
            "direct_window_ms": self.direct_window_ms,
            "prominence_window_ms": self.prominence_window_ms,
            "minimum_direct_prominence_db": self.minimum_direct_prominence_db,
            "earlier_arrival_window_ms": self.earlier_arrival_window_ms,
            "earlier_arrival_margin_db": self.earlier_arrival_margin_db,
            "earlier_arrival_separation_ms": (
                self.earlier_arrival_separation_ms
            ),
            "maximum_removed_energy_fraction": (
                self.maximum_removed_energy_fraction
            ),
            "maximum_delay_ms": self.maximum_delay_ms,
            "maximum_advance_ms": self.maximum_advance_ms,
            "fade_in_samples": self.fade_in_samples,
        }

    @property
    def policy_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class ChannelAlignment:
    """What the alignment did to one channel, and whether it is trustworthy."""

    channel: int
    label: str
    distance_m: float
    geometric_arrival_sample: int
    anchor_sample: int
    detected_onset_sample: int
    shift_samples: int
    direct_prominence_db: float | None
    noise_estimate_relative_db: float | None
    earlier_arrival_relative_db: float | None
    earlier_arrival_separation_samples: int | None
    removed_energy_fraction: float
    #: None means the window is exactly silent, which is the intended outcome
    #: after alignment and the strongest form of passing the causality gate.
    prearrival_relative_db_before: float | None
    prearrival_relative_db_after: float | None
    status: str
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "label": self.label,
            "distance_m": self.distance_m,
            "geometric_arrival_sample": self.geometric_arrival_sample,
            "anchor_sample": self.anchor_sample,
            "detected_onset_sample": self.detected_onset_sample,
            "shift_samples": self.shift_samples,
            "direct_prominence_db": self.direct_prominence_db,
            "noise_estimate_relative_db": self.noise_estimate_relative_db,
            "earlier_arrival_relative_db": self.earlier_arrival_relative_db,
            "earlier_arrival_separation_samples": (
                self.earlier_arrival_separation_samples
            ),
            "removed_energy_fraction": self.removed_energy_fraction,
            "prearrival_relative_db_before": self.prearrival_relative_db_before,
            "prearrival_relative_db_after": self.prearrival_relative_db_after,
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _relative_db(value: float, reference: float) -> float | None:
    """Level ratio in dB, or None when either side is exactly zero.

    None is the honest answer for a silent pre-arrival window, and it keeps the
    reports JSON-serializable under ``allow_nan=False``; a sentinel like -inf
    would be neither.
    """
    if reference <= 0.0 or value <= 0.0:
        return None
    return 20.0 * math.log10(value / reference)


def _noise_estimate(signal: np.ndarray, sample_rate: int) -> float:
    """Stationary noise RMS, as conservatively as the record allows.

    The Lundeby floor is the better estimator but converges on only part of this
    reference — corpora that faded their tails before publication leave it
    nothing stationary to find. The trailing-decile RMS always exists, though it
    reads high when the tail still carries decaying response. Taking whichever is
    *larger* errs toward a high floor, which makes ``earlier_arrival`` lenient
    rather than prone to rejecting channels over ordinary noise.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return 0.0
    trailing = x[max(0, int(0.9 * x.size)) :]
    proxy = (
        float(np.sqrt(np.mean(np.square(trailing, dtype=np.float64))))
        if trailing.size
        else 0.0
    )
    estimate = estimate_noise_floor_lundeby(x, sample_rate)
    lundeby = (
        math.sqrt(float(estimate.noise_power))
        if estimate.correction_applied and estimate.noise_power > 0.0
        else 0.0
    )
    return max(proxy, lundeby)


def _loudest_short_window_rms(
    signal: np.ndarray, window_samples: int
) -> tuple[float, int] | None:
    """Largest RMS over any window of ``window_samples``, and where it starts."""
    x = np.asarray(signal, dtype=np.float64)
    window = max(1, int(window_samples))
    if x.size == 0:
        return None
    if x.size < window:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 0
    power = np.square(x, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    sums = cumulative[window:] - cumulative[:-window]
    index = int(np.argmax(sums))
    return float(np.sqrt(sums[index] / window)), index


def align_measured_channel(
    signal: np.ndarray,
    sample_rate: int,
    *,
    distance_m: float,
    sound_speed_m_s: float,
    policy: MeasuredAlignmentPolicy | None = None,
) -> tuple[np.ndarray, ChannelAlignment]:
    """Move a measured channel's direct path onto its geometric arrival sample.

    Returns the conditioned channel and the record of what was done. A rejected
    channel is still returned conditioned, so callers can inspect it; the caller
    decides what a rejection means for the item.
    """

    policy = policy or MeasuredAlignmentPolicy()
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    reasons: list[str] = []

    magnitude = np.abs(x)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    total_energy = float(np.square(x, dtype=np.float64).sum())
    if not math.isfinite(distance_m) or distance_m < 0.0:
        reasons.append("invalid_distance")
    if peak <= 0.0 or total_energy <= 0.0:
        reasons.append("silent_channel")
    if reasons:
        return x, ChannelAlignment(
            channel=-1,
            label="",
            distance_m=float(distance_m),
            geometric_arrival_sample=0,
            anchor_sample=-1,
            detected_onset_sample=-1,
            shift_samples=0,
            direct_prominence_db=None,
            noise_estimate_relative_db=None,
            earlier_arrival_relative_db=None,
            earlier_arrival_separation_samples=None,
            removed_energy_fraction=0.0,
            prearrival_relative_db_before=None,
            prearrival_relative_db_after=None,
            status="rejected",
            rejection_reasons=tuple(sorted(set(reasons))),
        )

    target = max(0, int(math.floor(distance_m / sound_speed_m_s * sample_rate)))

    anchor_end = magnitude.size
    if policy.anchor_search_ms is not None:
        anchor_end = min(
            magnitude.size,
            max(1, int(round(policy.anchor_search_ms * sample_rate / 1000.0))),
        )
    anchor = int(np.argmax(magnitude[:anchor_end]))
    anchor_peak = float(magnitude[anchor])

    prearrival_before = (
        float(magnitude[:target].max()) if target and target <= magnitude.size else 0.0
    )

    noise = _noise_estimate(x, sample_rate)
    onset_threshold = max(
        anchor_peak * 10.0 ** (policy.onset_threshold_db_below_peak / 20.0),
        noise * 10.0 ** (policy.onset_margin_above_noise_db / 20.0),
    )
    above = np.flatnonzero(magnitude[: anchor + 1] >= onset_threshold)
    onset = int(above[0]) if above.size else anchor
    earlier_window = max(
        1, int(round(policy.earlier_arrival_window_ms * sample_rate / 1000.0))
    )
    loudest = _loudest_short_window_rms(x[:onset], earlier_window) if onset else None
    earlier_relative_db: float | None = None
    earlier_separation_samples: int | None = None
    if loudest is not None and noise > 0.0:
        earlier_rms, earlier_index = loudest
        earlier_relative_db = _relative_db(earlier_rms, noise)
        earlier_separation_samples = max(0, onset - (earlier_index + earlier_window))
        separation_limit = max(
            1, int(round(policy.earlier_arrival_separation_ms * sample_rate / 1000.0))
        )
        if (
            earlier_relative_db is not None
            and earlier_relative_db > policy.earlier_arrival_margin_db
            and earlier_separation_samples > separation_limit
        ):
            reasons.append("earlier_arrival")

    direct_samples = max(1, int(round(policy.direct_window_ms * sample_rate / 1000.0)))
    prominence_samples = max(
        1, int(round(policy.prominence_window_ms * sample_rate / 1000.0))
    )
    window = x[anchor + direct_samples : anchor + direct_samples + prominence_samples]
    prominence_db: float | None = None
    if window.size:
        reverberant_rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        prominence_db = _relative_db(anchor_peak, reverberant_rms)
        if prominence_db < policy.minimum_direct_prominence_db:
            reasons.append("weak_direct_path")

    # The impulse-response start is placed a fade length *after* the geometric
    # arrival so the mute's ramp lands on pre-onset signal only. The direct path
    # and its precursor pass through untouched, at the cost of a documented
    # arrival bias of fade_in_samples (0.25 ms at 16 kHz, well inside the 1 ms
    # arrival tolerance M6 QC applies).
    fade = max(0, int(policy.fade_in_samples))
    shift = (target + fade) - onset
    maximum_delay = int(round(policy.maximum_delay_ms * sample_rate / 1000.0))
    maximum_advance = int(round(policy.maximum_advance_ms * sample_rate / 1000.0))
    if shift > maximum_delay or -shift > maximum_advance:
        reasons.append("implausible_shift")

    if shift >= 0:
        aligned = np.concatenate([np.zeros(shift, dtype=np.float64), x])
    else:
        aligned = x[-shift:].copy()
    aligned[:target] = 0.0
    fade = min(fade, max(0, aligned.size - target))
    if fade > 0:
        ramp = 0.5 * (
            1.0 - np.cos(np.pi * (np.arange(1, fade + 1) / (fade + 1.0)))
        )
        aligned[target : target + fade] *= ramp

    aligned_energy = float(np.square(aligned, dtype=np.float64).sum())
    removed_fraction = max(0.0, (total_energy - aligned_energy) / total_energy)
    if removed_fraction > policy.maximum_removed_energy_fraction:
        reasons.append("removed_energy")

    aligned_peak = float(np.abs(aligned).max()) if aligned.size else 0.0
    prearrival_after = (
        float(np.abs(aligned[:target]).max()) if target and target <= aligned.size
        else 0.0
    )

    return aligned, ChannelAlignment(
        channel=-1,
        label="",
        distance_m=float(distance_m),
        geometric_arrival_sample=target,
        anchor_sample=anchor,
        detected_onset_sample=onset,
        shift_samples=int(shift),
        direct_prominence_db=prominence_db,
        noise_estimate_relative_db=(
            _relative_db(noise, peak) if noise > 0.0 else None
        ),
        earlier_arrival_relative_db=earlier_relative_db,
        earlier_arrival_separation_samples=earlier_separation_samples,
        removed_energy_fraction=removed_fraction,
        prearrival_relative_db_before=_relative_db(prearrival_before, peak),
        prearrival_relative_db_after=_relative_db(prearrival_after, aligned_peak),
        status="rejected" if reasons else "aligned",
        rejection_reasons=tuple(sorted(set(reasons))),
    )


@dataclass(frozen=True)
class MeasuredSourceItem:
    """One published multi-path measured record, as the corpus view stores it."""

    item_id: str
    corpus: str
    room: str
    audio_path: Path
    metadata_path: Path
    channel_map: tuple[Mapping[str, Any], ...]
    rt60_s: float | None


@dataclass(frozen=True)
class ItemAlignment:
    """Alignment outcome for a whole item, and the audio it produced."""

    item: MeasuredSourceItem
    channels: tuple[ChannelAlignment, ...]
    status: str
    audio: np.ndarray | None = None
    sample_rate: int = 0

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    reason
                    for channel in self.channels
                    for reason in channel.rejection_reasons
                }
            )
        )


def align_measured_item(
    item: MeasuredSourceItem,
    *,
    environment: EnvironmentConfig | None = None,
    policy: MeasuredAlignmentPolicy | None = None,
) -> ItemAlignment:
    """Align every channel of one measured item onto its geometric arrival.

    The channels of one item are independent source-receiver transfer paths
    rather than a synchronized array, so each carries its own time origin and is
    aligned on its own. One rejected channel rejects the item: M6 QC fails an
    item on any channel's per-channel failure, so admitting a partially aligned
    item would only move the rejection later.
    """

    import soundfile as sf

    policy = policy or MeasuredAlignmentPolicy()
    environment = environment or ASSUMED_MEASURED_ENVIRONMENT
    sound_speed = float(environment.sound_speed_m_s)

    audio, sample_rate = sf.read(item.audio_path, always_2d=True, dtype="float64")
    aligned_channels: list[np.ndarray] = []
    records: list[ChannelAlignment] = []
    for position, entry in enumerate(item.channel_map):
        index = int(entry["channel"])
        if index >= audio.shape[1]:
            records.append(
                ChannelAlignment(
                    channel=position,
                    label=str(entry.get("label", f"ch{position}")),
                    distance_m=float(entry.get("distance_m", float("nan"))),
                    geometric_arrival_sample=0,
                    anchor_sample=-1,
                    detected_onset_sample=-1,
                    shift_samples=0,
                    direct_prominence_db=None,
                    noise_estimate_relative_db=None,
                    earlier_arrival_relative_db=None,
                    earlier_arrival_separation_samples=None,
                    removed_energy_fraction=0.0,
                    prearrival_relative_db_before=None,
                    prearrival_relative_db_after=None,
                    status="rejected",
                    rejection_reasons=("channel_absent_from_audio",),
                )
            )
            continue
        conditioned, record = align_measured_channel(
            audio[:, index],
            int(sample_rate),
            distance_m=float(entry.get("distance_m", float("nan"))),
            sound_speed_m_s=sound_speed,
            policy=policy,
        )
        aligned_channels.append(conditioned)
        records.append(
            ChannelAlignment(
                **{
                    **record.to_dict(),
                    "channel": position,
                    "label": str(entry.get("label", f"ch{position}")),
                    "rejection_reasons": record.rejection_reasons,
                }
            )
        )

    status = (
        "aligned"
        if records and all(record.status == "aligned" for record in records)
        else "rejected"
    )
    if not aligned_channels or len(aligned_channels) != len(item.channel_map):
        return ItemAlignment(item=item, channels=tuple(records), status="rejected")

    length = max(channel.size for channel in aligned_channels)
    stacked = np.zeros((length, len(aligned_channels)), dtype=np.float64)
    for index, channel in enumerate(aligned_channels):
        stacked[: channel.size, index] = channel
    return ItemAlignment(
        item=item,
        channels=tuple(records),
        status=status,
        audio=stacked,
        sample_rate=int(sample_rate),
    )


_ITEM_STEM = re.compile(r"^(?P<room>.+)_(?P<index>\d+)$")


def scan_measured_corpus_view(
    root: str | Path,
    *,
    corpora: Sequence[str] | None = None,
    limit_per_corpus: int | None = None,
) -> tuple[MeasuredSourceItem, ...]:
    """Enumerate a ``<root>/<corpus>_<room>_<index>.{wav,json}`` corpus view."""

    view_root = Path(root)
    wanted = set(corpora) if corpora else None
    found: list[MeasuredSourceItem] = []
    for metadata_path in sorted(view_root.glob("*.json")):
        stem = metadata_path.stem
        match = _ITEM_STEM.match(stem)
        audio_path = metadata_path.with_suffix(".wav")
        if match is None or not audio_path.is_file():
            continue
        room = match.group("room")
        corpus = stem.split("_", 1)[0]
        if wanted is not None and corpus not in wanted:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scene = metadata.get("scene")
        if not isinstance(scene, Mapping):
            continue
        channel_map = scene.get("channel_map")
        if not isinstance(channel_map, list) or not channel_map:
            continue
        if not all(isinstance(entry, Mapping) and "channel" in entry for entry in channel_map):
            continue
        rt60 = scene.get("rt60")
        found.append(
            MeasuredSourceItem(
                item_id=stem,
                corpus=corpus,
                room=room,
                audio_path=audio_path,
                metadata_path=metadata_path,
                channel_map=tuple(dict(entry) for entry in channel_map),
                rt60_s=float(rt60) if isinstance(rt60, (int, float)) else None,
            )
        )
    if limit_per_corpus is None:
        return tuple(found)
    # Round-robin over rooms rather than truncating the sorted list: item ids sort
    # room-major, so a prefix of one corpus is a prefix of one *room*, and a
    # subset drawn that way collapses the room-disjoint split it is meant to
    # sample. Interleaving keeps a small subset representative of the corpus.
    by_room: dict[str, dict[str, list[MeasuredSourceItem]]] = {}
    for item in found:
        by_room.setdefault(item.corpus, {}).setdefault(item.room, []).append(item)
    limited: list[MeasuredSourceItem] = []
    for corpus in sorted(by_room):
        rooms = [by_room[corpus][room] for room in sorted(by_room[corpus])]
        taken = 0
        depth = 0
        while taken < limit_per_corpus and any(depth < len(r) for r in rooms):
            for room_items in rooms:
                if taken >= limit_per_corpus:
                    break
                if depth < len(room_items):
                    limited.append(room_items[depth])
                    taken += 1
            depth += 1
    return tuple(sorted(limited, key=lambda value: value.item_id))


def _measured_scene(
    item: MeasuredSourceItem,
    alignment: ItemAlignment,
    environment: EnvironmentConfig,
    policy: MeasuredAlignmentPolicy,
) -> dict[str, Any]:
    """Scene block for an aligned measured item.

    Carries the environment QC needs to recompute the geometric arrival and the
    alignment that was applied, so the conditioning is auditable from the item
    alone rather than only from the ingest report.
    """
    return {
        "scene_id": item.item_id,
        "origin": "real",
        "corpus": item.corpus,
        "room": item.room,
        "rt60": item.rt60_s,
        "environment": {
            "temperature_c": environment.temperature_c,
            "relative_humidity_percent": environment.relative_humidity_percent,
            "pressure_pa": environment.pressure_pa,
            "sound_speed_m_s": environment.sound_speed_m_s,
            "provenance": "assumed_by_measured_ingest_no_corpus_reports_conditions",
        },
        "channel_map": [
            {
                "channel": record.channel,
                "label": record.label,
                "distance_m": record.distance_m,
            }
            for record in alignment.channels
        ],
        "measured_time_origin": {
            "policy_id": policy.policy_id,
            "policy_sha256": policy.policy_sha256,
            "shift_samples_by_channel": [
                record.shift_samples for record in alignment.channels
            ],
            "geometric_arrival_sample_by_channel": [
                record.geometric_arrival_sample for record in alignment.channels
            ],
        },
    }


@dataclass
class MeasuredIngestReport:
    """Everything the ingest decided, per corpus and per rejected item."""

    root: str
    bank_root: str
    environment: Mapping[str, Any]
    alignment_policy: Mapping[str, Any]
    counts: dict[str, int] = field(default_factory=dict)
    per_corpus: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    qc_summary: Mapping[str, Any] | None = None
    qc_audit_valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEASURED_INGEST_SCHEMA_VERSION,
            "root": self.root,
            "bank_root": self.bank_root,
            "environment": dict(self.environment),
            "alignment_policy": dict(self.alignment_policy),
            "counts": dict(self.counts),
            "per_corpus": {
                corpus: dict(value) for corpus, value in sorted(self.per_corpus.items())
            },
            "rejections": list(self.rejections),
            "qc_summary": dict(self.qc_summary) if self.qc_summary else None,
            "qc_audit_valid": self.qc_audit_valid,
        }


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(finite, q)) if finite else None


def build_measured_m6_bank(
    source_root: str | Path,
    bank_root: str | Path,
    *,
    corpora: Sequence[str] | None = None,
    limit_per_corpus: int | None = None,
    split_seed: int = 20260804,
    bank_id: str = "puresound-m6-measured",
    code_revision: str = "unknown",
    environment: EnvironmentConfig | None = None,
    alignment_policy: MeasuredAlignmentPolicy | None = None,
    qc_policy: RIRBankQCPolicy | None = None,
    qc_workers: int = 1,
    run_qc: bool = True,
) -> MeasuredIngestReport:
    """Align a measured corpus view into a bank and publish its M6.3 QC release.

    Items whose alignment is refused never enter the manifest — they are recorded
    in the report instead, so the bank contains only items whose direct path was
    located, while the reason a corpus lost items stays visible. Items that align
    but fail acoustic QC do enter the manifest and are quarantined by
    ``run_rir_bank_qc`` exactly as synthetic items are.
    """

    view_root = Path(source_root)
    root = Path(bank_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"measured bank output already exists: {root}")
    environment = environment or ASSUMED_MEASURED_ENVIRONMENT
    alignment_policy = alignment_policy or MeasuredAlignmentPolicy()

    import soundfile as sf

    sources = scan_measured_corpus_view(
        view_root, corpora=corpora, limit_per_corpus=limit_per_corpus
    )
    if not sources:
        raise ValueError(f"no usable measured items under {view_root}")

    split_policy = BankSplitPolicy(seed=int(split_seed))
    report = MeasuredIngestReport(
        root=str(view_root),
        bank_root=str(root),
        environment={
            "temperature_c": environment.temperature_c,
            "relative_humidity_percent": environment.relative_humidity_percent,
            "pressure_pa": environment.pressure_pa,
            "sound_speed_m_s": environment.sound_speed_m_s,
            "provenance": (
                "assumed_by_measured_ingest_no_corpus_reports_conditions"
            ),
        },
        alignment_policy=alignment_policy.to_dict(),
    )
    root.mkdir(parents=True, exist_ok=True)

    items: list[RIRBankItem] = []
    statistics: dict[str, dict[str, list[float]]] = {}
    for index, source in enumerate(sorted(sources, key=lambda value: value.item_id)):
        alignment = align_measured_item(
            source, environment=environment, policy=alignment_policy
        )
        bucket = statistics.setdefault(
            source.corpus,
            {"shift": [], "prominence": [], "removed": [], "prearrival_after": []},
        )
        for record in alignment.channels:
            bucket["shift"].append(float(record.shift_samples))
            if record.direct_prominence_db is not None:
                bucket["prominence"].append(float(record.direct_prominence_db))
            bucket["removed"].append(float(record.removed_energy_fraction))
            # Only non-silent windows are collected: a None reading is the
            # intended outcome, so the list length is the count of channels that
            # did *not* end up causal, and an empty list is a clean pass.
            if record.prearrival_relative_db_after is not None:
                bucket["prearrival_after"].append(
                    float(record.prearrival_relative_db_after)
                )
        per_corpus = report.per_corpus.setdefault(
            source.corpus,
            {"seen": 0, "aligned": 0, "rejected": 0, "rejection_reasons": {}},
        )
        per_corpus["seen"] += 1
        if alignment.status != "aligned" or alignment.audio is None:
            per_corpus["rejected"] += 1
            for reason in alignment.rejection_reasons:
                per_corpus["rejection_reasons"][reason] = (
                    per_corpus["rejection_reasons"].get(reason, 0) + 1
                )
            report.rejections.append(
                {
                    "item_id": source.item_id,
                    "corpus": source.corpus,
                    "room": source.room,
                    "reasons": list(alignment.rejection_reasons),
                    "channels": [
                        record.to_dict()
                        for record in alignment.channels
                        if record.status == "rejected"
                    ],
                }
            )
            continue
        per_corpus["aligned"] += 1

        acoustic_space_id = f"real:{source.corpus}:{source.room}"
        room_dir = root / "items" / source.room
        room_dir.mkdir(parents=True, exist_ok=True)
        audio_path = room_dir / f"{source.item_id}.wav"
        metadata_path = room_dir / f"{source.item_id}.json"
        sf.write(
            audio_path,
            alignment.audio,
            alignment.sample_rate,
            subtype="FLOAT",
        )
        scene = _measured_scene(source, alignment, environment, alignment_policy)
        metadata = {
            "sample_id": source.item_id,
            "room_id": source.room,
            "scene": scene,
            "measured_source": {
                "corpus": source.corpus,
                "audio_path": str(source.audio_path),
                "audio_sha256": sha256_file(source.audio_path),
                "metadata_sha256": sha256_file(source.metadata_path),
            },
            "measured_alignment": {
                "policy_id": alignment_policy.policy_id,
                "policy_sha256": alignment_policy.policy_sha256,
                "channels": [record.to_dict() for record in alignment.channels],
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        info = sf.info(audio_path)
        items.append(
            RIRBankItem(
                item_id=source.item_id,
                room_id=source.room,
                acoustic_space_id=acoustic_space_id,
                scene_id=source.item_id,
                split=split_policy.assign(acoustic_space_id),
                generation_seed=index,
                origin="real",
                renderer_profile_id=f"measured-{source.corpus}",
                signal_variant="measured",
                level_policy="native_measured",
                rir_path=audio_path.relative_to(root).as_posix(),
                metadata_path=metadata_path.relative_to(root).as_posix(),
                rir_sha256=sha256_file(audio_path),
                metadata_sha256=sha256_file(metadata_path),
                scene_sha256=canonical_json_sha256(scene),
                sample_rate=info.samplerate,
                channel_count=info.channels,
                frame_count=info.frames,
                qc_status="pending",
            )
        )

    for corpus, bucket in statistics.items():
        summary = report.per_corpus.setdefault(corpus, {})
        summary["alignment"] = {
            "shift_samples_p5": _percentile(bucket["shift"], 5),
            "shift_samples_p50": _percentile(bucket["shift"], 50),
            "shift_samples_p95": _percentile(bucket["shift"], 95),
            "direct_prominence_db_p50": _percentile(bucket["prominence"], 50),
            "removed_energy_fraction_p95": _percentile(bucket["removed"], 95),
            "channels_with_residual_prearrival": len(bucket["prearrival_after"]),
            "residual_prearrival_relative_db_max": (
                max(bucket["prearrival_after"]) if bucket["prearrival_after"] else None
            ),
        }

    report.counts = {
        "source_items": len(sources),
        "aligned_items": len(items),
        "rejected_items": len(sources) - len(items),
    }
    if not items:
        raise ValueError(
            "no measured item survived alignment; see the ingest report rejections"
        )

    _publish_measured_manifest(
        root,
        items,
        split_policy=split_policy,
        bank_id=bank_id,
        code_revision=code_revision,
        environment=environment,
        alignment_policy=alignment_policy,
    )
    if run_qc:
        report.qc_summary = run_rir_bank_qc(
            root, policy=qc_policy, workers=int(qc_workers)
        )
        report.qc_audit_valid = bool(audit_rir_bank_qc_release(root)["valid"])
    report_path = root / "measured_ingest_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _publish_measured_manifest(
    root: Path,
    items: Sequence[RIRBankItem],
    *,
    split_policy: BankSplitPolicy,
    bank_id: str,
    code_revision: str,
    environment: EnvironmentConfig,
    alignment_policy: MeasuredAlignmentPolicy,
) -> RIRBankManifest:
    """Write the draft manifest ``run_rir_bank_qc`` consumes.

    Each corpus becomes its own renderer profile: the "renderer" of a measured
    item is the corpus's measurement chain, and the tilt and noise-floor studies
    both showed those chains differ enough that pooling them under one identity
    would hide the difference that matters most about this data.
    """

    empty = [
        split
        for split in BANK_SPLITS
        if not any(item.split == split for item in items)
    ]
    if empty:
        spaces = sorted({item.acoustic_space_id for item in items})
        raise ValueError(
            f"the room-disjoint split left {', '.join(empty)} empty: "
            f"{len(spaces)} acoustic spaces is too few for the "
            f"{split_policy.train_fraction:g}/{split_policy.validation_fraction:g}/"
            f"{split_policy.test_fraction:g} split at seed {split_policy.seed}. "
            "Ingest more rooms, or choose a different split seed."
        )
    split_indexes = write_split_indexes(root, items)
    corpora = sorted({item.renderer_profile_id: item for item in items}.keys())
    profiles = tuple(
        BankRendererProfile(
            profile_id=profile_id,
            renderer_id=f"published_measurement_chain.{profile_id[len('measured-'):]}",
            renderer_version="as_published",
            low_backend="measured",
            high_backend="measured",
            scene_schema_version=MEASURED_INGEST_SCHEMA_VERSION,
            renderer_config_sha256=alignment_policy.policy_sha256,
            # A measurement chain is the strongest evidence there is about its
            # own room, but M6 reserves "production_approved" for a profile with
            # an approval record, and ingesting data does not create one.
            evidence_tier="development",
        )
        for profile_id in corpora
    )
    generator = BankGeneratorProvenance(
        generator_id="puresound.audio.rir.bank.measured_ingest",
        generator_version=MEASURED_INGEST_SCHEMA_VERSION,
        code_revision=code_revision,
        config_sha256=canonical_json_sha256(
            {
                "alignment_policy": alignment_policy.to_dict(),
                "environment": {
                    "temperature_c": environment.temperature_c,
                    "relative_humidity_percent": (
                        environment.relative_humidity_percent
                    ),
                    "pressure_pa": environment.pressure_pa,
                    "sound_speed_m_s": environment.sound_speed_m_s,
                },
            }
        ),
        task_plan_sha256=canonical_json_sha256(task_plan_rows(items)),
        seed=int(split_policy.seed),
    )
    manifest = RIRBankManifest(
        bank_id=bank_id,
        release_status="draft",
        split_policy=split_policy,
        generator=generator,
        renderer_profiles=profiles,
        items=tuple(items),
        split_indexes=split_indexes,
    ).with_content_sha256()
    manifest_path = root / "rir_bank_manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
