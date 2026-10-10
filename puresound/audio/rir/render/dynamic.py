"""Offline moving-source rendering in a static room.

Each coherent path (direct sound and image reflections up to the scene's
order) is split into a short path filter and a travel time.  The filter —
spreading, directivity, wall reflection, obstacle insertion loss — is
evaluated every 40 ms at the source's emission-time pose and interpolated on
the 10 ms geometry grid.  The travel time is applied as a continuous
emission-to-reception time map read with a band-limited fractional delay, so
Doppler and arrival times follow the motion without frame switching.  The
diffuse late field is the project's energy-matched PathEvent-FDN coupling
evaluated at poses along each path and blended between them.  The algorithm,
its references and limits are documented in
``docs/algorithms/audio/dynamic_scene.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import hashlib

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve

from puresound.audio.rir.path_events import (
    generate_scene_shoebox_path_events,
    material_absorption_relaxation_models,
    render_path_events,
)
from puresound.audio.rir.path_events.band_filter import BAND_FILTER_TAPS
from puresound.audio.rir.path_events.fractional_delay import (
    WINDOWED_SINC_BETA,
    WINDOWED_SINC_HALF_WIDTH,
    fractional_read,
    windowed_sinc_kernel,
)
from puresound.audio.rir.physics.propagation import air_adjusted_rt60_s
from puresound.audio.rir.render.coupling import (
    PATH_EVENT_FDN_COUPLING_POLICY,
    couple_path_event_rir_with_fdn,
    transition_samples,
)
from puresound.audio.rir.scene.dynamic import TALKER_ROLES, DynamicSceneSpec
from puresound.audio.rir.scene.schema import Pose

DYNAMIC_RENDER_VERSION = "puresound.dynamic_geometric_fdn.v2.1"
GEOMETRY_STEP_S = 0.01
PATH_FILTER_STEP_S = 0.04
EARLY_WINDOW_S = 0.05
DIRECT_WINDOW_S = 0.006
# How long after its direct sound each reference type keeps a source's
# arrivals, as training's wav_apply_rir does; None keeps everything and 0
# nothing (the anechoic reference is the dry signal moved to the arrival).
REFERENCE_WINDOWS_S = {
    "early": EARLY_WINDOW_S,
    "direct": DIRECT_WINDOW_S,
    "full": None,
    "anechoic": 0.0,
}
MIXING_TIME_S = 0.024
TRANSITION_S = 0.016
BOUNDARY_TAIL_S = 0.024
OCCLUSION_MODEL = "fresnel_kirchhoff"
# The late field's energy is extrapolated from the coherent tail of a
# second-order path field, whatever order the scene renders coherently, so
# the diffuse level belongs to the room rather than to ``max_order``.
LATE_FIELD_ORDER = 2
# Late responses are computed at poses this close along each path.
LATE_POSE_SPACING_M = 1.0
LATE_POSE_TURN_DEG = 45.0


def near_weights(distance, radius=1.0):
    """Raised cosine across a 0.2 metre band centred on the near boundary."""
    x = np.clip((np.asarray(distance) - (radius - 0.1)) / 0.2, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def transport(signals, delay_samples, length, *, block_size=4096):
    """Move emitted samples to their reception times.

    ``delay_samples[e]`` is the travel time of the sample emitted at ``e``;
    output sample ``r`` reads the emission instant ``e`` solving
    ``e + delay(e) = r`` with a windowed-sinc fractional read.  No energy
    Jacobian is applied: at walking speeds (Mach < 0.006) the Doppler
    amplitude factor is below 0.05 dB.
    """
    values = np.asarray(signals, dtype=np.float64)
    delay = np.broadcast_to(np.asarray(delay_samples, dtype=np.float64), values.shape[-1:])
    if np.ptp(delay) == 0.0:
        # A constant delay is a time-invariant filter: convolve with its kernel.
        start, kernel = windowed_sinc_kernel(float(delay[0]))
        moved = np.zeros(values.shape[:-1] + (length,))
        full = fftconvolve(np.atleast_2d(values), kernel[None, :], axes=-1)
        lo, hi = max(start, 0), min(length, start + full.shape[-1])
        moved.reshape(-1, length)[:, lo:hi] = full[:, lo - start : hi - start]
        return moved
    received = np.arange(values.shape[-1], dtype=np.float64) + delay
    if np.any(np.diff(received) <= 0):
        raise ValueError("non-monotone propagation time map")
    return fractional_read(
        values, _emission_positions(received, length), block_size=block_size
    )


@dataclass
class DynamicSceneRender:
    """The microphone mixture, each source at the microphone (``stems``), each
    source's reference of the scene's RIR type (``reference_stems``), and the
    three references a model is scored against: ``target`` (target talkers),
    ``speech`` (every talker) and ``near`` (talkers in the near region)."""

    mixture: np.ndarray
    stems: dict[str, np.ndarray]
    reference_stems: dict[str, np.ndarray]
    references: dict[str, np.ndarray]
    timeline: dict
    metadata: dict
    source_audio: dict[str, np.ndarray]

    def audio(self):
        return {
            "input": self.mixture,
            **{f"source-{k}": v for k, v in self.stems.items()},
            **{f"reference-source-{k}": v for k, v in self.reference_stems.items()},
            **{f"reference-{k}": v for k, v in self.references.items()},
        }


def render_dynamic_scene(
    spec: DynamicSceneSpec,
    assets: dict[str, np.ndarray],
    *,
    progress: Callable[[float, str], None] | None = None,
    block_size: int = 4096,
) -> DynamicSceneRender:
    """Render 16 kHz mono source assets; source audio is never looped implicitly.

    Travel times follow the geometry every 10 ms; path filters are evaluated
    every 40 ms and interpolated. ``block_size`` bounds read temporaries
    without changing the resulting audio.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be positive")
    notify = progress or (lambda value, phase: None)
    notify(0, "geometry")
    sr = spec.sample_rate
    n = int(round(spec.duration_s * sr))
    environment = spec.room.environment
    # Material decay plus air absorption, the target the static FDN backends use.
    rt60 = {
        float(k): air_adjusted_rt60_s(
            float(v),
            float(k),
            float(environment.sound_speed_m_s),
            temperature_c=float(environment.temperature_c),
            relative_humidity_percent=float(environment.relative_humidity_percent),
            pressure_pa=float(environment.pressure_pa),
        )
        for k, v in spec.room.predicted_octave_rt60_s(
            [125, 250, 500, 1000, 2000, 4000]
        ).items()
    }
    if max(rt60.values()) > 10:
        raise ValueError("room decay exceeds the supported 10 second tail")
    sound_speed = environment.sound_speed_m_s
    diagonal_delay = np.linalg.norm(spec.room.dimensions_m) / sound_speed
    max_path_delay = max(3 * diagonal_delay, diagonal_delay + BOUNDARY_TAIL_S)
    tail = max(max(rt60.values()) if spec.late_reverb else 0, BOUNDARY_TAIL_S)
    length = n + int(np.ceil((tail + max_path_delay) * sr)) + 4
    context = _RenderContext(spec, rt60, length, block_size)
    stems, reference_stems, near_stems, timeline = {}, {}, {}, {}
    hashes, source_audio, seeds = {}, {}, {}
    for si, source in enumerate(spec.sources):
        raw = np.asarray(assets[source.asset_id], dtype=np.float32).astype(np.float64)
        if raw.ndim != 1 or not np.isfinite(raw).all() or raw.size > 120 * sr:
            raise ValueError("assets must be finite mono audio of at most two minutes")
        source_audio[source.asset_id] = raw.astype(np.float32)
        hashes[source.asset_id] = hashlib.sha256(
            raw.astype("<f4").tobytes()
        ).hexdigest()
        # Each talker gets its own FDN so the late tails are not one filter.
        seeds[source.source_id] = int(
            np.random.SeedSequence([spec.seed, si]).generate_state(1)[0] & 0x7FFFFFFF
        )
        low, high = si / len(spec.sources), (si + 1) / len(spec.sources)
        rendered = _render_source(
            context,
            source,
            _emission(spec, source, raw, n),
            seeds[source.source_id],
            lambda value, phase: notify(low + value * (high - low), phase),
        )
        stems[source.source_id] = rendered["full"]
        reference_stems[source.source_id] = rendered["reference"]
        near_stems[source.source_id] = rendered["near"]
        timeline[source.source_id] = rendered["timeline"]
    mixture = sum(stems.values(), np.zeros(length))
    gain = min(1.0, 0.95 / max(np.max(np.abs(mixture)), 1e-12))
    for group in (stems, reference_stems, near_stems):
        for key in group:
            group[key] = (group[key] * gain).astype(np.float32)
    mixture = sum(stems.values(), np.zeros(length, dtype=np.float32))

    def total(group, roles):
        keys = [s.source_id for s in spec.sources if s.role in roles]
        return sum((group[k] for k in keys), np.zeros(length, dtype=np.float32))

    references = {
        "target": total(reference_stems, ("target",)),
        "speech": total(reference_stems, TALKER_ROLES),
        "near": total(near_stems, TALKER_ROLES),
    }
    metadata = {
        "renderer": DYNAMIC_RENDER_VERSION,
        "scene": spec.to_dict(),
        "sample_rate": sr,
        "source_duration_s": spec.duration_s,
        "speech_playback": "complete_clips_only; repeat fits whole clips then silence",
        "duration_s": length / sr,
        "tail_s": (length - n) / sr,
        "max_propagation_and_predelay_s": max_path_delay,
        "input_gain": gain,
        "asset_hashes": hashes,
        "geometry_step_s": GEOMETRY_STEP_S,
        "path_filter_step_s": PATH_FILTER_STEP_S,
        "reference_rir": spec.reference_rir,
        "reference_window_s": REFERENCE_WINDOWS_S[spec.reference_rir],
        "fractional_delay": (
            f"kaiser_windowed_sinc_{2 * WINDOWED_SINC_HALF_WIDTH}"
            f"_beta{WINDOWED_SINC_BETA:g}"
        ),
        "occlusion_model": OCCLUSION_MODEL,
        "source_directivity": {
            tr.transducer_id: tr.directivity_id for tr in spec.room.sources
        },
        "late_field": (
            "energy-matched PathEvent-FDN coupling at poses along each path, "
            "blended with energy-preserving weights"
            if spec.late_reverb
            else "none"
        ),
        "late_coupling_policy": PATH_EVENT_FDN_COUPLING_POLICY,
        "late_field_path_order": LATE_FIELD_ORDER,
        "late_pose_spacing": {"metres": LATE_POSE_SPACING_M, "degrees": LATE_POSE_TURN_DEG},
        "mixing_time_s": MIXING_TIME_S,
        "transition_s": TRANSITION_S,
        "fdn_seeds": seeds,
        "wave_solver": False,
        "rt60_s": rt60,
    }
    return DynamicSceneRender(
        mixture,
        stems,
        reference_stems,
        references,
        {"times_s": context.times.tolist(), "sources": timeline},
        metadata,
        source_audio,
    )


class _RenderContext:
    """Per-scene constants and helpers shared by every source."""

    def __init__(self, spec, rt60, length, block_size):
        self.spec = spec
        self.rt60 = rt60
        self.length = length
        self.block_size = block_size
        self.sr = spec.sample_rate
        self.sound_speed = spec.room.environment.sound_speed_m_s
        self.mic = np.array(spec.room.mic_pos)
        self.times = np.unique(
            np.r_[
                np.arange(0, spec.duration_s, GEOMETRY_STEP_S),
                spec.duration_s,
                [k.time_s for s in spec.sources for k in s.keyframes],
            ]
        )
        self.boundary_tail = int(BOUNDARY_TAIL_S * self.sr)
        self.taps = self.boundary_tail + BAND_FILTER_TAPS + 4
        # Filtered signals run one filter length past the source end.
        self.filtered_length = int(round(spec.duration_s * self.sr)) + self.taps
        self.boundaries, _ = material_absorption_relaxation_models(spec.room)
        self.surface_models = {
            s.surface_id: self.boundaries[s.boundary] for s in spec.room.surfaces
        }
        origin = 10 * self.taps
        _, start, end = transition_samples(
            2 * origin, self.sr, origin, MIXING_TIME_S, TRANSITION_S
        )
        self.transition = (start - origin, end - origin)

    def events_at(self, source, state, max_order=None):
        """Path events of ``source`` at pose ``(x, y, z, yaw, pitch)``."""
        transducers = list(self.spec.room.sources)
        index = next(
            i
            for i, tr in enumerate(transducers)
            if tr.transducer_id == source.source_id
        )
        transducers[index] = replace(
            transducers[index],
            pose=Pose(state[:3].tolist(), [float(state[3]), float(state[4]), 0]),
        )
        return generate_scene_shoebox_path_events(
            replace(self.spec.room, sources=transducers),
            source_index=index,
            max_order=self.spec.max_order if max_order is None else max_order,
            occlusion_model=OCCLUSION_MODEL,
            boundary_admittance_models=self.boundaries,
        ).events

    def early_weight(self, offsets):
        """The coupling's equal-power early weight ``offsets`` samples after
        the direct arrival (1 everywhere without a late field)."""
        if not self.spec.late_reverb:
            return np.ones_like(offsets)
        start, end = self.transition
        phase = np.clip((offsets - start) / (end - start), 0.0, 1.0)
        return np.cos(0.5 * np.pi * phase)


def _emission(spec, source, raw, n):
    """Place complete speech clips; noise may fill a partial final cycle.

    Speech that cannot fit even once is an editing error, not permission to
    cut an utterance. After the last complete repetition the source is quiet;
    propagation and reverberation still render normally over the tail.
    """
    sr = spec.sample_rate
    dry = np.zeros(n)
    start = int(round(source.start_s * sr))
    available = n - start
    speech = source.role in TALKER_ROLES
    if speech and len(raw) > available:
        raise ValueError(
            f"{source.source_id}: complete speech clip needs "
            f"{len(raw) / sr:.3f} s after its start; only {available / sr:.3f} s "
            "remain. Increase scene duration, move the start earlier, or choose "
            "a shorter complete recording."
        )
    count = min(len(raw), n - start)
    emission = raw[:count]
    if source.repeat and raw.size:
        # Repetition is explicit in the scene. Taper loop seams without
        # normalizing source energy or changing the saved original asset.
        cycle = raw.copy()
        ramp = min(int(0.01 * sr), len(cycle) // 2)
        if ramp:
            fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
            cycle[:ramp] *= fade
            cycle[-ramp:] *= fade[::-1]
        count = (available // len(raw)) * len(raw) if speech else available
        emission = np.resize(cycle, count)
    dry[start : start + count] = emission * 10 ** (source.gain_db / 20)
    return dry


def _render_source(context, source, dry, seed, notify):
    """One source at the microphone: its full stem, its reference of the
    scene's RIR type, and that reference weighted by the near region."""
    spec, sr, times = context.spec, context.sr, context.times
    window = REFERENCE_WINDOWS_S[spec.reference_rir]
    pose = source.pose_at(times)
    distance = np.linalg.norm(pose[:, :3] - context.mic, axis=1)
    direct = distance / context.sound_speed * sr
    weight = (
        near_weights(distance, spec.near_radius_m)
        if source.role in TALKER_ROLES
        else np.zeros(len(times))
    )
    emitted = np.zeros(context.filtered_length)
    emitted[: len(dry)] = dry
    emission_times = np.arange(context.filtered_length) / sr
    weighted = emitted * np.interp(emission_times, times, weight)
    near_active = bool(np.any(weighted))
    direct_delay = np.interp(emission_times, times, direct)

    # A path keeps its identity (type and image order) along the trajectory;
    # the order in which a second-order path meets its two walls can swap as
    # the source moves, so it is not part of the identity.  The filter is
    # evaluated on update frames and interpolated between them; the image
    # position is interpolated too, which is exact: images are affine in the
    # source position, and the source moves linearly between keyframes, all of
    # which are update frames.
    update = _update_frames(times, source)
    events_by_pose, tracks = {}, {}
    for count, j in enumerate(update):
        key = tuple(np.round(pose[j], 9))
        if key not in events_by_pose:
            events_by_pose[key] = context.events_at(source, pose[j])
        for event in events_by_pose[key]:
            identity = (event.path_type, tuple(event.image_order_xyz))
            tracks.setdefault(identity, {})[j] = (key, event)
        notify(0.3 * (count + 1) / len(update), "geometry")

    # The filter changes linearly between update frames, so those frames are
    # the cross-fade knots; the last kernel holds over the filter tail.
    knots = np.r_[np.round(times[update] * sr).astype(int), context.filtered_length]
    inputs = _SegmentFilter(np.stack([emitted, weighted]), knots, context.taps)
    out = np.zeros((3, context.length))
    kernels_by_pose = {}
    for pi, slots in enumerate(tracks.values()):
        occupied = np.array(sorted(slots))
        images = np.array(
            [
                context.mic
                - slots[j][1].distance_m * np.asarray(slots[j][1].arrival_direction_unit)
                for j in occupied
            ]
        )
        image_track = np.stack(
            [np.interp(times, times[occupied], images[:, axis]) for axis in range(3)],
            axis=1,
        )
        delays = np.linalg.norm(image_track - context.mic, axis=1) / context.sound_speed * sr
        update_kernels = np.zeros((len(update), context.taps))
        for row, j in enumerate(update):
            if j not in slots:
                continue
            key, event = slots[j]
            cache_key = (event.event_id, key)
            if cache_key not in kernels_by_pose:
                kernels_by_pose[cache_key] = render_path_events(
                    [event],
                    sample_rate_hz=sr,
                    num_samples=context.taps,
                    include_propagation_delay=False,
                    surface_admittance_models=context.surface_models,
                    maximum_boundary_filter_tail_samples=context.boundary_tail,
                )
            update_kernels[row] = kernels_by_pose[cache_key]
        _bridge_gaps(update_kernels, times[update], np.isin(update, occupied))
        after_direct = (
            delays[update, None]
            + np.arange(context.taps)[None, :]
            - direct[update, None]
        )
        coherent = update_kernels * context.early_weight(after_direct)
        if not np.any(coherent):
            continue
        reference = coherent if window is None else coherent * (after_direct < window * sr)
        coherent = np.vstack([coherent, coherent[-1:]])
        reference = np.vstack([reference, reference[-1:]])
        filtered_full = inputs.apply(0, coherent)
        # With a late field the coherent part fades out 32 ms after the direct
        # sound, so an early (or full) reference usually equals it.
        same = np.array_equal(reference, coherent)
        channels = [filtered_full]
        if not same and np.any(reference):
            channels.append(inputs.apply(0, reference))
        if near_active and np.any(reference):
            channels.append(inputs.apply(1, reference))
        moved = transport(
            np.stack(channels),
            np.interp(emission_times, times, delays),
            context.length,
            block_size=context.block_size,
        )
        out[0] += moved[0]
        if same:
            out[1] += moved[0]
        elif np.any(reference):
            out[1] += moved[1]
        if near_active and np.any(reference):
            out[2] += moved[-1]
        notify(0.3 + 0.6 * (pi + 1) / len(tracks), "reflections")
    if spec.late_reverb:
        notify(0.9, "reverberation")
        out += _late_field(context, source, emitted, weighted, direct_delay, seed, window)
    if spec.reference_rir == "anechoic":
        dry_at_arrival = transport(
            np.stack([emitted, weighted]), direct_delay, context.length, block_size=context.block_size
        )
        out[1] += dry_at_arrival[0]
        out[2] += dry_at_arrival[1]
    return {
        "full": out[0],
        "reference": out[1],
        "near": out[2],
        "timeline": {
            "poses": pose.tolist(),
            "distance_m": distance.tolist(),
            "near_weight": weight.tolist(),
            "role": source.role,
        },
    }


def _late_field(context, source, emitted, weighted, direct_delay, seed, window):
    """Diffuse field from coupled responses along the path, blended on input.

    At poses along the path (the keyframes, plus enough between them that
    neighbours are at most 1 m and 45 degrees apart) the static PathEvent RIR
    is coupled to an FDN whose tail energy follows the room's RT60
    (``couple_path_event_rir_with_fdn``); its diffuse component, re-timed to
    start at the direct arrival, is that pose's late response.  An emitted
    sample drives the two responses around its emission time with linear
    weights rescaled so the blend's energy, too, is linear between them:
    neighbouring responses are only partly correlated, and plain linear
    weights would lose up to 3 dB half-way.  Returns the full, reference and
    near-reference contributions; the references keep the field's first
    ``window`` seconds after the direct sound (all of it for None).
    """
    sr = context.sr
    times, poses = _late_poses(source)
    response_length = int(np.ceil(max(context.rt60.values()) * sr)) + context.taps
    responses = []
    for state in poses:
        arrival = np.linalg.norm(state[:3] - context.mic) / context.sound_speed * sr
        start = int(round(arrival))
        rir = render_path_events(
            context.events_at(source, state, LATE_FIELD_ORDER),
            sample_rate_hz=sr,
            num_samples=start + response_length,
            fractional_delay="windowed_sinc",
            surface_admittance_models=context.surface_models,
            maximum_boundary_filter_tail_samples=context.boundary_tail,
        )
        diffuse = couple_path_event_rir_with_fdn(
            rir,
            sr,
            start,
            context.rt60,
            mixing_time_s=MIXING_TIME_S,
            transition_duration_s=TRANSITION_S,
            seed=seed,
        ).diffuse_component
        responses.append(
            fractional_read(diffuse, np.arange(response_length) + arrival)
        )
    received = np.arange(context.filtered_length) + direct_delay
    excitation = transport(
        np.stack([emitted, weighted]),
        direct_delay,
        context.length,
        block_size=context.block_size,
    )
    shares = _blend_weights(
        _emission_positions(received, context.length), times * sr, responses
    )
    out = np.zeros((3, context.length))
    for response, share in zip(responses, shares):
        if not np.any(share):
            continue
        drive, drive_near = excitation * share[None, :]
        out[0] += fftconvolve(drive, response)[: context.length]
        kept = response if window is None else response[: int(round(window * sr))]
        if not kept.size:
            continue
        out[1] += fftconvolve(drive, kept)[: context.length]
        out[2] += fftconvolve(drive_near, kept)[: context.length]
    return out


def _late_poses(source):
    """Times and poses for late responses: every keyframe, and evenly spaced
    poses between two keyframes that move or turn far apart."""
    keys = source.keyframes
    times = []
    for a, b in zip(keys, keys[1:]):
        moved = np.linalg.norm(np.subtract(b.position_m, a.position_m))
        turned = max(
            abs((b.yaw_deg - a.yaw_deg + 180.0) % 360.0 - 180.0),
            abs((b.pitch_deg - a.pitch_deg + 180.0) % 360.0 - 180.0),
        )
        steps = max(1, int(np.ceil(moved / LATE_POSE_SPACING_M)), int(np.ceil(turned / LATE_POSE_TURN_DEG)))
        times.extend(np.linspace(a.time_s, b.time_s, steps + 1)[:-1])
    times = np.array([*times, keys[-1].time_s])
    return times, source.pose_at(times)


def _blend_weights(positions, knots, responses):
    """Per-response weights over emission ``positions``: linear between the two
    knots around each position, scaled so the energy of the weighted sum of
    the two responses is linear between their energies; held past the ends."""
    if len(responses) == 1:
        return [np.ones_like(positions)]
    energy = np.array([np.dot(r, r) for r in responses])
    cross = np.array([np.dot(a, b) for a, b in zip(responses, responses[1:])])
    index = np.clip(np.searchsorted(knots, positions, side="right") - 1, 0, len(knots) - 2)
    u = np.clip((positions - knots[index]) / (knots[index + 1] - knots[index]), 0.0, 1.0)
    e0, e1, c = energy[index], energy[index + 1], cross[index]
    plain = (1 - u) ** 2 * e0 + u**2 * e1 + 2 * u * (1 - u) * c
    floor = 0.25 * ((1 - u) ** 2 * e0 + u**2 * e1)
    scale = np.sqrt(((1 - u) * e0 + u * e1) / np.maximum(plain, np.maximum(floor, 1e-30)))
    shares = [np.zeros_like(positions) for _ in responses]
    for k in range(len(responses) - 1):
        here = index == k
        shares[k][here] += (1 - u[here]) * scale[here]
        shares[k + 1][here] += u[here] * scale[here]
    return shares


def _bridge_gaps(kernels, times, present):
    """Interpolate rows missing between present ones: a path drops out of one
    update frame only where its reflection point sits exactly on an edge."""
    rows = np.flatnonzero(present)
    gaps = np.flatnonzero(~present)
    gaps = gaps[(gaps > rows[0]) & (gaps < rows[-1])] if rows.size else gaps[:0]
    if not gaps.size:
        return
    upper = rows[np.searchsorted(rows, gaps)]
    lower = rows[np.searchsorted(rows, gaps) - 1]
    fraction = ((times[gaps] - times[lower]) / (times[upper] - times[lower]))[:, None]
    kernels[gaps] = kernels[lower] * (1 - fraction) + kernels[upper] * fraction


def _update_frames(times, source):
    """Frames on which path filters are evaluated: every ``PATH_FILTER_STEP_S``,
    the last frame, and the source's keyframe instants."""
    stride = max(1, int(round(PATH_FILTER_STEP_S / GEOMETRY_STEP_S)))
    keyframes = np.searchsorted(times, [k.time_s for k in source.keyframes])
    return np.unique(
        np.r_[np.arange(0, len(times), stride), len(times) - 1, keyframes]
    )


class _SegmentFilter:
    """Overlap-save filtering whose FIR changes linearly between knots.

    Segment ``j`` spans ``[knots[j], knots[j + 1])`` and cross-fades the
    outputs of kernels ``j`` and ``j + 1``; because filtering is linear in the
    taps this equals filtering with linearly interpolated taps.  The input
    segments are transformed once and shared by every path.
    """

    def __init__(self, signals, knots, taps):
        self.signals = signals
        self.length = signals.shape[1]
        starts, stops = knots[:-1], np.minimum(knots[1:], self.length)
        keep = stops > starts
        self.rows = np.flatnonzero(keep)
        starts, counts = starts[keep], (stops - starts)[keep]
        self.size = next_fast_len(int(counts.max()) + taps - 1)
        index = starts[:, None] - (taps - 1) + np.arange(self.size)[None, :]
        inside = (index >= 0) & (index < self.length)
        segments = np.where(
            inside[None], signals[:, np.clip(index, 0, self.length - 1)], 0.0
        )
        self.spectra = rfft(segments, axis=-1)
        within = np.arange(self.size)[None, :] - (taps - 1)
        self.mask = (within >= 0) & (within < counts[:, None])
        self.fade = np.where(self.mask, within / counts[:, None], 0.0)
        self.target = (starts[:, None] + within)[self.mask]

    def apply(self, signal_index, kernels):
        """Filter input ``signal_index`` with one kernel row per knot."""
        if np.all(kernels == kernels[0]):
            return fftconvolve(self.signals[signal_index], kernels[0])[: self.length]
        response = rfft(kernels, n=self.size, axis=-1)
        spectra = self.spectra[signal_index]
        first = irfft(spectra * response[self.rows], n=self.size, axis=-1)
        second = irfft(spectra * response[self.rows + 1], n=self.size, axis=-1)
        out = np.zeros(self.length)
        out[self.target] = (first + self.fade * (second - first))[self.mask]
        return out


def _emission_positions(received, length):
    """Emission instant of each output sample; constant delay off the ends."""
    output = np.arange(length, dtype=np.float64)
    emitted = np.arange(received.size, dtype=np.float64)
    positions = np.interp(output, received, emitted)
    before, after = output < received[0], output > received[-1]
    positions[before] = output[before] - received[0]
    positions[after] = emitted[-1] + output[after] - received[-1]
    return positions
