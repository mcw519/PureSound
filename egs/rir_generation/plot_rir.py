#!/usr/bin/env python
"""Visualize a generated hybrid RIR. One CLI, four subcommands:

  overview   room geometry (floor plan + 3D) + per-channel RIR waveforms + EDC
  paths      geometric sound paths to the mic (image-source reflections)
  field      illustrative 2D wave-field animation (reflection + diffraction)
  low-field  actual low-frequency modal pressure-field animation

Examples:
  python plot_rir.py overview --rir egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k/room_000000/room_000000_000000.wav
  python plot_rir.py paths    --rir <rir.wav> --channel 2 --order 2
  python plot_rir.py field    --rir <rir.wav> --channel 2 --nx 340 --t-ms 40 --gif
  python plot_rir.py low-field --rir <rir.wav> --channel 0 --t-ms 80 --gif

Notes:
  * Room acoustics has no *refraction* (that needs a medium gradient the
    simulator does not model). What you see is reflection (off walls) and, in
    `field`, diffraction/scattering around furniture.
  * `field` is a standalone 2D FDTD for visualization only — NOT the pipeline's
    low-band modal solver, which runs on an empty box without obstacles.
  * `low-field` re-runs the pipeline's modal recurrence and reconstructs one
    physical z-slice from the modal pressure state. It is intentionally an
    opt-in diagnostic because snapshots are not stored in normal RIR banks.
"""
import argparse
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath

# Stable per-channel colors (near = warm, far = cool).
CH_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
SPEED = 343.0


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_scene(args):
    json_path = args.json or args.rir.with_suffix(".json")
    return json.loads(json_path.read_text())


def _scene_and_config_from_metadata(metadata, duration_s):
    """Reconstruct the low-band scene/config needed for a field diagnostic."""
    from puresound.audio.rir.contracts import HybridRIRConfig
    from puresound.audio.rir.scene.sampling import HybridRIRScene, PolygonObstacle
    from puresound.audio.rir.scene.schema import RoomSceneV2

    scene_data = metadata["scene"]
    if scene_data.get("schema_version") == "rir_scene.v2":
        scene = RoomSceneV2.from_dict(scene_data)
    else:
        obstacles = [
            PolygonObstacle(
                footprint=item["footprint"],
                z_min=float(item.get("z_min", 0.0)),
                z_max=float(item.get("z_max", scene_data["room_dim"][2])),
                material=str(item.get("material", "unknown")),
                absorption=float(item.get("absorption", 0.0)),
                scattering=float(item.get("scattering", 0.0)),
            )
            for item in scene_data.get("obstacles", [])
        ]
        scene = HybridRIRScene(
            room_dim=list(scene_data["room_dim"]),
            rt60=float(scene_data["rt60"]),
            mic_pos=list(scene_data["mic_pos"]),
            source_pos=[list(item) for item in scene_data["source_pos"]],
            source_labels=list(scene_data.get("source_labels", [])),
            obstacles=obstacles,
        )

    allowed = {item.name for item in dataclass_fields(HybridRIRConfig)}
    config_values = {
        key: value for key, value in metadata["config"].items() if key in allowed
    }
    config_values["duration"] = float(duration_s)
    return scene, HybridRIRConfig(**config_values)


def load_rir(rir_path: Path):
    from scipy.io import wavfile

    sr, data = wavfile.read(rir_path)
    if data.dtype.kind in "iu":  # integer PCM -> float
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return sr, data.T  # [channels, samples]


def schroeder_db(h: np.ndarray) -> np.ndarray:
    """Backward-integrated energy decay curve in dB, normalized to 0 dB peak."""
    energy = h.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc = np.maximum(edc, edc.max() * 1e-12)
    return 10.0 * np.log10(edc / edc.max())


def default_out(args, suffix: str) -> Path:
    if args.output:
        return args.output
    return args.rir.with_name(args.rir.stem + suffix)


# --------------------------------------------------------------------------- #
# overview
# --------------------------------------------------------------------------- #
def draw_floor_plan(ax, scene):
    room = scene["room_dim"]
    ax.add_patch(plt.Rectangle((0, 0), room[0], room[1], fill=False, ec="black", lw=1.5))
    objects = list(scene.get("obstacles", []))
    if not objects:
        objects = [
            {
                "footprint": obj["footprint"],
                "material": obj.get("family", obj.get("material_id", "object")),
            }
            for obj in scene.get("objects", [])
        ]
    for obs in objects:
        foot = np.asarray(obs["footprint"])
        ax.add_patch(Polygon(foot, closed=True, facecolor="0.7",
                             edgecolor="0.4", alpha=0.6, lw=0.8))
        cx, cy = foot.mean(axis=0)
        ax.text(cx, cy, obs["material"], ha="center", va="center", fontsize=6)

    mic = scene["mic_pos"]
    ax.scatter([mic[0]], [mic[1]], marker="^", s=140, c="black", zorder=5, label="mic")
    for ch in scene["channel_map"]:
        x, y, _ = ch["source_pos"]
        c = CH_COLORS[ch["channel"] % len(CH_COLORS)]
        ax.scatter([x], [y], s=70, c=c, zorder=5)
        ax.annotate(f"{ch['label']}\n{ch['distance_m']:.2f} m", (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color=c)
        ax.plot([mic[0], x], [mic[1], y], c=c, lw=0.7, ls="--", alpha=0.6)

    ax.set_aspect("equal")
    ax.set_xlim(-0.3, room[0] + 0.3)
    ax.set_ylim(-0.3, room[1] + 0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Floor plan  ({room[0]:.1f} × {room[1]:.1f} × {room[2]:.1f} m, "
                 f"RT60={scene['rt60']:.2f}s)")
    ax.legend(loc="upper right", fontsize=7)


def draw_3d(ax, scene):
    x, y, z = scene["room_dim"]
    corners = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), c="0.6", lw=0.8)
    mic = scene["mic_pos"]
    ax.scatter([mic[0]], [mic[1]], [mic[2]], marker="^", s=120, c="black", label="mic")
    for ch in scene["channel_map"]:
        px, py, pz = ch["source_pos"]
        c = CH_COLORS[ch["channel"] % len(CH_COLORS)]
        ax.scatter([px], [py], [pz], s=60, c=c)
        ax.plot([mic[0], px], [mic[1], py], [mic[2], pz], c=c, lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D scene")


def draw_waveforms(ax, sr, rir, scene):
    t = np.arange(rir.shape[1]) / sr * 1000.0  # ms
    chmap = {c["channel"]: c for c in scene["channel_map"]}
    offset, step = 0.0, 2.2
    yticks, ylabels = [], []
    for ch in range(rir.shape[0]):
        peak = np.max(np.abs(rir[ch])) or 1.0
        c = CH_COLORS[ch % len(CH_COLORS)]
        ax.plot(t, rir[ch] / peak + offset, c=c, lw=0.6)
        label = chmap.get(ch, {}).get("label", f"ch{ch}")
        dist = chmap.get(ch, {}).get("distance_m")
        yticks.append(offset)
        ylabels.append(label + (f"\n{dist:.2f} m" if dist else ""))
        offset += step
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("time (ms)")
    ax.set_title("RIR per channel (peak-normalized)")
    ax.set_xlim(0, min(t[-1], 150))
    ax.grid(True, axis="x", alpha=0.3)


def draw_edc(ax, sr, rir, scene):
    t = np.arange(rir.shape[1]) / sr * 1000.0
    chmap = {c["channel"]: c for c in scene["channel_map"]}
    for ch in range(rir.shape[0]):
        c = CH_COLORS[ch % len(CH_COLORS)]
        ax.plot(t, schroeder_db(rir[ch]), c=c, lw=0.9,
                label=chmap.get(ch, {}).get("label", f"ch{ch}"))
    ax.axhline(-60, c="0.5", ls=":", lw=0.8)
    ax.text(t[-1], -60, " -60 dB", va="center", fontsize=7, color="0.4")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (dB)")
    ax.set_ylim(-80, 2)
    ax.set_title("Energy decay (Schroeder)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def cmd_overview(args):
    metadata = load_scene(args)
    scene = metadata["scene"]
    sr, rir = load_rir(args.rir)
    fig = plt.figure(figsize=(15, 10))
    draw_floor_plan(fig.add_subplot(2, 2, 1), scene)
    draw_3d(fig.add_subplot(2, 2, 2, projection="3d"), scene)
    draw_waveforms(fig.add_subplot(2, 2, 3), sr, rir, scene)
    draw_edc(fig.add_subplot(2, 2, 4), sr, rir, scene)
    high_backend = metadata.get("bands", {}).get("high", {}).get("backend", "unknown")
    low_backend = metadata.get("bands", {}).get("low", {}).get("backend", "unknown")
    fig.suptitle(
        f"{args.rir.stem}   sr={sr} Hz   {rir.shape[0]} ch × {rir.shape[1]} samp\n"
        f"low={low_backend}   high={high_backend}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = default_out(args, "_overview.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# paths (image-source reflections)
# --------------------------------------------------------------------------- #
def _reflect_point(src, axis, coord):
    img = src.copy()
    img[axis] = 2.0 * coord - src[axis]
    return img


def _wall_hit(img, mic, axis, coord, room):
    denom = mic[axis] - img[axis]
    if abs(denom) < 1e-9:
        return None
    t = (coord - img[axis]) / denom
    if not (0.0 <= t <= 1.0):
        return None
    pt = img + t * (mic - img)
    other = 1 - axis
    if not (0.0 <= pt[other] <= room[other]):
        return None
    return pt


def _path_length(points):
    pts = np.asarray(points)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def cmd_paths(args):
    scene = load_scene(args)["scene"]
    room = np.asarray(scene["room_dim"][:2])
    mic = np.asarray(scene["mic_pos"][:2])
    chan = scene["channel_map"][args.channel]
    src = np.asarray(chan["source_pos"][:2])
    color = CH_COLORS[args.channel % len(CH_COLORS)]
    walls = [(0, 0.0), (0, room[0]), (1, 0.0), (1, room[1])]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.add_patch(plt.Rectangle((0, 0), room[0], room[1], fill=False, ec="black", lw=1.8))
    for obs in scene.get("obstacles", []):
        ax.add_patch(Polygon(np.asarray(obs["footprint"]), closed=True,
                             facecolor="0.75", edgecolor="0.4", alpha=0.6, lw=0.8))

    ax.plot([src[0], mic[0]], [src[1], mic[1]], c=color, lw=2.2,
            label=f"direct  {_path_length([src, mic]):.2f} m")

    for axis, coord in walls:  # first order
        hit = _wall_hit(_reflect_point(src, axis, coord), mic, axis, coord, room)
        if hit is None:
            continue
        ax.plot([src[0], hit[0], mic[0]], [src[1], hit[1], mic[1]],
                c=color, lw=1.0, ls="--", alpha=0.75)
        ax.scatter([hit[0]], [hit[1]], c=color, s=18, zorder=4)

    if args.order >= 2:  # second order (distinct wall pairs)
        for a1, c1 in walls:
            img1 = _reflect_point(src, a1, c1)
            for a2, c2 in walls:
                if a1 == a2 and c1 == c2:
                    continue
                hit2 = _wall_hit(_reflect_point(img1, a2, c2), mic, a2, c2, room)
                if hit2 is None:
                    continue
                hit1 = _wall_hit(img1, hit2, a1, c1, room)
                if hit1 is None:
                    continue
                ax.plot([src[0], hit1[0], hit2[0], mic[0]],
                        [src[1], hit1[1], hit2[1], mic[1]],
                        c=color, lw=0.6, ls=":", alpha=0.45)

    ax.scatter([mic[0]], [mic[1]], marker="^", s=170, c="black", zorder=6, label="mic")
    ax.scatter([src[0]], [src[1]], s=110, c=color, zorder=6,
               label=f"{chan['label']}  (d={chan['distance_m']:.2f} m)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, room[0] + 0.4)
    ax.set_ylim(-0.4, room[1] + 0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{args.rir.stem}  —  {chan['label']} reflection paths "
                 f"(order≤{args.order})\nsolid=direct, dashed=1st, dotted=2nd")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = default_out(args, f"_paths_ch{args.channel}.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# field (2D FDTD wave propagation)
# --------------------------------------------------------------------------- #
def _rasterize_air_mask(room, obstacles, nx, ny):
    xs = (np.arange(nx) + 0.5) * room[0] / nx
    ys = (np.arange(ny) + 0.5) * room[1] / ny
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    air = np.ones((ny, nx), dtype=bool)
    for obs in obstacles:
        inside = MplPath(np.asarray(obs["footprint"])).contains_points(pts).reshape(ny, nx)
        air &= ~inside
    return air


def _laplacian_neumann(p, air):
    """5-point Laplacian with rigid (zero-gradient) boundaries at non-air cells."""
    lap = -4.0 * p
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        rolled = np.roll(p, shift, axis=axis)
        rolled_air = np.roll(air, shift, axis=axis)
        neighbor = np.where(rolled_air, rolled, p)
        if axis == 0:
            neighbor[0 if shift == 1 else -1, :] = p[0 if shift == 1 else -1, :]
        else:
            neighbor[:, 0 if shift == 1 else -1] = p[:, 0 if shift == 1 else -1]
        lap += neighbor
    return lap


def _style_field_ax(ax, room, obstacles, src, mic):
    # RdBu_r is white at zero, so white markers vanish — use lime + black outlines.
    for obs in obstacles:
        ax.add_patch(Polygon(np.asarray(obs["footprint"]), closed=True,
                             facecolor="0.35", edgecolor="black", lw=1.4,
                             alpha=0.85, zorder=4))
    ax.scatter([src[0]], [src[1]], s=90, edgecolor="black", linewidth=1.2,
               facecolor="#00ff88", zorder=5)
    ax.scatter([mic[0]], [mic[1]], marker="^", s=120, edgecolor="black",
               linewidth=1.2, facecolor="#00ff88", zorder=5)
    ax.set_xlim(0, room[0])
    ax.set_ylim(0, room[1])
    ax.set_aspect("equal")


def cmd_field(args):
    scene = load_scene(args)["scene"]
    room = np.asarray(scene["room_dim"][:2])
    mic = np.asarray(scene["mic_pos"][:2])
    chan = scene["channel_map"][args.channel]
    src = np.asarray(chan["source_pos"][:2])
    obstacles = scene.get("obstacles", [])

    nx = args.nx
    dx = room[0] / nx
    ny = int(round(room[1] / dx))
    air = _rasterize_air_mask(room, obstacles, nx, ny)

    courant = 0.5  # < 1/sqrt(2): CFL-stable for 2D explicit leapfrog
    dt = courant * dx / SPEED
    n_steps = int(args.t_ms * 1e-3 / dt)
    c2 = (SPEED * dt / dx) ** 2

    def cell(pos):
        return (min(int(pos[1] / room[1] * ny), ny - 1),
                min(int(pos[0] / room[0] * nx), nx - 1))

    si, sj = cell(src)

    # Ricker wavelet (bipolar) source: shows compressions (red) and rarefactions
    # (blue), low enough in frequency for the coarse grid to stay clean.
    t0, width = 16, 5.0
    tt = np.arange(n_steps)
    a = ((tt - t0) / width) ** 2
    pulse = (1.0 - a) * np.exp(-a / 2.0)

    p_prev = np.zeros((ny, nx))
    p_cur = np.zeros((ny, nx))
    frames, frame_times = [], []
    for n in range(n_steps):
        p_next = (2.0 * p_cur - p_prev + c2 * _laplacian_neumann(p_cur, air)) * air
        p_next[si, sj] += pulse[n]
        p_prev, p_cur = p_cur, p_next
        if n % args.frame_stride == 0:
            frames.append(p_cur.copy())
            frame_times.append(n * dt * 1e3)

    # Per-frame display scale: the pulse is huge at the source but spreads thin,
    # so a single global vmax would wash out everything after the first frames.
    floor = max(np.abs(f).max() for f in frames) * 1e-3 or 1.0
    vmaxes = [max(np.percentile(np.abs(f), 99.8), floor) for f in frames]
    extent = [0, room[0], 0, room[1]]

    # --- MP4 / GIF ---
    out = default_out(args, f"_field_ch{args.channel}.mp4")
    fig, ax = plt.subplots(figsize=(8, 8 * room[1] / room[0] + 0.6))
    im = ax.imshow(frames[0], origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-vmaxes[0], vmax=vmaxes[0], interpolation="bilinear")
    _style_field_ax(ax, room, obstacles, src, mic)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = ax.set_title("")

    def update(k):
        im.set_data(frames[k])
        im.set_clim(-vmaxes[k], vmaxes[k])
        title.set_text(f"{chan['label']} pulse — t = {frame_times[k]:5.1f} ms")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / args.fps, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
    print(f"wrote {out}  ({len(frames)} frames, grid {ny}x{nx}, dt={dt*1e3:.3f} ms)")
    if args.gif:
        gif = out.with_suffix(".gif")
        anim.save(gif, writer=animation.PillowWriter(fps=args.fps))
        print(f"wrote {gif}")
    plt.close(fig)

    # --- contact sheet (6 frames, for quick inline viewing) ---
    sheet = out.with_name(out.stem + "_sheet.png")
    idxs = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig2, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, k in zip(axes.ravel(), idxs):
        ax.imshow(frames[k], origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmaxes[k], vmax=vmaxes[k], interpolation="bilinear")
        _style_field_ax(ax, room, obstacles, src, mic)
        ax.set_title(f"t = {frame_times[k]:.1f} ms", fontsize=10)
    fig2.suptitle(f"{args.rir.stem} — {chan['label']} 2D wave field "
                  f"(reflection + diffraction around furniture)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    fig2.savefig(sheet, dpi=120)
    print(f"wrote {sheet}")


# --------------------------------------------------------------------------- #
# low-field (actual modal pressure slice)
# --------------------------------------------------------------------------- #
def _make_low_field_backend(metadata, args):
    from puresound.audio.rir.render.low_frequency import (
        GpuARDPytARDBackend,
        GpuARDPytARDCuPyBackend,
    )

    recorded = metadata.get("bands", {}).get("low", {}).get("backend", "")
    name = recorded if args.backend == "auto" else args.backend
    material = bool(
        metadata.get("bands", {}).get("low", {}).get("modal_damping")
    ) or name.endswith("-material")
    if name in {"GpuARDPytARDCuPyBackend", "pytard-cupy", "pytard-cupy-material"}:
        return GpuARDPytARDCuPyBackend(
            low_sample_rate=int(args.low_sample_rate),
            spatial_samples_per_wave_length=int(args.spatial_samples_per_wavelength),
            material_modal_damping=material,
            calibrate_output=False,
            apply_rt60_decay=False,
        ), "pytard-cupy"
    if name in {"GpuARDPytARDBackend", "pytard", "pytard-material"}:
        return GpuARDPytARDBackend(
            low_sample_rate=int(args.low_sample_rate),
            spatial_samples_per_wave_length=int(args.spatial_samples_per_wavelength),
            material_modal_damping=material,
            calibrate_output=False,
            apply_rt60_decay=False,
        ), "pytard"
    raise ValueError(
        "low-field currently supports the pytARD modal backends only; "
        f"metadata/backend={recorded!r}, requested={name!r}"
    )


def _low_field_output_paths(args):
    output = args.output or args.rir.with_name(
        f"{args.rir.stem}_low_field_ch{args.channel}.mp4"
    )
    diagnostic = args.diagnostic or output.with_suffix(".npz")
    return output, diagnostic


def _draw_low_field_overlay(ax, room, source, mic):
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            room[0],
            room[1],
            fill=False,
            edgecolor="black",
            linewidth=1.6,
            zorder=4,
        )
    )
    ax.scatter(
        [source[0]],
        [source[1]],
        s=105,
        facecolor="#00ff88",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label="source",
    )
    ax.scatter(
        [mic[0]],
        [mic[1]],
        marker="^",
        s=135,
        facecolor="#00ff88",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label="receiver",
    )
    ax.set_xlim(0.0, room[0])
    ax.set_ylim(0.0, room[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def cmd_low_field(args):
    metadata = load_scene(args)
    scene, config = _scene_and_config_from_metadata(
        metadata,
        duration_s=float(args.t_ms) * 1e-3,
    )
    backend, backend_label = _make_low_field_backend(metadata, args)
    room = np.asarray(scene.room_dim, dtype=np.float64)
    source = np.asarray(scene.source_pos[args.channel], dtype=np.float64)
    mic = np.asarray(scene.mic_pos, dtype=np.float64)
    slice_z = float(args.slice_z) if args.slice_z is not None else float(mic[2])
    capture = {
        "source_index": int(args.channel),
        "z_m": slice_z,
        "stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
    }
    _low_rir, capture = backend.simulate_with_pressure_field(
        scene,
        config,
        capture,
    )
    frames = np.asarray(capture["frames"], dtype=np.float32)
    times_s = np.asarray(capture["times_s"], dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise RuntimeError("low-field solver returned no pressure-field frames")

    output, diagnostic = _low_field_output_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostic,
        frames=frames,
        times_s=times_s,
        room_dim=room,
        source=source,
        mic=mic,
        slice_z=np.asarray(capture["z_m"]),
        grid_shape_zyx=np.asarray(capture["grid_shape_zyx"], dtype=np.int64),
        dt_s=np.asarray(capture["dt_s"]),
        source_index=np.asarray(args.channel, dtype=np.int64),
        backend=np.asarray(backend_label),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "rir": str(args.rir),
                    "backend": backend_label,
                    "recorded_backend": metadata.get("bands", {})
                    .get("low", {})
                    .get("backend"),
                    "note": "actual modal pressure slice; pre-output calibration",
                },
                ensure_ascii=False,
            )
        ),
    )
    print(f"wrote {diagnostic}")

    vmax = float(np.percentile(np.abs(frames), 99.7))
    vmax = max(vmax, float(np.max(np.abs(frames))) * 1e-3, 1e-12)
    extent = [0.0, room[0], 0.0, room[1]]
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    im = ax.imshow(
        frames[0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )
    _draw_low_field_overlay(ax, room, source, mic)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("modal pressure (relative solver units)")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)

    def update(index):
        im.set_data(frames[index])
        title.set_text(
            f"{args.rir.stem} — low-frequency modal pressure field\n"
            f"{backend_label}, channel={args.channel}, z={float(capture['z_m']):.2f} m, "
            f"t={times_s[index] * 1e3:6.2f} ms"
        )
        return im, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / float(args.fps),
        blit=False,
    )
    try:
        anim.save(
            output,
            writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for MP4 output; install it or use --gif"
        ) from exc
    print(
        f"wrote {output} ({len(frames)} frames, "
        f"grid {frames.shape[1]}x{frames.shape[2]}, "
        f"solver dt={float(capture['dt_s']) * 1e3:.4f} ms)"
    )
    if args.gif:
        gif = output.with_suffix(".gif")
        anim.save(gif, writer=animation.PillowWriter(fps=args.fps))
        print(f"wrote {gif}")
    plt.close(fig)

    sheet = output.with_name(output.stem + "_sheet.png")
    indices = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig2, axes = plt.subplots(2, 3, figsize=(14, 9))
    for axis, index in zip(axes.ravel(), indices):
        axis.imshow(
            frames[index],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        _draw_low_field_overlay(axis, room, source, mic)
        axis.set_title(f"t = {times_s[index] * 1e3:.2f} ms", fontsize=10)
    fig2.suptitle(
        f"{args.rir.stem} — actual low-frequency modal pressure field "
        f"(channel {args.channel}, z={float(capture['z_m']):.2f} m)",
        fontsize=12,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    fig2.savefig(sheet, dpi=120)
    print(f"wrote {sheet}")
    plt.close(fig2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--rir", required=True, type=Path, help="RIR WAV")
        sp.add_argument("--json", type=Path, help="metadata JSON (default: RIR with .json)")
        sp.add_argument("--output", type=Path, help="output path (default: next to RIR)")
        sp.add_argument("--dpi", type=int, default=130)

    sp = sub.add_parser("overview", help="geometry + waveforms + energy decay")
    add_common(sp)
    sp.set_defaults(func=cmd_overview)

    sp = sub.add_parser("paths", help="image-source reflection paths")
    add_common(sp)
    sp.add_argument("--channel", type=int, default=2, help="source channel 0-4")
    sp.add_argument("--order", type=int, default=2, choices=[1, 2],
                    help="max reflection order to draw")
    sp.set_defaults(func=cmd_paths)

    sp = sub.add_parser("field", help="2D FDTD wave-field animation")
    add_common(sp)
    sp.add_argument("--channel", type=int, default=2, help="source channel 0-4")
    sp.add_argument("--nx", type=int, default=340, help="grid cells along x")
    sp.add_argument("--t-ms", type=float, default=40.0, help="sim duration (ms)")
    sp.add_argument("--fps", type=int, default=25)
    sp.add_argument("--frame-stride", type=int, default=6, help="capture every Nth step")
    sp.add_argument("--gif", action="store_true", help="also write a .gif")
    sp.set_defaults(func=cmd_field)

    sp = sub.add_parser(
        "low-field",
        help="actual low-frequency modal pressure-field animation",
    )
    add_common(sp)
    sp.add_argument("--channel", type=int, default=0, help="source channel 0-4")
    sp.add_argument(
        "--backend",
        choices=["auto", "pytard", "pytard-cupy"],
        default="auto",
        help="modal backend; auto uses the backend recorded in metadata",
    )
    sp.add_argument(
        "--low-sample-rate",
        type=int,
        default=16000,
        help="low solver sample rate used during the diagnostic re-run",
    )
    sp.add_argument(
        "--spatial-samples-per-wavelength",
        type=int,
        default=2,
        help="modal grid density, matching M6 pytARD generation",
    )
    sp.add_argument(
        "--slice-z",
        type=float,
        default=None,
        help="z height of the 2-D pressure slice in metres (default: mic height)",
    )
    sp.add_argument(
        "--t-ms",
        type=float,
        default=80.0,
        help="duration of the diagnostic animation in milliseconds",
    )
    sp.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="capture every Nth solver step before max-frame limiting",
    )
    sp.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="maximum number of pressure frames held in memory",
    )
    sp.add_argument("--fps", type=int, default=25)
    sp.add_argument("--gif", action="store_true", help="also write a .gif")
    sp.add_argument(
        "--diagnostic",
        type=Path,
        help="optional .npz path for reusable pressure snapshots",
    )
    sp.set_defaults(func=cmd_low_field)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
