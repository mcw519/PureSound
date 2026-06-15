#!/usr/bin/env python
"""Visualize a generated hybrid RIR. One CLI, three subcommands:

  overview   room geometry (floor plan + 3D) + per-channel RIR waveforms + EDC
  paths      geometric sound paths to the mic (image-source reflections)
  field      illustrative 2D wave-field animation (reflection + diffraction)

Examples:
  python rir_viz.py overview --rir exp/hybrid_rir_16k/room_000000/room_000000_000000.wav
  python rir_viz.py paths    --rir <rir.wav> --channel 2 --order 2
  python rir_viz.py field    --rir <rir.wav> --channel 2 --nx 340 --t-ms 40 --gif

Notes:
  * Room acoustics has no *refraction* (that needs a medium gradient the
    simulator does not model). What you see is reflection (off walls) and, in
    `field`, diffraction/scattering around furniture.
  * `field` is a standalone 2D FDTD for visualization only — NOT the pipeline's
    low-band modal solver, which runs on an empty box without obstacles.
"""
import argparse
import json
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
    for obs in scene.get("obstacles", []):
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
    scene = load_scene(args)["scene"]
    sr, rir = load_rir(args.rir)
    fig = plt.figure(figsize=(15, 10))
    draw_floor_plan(fig.add_subplot(2, 2, 1), scene)
    draw_3d(fig.add_subplot(2, 2, 2, projection="3d"), scene)
    draw_waveforms(fig.add_subplot(2, 2, 3), sr, rir, scene)
    draw_edc(fig.add_subplot(2, 2, 4), sr, rir, scene)
    fig.suptitle(f"{args.rir.stem}   sr={sr} Hz   {rir.shape[0]} ch × {rir.shape[1]} samp",
                 fontsize=12)
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
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
