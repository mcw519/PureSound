# rir_generation

Tooling for the room-impulse-response (RIR) side of near/far speech
augmentation. It covers the whole pipeline: **generate** synthetic hybrid
wave/geometric RIRs, **inspect / apply** them, and **package** them — whether
synthesised here or measured in a real room — into training banks that
[`PreGeneratedRoomBank`](../../puresound/audio/rir_bank.py) consumes.

Each room item has **one microphone and five sources** (2 near, 3 far), written
as a 5-channel RIR WAV plus a JSON sidecar:

| WAV channel | Source label | Horizontal distance from mic |
|-------------|--------------|-------------------|
| 0 | `near_0` | `< 1 m` |
| 1 | `near_1` | `< 1 m` |
| 2 | `far_0`  | `> 2 m` |
| 3 | `far_1`  | `> 2 m` |
| 4 | `far_2`  | `> 2 m` |

The low band (`20 Hz`–`1000 Hz`) is a wave simulation (vendored `gpuard/pytARD`
modal model, solved with an exact batched modal recurrence); the high band
(`1000 Hz`–Nyquist) is a Pyroomacoustics geometric simulation; the two are glued
with a causal Linkwitz-Riley crossover. Furniture obstacles are sampled with
area-aware counts and material-specific footprint/height profiles, then applied
as height-aware high-frequency occlusion and scattering. See
[`docs/audio/hybrid_rir.md`](../../docs/audio/hybrid_rir.md) for the acoustic
model and tuning details.

## Scripts

The bold tag marks each script's stage in the pipeline: **Generate** synthetic
RIRs → **Apply / Inspect** them → **Bank** them for training.

| Script | Purpose |
|--------|---------|
| `generate_hybrid_rir.py` | **Generate** — sample rooms and write 5-channel RIR WAV + metadata JSON. |
| `apply_rir_to_wav.py` | **Apply** — convolve a dry WAV with a generated RIR WAV. |
| `rir_viz.py` | **Inspect** — visualize one RIR: room geometry, reflection paths, or a 2D wave-field animation. |
| `rir_stats.py` | **Inspect** — print distribution stats for a folder of RIRs (room dims, RT60, distances, obstacles) to the terminal. |
| `filter_rir_levels.py` | **Bank** — slice a generated bank into cumulative curriculum levels (core/expand/stress) by RT60 + DRR; symlink-only views. |
| `real_rir_to_bank.py` | **Bank** — convert a real measured RIR dataset (e.g. BUT ReverbDB) into a `PreGeneratedRoomBank` folder. |

## Install

```bash
# CPU (pytARD is already vendored in this repo)
pip install pyroomacoustics

# Optional GPU low band (CUDA 12)
uv pip install cupy-cuda12x
```

## Generate

Smoke test without running pytARD (still requires Pyroomacoustics for the high
band):

```bash
python generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir_smoke \
  --n-rooms 2 --rir-per-room 1 \
  --low-backend analytic
```

Dataset on CPU (recommended for a single machine — set `--num-workers` near the
core count):

```bash
python generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir \
  --n-rooms 1000 --rir-per-room 4 \
  --sample-rate 16000 --duration 1.6 \
  --low-backend pytard \
  --num-workers 22 \
  --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2
```

GPU low band (one worker per device):

```bash
python generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir_gpu \
  --n-rooms 1000 --rir-per-room 4 \
  --sample-rate 16000 --duration 1.6 \
  --low-backend pytard-cupy \
  --gpu-devices 0,1 --num-workers 2 \
  --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2
```

Only `pytard-cupy` uses the GPU; the `pytard`/`analytic` low backends and the
Pyroomacoustics high band are always CPU. Because the modal solve made the GPU
low band only a few seconds per RIR, a high core-count machine running CPU
multi-worker can out-throughput a small number of GPUs — benchmark both.

### Key options

| Option | Meaning |
|--------|---------|
| `--n-rooms`, `--rir-per-room` | Number of rooms, and RIRs per room (same geometry, resampled mic/source positions). |
| `--num-workers` | Parallel generation workers. Defaults to one per `--gpu-devices` entry, else 1. |
| `--gpu-devices` | Comma-separated CUDA devices for `pytard-cupy` workers, e.g. `0,1`. |
| `--low-backend` | `pytard` (CPU), `pytard-cupy` (GPU), or `analytic` (fast smoke fallback). |
| `--sample-rate`, `--duration` | Output sample rate (Hz) and RIR length (s). |
| `--rt60 MIN MAX` | RT60 range; one value is sampled per room and drives both bands. |
| `--obstacles MIN MAX` | Hard cap for sampled furniture count; actual count also tracks room floor area and coverage limits. |
| `--crossover-hz` | Low/high crossover frequency. |
| `--pytard-low-sample-rate` | Internal wave simulation rate (resampled to `--sample-rate` afterwards). |
| `--pytard-spatial-samples-per-wavelength` | Grid resolution; `2` is practical, larger is slower. |
| `--seed` | RNG seed. Scene sampling is deterministic and independent of `--num-workers`. |
| `--resume` | Skip RIRs already written to `--output-dir` and generate only the rest. |

Run `python generate_hybrid_rir.py --help` for the full list.

### Resume an interrupted run

If a run is killed partway through, rerun the **same command** with `--resume`
added. It rescans `--output-dir`, skips every room/RIR that already has both a
`.wav` and a parseable `.json` (a half-written item is regenerated), and
continues with the remainder; the progress bar starts at the already-finished
count:

```bash
python generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir \
  --n-rooms 1000 --rir-per-room 4 \
  --sample-rate 16000 --duration 1.6 \
  --low-backend pytard --num-workers 22 \
  --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2 \
  --resume
```

Keep `--seed`, `--n-rooms`, `--rir-per-room`, and the room/RT60/obstacle options
identical to the original run. Scene sampling is deterministic, so the resumed
items reuse the exact room/mic/source geometry the uninterrupted run would have
produced (the high-band acoustic realization is randomized per RIR regardless of
resume).

### Output layout

Items are grouped by `room_id`, each RIR/metadata pair named `{room_id}_{index}`:

```text
exp/hybrid_rir/
  room_000000/
    room_000000_000000.wav   # [5, samples] RIR, PCM_F
    room_000000_000000.json  # config, scene geometry, per-channel distances
    room_000000_000001.wav
    room_000000_000001.json
```

## Apply a RIR to dry audio

Convolve a dry mono/multi-channel WAV with a generated RIR. Multi-channel dry
input is mixed down to mono by default.

```bash
# One wet channel per RIR channel
python apply_rir_to_wav.py \
  --wav path/to/dry.wav \
  --rir exp/hybrid_rir/room_000000/room_000000_000000.wav \
  --output exp/hybrid_rir/room_000000/wet_5ch.wav \
  --rir-mode full --length-mode same \
  --output-layout rir-channels --peak-normalize

# Single mono mixture (sum all RIR channels)
python apply_rir_to_wav.py \
  --wav path/to/dry.wav \
  --rir exp/hybrid_rir/room_000000/room_000000_000000.wav \
  --output exp/hybrid_rir/room_000000/wet_mono.wav \
  --output-layout mono-sum --peak-normalize
```

Useful flags: `--rir-mode {full,early,direct}` (trim to direct path or early
reflections), `--dry-wet` (blend dry/wet), and `--metadata PATH` (write a
convolution metadata JSON).

## Build a training RIR bank

Both bank scripts emit folders that
[`PreGeneratedRoomBank`](../../puresound/audio/rir_bank.py) indexes directly:
same-stem `.wav`/`.json` pairs whose JSON carries a `scene.channel_map` giving a
near/far label and a distance per channel. Training recipes point at one of
these folders. `filter_rir_levels.py` reshapes a bank you generated above;
`real_rir_to_bank.py` builds one from RIRs measured in real rooms.

### Curriculum levels from a generated bank — `filter_rir_levels.py`

Slices an existing hybrid RIR bank into three **cumulative** difficulty levels
for curriculum training, judged by RT60 and the measured near/far DRR
(direct-to-reverberant ratio) separation. The conservative gap per item is
`min(DRR over near channels) − max(DRR over far channels)`, so every near/far
pair in a level clears the printed bound.

| Level | RT60 | Worst-case near/far DRR gap |
|-------|------|------------------------------|
| `core`   | 0.20–0.45 s | ≥ 6 dB |
| `expand` | 0.20–0.65 s | ≥ 3 dB |
| `stress` | all remaining valid items | — |

Output folders hold **relative symlinks only** (no RIR audio is copied), each
under an `items/` child, plus a per-level `manifest.json` (RT60, near/far DRR,
gap per item) and a top-level `summary.json` carrying the thresholds and
RT60/gap percentiles.

```bash
# Measure and report only; nothing is written to disk
python filter_rir_levels.py \
  exp/hybrid_rir_16k exp/hybrid_rir_16k_levels --dry-run

# Build the core/expand/stress views
python filter_rir_levels.py \
  exp/hybrid_rir_16k exp/hybrid_rir_16k_levels \
  --drr-window-ms 2.5 --workers 16
```

Set `--drr-window-ms` to match the direct-path window your recipe config uses.
The output path must not already exist — there is no overwrite option by design.

### Real measured RIRs → a bank — `real_rir_to_bank.py`

Converts a real measured RIR dataset into a bank to narrow the synthetic→real
domain gap. It runs in two stages decoupled by a JSON-Lines manifest, so the
exact bank emission is independent of any one dataset's on-disk layout:

* **Stage A — scan** a specific dataset into `manifest.jsonl`, one JSON object
  per RIR: `room_id`, `rir_path`, `channel`, optional `rt60`, and a distance
  given either as `distance_m` or as `src_xyz`/`mic_xyz` (Euclidean). This
  scanner is dataset-specific and best-effort — verify it against your download.
* **Stage B — assemble** the manifest into a bank. Within each `room_id`, RIRs
  split near/far at `--d0` metres and group into multi-channel items (≤2 near,
  ≤3 far). This stage is the definitive, self-tested one.

By acoustic reciprocity a measured RIR is identical read source→mic or
mic→source, so a dataset with one loudspeaker and many microphones maps cleanly
onto the bank's "one receiver, many sources at various distances" item — the
distance is just `‖loudspeaker − microphone‖`.

```bash
# Stage B only — you supply the manifest
python real_rir_to_bank.py from-manifest \
  --manifest manifest.jsonl --output exp/real_rir_bank \
  --d0 1.0 --target-sr 16000

# BUT ReverbDB end-to-end (scan → manifest → bank)
python real_rir_to_bank.py but \
  --input /path/to/BUT_ReverbDB --output exp/real_rir_bank --d0 1.0

# Self-test Stage B against PreGeneratedRoomBank (no dataset needed)
python real_rir_to_bank.py self-test
```

Only the BUT ReverbDB scanner ships today. To add another dataset, write a
`scan_*` that emits the manifest schema above, then reuse Stage B unchanged.

## Visualize a RIR

`rir_viz.py` is one CLI with three subcommands. Each reads a RIR WAV and its
`.json` sidecar, and writes its output next to the RIR by default (override with
`--output`). Common flags: `--rir` (required), `--json`, `--output`, `--dpi`.

| Subcommand | Output | Shows |
|------------|--------|-------|
| `overview` | `*_overview.png` | Floor plan + 3D scene + per-channel RIR waveforms + Schroeder energy decay. |
| `paths` | `*_paths_chN.png` | Geometric sound paths to the mic (image-source: direct + 1st/2nd-order wall reflections). |
| `field` | `*_field_chN.mp4` (+ `.gif`, `_sheet.png`) | Illustrative 2D wave-field animation: reflection off walls and diffraction/scattering around furniture. |

```bash
RIR=exp/hybrid_rir/room_000000/room_000000_000000.wav

# Geometry + waveforms + energy decay
python rir_viz.py overview --rir "$RIR"

# Reflection paths for source channel 2 (far_0), up to 2nd order
python rir_viz.py paths --rir "$RIR" --channel 2 --order 2

# 2D wave-field animation for channel 2 (MP4 + GIF + contact sheet)
python rir_viz.py field --rir "$RIR" --channel 2 --nx 340 --t-ms 40 --gif
```

`paths`/`field` take `--channel` (0–4, see the channel table above); `paths`
takes `--order {1,2}`; `field` takes `--nx` (grid resolution), `--t-ms`
(duration), `--fps`, `--frame-stride`, and `--gif`.

### Example outputs

All three are rendered from the same item (`room_000000_000000`, a
`7.5 × 4.1 × 3.5 m`, `RT60 = 0.81 s` room), with `paths`/`field` on channel 2
(`far_0`, `2.53 m` from the mic).

**`overview`** — floor plan + 3D scene + per-channel RIR waveforms + Schroeder
energy decay:

![rir_viz overview output](assets/overview.png)

**`paths`** — image-source reflection paths to the mic (solid = direct,
dashed = 1st-order, dotted = 2nd-order wall reflections):

![rir_viz paths output for far_0](assets/paths_ch2.png)

**`field`** — 2D FDTD wave-field animation (red = compression, blue =
rarefaction); note the wave reflecting off the walls and bending around the
furniture footprints:

![rir_viz field animation for far_0](assets/field_ch2.gif)

> The GIF here is downsized (`--nx 190`, sparse frames) to keep the repo light;
> the default `--nx 340` render is sharper. `field` also writes an MP4 and a
> 6-frame `_sheet.png` contact sheet, not shown here.

A caveats worth stating plainly:

* **`field` is a standalone 2D FDTD for visualization only** — not the
  pipeline's low-band modal solver, which runs on an empty box without
  obstacles. It illustrates wave behavior; it is not the exact RIR computation.

### How the wave-field (`field`) is computed

The animation is a small 2D **FDTD** (finite-difference time-domain) solver on a
horizontal slice at mic height. Five steps:

**1. Physics — the 2D wave equation.** The acoustic pressure `p(x, y, t)` obeys

```
∂²p/∂t² = c²·∇²p ,   c = 343 m/s ,   ∇²p = ∂²p/∂x² + ∂²p/∂y²
```

i.e. a point's pressure *accelerates* in proportion to how it differs from its
surroundings (the Laplacian), which is what makes energy spread outward as waves.

**2. Discretization — a grid you can step cell by cell.** The room slice becomes
an `ny × nx` grid (`--nx`), time becomes steps `dt`. Central differences turn the
derivatives into a leapfrog update of the whole field per step:

```
p^{n+1} = 2·p^n − p^{n-1} + C²·∇²p^n ,   C = c·dt/dx
```

with the Laplacian as the 5-point stencil (each cell vs. its 4 neighbors). This
is the one line in `cmd_field`:

```python
p_next = (2.0 * p_cur - p_prev + c2 * _laplacian_neumann(p_cur, air)) * air
```

**Stability (CFL).** An explicit scheme only stays bounded if a wave travels less
than one cell per step; in 2D that means `C ≤ 1/√2 ≈ 0.707`. The code fixes
`courant = 0.5` and derives `dt = 0.5·dx/c`.

**3. Boundaries — rigid walls and furniture.** A rigid surface means zero normal
pressure gradient (`∂p/∂n = 0`, Neumann). `_laplacian_neumann` enforces it with
one trick: when a neighbor is wall/furniture, substitute the center value
(`np.where(rolled_air, rolled, p)`), so that direction's pressure difference is
zero and the wave reflects. Furniture footprints from the JSON are rasterized
into the boolean `air` mask (`_rasterize_air_mask`); waves reflect off them and
diffract around their edges.

**4. Source — a Ricker wavelet.** A bipolar pulse (2nd derivative of a Gaussian)
is injected at the source cell each step. Bipolar so the animation shows
compressions (red) and rarefactions (blue); band-limited and low enough in
frequency that its wavelength spans many cells, which avoids grid dispersion
ripples.

**5. Rendering.** Every `--frame-stride` steps a snapshot is saved. Frames use the
`RdBu_r` colormap (red = positive pressure, blue = negative, white = zero). Each
frame is normalized to its own 99.8th-percentile amplitude, because the pulse is
huge at the source but spreads thin — a single fixed color scale would wash out
later frames. `FuncAnimation` + `FFMpegWriter` write the MP4 (plus an optional
GIF and a 6-frame contact sheet).

## Third-party libraries and projects

These scripts stand on the following external code. Generic numerics and audio
I/O (`numpy`, `scipy`, `torch`/`torchaudio`) are repo-wide dependencies declared
in [`pyproject.toml`](../../pyproject.toml); the acoustics- and rendering-specific
ones are called out here with their role and which script pulls them in.

| Library | Role here | Used by |
|---------|-----------|---------|
| [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) | High-band **geometric (image-source)** room simulation | `generate_hybrid_rir.py` (via `puresound.audio.hybrid_rir`) |
| [CuPy](https://cupy.dev/) (`cupy-cuda12x`) | Optional **GPU** acceleration of the low-band modal solve (`--low-backend pytard-cupy`) | `generate_hybrid_rir.py` |
| [PyTorch / torchaudio](https://pytorch.org/audio/) | Tensors, WAV I/O, resampling, FFT convolution | `generate_hybrid_rir.py`, `apply_rir_to_wav.py`, `real_rir_to_bank.py` |
| [SciPy](https://scipy.org/) | Modal DCT/IDCT transforms; WAV-read fallback | low-band solver, `rir_viz.py` |

`pyroomacoustics` and `cupy-cuda12x` are **optional extras**, not installed by
default — see [Install](#install) (the `hybrid-rir` / `hybrid-rir-gpu` extras in
`pyproject.toml`). `ffmpeg` is a system binary Matplotlib shells out to for MP4;
without it, `field` can still write the GIF and contact sheet.

### Vendored: gpuard/pytARD — low-band wave solver

The low band (`20 Hz`–crossover) is solved with **Adaptive Rectangular
Decomposition (ARD)** from [`gpuard/pytARD`](https://github.com/gpuard/pytARD),
vendored under
[`puresound/third_party/pytARD/`](../../puresound/third_party/pytARD). The
adapter in [`puresound/audio/hybrid_rir.py`](../../puresound/audio/hybrid_rir.py)
wraps pytARD's 3D partition modules and replaces its per-step FFT loop with an
exact batched modal recurrence (optionally CuPy-accelerated) **without modifying
the vendored source**. pytARD is licensed **AGPL-3.0** (see its bundled
`LICENSE`); that license governs the vendored subtree.

### External datasets (downloaded separately, not bundled)

`real_rir_to_bank.py` converts *real measured* RIR corpora into banks. It ships a
scanner for the [**BUT Speech@FIT Reverb Database**](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database)
(Brno University of Technology; CC-BY 4.0) — specifically the `rel_19_06`
*RIR-Only* release. The dataset is downloaded separately and is not part of this
repo; only the scanner and the manifest→bank converter live here. The
docstring's "BUT, UPV, ..." notes other one-loudspeaker/many-microphone corpora
that the same Stage-A→Stage-B path can target once a `scan_*` is added.
