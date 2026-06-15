# rir_generation

Generate hybrid wave/geometric room impulse responses (RIRs) for near/far
speech augmentation, and convolve dry audio with them.

Each room item has **one microphone and five sources** (2 near, 3 far), written
as a 5-channel RIR WAV plus a JSON sidecar:

| WAV channel | Source label | Distance from mic |
|-------------|--------------|-------------------|
| 0 | `near_0` | `< 1 m` |
| 1 | `near_1` | `< 1 m` |
| 2 | `far_0`  | `> 2 m` |
| 3 | `far_1`  | `> 2 m` |
| 4 | `far_2`  | `> 2 m` |

The low band (`20 Hz`–`1000 Hz`) is a wave simulation (vendored `gpuard/pytARD`
modal model, solved with an exact batched modal recurrence); the high band
(`1000 Hz`–Nyquist) is a Pyroomacoustics geometric simulation; the two are glued
with a causal Linkwitz-Riley crossover. Furniture obstacles add height-aware
occlusion and scattering. See [`docs/audio/hybrid_rir.md`](../../docs/audio/hybrid_rir.md)
for the acoustic model and tuning details.

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_hybrid_rir.py` | Sample rooms and write 5-channel RIR WAV + metadata JSON. |
| `apply_rir_to_wav.py` | Convolve a dry WAV with a generated RIR WAV. |
| `rir_viz.py` | Visualize a RIR: room geometry, reflection paths, or a 2D wave-field animation. |

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
| `--crossover-hz` | Low/high crossover frequency. |
| `--pytard-low-sample-rate` | Internal wave simulation rate (resampled to `--sample-rate` afterwards). |
| `--pytard-spatial-samples-per-wavelength` | Grid resolution; `2` is practical, larger is slower. |
| `--seed` | RNG seed. Scene sampling is deterministic and independent of `--num-workers`. |

Run `python generate_hybrid_rir.py --help` for the full list.

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
