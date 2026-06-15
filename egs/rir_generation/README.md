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
