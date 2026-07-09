# puresound.audio.hybrid_rir

Hybrid RIR generation for near/far speech augmentation.

The generator creates one room item with one microphone and five sources:

| WAV channel | Source label | Horizontal distance from mic |
|-------------|--------------|-------------------|
| 0 | `near_0` | `< 1m` |
| 1 | `near_1` | `< 1m` |
| 2 | `far_0` | `> 2m` |
| 3 | `far_1` | `> 2m` |
| 4 | `far_2` | `> 2m` |

## Acoustic Model

- Low band: vendored `gpuard/pytARD` wave model, `20 Hz` to `1000 Hz`. The
  ARD room is lossless (rigid walls), so the scene RT60 is imposed as a decay
  envelope keyed to the physical direct-path arrival (`distance / c`). The
  impulse excitation is widened so the band carries energy up to the crossover
  (pytARD's `Unit` halves the requested cutoff internally).
- The low band reuses pytARD's voxelization, modal frequencies, and `Unit`
  impulse, but the per-time-step solve is an **exact modal recurrence**
  (`_solve_modal_ard`) instead of pytARD's per-step 3D DCT/IDCT loop. Because
  the excitation is a single voxel (the forward DCT collapses to a precomputed
  basis) and only one microphone voxel is read back (the inverse DCT at that
  point is a dot product), both transforms drop out of the hot loop. The result
  is numerically identical to pytARD's loop (relative error `~1e-15`) but ~30x
  faster on CPU. All five sources are integrated as one batched update.
- Optional GPU low band: `GpuARDPytARDCuPyBackend` runs that same modal
  recurrence on CuPy arrays. With the DCTs removed, the loop is elementwise and
  GPU-resident (no per-step host transfer), so it is throughput-bound rather
  than latency-bound.
- High band: Pyroomacoustics geometric backend, `1000 Hz` to Nyquist. Wall
  absorption and ISM order are derived from the scene RT60 via
  `pra.inverse_sabine`, so the high-band reverberation time tracks the requested
  RT60 (ray tracing fills the diffuse tail). The direct path is shifted to the
  true geometric arrival time so it aligns with the low band.
- Crossover: causal fourth-order Linkwitz-Riley filters at `1000 Hz` (Butterworth
  applied twice). The complementary low/high pass share a phase response, so the
  two independently simulated bands stay time-aligned and sum to a flat
  magnitude while keeping the RIR causal (no pre-ringing).
- Furniture: random polygon-prism obstacles are stored in metadata and applied
  to the high-frequency RIR as deterministic, height-aware occlusion and
  scattering (a low obstacle does not shadow a path above its top). Obstacle
  counts are area-aware, furniture profiles constrain footprint/height/material,
  and the sampler rejects overlapping or over-crowded layouts. Low-frequency
  obstacle diffraction is not modeled; metadata marks this as a high-frequency
  post-process.

`gpuard/pytARD` is vendored under `puresound/third_party/pytARD` because the
upstream project is not packaged as a normal pip dependency. PureSound imports
that source tree for its voxelization, modal frequencies, and `Unit` impulse,
then runs a single batched modal solve over all five sources and returns a
`[5, samples]` low-frequency RIR array.

## Example

```python
from puresound.audio.hybrid_rir import (
    GpuARDPytARDBackend,
    HybridRIRConfig,
    PyroomacousticsHighFrequencyBackend,
    generate_hybrid_rir,
    write_hybrid_rir_dataset_item,
)

config = HybridRIRConfig(sample_rate=48000, duration=1.5)
rir, metadata = generate_hybrid_rir(
    config=config,
    low_backend=GpuARDPytARDBackend(),
    high_backend=PyroomacousticsHighFrequencyBackend(),
    seed=0,
)
write_hybrid_rir_dataset_item("hybrid_rir_db", "room_000000", rir, metadata, 48000)
```

For a smoke test without running pytARD:

```bash
python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir_smoke \
  --n-rooms 2 \
  --rir-per-room 1 \
  --low-backend analytic
```

For the intended dataset path, install Pyroomacoustics. pytARD is already
vendored in this repository:

```bash
pip install pyroomacoustics
python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir \
  --n-rooms 1000 \
  --rir-per-room 1 \
  --low-backend pytard \
  --num-workers 22 \
  --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2
```

The CPU `pytard` backend is single-threaded per RIR but the modal solve makes
each RIR cheap (~30s for `1.6s` @ `16 kHz`), so the fastest path on one machine
is usually CPU multi-worker: set `--num-workers` near the core count. Each
worker is a separate process; tune `OMP_NUM_THREADS`/`MKL_NUM_THREADS` in the
environment if your BLAS stack oversubscribes CPU threads. The scene sampling
is deterministic in the main process, so `--num-workers` changes speed only,
not the generated rooms.

Use `--rir-per-room M` to generate multiple RIRs from the same sampled room
geometry. Each room keeps the same dimensions, RT60, and obstacles while
resampling non-overlapping microphone/source positions. Output directories are
grouped by `room_id`, and each RIR/metadata pair is named `{room_id}_{index}`,
for example `room_000000/room_000000_000003.wav` and
`room_000000/room_000000_000003.json`.

Microphone and speech-source heights are sampled from deployment-like ranges
instead of the full room height. Near/far labels are sampled by horizontal
source-to-mic distance; metadata stores both `distance_m` (3D propagation
distance) and `horizontal_distance_m` (near/far geometry).

`--pytard-low-sample-rate` controls the internal wave simulation rate. The RIR
is resampled to `--sample-rate` after pytARD finishes.
`--pytard-spatial-samples-per-wavelength` trades speed for accuracy; `2` is
practical for dataset generation, while larger values are much slower.

For GPU acceleration of the pytARD low band, install CuPy for your CUDA runtime:

```bash
uv pip install cupy-cuda12x
```

Then select the CuPy backend:

```bash
python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir_gpu \
  --n-rooms 1000 \
  --rir-per-room 4 \
  --low-backend pytard-cupy \
  --gpu-devices 0,1 \
  --num-workers 2 \
  --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2
```

`cupy-cuda12x` matches the CUDA 12 PyTorch stack used by this project. If your
machine uses a different CUDA runtime, install the matching CuPy wheel manually
and keep `--low-backend pytard-cupy`.
`--gpu-devices` assigns worker pools to physical CUDA devices via
`CUDA_VISIBLE_DEVICES`; when omitted, `--num-workers` processes share the
currently visible device set.

Only `--low-backend pytard-cupy` runs on the GPU; the `pytard`/`analytic` low
backends and the Pyroomacoustics high band are always CPU. Prefer one worker per
GPU device because each process owns its own CUDA context. Because the high band
stays on CPU, benchmark CPU and GPU worker counts for your hardware.

## Tuning

A single RT60 is sampled per room (`--rt60 MIN MAX`) and drives both bands: the
high band through `pra.inverse_sabine`, the low band through the decay envelope.
`--pra-max-order` caps the image-source order derived from the RT60 (ray tracing
covers the diffuse tail), trading accuracy for speed.

The crossover glues two independently scaled simulators, so the low band is
level-matched to the high band in the overlap region before summing:

- `--crossover-target-db` (default `0.0`): target low-band RMS relative to the
  high-band RMS near the crossover. `0 dB` makes the bands equal so the
  Linkwitz-Riley pair sums flat; negative values deliberately attenuate the low
  band.
- `--crossover-max-gain` (default `2.0`): upper bound on the matching gain, so a
  weak low band cannot be boosted without limit.

The low-band tail level and decay are controlled by
`--pytard-calibration-peak` (peak used to normalize raw pytARD output) and
`--pytard-rt60-decay-scale` (scales the scene RT60 when damping the lossless
wave tail).

Each room directory contains one WAV/metadata pair per generated RIR:

```text
room_000000/
  room_000000_000000.wav
  room_000000_000000.json
  room_000000_000001.wav
  room_000000_000001.json
```

## Apply RIR to WAV

Use `apply_rir_to_wav.py` to convolve a dry mono/multi-channel WAV with a
generated RIR WAV. Multi-channel dry WAVs are mixed down to mono by default.

```bash
python egs/rir_generation/apply_rir_to_wav.py \
  --wav path/to/dry.wav \
  --rir exp/hybrid_rir/room_000000/room_000000_000000.wav \
  --output exp/hybrid_rir/room_000000/wet_5ch.wav \
  --rir-mode full \
  --length-mode same \
  --output-layout rir-channels \
  --peak-normalize
```

`--output-layout rir-channels` keeps one wet channel per RIR channel. For a
single mono mixture, use:

```bash
python egs/rir_generation/apply_rir_to_wav.py \
  --wav path/to/dry.wav \
  --rir exp/hybrid_rir/room_000000/room_000000_000000.wav \
  --output exp/hybrid_rir/room_000000/wet_mono.wav \
  --output-layout mono-sum \
  --peak-normalize
```
