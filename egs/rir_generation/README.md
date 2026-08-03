# RIR generation

This directory contains the public tools for generating, inspecting, and
packaging room impulse responses (RIRs) for near/far speech augmentation.
The recommended path for training data is the M6 bank entry point, which
combines deterministic generation, per-item QC, and release packaging.

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

## What one simulated item is

One item is a five-channel RIR WAV plus a same-stem JSON sidecar:

| Channel | Source | Intended distance |
|---:|---|---|
| 0 | `near_0` | near, usually `< 1 m` |
| 1 | `near_1` | near, usually `< 1 m` |
| 2 | `far_0` | far, usually `> 2 m` |
| 3 | `far_1` | far, usually `> 2 m` |
| 4 | `far_2` | far, usually `> 2 m` |

The receiver is one microphone. A room/acoustic-space sample contains room
dimensions, surface/material causes, furniture/obstacles, microphone position,
source positions, renderer settings, and realized acoustic metrics. Multiple
items may share an acoustic space while varying the microphone/source layout;
M6 keeps train/validation/test assignments room- and acoustic-space-disjoint.

The normal hybrid renderer combines a low-frequency wave/modal component with
a high-frequency geometric component through a causal crossover. The JSON
sidecar is part of the data contract; do not copy WAV files without their
metadata. A finite modal/voxel low-band solve can leave a tiny numerical
precursor, so the generator zeros each low-band channel before
`floor(distance / sound_speed * sample_rate)` while preserving the arrival
sample itself; this is the causality contract used by M6 QC.

## What the M-series means

The M labels are milestones in the renderer and data-bank roadmap. They are
not quality scores and a later M does not automatically make every earlier
backend production-ready.

| Milestone | Meaning | Practical status |
|---|---|---|
| **M0** | Frozen RT60-driven hybrid baseline (`v0`), useful as a regression/reference condition. | Reference only; not the preferred training recipe. |
| **M1** | Material-first scene sampling (`v1`): room type, surface/material priors, obstacles, and correlated acoustic causes. | Recommended physical scene foundation. |
| **M2** | Complex impedance and frequency-dependent modal loss, including impedance measurements and residue calibration. | Experimental/opt-in; production mapping is not frozen. |
| **M3** | Coherent PathEvents for early/direct/reflected wave paths. | Implemented and opt-in as a high-band backend. |
| **M4** | Spatial late field: PathEvents coupled to a multiband late-field/FDN realization. | Implemented and opt-in; not the default bank renderer. |
| **M5** | Controlled measured-room campaigns, inverse calibration, constrained residuals, and spatial calibration. | Implementation gates exist; empirical room evidence is still required. |
| **M6** | Deterministic training-bank contract, QC, variants, evaluation, and immutable production decision. | Candidate-bank pipeline is usable; production promotion remains evidence-gated. |

### M6 sub-milestones

1. **M6.1 — contract:** versioned manifest, provenance, hashes, and deterministic
   train/validation/test split.
2. **M6.2 — generation:** complete task plan, per-item seeds, parallel/resume
   generation, and post-generation audit.
3. **M6.3 — QC:** physical per-item checks, pass-only indexes, and quarantine
   for failed or non-evaluable items.
4. **M6.4 — release:** immutable calibrated and peak-normalized variants plus
   training recipes.
5. **M6.5 — evaluation:** acoustic distributions, throughput, listening, and
   downstream-model evidence contracts.
6. **M6.6 — decision:** append-only promotion certificate binding the release,
   evidence hashes, renderer approval, and sign-offs.

M6.1–M6.6 implementation gates are present. The current synthetic release is
still a **candidate**, not a production-approved bank: measured/mixed assets,
controlled listening, downstream-model evidence, and production approvals are
not bundled automatically.

### Post-review hardening (2026-08-02)

The bank path now seeds both NumPy and libroom for Pyroomacoustics ray tracing,
binds resume identity to the code revision and runtime package versions, and
tests serial, parallel, fresh-run, and resume reproducibility on the actual
default high backend. The low-frequency production recommendation uses a
single-sample causal excitation rather than the former bipolar excitation that
created fixed spectral comb nulls.

The M4 alternative now covers DC through Nyquist with an endpoint-complete FDN
filterbank, derives late-tail energy from material RT60 instead of the
order-truncated path tail, preserves one shared spatial-field gain, and applies
material boundary phase priors, air absorption, source directivity, and
path-local obstacle effects. These are renderer changes, not merely parameter
tuning of Pyroomacoustics.

M6 admission and evidence validation were also tightened: late-arrival and
octave-decay gates are active, calibrated float RIRs no longer inherit a false
unit-peak limit, unsafe item paths and manifestless M6 layouts fail closed,
variant audio lineage is sample-verified, downstream confidence intervals are
recomputed, and production certificates re-run the bound release/evidence
audits. Existing banks generated before these fixes must not be presented as
post-hardening results; generate a new output directory.

### Hardened matched preflight result (2026-08-02)

The first post-hardening smoke campaign is complete in
[`exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/`](exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/).
It used 30 matched rooms × 2 items (60 items / 300 channels per backend),
`v1/mixed`, seed `1337`, calibrated `16 kHz / 1.6 s` output, and the GPU low
backend `pytard-cupy-material`. Every scene hash, room/acoustic-space ID,
split, seed, and audio shape matched across Pyroomacoustics and M4; both banks
had 60/60 QC PASS, zero quarantines, and passing release audits. The actual
calibrated train readers also loaded 56 PASS items from each release.

The paired medians show what the backends currently change:

| Metric | Pyroomacoustics | PathEvents-M4 | M4 − Pyroom |
|---|---:|---:|---:|
| DRR | -4.83 dB | -3.24 dB | +1.59 dB |
| C50 | 6.10 dB | 10.28 dB | +4.18 dB |
| C80 | 8.03 dB | 15.01 dB | +6.97 dB |
| T20 | 0.99 s | 0.47 s | -0.52 s |
| absolute T20 − scene RT60 error | 0.327 s | 0.080 s | M4 closer |

M4 is drier and more early-energy dominant in this sample, while its broadband
T20 tracks the sampled scene RT60 more closely. Causal arrival, exact-zero final
sample, the fixed 390 Hz comb check, and M4 high-tail coverage all passed. This
is evidence that the hardened pipeline and M4 algorithm exercise the intended
mechanisms; it is not a measured-room realism winner. Validation and test each
contain only two items per backend, the code revision is dirty, and no measured
RIR, human listening, or downstream-model result is included. Both releases are
therefore **candidate** only, Pyroomacoustics remains the default, and the full
4,000-item pilot is still required.

The observed aggregate generation cost was about 6.5 s/item for Pyroomacoustics
with two workers and 8.2 s/item for M4 with eight workers after resume. M4 is
currently CPU PathEvent/material-boundary limited; the GPU mainly accelerates
the shared low-frequency solve, so adding workers does not make the M4 high band
GPU-bound.

The machine-readable conclusion, including manifest/QC/release hashes and
provenance warnings, is [`preflight_validation_summary.json`](exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/preflight_validation_summary.json).

## Closest available training-data simulation

The closest current simulation to something usable as training data is:

| Setting | Recommended value | Why |
|---|---|---|
| Bank pipeline | `generate_m6_bank.py` | Produces manifest, QC indexes, and release recipes. |
| Scene | `--scene-version v1 --room-type mixed` | Material-first, correlated room/finish/obstacle causes. |
| Output | `--output-mode calibrated --record-realized-metrics` | Stable level semantics and auditable acoustic metrics. |
| Low band | `--low-backend pytard-material` | CPU wave/modal solver with material-frequency-dependent modal damping and a causal delta excitation without the former fixed comb signature. |
| High band | `--backend pyroomacoustics` | Default geometric renderer; best-controlled baseline today. |
| Sample rate / duration | `16 kHz / 1.6 s` | Matches the recommended speech-augmentation bank setup. |
| Bank size | `1000 rooms × 4 items = 4000 items` | Enough for a first training pilot while preserving room-disjoint splits. |
| Seed / workers | `1337 / 8` | Reproducible content; worker count does not change item identity. |
| Training recipe | `synthetic_calibrated`, `split: train` | Uses only QC-passed train items. |

This is the recommended **synthetic candidate**. It is not a claim that the
simulation is equivalent to a measured room. The M4 high backend
(`path-events-m4`) is useful for a matched ablation, but should not silently
replace the Pyroomacoustics baseline in the first training run.

### Generate the recommended M6 candidate

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot \
  --backend pyroomacoustics \
  --n-rooms 1000 \
  --rir-per-room 4 \
  --num-workers 8 \
  --seed 1337 \
  --sample-rate 16000 \
  --duration 1.6 \
  --scene-version v1 \
  --room-type mixed \
  --output-mode calibrated \
  --low-backend pytard-material \
  --record-realized-metrics
```

The one command runs M6.2 generation, M6.3 QC, and M6.4 release packaging.
It creates:

```text
egs/rir_generation/exp/rir_realism/m6/training_pilot/
├── pyroomacoustics_bank/       # WAV/JSON items, manifest, split indexes, QC
└── pyroomacoustics_release/    # audited training variants and recipes
```

An interrupted generation can be rerun with the same arguments; M6 resume
checks task/config/scene/audio identity before skipping an item. A release
directory is never overwritten; choose a new `--output-dir` for a new bank.

To build a matched M4 high-band candidate, keep every setting and seed the same
and change only the output directory and renderer:

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot_m4 \
  --backend path-events-m4 \
  --n-rooms 1000 --rir-per-room 4 --num-workers 8 --seed 1337 \
  --sample-rate 16000 --duration 1.6 \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --low-backend pytard-material \
  --record-realized-metrics
```

Use this M4 bank as a matched realism ablation first. Do not silently mix it
with the baseline bank or label it production-approved without the empirical
M4/M5 evidence gates.

### Same-scene backend comparison

The overview below replaces the original single-backend illustration. Both
panels use the same v1 scene, source/receiver geometry, `16 kHz / 1.6 s`
configuration, calibrated output, and the same low backend. Only the high-band
renderer changes:

- left: Pyroomacoustics high-frequency renderer;
- right: M4 PathEvent early response plus FDN late field.

The scene metadata hash is
`6bb48bb7b7f5bca0c61a7765544d8568ac946faf955ca87a358d43dd9a69d1df` in both
versions. The waveform and Schroeder decay panels therefore show a matched
backend ablation rather than two independently sampled rooms.

Regenerate the figures with:

```bash
for backend in pyroomacoustics path-events-m4; do
  PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_hybrid_rir.py \
    --output-dir /tmp/readme_assets/$backend --n-rooms 1 --rir-per-room 1 \
    --sample-rate 16000 --duration 1.6 --scene-version v1 --room-type mixed \
    --output-mode calibrated --low-backend pytard-material \
    --high-backend $backend --seed 1423 --record-realized-metrics
done
```

The seed is 1423 rather than the pipeline default, chosen so the pictured room
sits at the bank's median reverberation instead of in its long tail: across the
100-room pilot the scalar scene RT60 has a median of 0.56 s but a 95th
percentile of 2.34 s, and this scene is 0.55 s with an RT60 that falls with
frequency like the median. Seed 1337's first room happens to be a 2.82 s
all-hard-surface outlier and made a misleading illustration.

![Matched Pyroomacoustics versus M4 overview](assets/overview.png)

The path view and the pressure-field animation use this same scene. Paths are
geometry-only; the animation is the actual low-frequency modal pressure slice,
so it is shared by both high-band variants.

![Matched scene reflection paths](assets/paths_ch2.png)

![Low-frequency modal pressure-field animation](assets/field_ch2.gif)

### Hear one M4 item

A RIR on its own is a click. These are one dry LibriSpeech utterance convolved
with a near and a far channel of the same M4 item pictured above, so the pair is
a distance comparison inside one room:

| File | Content |
|---|---|
| [`m4_sample_dry.wav`](assets/m4_sample_dry.wav) | dry source, no room |
| [`m4_sample_near_near_0.wav`](assets/m4_sample_near_near_0.wav) | through `near_0` at 0.45 m |
| [`m4_sample_far_far_2.wav`](assets/m4_sample_far_far_2.wav) | through `far_2` at 3.89 m |
| [`m4_sample_rir.wav`](assets/m4_sample_rir.wav) | the five-channel RIR itself, float32, calibrated levels |

The two convolved files share one gain so their level relationship survives;
the dry reference is scaled separately, because a calibrated RIR's amplitude
encodes a reference SPL rather than a listening level. Most of the audible
distance cue is the direct-to-reverberant ratio, not level: in a room this
reverberant the diffuse field is nearly distance-independent, so the two land
within about a dB of each other in RMS while sounding very different.

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/build_readme_sample.py \
  --rir /tmp/readme_assets/path-events-m4/room_000000/room_000000_000000.wav
```

### Use a GPU for the low band

Install the CUDA-matched CuPy package and select the
`pytard-cupy-material` backend:

```bash
uv pip install cupy-cuda12x

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot_gpu \
  --backend pyroomacoustics \
  --n-rooms 1000 --rir-per-room 4 \
  --num-workers 1 --gpu-devices 0 \
  --low-backend pytard-cupy-material \
  --seed 1337 --sample-rate 16000 --duration 1.6 \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --record-realized-metrics
```

Only the low-frequency modal solve uses the GPU; the high-frequency
Pyroomacoustics or PathEvents renderer remains CPU-based. Prefer one worker per
GPU (`--gpu-devices 0,1 --num-workers 2` for two GPUs) because each CuPy worker
owns a separate CUDA context and memory pool.

### Small smoke run

M6 needs all three splits, so use at least a few rooms. This is a pipeline
check, not a training configuration:

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/smoke \
  --backend pyroomacoustics \
  --n-rooms 6 \
  --rir-per-room 1 \
  --num-workers 1 \
  --low-backend analytic \
  --duration 0.4
```

### Consume the release for augmentation

Use the release reader with an explicit recipe and split:

```yaml
pregenerated:
  used: true
  bank_type: release
  folder: egs/rir_generation/exp/rir_realism/m6/training_pilot/pyroomacoustics_release
  recipe_id: synthetic_calibrated
  split: train
  usage_role: train
  require_production: false
```

`usage_role` must match the dataset role and split, so a training dataset cannot
silently consume `validation` or `test`. `require_production: false` is
intentional while M6.5 empirical evidence and the M6.6 promotion certificate
are still open. Never point a training job at the unsplit bank root when an M6
manifest is present.

## Public commands

Run these commands from the repository root. Use `--help` for all options.

| Command | Use |
|---|---|
| `generate_hybrid_rir.py` | Generate individual simulated hybrid RIRs. |
| `generate_m6_bank.py` | Generate, QC, and package an M6 synthetic candidate. |
| `render_spatial_rir.py` | Render receiver-array, FOA, or optional BRIR outputs. |
| `plot_rir.py` | Plot waveforms, EDCs, image-source paths, and actual low-band pressure fields. |
| `inspect_bank.py` | Summarize metadata distributions for one RIR folder. |
| `compare_bank_acoustics.py` | Compare DRR, C50, decay, and spectral statistics across banks. |
| `compare_modal_acoustics.py` | Compare low-frequency modal peak/spacing/bandwidth/Q statistics. |

Examples:

```bash
PYTHONPATH=. python egs/rir_generation/generate_hybrid_rir.py --help
PYTHONPATH=. python egs/rir_generation/generate_m6_bank.py --help
PYTHONPATH=. python egs/rir_generation/plot_rir.py --help
PYTHONPATH=. python egs/rir_generation/inspect_bank.py --help
```

### Actual low-frequency pressure-field animation

The `field` subcommand is a self-contained 2-D FDTD illustration. To inspect
the pressure field generated by the low-frequency modal recurrence itself, use
`low-field`:

```bash
PYTHONPATH=. python egs/rir_generation/plot_rir.py low-field \
  --rir egs/rir_generation/exp/rir_realism/m6/training_pilot_gpu/pyroomacoustics_bank/room_000349/room_000349_000000.wav \
  --backend pytard --channel 0 --t-ms 80 --gif
```

This re-runs one sample, reconstructs a fixed-height `p(x, y, z_slice, t)`
slice from the modal state, and writes an MP4, a GIF when requested, a contact
sheet, and a reusable `.npz` diagnostic. The normal M6 generator does not
store field snapshots. The animation uses relative solver pressure units and
does not include the final output calibration or the high-frequency band.
`--slice-z` selects the slice height; by default it uses the receiver height.
Use `--low-sample-rate` and `--spatial-samples-per-wavelength` to match the
generation recipe when those values are not recorded in the sidecar metadata.

## Directory layout

```text
egs/rir_generation/
├── generate_hybrid_rir.py       # individual simulated RIRs
├── generate_m6_bank.py          # public M6 one-command pipeline
├── render_spatial_rir.py        # spatial/array rendering
├── plot_rir.py                  # plots
├── inspect_bank.py              # folder statistics
├── compare_*_acoustics.py       # bank comparisons
├── examples/                    # reproducible recipes
├── tools/                       # audition, measured-RIR, and bank helpers
├── exp/rir_realism/             # retained M0–M6 reference artifacts
└── phases/
    ├── m0_baseline/
    ├── m1_material/
    ├── m2_impedance/
    ├── m3_wave_path/
    ├── m4_spatial_late_field/
    ├── m5_calibration/
    └── m6_bank/
```

Phase scripts, configs, reports, and fixtures are kept under their milestone
directory. Retained generated evidence is grouped under
[`exp/rir_realism/`](exp/rir_realism/); the root is intentionally reserved for
stable user-facing tools.

## Dependencies and references

The CPU M6 path uses the repository's `pytard` implementation and
Pyroomacoustics for the high band:

```bash
pip install pyroomacoustics
```

Optional CUDA acceleration is available through the `pytard-cupy` backend when
the matching CuPy package is installed. It is not required for the recommended
CPU pilot.

Further reading:

- [`docs/audio/rir_realism_algorithm_zh-TW.md`](../../docs/audio/rir_realism_algorithm_zh-TW.md) — full physical/algorithmic notes.
- [`docs/audio/rir_bank_v2_zh-TW.md`](../../docs/audio/rir_bank_v2_zh-TW.md) — M6 contract and evidence rules.
- [`docs/audio/hybrid_rir.md`](../../docs/audio/hybrid_rir.md) — hybrid renderer details.
- [`docs/audio/rir_scene_v2.md`](../../docs/audio/rir_scene_v2.md) — M1 scene/material schema.
- [`RIR_REALISM_PLAN.md`](../../RIR_REALISM_PLAN.md) — project roadmap.
