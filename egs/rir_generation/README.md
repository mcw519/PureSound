# RIR generation

Public tools for generating, inspecting, and packaging room impulse responses
(RIRs) for near/far speech augmentation. The recommended path for training
data is the M6 bank entry point, which combines deterministic generation,
per-item QC, and release packaging in one command.

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

This README covers **usage**. How each algorithm maps to the code is in
[`docs/audio/rir_realism_algorithm_zh-TW.md`](../../docs/audio/rir_realism_algorithm_zh-TW.md);
experiment records, review findings, and plans are in
[`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md).

> **Always run with `.venv/bin/python`** (or an environment with
> `pyroomacoustics`/`rir_generator` installed and matching numpy ABI). A wrong
> interpreter produces collection errors and validator failures that look like
> code bugs but are not.

## What one item is

One item is a five-channel RIR WAV plus a same-stem JSON sidecar:

| Channel | Source | Intended distance |
|---:|---|---|
| 0 | `near_0` | near, usually `< 1 m` |
| 1 | `near_1` | near, usually `< 1 m` |
| 2 | `far_0` | far, usually `> 2 m` |
| 3 | `far_1` | far, usually `> 2 m` |
| 4 | `far_2` | far, usually `> 2 m` |

The receiver is one microphone; channels are independent source-to-microphone
paths, not a synchronized array. The JSON sidecar is part of the data contract
(channel map, distances, level policy, renderer provenance) — never separate a
WAV from its metadata. M6 keeps train/validation/test assignments room- and
acoustic-space-disjoint.

## The M vocabulary

M labels are roadmap milestones, not quality scores:

| Milestone | Meaning |
|---|---|
| M0 | Frozen RT60-driven baseline (`v0`), regression reference only |
| M1 | Material-first scene sampling (`v1`): correlated room/finish/obstacle causes |
| M2 | Complex impedance / frequency-dependent modal loss (experimental, opt-in) |
| M3 | Coherent PathEvents high band (opt-in) |
| M4 | PathEvents early field + multiband FDN late field — **the default M6 high band** |
| M5 | Measured-room campaigns and inverse calibration (contract frozen; empirical evidence open) |
| M6 | Training-bank contract, QC, release, evaluation, production decision |

The current synthetic release is a **candidate**: production promotion is
still gated on human listening and downstream evidence
(`RIR_EXP_LOG.md` §6).

## Generate the recommended M6 candidate

Run from the repository root. `--backend path-events-m4` is the default and is
shown explicitly only for clarity; the default was chosen on measured decay
shape (its octave decay sits 8× closer to measured rooms than
pyroomacoustics — `RIR_EXP_LOG.md` §6.6.6).

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot \
  --backend path-events-m4 \
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

One command runs M6.2 generation, M6.3 QC, and M6.4 release packaging:

```text
<output-dir>/
├── path-events-m4_bank/       # WAV/JSON items, manifest, split indexes, QC
└── path-events-m4_release/    # audited training variants and recipes
```

An interrupted run can be re-invoked with the same arguments: resume checks
task/config/scene/audio identity (bound to the code revision) before skipping
an item. A release directory is never overwritten — choose a new
`--output-dir` for a new bank.

### A/B arm

`--backend pyroomacoustics` renders the geometric arm on the same scenes when
the seed and low backend are identical. The matched-pilot wrapper enforces
that and refuses to overwrite:

```bash
for backend in path-events-m4 pyroomacoustics; do
  PURESOUND_M6_PILOT_ROOMS=100 \
  bash egs/rir_generation/phases/m6_bank/scripts/generate_m6_training_pilot.sh \
    "$backend" <pilot-root>
done
```

### GPU for the low band

Only the low-frequency solve can use the GPU; the high band is always CPU.
Prefer one worker per GPU (each CuPy worker owns a CUDA context):

```bash
uv pip install cupy-cuda12x

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir <out> \
  --n-rooms 1000 --rir-per-room 4 \
  --num-workers 2 --gpu-devices 0,1 \
  --low-backend pytard-cupy-material \
  --seed 1337 --sample-rate 16000 --duration 1.6 \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --record-realized-metrics
```

### Small smoke run

M6 needs a non-empty train/validation/test split, so use at least a few rooms.
Pipeline check only, not a training configuration:

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir <out>/smoke \
  --n-rooms 6 --rir-per-room 1 --num-workers 1 --duration 0.4
```

Expected: `status=candidate`, QC 6/6 pass, all three splits non-empty. Two
fresh runs with the same arguments produce identical `manifest_sha256`.

## Ingest measured RIRs (unlocks `real_native`)

Published measured corpora fail M6 QC as-is because their time origin is the
direct arrival, not the emission instant. The ingest re-inserts the
propagation delay, rejects channels whose direct path cannot be located, and
runs the same item QC as synthetic banks
(`RIR_EXP_LOG.md` §5 for the method and validation):

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/ingest_measured_m6_variant.py \
  --source <corpus-view>/items \
  --bank <out>/measured_bank \
  --pruned-bank <out>/measured_pruned \
  --workers 8 --code-revision "$(git rev-parse --short HEAD)" \
  --report <out>/measured_ingest.json
```

`--pruned-bank` writes the quarantine-free copy a release variant requires;
the unpruned bank stays as the record of what was dropped and why. Feed it to
the release builder to make the real and mixed recipes ready:

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_variant_release.py \
  --source-bank <synthetic-qc-bank> --output-dir <out>/release \
  --measured-bank <out>/measured_pruned --qc-workers 8
```

## Produce the M6.6 evidence chain

Renderer approval must be stamped **before** QC (the QC summary is bound to
the manifest hash), so the release is a three-pass flow:

```bash
# 1) evaluate: throughput report + first evaluation (the basis for approval)
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py \
  --release <release_1> --evidence-root <ev> \
  --generation-audit <bank>/rir_bank_generation_audit.json --pass evaluate

# 2) approve: prune both banks, stamp renderer approvals, rebuild the release
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py \
  --release <release_1> --evidence-root <ev> --pass approve \
  --synthetic-bank <bank> --measured-bank <measured_pruned> \
  --rebuild-release <release_2> --approver-id "<who>" --qc-workers 8

# 3) attest: listening assignment, sign-offs, bundle, production decision
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py \
  --release <release_2> --evidence-root <ev> --pass attest \
  --generation-audit <bank>/rir_bank_generation_audit.json \
  --reviewer-id "<who>" --participants 24 --report <ev>/summary.json
```

Exit 0 means production-ready; exit 3 means still blocked, with all thirteen
checks printed. Without `--listening-responses` the listening report is a
`contract_fixture` dry run — it validates the pipeline and correctly does
**not** count as empirical human evidence.

## Consume a release for augmentation

```yaml
pregenerated:
  used: true
  bank_type: release
  folder: <output-dir>/path-events-m4_release
  recipe_id: synthetic_calibrated   # real_native / mixed_calibrated_real when ready
  split: train
  usage_role: train
  require_production: false
```

`usage_role` must match the dataset role and split, so a training job cannot
silently consume validation or test rooms. `require_production: false` is
intentional while the M6.6 promotion certificate is open. Never point a
training job at an unsplit bank root when an M6 manifest is present.

## Inspect and visualize

| Command | Use |
|---|---|
| `generate_hybrid_rir.py` | Generate individual simulated hybrid RIRs (low-level tool). |
| `generate_m6_bank.py` | Generate, QC, and package an M6 candidate (recommended entry). |
| `render_spatial_rir.py` | Receiver-array, FOA, or optional BRIR outputs. |
| `plot_rir.py` | Waveforms, EDCs, image-source paths, low-band pressure fields. |
| `inspect_bank.py` | Metadata distributions for one RIR folder. |
| `compare_bank_acoustics.py` | DRR/C50/decay/spectral statistics across banks. |
| `compare_modal_acoustics.py` | Low-frequency modal peak/spacing/Q statistics. |
| `build_readme_sample.py` | Render the listenable dry/near/far sample set. |

All support `--help`. The low-band pressure-field animation replays one item's
modal state:

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/plot_rir.py low-field \
  --rir <bank>/room_000349/room_000349_000000.wav \
  --backend pytard --channel 0 --t-ms 80 --gif
```

### Same-scene backend comparison

Both panels use the same v1 scene (`scene_sha256`
`6bb48bb7…a69d1df`), the same low band, and `16 kHz / 1.6 s` calibrated
output; only the high-band renderer changes (left: pyroomacoustics, right:
M4). Regenerate with:

```bash
for backend in pyroomacoustics path-events-m4; do
  PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_hybrid_rir.py \
    --output-dir /tmp/readme_assets/$backend --n-rooms 1 --rir-per-room 1 \
    --sample-rate 16000 --duration 1.6 --scene-version v1 --room-type mixed \
    --output-mode calibrated --low-backend pytard-material \
    --high-backend $backend --seed 1423 --record-realized-metrics
done
```

(Seed 1423 puts the pictured room at the bank's median reverberation rather
than in its long tail.)

![Matched Pyroomacoustics versus M4 overview](assets/overview.png)

![Matched scene reflection paths](assets/paths_ch2.png)

![Low-frequency modal pressure-field animation](assets/field_ch2.gif)

### Hear one M4 item

One dry LibriSpeech utterance through a near and a far channel of the same M4
item — a distance comparison inside one room. The two convolved files share
one gain (their level relationship survives); the dry reference is scaled
separately because a calibrated RIR's amplitude encodes a reference SPL, not a
listening level.

| File | Content |
|---|---|
| [`m4_sample_dry.wav`](assets/m4_sample_dry.wav) | dry source, no room |
| [`m4_sample_near_near_0.wav`](assets/m4_sample_near_near_0.wav) | through `near_0` at 0.45 m |
| [`m4_sample_far_far_2.wav`](assets/m4_sample_far_far_2.wav) | through `far_2` at 3.89 m |
| [`m4_sample_rir.wav`](assets/m4_sample_rir.wav) | the five-channel RIR itself |

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/build_readme_sample.py \
  --rir /tmp/readme_assets/path-events-m4/room_000000/room_000000_000000.wav
```

## Directory layout

```text
egs/rir_generation/
├── generate_hybrid_rir.py       # individual simulated RIRs (low-level)
├── generate_m6_bank.py          # public M6 one-command pipeline
├── render_spatial_rir.py        # spatial/array rendering
├── plot_rir.py                  # plots and field animation
├── inspect_bank.py              # folder statistics
├── compare_*_acoustics.py       # bank comparisons
├── build_readme_sample.py       # listenable sample set
├── examples/                    # reproducible recipes
├── tools/                       # audition and bank helpers
├── exp/                         # experiment outputs (not tracked)
└── phases/                      # per-milestone validators, configs, reports
    └── m6_bank/scripts/         # QC, release, ingest, evidence, pilot CLIs
```

## Dependencies

The default backend (`path-events-m4`) has no third-party renderer
dependency. `pyroomacoustics` is required only for the A/B arm; CUDA CuPy is
optional for the low band:

```bash
pip install pyroomacoustics   # A/B arm only
uv pip install cupy-cuda12x   # GPU low band, optional
```

## Further reading

- [`docs/audio/rir_realism_algorithm_zh-TW.md`](../../docs/audio/rir_realism_algorithm_zh-TW.md) — algorithm ↔ code map.
- [`docs/audio/rir_bank_v2_zh-TW.md`](../../docs/audio/rir_bank_v2_zh-TW.md) — M6 contract and evidence rules.
- [`docs/audio/rir_bank.md`](../../docs/audio/rir_bank.md) — training-side loaders.
- [`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md) — experiment records, review findings, plans.
