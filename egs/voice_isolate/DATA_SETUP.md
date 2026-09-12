# Data setup — from nothing to a trainable recipe

繁體中文版本：[`DATA_SETUP.zh-TW.md`](DATA_SETUP.zh-TW.md)

`config/train_dpcrn.yaml` trains from scratch in one run, but it does not ship
data. This page is the chain from public corpora to the paths that recipe reads.
Every path in the recipe that starts with `/path/to/` is a placeholder you must
replace.

## What the default recipe needs

| recipe key | what it wants | built by |
|---|---|---|
| `dataset.train_metafile` / `valid_metafile` | speech metafile, speaker-disjoint | §1 |
| `augmentation_noise.noise_folder` | a noise corpus | §1 |
| `augmentation_real_far.pool_manifest` | real distant recordings | §2 |
| `augmentation_real_near.pool_manifest` | real near-field recordings | §2 |
| `rir_bank.banks[core/expand/wide].folder` | synthetic RIR bank, sliced by difficulty | §3 |
| `rir_bank.banks[real].folder` | measured RIRs as a bank | §4 |

Run everything from the repository root.

## 1. Speech and noise

DNS-5 provides both. Fetch and resample once:

```bash
uv run python egs/voice_isolate/prepare_dns_challenge.py --help
```

Then build the metafile, cut it into contiguous chapters (the curriculum trains
on rows up to 30 s, and a chapter corpus is what makes long rows possible), and
split it so no speaker crosses train/dev:

```bash
uv run python egs/voice_isolate/prepare_metafile.py ...        # -> data/dns5-read.list

uv run python egs/voice_isolate/scripts/split_by_speaker.py data/dns5-read.list \
    --train-out data/dns5-read.train.list --dev-out data/dns5-read.dev.list \
    --dev-speaker-frac 0.05 --seed 0

uv run python egs/voice_isolate/scripts/build_chapter_corpus.py \
    --metafile data/dns5-read.train.list \
    --out-dir /your/scratch/dns5_read_16k_chapters \
    --out-metafile data/dns5-read.chapters.train.list
```

Point `augmentation_noise.noise_folder` at the DNS-5 noise directory.

## 2. Real recording pools

Two pools, from the same corpus, split by distance. VOiCES is public
(CC-BY). The far pool supplies distant interferers; the near pool supplies the
real close-talk rows that keep the model from deleting the user.

```bash
uv run python egs/voice_isolate/scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 1.0 \
    --out data/realfar_pool/voices.train.jsonl

uv run python egs/voice_isolate/scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 0 --max-distance 1.0 \
    --out data/realfar_pool/voices.near.train.jsonl
```

These emit finished waveforms, not RIRs, and split train/held-out by speaker.

## 3. Synthetic RIR bank, sliced into difficulty levels

Generate one bank, then cut the `core` / `expand` / `wide` views the curriculum
walks through. The views are symlink-only, so they cost nothing to keep.

```bash
# generate (see egs/rir_generation/README.md for the full argument set)
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
    --output-dir /your/scratch/hybrid_rir_16k \
    --backend path-events-m4 --n-rooms 1000 --rir-per-room 4 \
    --num-workers 8 --seed 1337 --sample-rate 16000 --duration 1.6 \
    --scene-version v1 --room-type mixed --output-mode calibrated \
    --record-realized-metrics

# slice it into levels: core / expand / wide / stress / all
uv run python egs/rir_generation/tools/bank/build_bank_view.py levels \
    /your/scratch/hybrid_rir_16k \
    /your/scratch/hybrid_rir_16k_levels
```

`levels` cuts by RT60 and by the worst-case near/far DRR gap, and the levels are
cumulative, so the curriculum can walk them in order:

| level | RT60 | worst-case near/far DRR gap |
|---|---|---|
| `core` | 0.20–0.45 s | ≥ 6 dB |
| `expand` | 0.20–0.65 s | ≥ 3 dB |
| `wide` | 0.20–0.85 s | ≥ 3 dB |
| `stress` | anything in none of the above | — |

Point `rir_bank.banks[core/expand/wide].folder` at
`/your/scratch/hybrid_rir_16k_levels/{core,expand,wide}`.

## 4. Measured RIRs

The `real` bank entry wants measured impulse responses, not synthetic ones.
Several public corpora work (ACE, dEchorate, BUT ReverbDB, BRUDEX, AIR); check
each one's licence for your use.

```bash
uv run python egs/rir_generation/tools/measured/real_rir_to_bank.py --help
uv run python egs/rir_generation/tools/bank/build_bank_view.py levels \
    /your/scratch/real_rir_16k /your/scratch/real_rir_16k_levels
```

Point `rir_bank.banks[real].folder` at the resulting view.

## 5. Check before you commit a GPU to it

```bash
uv run python egs/voice_isolate/scripts/check_training_data.py \
    egs/voice_isolate/config/train_dpcrn.yaml --n 64 --dump 8
```

This reports manifest speaker-disjointness, path existence, the near/far DRR gap
and distance/RT60 distributions measured on real sampled items, and dumps a few
mixtures to listen to. A recipe that passes this is one whose paths are right and
whose near/far contrast actually exists in the sampled data.

Then confirm the pipeline can learn at all before a long run:

```bash
uv run python egs/voice_isolate/scripts/overfit_check.py \
    egs/voice_isolate/config/train_dpcrn.yaml --steps 800 --device cuda
```

More tools, and what each one is for: [`scripts/README.md`](scripts/README.md).
