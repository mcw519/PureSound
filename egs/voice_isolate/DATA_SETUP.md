# Data setup — from nothing to a trainable recipe

Traditional Chinese: [`DATA_SETUP.zh-TW.md`](DATA_SETUP.zh-TW.md)

`config/train_dpcrn.yaml` trains from scratch in one run, but it does not ship
data. This page is the chain from public corpora to the paths that recipe reads.
Every path in the recipe that starts with `/path/to/` is a placeholder to
replace; `data/...` paths are relative to `egs/voice_isolate/`.

## What the default recipe needs

| Recipe key | What it wants | Built in |
|---|---|---|
| `dataset.train_metafile` / `valid_metafile` | speaker-disjoint chapter-length speech metafiles | §1 |
| `augmentation_noise.noise_folder` | a noise corpus at 16 kHz | §1 |
| `augmentation_realfar.pool_manifest` | real distant recordings | §2 |
| `augmentation_realnear.pool_manifest` | real near-field recordings | §2 |
| `augmentation_reverb.simulator.pregenerated.banks` `core` / `expand` / `wide` | a synthetic RIR bank, sliced by difficulty | §3 |
| `augmentation_reverb.simulator.pregenerated.banks` `real` | measured RIRs as a bank | §4 |

Run the commands from the repository root unless a block says otherwise.

## 1. Speech and noise

DNS-5 provides both. Corpus preparation is library code shared with every other
recipe -- see [`docs/usage/data_preparation.md`](../../docs/usage/data_preparation.md).
One command scans the read-speech subset, converts it to 16 kHz once, and writes
a speaker-disjoint train/dev pair:

```bash
uv run python -m puresound.dataset.corpus.dns_challenge speech /path/to/audio/dns-5 \
    --output-dir egs/voice_isolate/data --id-prefix dns-read --subset read_speech \
    --utt-id-style stem \
    --train-metafile egs/voice_isolate/data/dns5-read.train.list \
    --valid-metafile egs/voice_isolate/data/dns5-read.dev.list \
    --valid-ratio 0.05 --seed 0 \
    --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/clean_fullband
```

The utterance ids stay the corpus's own file names (`--utt-id-style stem`)
because the next step reads the segment index out of them. The recipe trains on
rows up to 30 s, so stitch the 10 s segments back into whole chapters, for both
splits:

```bash
for split in train dev; do
  uv run python egs/voice_isolate/scripts/build_chapter_corpus.py \
      --metafile egs/voice_isolate/data/dns5-read.$split.list \
      --out-dir /your/scratch/dns5_read_16k_chapters/$split \
      --out-metafile egs/voice_isolate/data/dns5-read.chapters.$split.list
done
```

For noise, convert the DNS-5 noise folder once and point
`augmentation_noise.noise_folder` at the converted tree:

```bash
uv run python -m puresound.dataset.corpus.dns_challenge noise /path/to/audio/dns-5 \
    --output-dir egs/voice_isolate/data --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband
```

## 2. Real recording pools

Two pools from VOiCES (public, CC BY), split by loudspeaker-to-microphone
distance. The far pool supplies distant interferers; the near pool supplies the
real close-talk rows that keep the model from deleting the user.

```bash
cd egs/voice_isolate
uv run python scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 1.0 \
    --out data/realfar_pool/voices.train.jsonl

uv run python scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 0 --max-distance 1.0 \
    --out data/realfar_pool/voices.near.train.jsonl
```

These are finished recordings, not RIRs, split train/held-out by speaker.
`--distractor none` (the default) keeps clean lone far speakers, the right
interferer for near-field isolation; `--stats-only` prints the pool without
writing it.

## 3. Synthetic RIR bank, sliced into difficulty levels

Generate one bank, then cut the `core` / `expand` / `wide` views the curriculum
walks through. The views are symlink-only, so they cost nothing to keep. See
[`egs/rir_generation/README.md`](../rir_generation/README.md) for the full
argument set and the interpreter requirement.

```bash
# generate: writes <out>/path-events-m4_bank/ (items) and <out>/path-events-m4_release/
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
    --output-dir /your/scratch/hybrid_rir_16k \
    --backend path-events-m4 --n-rooms 1000 --rir-per-room 4 \
    --num-workers 8 --seed 1337 --sample-rate 16000 --duration 1.6 \
    --scene-version v1 --room-type mixed --output-mode calibrated \
    --record-realized-metrics

# slice the bank into levels: core / expand / wide / stress / all
PYTHONPATH=. .venv/bin/python egs/rir_generation/tools/bank/build_bank_view.py levels \
    /your/scratch/hybrid_rir_16k/path-events-m4_bank \
    /your/scratch/hybrid_rir_16k_levels
```

The generator renders the low band with `--low-backend pytard-material` by default,
which needs a [pytARD](https://github.com/gpuard/pytARD) checkout you install yourself
(AGPL-3.0, not on PyPI; point `PURESOUND_PYTARD_ROOT` at it -- see the
[dependencies](../rir_generation/README.md#dependencies)). To avoid that dependency, add
`--low-backend analytic-material`: a rectangular-room modal model that needs nothing
installed, so the bank will not be identical to one built with the default.

`levels` measures each item's RT60 and worst-case near/far DRR gap
(`min(DRR_near) - max(DRR_far)`), and the levels are cumulative, so the curriculum
can walk them in order:

| Level | RT60 | Worst-case near/far DRR gap |
|---|---|---|
| `core` | 0.20–0.45 s | ≥ 6 dB |
| `expand` | 0.20–0.65 s | ≥ 3 dB |
| `wide` | 0.20–0.85 s | ≥ 3 dB |
| `stress` | valid items in none of the above | -- |
| `all` | every valid item, unfiltered | -- |

`--drr-window-ms` must match the recipe's `drr_window_ms` (2.5 by default). Point
the `core`, `expand` and `wide` bank members at
`/your/scratch/hybrid_rir_16k_levels/{core,expand,wide}`.

## 4. Measured RIRs

The `real` bank member wants measured impulse responses, whose near channels are
genuine sub-metre responses. `real_rir_to_bank.py` converts a public corpus in
two steps -- a corpus-specific scan to a manifest, then the shared bank writer:

```bash
# scan one corpus: ace, brudex, dechorate, diffrir or slr28
PYTHONPATH=. .venv/bin/python egs/rir_generation/tools/measured/real_rir_to_bank.py scan dechorate \
    --input /path/to/dEchorate --staging /your/scratch/dechorate_wav \
    --manifest /your/scratch/dechorate.manifest.jsonl

# write the bank, at 16 kHz, split near/far by distance
PYTHONPATH=. .venv/bin/python egs/rir_generation/tools/measured/real_rir_to_bank.py from-manifest \
    --manifest /your/scratch/dechorate.manifest.jsonl \
    --output /your/scratch/real_rir_16k/dechorate --target-sr 16000
```

Repeat per corpus, then combine the banks into one training view with
`build_bank_view.py merge --source TAG=PATH ... --output DIR` and point the `real`
member at it. The script's `--help` documents each corpus's geometry and
distance handling. Keep BUT ReverbDB out of the training bank: benchmark stages
7b and 8 are built from it. Check each corpus's licence for your use.

## 5. Check before you commit a GPU to it

```bash
cd egs/voice_isolate
uv run python scripts/check_training_data.py config/train_dpcrn.yaml --n 64 --dump 8
```

This reports manifest speaker-disjointness, path existence, the near/far DRR gap
and the distance and RT60 distributions measured on real sampled items, and
dumps a few mixtures to listen to. A recipe that passes it is one whose paths are
right and whose near/far contrast actually exists in the sampled data.

Then confirm the pipeline can learn at all before a long run:

```bash
uv run python scripts/overfit_check.py config/train_dpcrn.yaml --steps 800 --device cuda
```

More tools, and what each one is for: [`scripts/README.md`](scripts/README.md).
