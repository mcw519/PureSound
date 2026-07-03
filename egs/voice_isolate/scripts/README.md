# scripts — voice_isolate tooling

All scripts used by this recipe (data prep, eval, inference). Unused/dead scripts have been removed.
Everything below is invoked from the repo root unless noted; eval scripts default to `--device cpu`
(don't fight the training GPU).

## 1. Data prep

| script | purpose |
|---|---|
| `split_by_speaker.py` | Speaker-disjoint train/dev split for a metafile (regroups by `spkid` so dev speakers never leak into train). |
| `validate_data.py` | Sanity-checks the data pipeline against a config: manifest speaker-disjointness + path existence, scene-level DRR/RT60 separability (foreground vs interferer), and a sample dump for human listening. |

```bash
uv run python egs/voice_isolate/scripts/split_by_speaker.py egs/voice_isolate/data/dns5-read.list \
    --train-out egs/voice_isolate/data/dns5-read.train.list --dev-out egs/voice_isolate/data/dns5-read.dev.list \
    --dev-speaker-frac 0.05 --seed 0

uv run python egs/voice_isolate/scripts/validate_data.py egs/voice_isolate/config/train_dpcrn_wide_antisup.yaml
```

## 2. Training-time validation

Run during/after training to check the model is actually separating.

| script | purpose |
|---|---|
| `run_valid.sh` | One-click: loss curves + in-domain SI-SDRi + PASS/MARGINAL/FAIL verdict. Calls `indomain_sisdri.py` + `dump_tb.py`. |
| `indomain_sisdri.py` | **Primary judge metric.** Reuses the training/valid synthetic pipeline, scores enhanced vs early-reverb target SI-SDRi; `--by-bucket` breaks out counter_level / 1N+0F / overlap (read the hard buckets, not the aggregate). |
| `dump_tb.py` | Prints tensorboard scalar curves as compact ASCII. |

```bash
bash egs/voice_isolate/scripts/run_valid.sh config/train_dpcrn_wide_antisup.yaml [device] [n_batches] [ckpt]
uv run python egs/voice_isolate/scripts/indomain_sisdri.py <config> --ckpt <ckpt> --device cpu --n-batches 40 --by-bucket --dump-distribution
```

## 3. Real-RIR WER benchmark (build once, eval any checkpoint)

Task-aligned real-acoustic near/far benchmark (BUT ReverbDB). Build the frozen set once, then eval any
checkpoint against it.

```bash
# 3a. build the frozen set (one-time)
uv run python egs/voice_isolate/scripts/build_but_wer_set.py --n-items 200 --out data_report/but_wer_set --seed 1234

# 3b. eval a checkpoint (repeat per checkpoint)
uv run python egs/voice_isolate/scripts/eval_but_wer.py config/eval_but_real.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
    --set-dir data_report/but_wer_set --device cuda
# stronger recognizer (lowers the reverb floor, reveals over-suppression whisper-small hides):
... --asr faster-whisper --asr-model large-v3
# Azure cloud STT (uv pip install azure-cognitiveservices-speech; SPEECH_KEY/SPEECH_REGION env):
SPEECH_KEY=... SPEECH_REGION=eastus ... --asr azure
```
`--asr` switches backend (`auto`/`faster-whisper`/`openai-whisper`/`azure`); cache/output filenames are
keyed by the actual backend, so switching never reuses a stale cache — **only compare WER within the
same backend** (enh-vs-mix, not across recognizers).

Over-suppression relief knobs (inference-only, default off): `--dry-blend a` (`out = a*enh + (1-a)*mix`,
`1.0` = off) and `--spec-floor f` (`|enh| >= f*|mix|` per bin, phase preserved, `0.0` = off; complex-mask
models only). Implemented in `siso.forward` (no-op by default, zero effect on training).

## 4. Dawn Chorus (reference metric) + ASR helper

`eval_dawn_chorus.py` — ai-coustics Dawn Chorus real-recording WER/SI-SDRi. **Note: Dawn's FG/BG is not
distance-defined, so it's cue-mismatched to this near/far task** — kept as a do-no-harm reference, not
the primary judge. Also exports `init_asr`/`wer_breakdown`, reused by `eval_but_wer.py`.

## 5. General diagnostics

`overfit_sanity.py` — small-batch memorization test: can the model learn to separate at all? Config-driven,
architecture-agnostic (use before trusting a new backbone/loss on a full run).

## 6. Inference / deployment

| script | purpose |
|---|---|
| `demo.py` | Gradio offline-enhance demo. Scans `trainer.work_folder` in the config for checkpoints (`config/infer_dpcrn.yaml` points at `../pretrained_ckpt/`, so all 6 pipeline checkpoints show up in the dropdown). |
| `streaming_onnx.py` | DPCRN streaming ONNX: `export` (config+ckpt → per-frame ONNX + manifest), `infer` (run on a wav), `benchmark` (real-time factor), `verify` (offline-vs-streaming parity, aligned + trimmed by the model's algorithmic latency — see `../pretrained_ckpt/README.md`). |

```bash
uv run python egs/voice_isolate/scripts/demo.py --config_path egs/voice_isolate/config/infer_dpcrn.yaml

uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_wide_antisup_ep19.onnx
```
