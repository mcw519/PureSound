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

`overfit_gate_sanity.py` — gate-only integration smoke: loads the real synthetic dataloader and
`wide-ep19`, captures one frozen DPCRN bottleneck batch, and verifies that only the causal gate head can
overfit its near-activity BCE. This validates plumbing only, not real-domain transfer.

```bash
uv run python egs/voice_isolate/scripts/overfit_gate_sanity.py \
    egs/voice_isolate/config/train_dpcrn_gate.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
    --device cpu --steps 50
```

## 6. Inference / deployment

| script | purpose |
|---|---|
| `demo.py` | Gradio demo, two tabs. **Offline / File**: upload or record a clip, enhance the whole file (PyTorch or ORT-streaming backend). **Realtime Mic**: live per-frame streaming through an `.onnx` model — press record and enhanced audio plays back live, with optional realtime STT (off by default). Both tabs scan `trainer.work_folder` for models (`config/infer_dpcrn.yaml` → `../pretrained_ckpt/`, so all pipeline checkpoints + the exported `pretrained_ckpt/streaming/*.onnx` show up). |
| `streaming_onnx.py` | DPCRN streaming ONNX: `export` (config+ckpt → per-frame ONNX + manifest), `infer` (run on a wav), `benchmark` (real-time factor), `verify` (offline-vs-streaming parity, aligned + trimmed by the model's algorithmic latency — see `../pretrained_ckpt/README.md`). |

```bash
uv run python egs/voice_isolate/scripts/demo.py --config_path egs/voice_isolate/config/infer_dpcrn.yaml

uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_wide_antisup_ep19.onnx
```

For the joint separator+gate checkpoint, use the gate training config so the demo
discovers the `exp/dpcrn_v2_sepgate` checkpoints. In the Offline / File tab, select
`Raw probability`, `Binary`, `Binary + EMA`, or `Binary + envelope` under
**Near-field gate**. Binary modes use the threshold slider; EMA and envelope
expose their smoothing parameters in the UI.
Gate application is currently supported by the PyTorch offline backend only.

```bash
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/train_dpcrn_v2_sepgate.yaml
```

**Realtime Mic tab notes.** ORT `.onnx` models only (the PyTorch backend does a whole-utterance
forward and can't stream). Each session gets its own `StreamingDparnOrt` runtime, so streaming state
(OLA + ONNX recurrent state) is per-session and reset on each new recording. Output latency = the
model's algorithmic look-ahead (`streaming_delay_frames`, ~30 ms for this recipe) plus the mic chunk
size (`stream_every`, 0.5 s). STT is **off by default**; when enabled it transcribes the enhanced audio
in fixed-length segments (default 4 s) via `faster-whisper` (or `openai-whisper`) with auto language
detection — use `tiny`/`base` for realtime, larger models lag.
