# scripts — voice_isolate tooling

Names follow what the script does:

| prefix | meaning |
|---|---|
| `build_*` | build a dataset, test set or pool once, then reuse it |
| `check_*` | sanity-check an input or a run before trusting it |
| `eval_*`  | score a checkpoint on one axis |
| `dump_*`  | print/export raw information |

Everything is invoked from the repo root unless noted. Eval scripts default to `--device cpu` so they
never fight the training GPU, and all of them accept the released inference knobs where applicable:
`--dry-blend a` (`out = a*enh + (1-a)*mix`, `1.0` = off) and `--spec-floor f` (`|enh| >= f*|mix|` per
bin, phase preserved, `0.0` = off; complex-mask models only). Both are inference-only and implemented
in `siso.forward`, so they have zero effect on training. The released `dpcrn_v8` checkpoint is scored
**with `--dry-blend 0.9`** — see `../pretrained_ckpt/README.md`.

## 1. Data prep

| script | purpose |
|---|---|
| `split_by_speaker.py` | Speaker-disjoint train/dev split for a metafile (regroups by `spkid` so dev speakers never leak into train). |
| `build_real_recording_pool.py` | Index real distant recordings (VOiCES) into the pool manifests the training pipeline draws from: far interferers (`--min-distance 1.0`) and real near-field keep rows (`--min-distance 0 --max-distance 1.0`). Emits finished waveforms, not RIRs, and splits train/held-out by speaker. |
| `check_training_data.py` | What a recipe will actually train on: manifest speaker-disjointness + path existence, near/far DRR-gap and distance/RT60 distributions measured on real sampled items, realized target-to-residual ratio, and a wav dump for listening. |

```bash
uv run python egs/voice_isolate/scripts/split_by_speaker.py egs/voice_isolate/data/dns5-read.list \
    --train-out egs/voice_isolate/data/dns5-read.train.list --dev-out egs/voice_isolate/data/dns5-read.dev.list \
    --dev-speaker-frac 0.05 --seed 0

uv run python egs/voice_isolate/scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 1.0 \
    --out egs/voice_isolate/data/realfar_pool/voices.train.jsonl

uv run python egs/voice_isolate/scripts/check_training_data.py \
    egs/voice_isolate/config/train_dpcrn.yaml --n 64 --dump 8
```

## 2. Training-time checks

| script | purpose |
|---|---|
| `check_training_run.sh` | One-click: finds the run's latest checkpoint, prints loss curves, runs the in-domain metric, prints PASS / MARGINAL / FAIL. |
| `eval_indomain.py` | **Primary judge metric.** Reuses the training/valid synthesis pipeline and scores enhanced vs early-reverb target SI-SDRi; `--by-bucket` breaks out counter_level / 1N+0F / overlap / distance buckets (read the hard buckets, not the aggregate). |
| `overfit_check.py` | Can the pipeline learn at all? Overfits one fixed batch with the real training loss (`--pick-hardest-of N` avoids batches where passthrough already wins). `--gate` instead freezes the separator and fits only the frame-level gate head. Plumbing test, not a generalization test. |
| `dump_loss_curves.py` | Prints tensorboard scalar curves as compact ASCII. |

```bash
bash egs/voice_isolate/scripts/check_training_run.sh config/train_dpcrn.yaml [device] [n_batches] [ckpt]
uv run python egs/voice_isolate/scripts/eval_indomain.py <config> --ckpt <ckpt> \
    --device cpu --n-batches 40 --by-bucket --dump-distribution
uv run python egs/voice_isolate/scripts/overfit_check.py \
    egs/voice_isolate/config/train_dpcrn.yaml --steps 800 --device cuda
```

## 3. Real-acoustic WER (build once, eval any checkpoint)

Task-aligned near/far benchmark on measured RIRs (BUT ReverbDB), with LibriTTS ground-truth
transcripts, so WER is not whisper-vs-whisper.

```bash
# 3a. build the frozen set (one-time)
uv run python egs/voice_isolate/scripts/build_wer_set.py --n-items 200 \
    --out data_report/but_wer_set --seed 1234

# 3b. score a checkpoint (repeat per checkpoint)
uv run python egs/voice_isolate/scripts/eval_wer.py config/exp/eval_but_real.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --set-dir data_report/but_wer_set --device cuda --asr faster-whisper --asr-model large-v3
# Azure cloud STT (uv pip install azure-cognitiveservices-speech; SPEECH_KEY/SPEECH_REGION env):
SPEECH_KEY=... SPEECH_REGION=eastus ... --asr azure
```

`--asr` switches backend (`auto`/`faster-whisper`/`openai-whisper`/`azure`); cache and output filenames
are keyed by the backend, so switching never reuses a stale cache — **only compare WER within the same
backend** (enh-vs-mix, not across recognizers). A weak recognizer hides over-suppression, so judge
deletion with a strong one.

## 4. Dawn Chorus (reference metric) + ASR helper

`eval_dawn_chorus.py` — Dawn Chorus real-recording WER/SI-SDRi. **Note: its FG/BG split is not
distance-defined, so it is cue-mismatched to this near/far task** — kept as a do-no-harm reference,
not the primary judge. Also exports `init_asr`/`wer_breakdown`, reused by `eval_wer.py`.

## 5. Real-recording behaviour (the deployment gates)

Simulated far-only probes over-estimate far-field suppression by more than 20 dB, and in-domain
metrics cannot see the keep-side failure at all. These two measure both sides on real recordings.

| script | purpose |
|---|---|
| `eval_far_suppression.py` | Suppress side: how much real far-field speech is suppressed, by distance. `--corpus voices` buckets by (room, mic) with spans from the recording's own energy; `--corpus realman` buckets by annotated distance with spans from the sample-aligned direct-path reference. Want very negative beyond 1 m, ~0 dB within 1 m. |
| `eval_keep_robustness.py` | Keep side: how much near speech survives as its capture chain moves away from the training one (dry → simulated RIR → measured RIR → real recording), with and without noise. A steep ladder means the keep decision is anchored on the capture chain instead of on distance. |
| `eval_realcase.py` | Hand-annotated keep/suppress spans on real clips (`windows.json`), scoring both failure directions separately. Scores a second system's output alongside when the case dir ships one. |
| `eval_turntaking.py` | Same two-sided scorecard, spans derived automatically from the target energy of a frozen turn-taking set. |
| `build_turntaking_set.py` | Builds that frozen (mix, target) turn-taking set; `--rir-folder` swaps in a measured-RIR bank for real-room turns. |
| `eval_gate.py` | Frame-level gate-head scorecard (recall / specificity / balanced accuracy / BCE) on a held-out simulated set — the only view that shows gate progress, since gate training leaves the mask path untouched. |

`eval_realcase.py` and `eval_turntaking.py` take `--gate` to apply a trained gate head to the output as
a per-frame gain, adding `gate_soft` / `gate_hard` rows next to the mask-only one.

```bash
uv run python egs/voice_isolate/scripts/eval_far_suppression.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --corpus voices --voices-root /path/to/VOiCES --device cuda --per-bucket 40

uv run python egs/voice_isolate/scripts/eval_keep_robustness.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt v6=egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt \
    --ckpt v7=egs/voice_isolate/pretrained_ckpt/dpcrn_v7.ckpt \
    --ckpt v8=egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --device cuda

uv run python egs/voice_isolate/scripts/eval_realcase.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --cases-dir egs/voice_isolate/data_report/qvf22_real_cases --device cpu
```

All of the above (plus the in-domain and WER stages) run in one shot per checkpoint:

```bash
cd egs/voice_isolate && bash run_full_benchmark.sh <ckpt> <tag> [device] [dry_blend]
```

## 6. Inference / deployment

| script | purpose |
|---|---|
| `demo.py` | Gradio demo: offline enhance, checkpoint dropdown (scans `trainer.work_folder` from `config/infer_dpcrn.yaml` → `../pretrained_ckpt/`, so every version and the exported `streaming/*.onnx` show up), knob sliders, spectrograms. |
| `streaming_onnx.py` | `export` / `infer` / `benchmark` / `verify` for the per-frame streaming ONNX. `verify` aligns by the reported algorithmic latency before scoring — a comparison that skips this reads the 30 ms delay as error. |

```bash
uv run python egs/voice_isolate/scripts/demo.py --config_path egs/voice_isolate/config/infer_dpcrn.yaml

uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.json --provider cpu
```

## Renamed / merged (older logs use the old names)

| old | now |
|---|---|
| `indomain_sisdri.py` | `eval_indomain.py` |
| `eval_but_wer.py` / `build_but_wer_set.py` | `eval_wer.py` / `build_wer_set.py` |
| `eval_gate_vad.py` | `eval_gate.py` |
| `validate_data.py` | `check_training_data.py` (rewritten: measures the sampled items, not a re-derived simulator) |
| `run_valid.sh` | `check_training_run.sh` |
| `dump_tb.py` | `dump_loss_curves.py` |
| `dump_turntaking_samples.py` | `build_turntaking_set.py` |
| `build_realfar_pool.py` | `build_real_recording_pool.py` |
| `probe_channel_keep.py` | `eval_keep_robustness.py` |
| `probe_retransmitted_farfield.py` + `probe_realman_farfield.py` | `eval_far_suppression.py --corpus {voices,realman}` |
| `eval_realcase_faronly.py` + `eval_realcase_gated.py` | `eval_realcase.py [--gate]` |
| `eval_turntaking_set.py` + `eval_turntaking_gated.py` | `eval_turntaking.py [--gate]` |
| `overfit_sanity.py` + `overfit_gate_sanity.py` | `overfit_check.py [--gate]` |
| `bench_qvf22_realcases.py` | removed — no-reference DNSMOS/RMS comparison on the same clips `eval_realcase.py` scores directly, and DNSMOS does not react to far-field leakage |
