# voice_isolate — pretrained checkpoints

Near-field (<1 m) foreground voice isolation, single channel, no enrollment: keep the near
speaker, suppress far/competing speakers and noise.

Versions are numbered in training order — **`dpcrn_v7.ckpt` is the current default**. Every
version shares the same architecture (DPCRN, complex ratio mask, 16 kHz, ~0.8 M params,
30 ms look-ahead), so one inference config loads all of them:

```bash
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/infer_dpcrn.yaml     # dropdown lists every version
```

## Default: `dpcrn_v7.ckpt` + `dry_blend = 0.9`

The runtime blend is **part of the released configuration**, not an optional extra:

```python
enhanced = model(wav, dry_blend=0.9)      # out = 0.9 * enhanced + 0.1 * input
```

It bounds attenuation at any point to −20 dB. That costs a little residual interferer and
buys a large drop in deletions on capture chains outside the training data — with it the
model *lowers* ASR error on real recordings (Dawn Chorus WER 0.174 vs 0.184 unprocessed);
without it, the same checkpoint raises WER on one of the reverberant test sets. Eval scripts
expose `--dry-blend`; the streaming manifest carries the value under `recommended_inference`.

Known limitation: a blend of 0.9 cannot produce full silence (−20 dB floor by construction),
and suppression of far speech recorded through chains very unlike the training corpora is
shallower than in-domain. For hard muting, the mask itself reaches about −38 dB, but that
path needs a gate rather than a blend.

## Versions

| version | recipe | warm-start | what the stage adds | judged result |
|---|---|---|---|---|
| `dpcrn_v1.ckpt` | `config/exp/train_dpcrn_curriculum_core.yaml` | cold | curriculum core: RT60 0.20–0.45, near/far DRR gap ≥ 6 dB | in-domain SI-SDRi median **+6.98** (ep39) |
| `dpcrn_v2.ckpt` | `config/exp/train_dpcrn_curriculum_expand.yaml` | v1 | widened curriculum: RT60 ≤ 0.65, DRR gap ≥ 3 dB | in-domain **+7.99**; measured-RIR set +5.46; first real-WER win (BUT enh 0.777 < mix 0.796) (ep59) |
| `dpcrn_v3.ckpt` | `config/exp/train_dpcrn_antisup_w1.yaml` | v2 | `OverSuppressionLoss` weight 1.0 (anti-deletion) | in-domain **+8.21**; BUT deletion 0.308 → 0.291 (ep19) |
| `dpcrn_v4.ckpt` | `config/exp/train_dpcrn_antisup_w2.yaml` | v3 | anti-suppression weight 2.0 | in-domain **+8.32**; BUT deletion → 0.276; most even across domains (ep19) |
| `dpcrn_v5.ckpt` | `config/exp/train_dpcrn_antisup_w3.yaml` | v4 | anti-suppression weight 3.0 | in-domain **+8.45**; best in deployment-level reverb, worst in extreme reverb — the domain split point (ep19) |
| `dpcrn_v6.ckpt` | `config/exp/train_dpcrn_wide_antisup.yaml` | v5 | wide RIR domain (RT60 0.20–0.85) + capture realism (media-voice interferer, HPF) | held-out unseen-room **+8.06**; **first streaming-verified** version (see below) (ep19) |
| **`dpcrn_v7.ckpt`** | **`config/train_dpcrn.yaml`** | v6 | real recordings on both sides of the decision (real far interferers + real <1 m keep rows), turn-taking far-solo supervision, distance/DRR aux head, channel-perturbation mask consistency | held-out real far-field **−9.45 dB, graded by distance** (1–2 m −3 → 5 m+ −38; v6: −1.76 flat), near-field keep flat (−0.11), **Dawn WER 0.174 < 0.184 raw**, deletion 0.088 ≈ raw floor, reverberant-office WER −0.024 vs mix, in-domain +8.18 (ep19, with `dry_blend 0.9`) |

`dpcrn_v6_gate.ckpt` — off the main line: `config/exp/train_dpcrn_gate.yaml` freezes v6 and
trains only a causal frame-level near/far VAD gate head (98,689 params). It reaches 0.90+
balanced accuracy on simulated data but the gate does not close on real recordings, so it is
an engineering reference for the gate path, not a deployable model. Its separator weights are
identical to v6; only 10 BatchNorm running-statistic buffers drifted during that run, so its
mask output is v6's up to those buffers.

**Choosing a version.** Take `dpcrn_v7.ckpt` with `dry_blend 0.9`. `dpcrn_v6.ckpt` is the
fallback: it needs no runtime knob and is the most conservative on far-field suppression
(it largely passes far speech through). v1–v5 are the training-history stages, kept so any
stage can be re-judged or re-warm-started; they are not deployment candidates.

**Judging convention.** The scheduler (`CosineAnnealingWarmRestarts`, `T_0=20`) restarts every
20 epochs, so checkpoints are only comparable at the cosine troughs — ep19/ep39/ep59. Every
number above comes from a trough epoch.

## `streaming/` — per-frame ONNX exports

`dpcrn_v6.{onnx,json}` and `dpcrn_v7.{onnx,json}`, built with `../scripts/streaming_onnx.py
export`. Both carry a **30 ms (3-frame) algorithmic latency** from the look-ahead, handled by
future-buffering baked into the graph as extra state (`puresound/streaming/dpcrn.py`), and both
are verified against the offline model once aligned by that latency: v6 88–105 dB, v7 63 dB
SI-SDR. CPU RTF 0.43. Load with `puresound.streaming.StreamingDpcrnOrt` or the SDK's
manifest-driven `PureSoundStreamingRuntime` (`processor: stft_frame_ort`).

Any offline↔streaming comparison **must** align by the reported latency and trim the edges,
otherwise the delay reads as error; `streaming_onnx.py verify` does this:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v7.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v7.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v7.json --provider cpu
```

Streaming with v7 applies the blend on the output: mix the enhanced frame with the input
frame delayed by `streaming_delay_frames`. It costs no extra latency.

## Re-export / re-train

```bash
# streaming ONNX from any checkpoint here
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v7.ckpt /tmp/model.onnx

# run the default recipe, warm-starting from v6 as v7 did
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn.yaml --training \
    --pretrained_ckpt_path egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt
```

Checkpoints here are the judged troughs pulled out of the full training history under `../exp/`
(symlinked to a work volume, gitignored, keeps every epoch). Per-version measurements in
context: `../EXPERIMENT_LOG.md`.

## Filename history

Earlier logs and reports use the pre-versioning names:

| old name | version |
|---|---|
| `dpcrn_curriculum_core_ep39.ckpt` | `dpcrn_v1.ckpt` |
| `dpcrn_curriculum_expand_ep59.ckpt` | `dpcrn_v2.ckpt` |
| `dpcrn_antisup_w1_ep19.ckpt` | `dpcrn_v3.ckpt` |
| `dpcrn_antisup_w2_ep19.ckpt` | `dpcrn_v4.ckpt` |
| `dpcrn_antisup_w3_ep19.ckpt` | `dpcrn_v5.ckpt` |
| `dpcrn_wide_antisup_ep19.ckpt` | `dpcrn_v6.ckpt` |
| `dpcrn_gate_synth_ep7.ckpt` | `dpcrn_v6_gate.ckpt` |
| `dpcrn_realE2E_v2c_ep19.ckpt` | `dpcrn_v7.ckpt` |

Training-run directories keep their original names (`exp/dpcrn_wide_antisup_0702`,
`exp/dpcrn_realE2E_v2c_0722`, …).
