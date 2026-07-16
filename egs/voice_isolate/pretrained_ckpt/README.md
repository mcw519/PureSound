# voice_isolate — pretrained checkpoints

One judged checkpoint per completed pipeline stage, pulled out of the full training history under
`../exp/` (symlinked to `/work/any_exp_link/puresound_exp`, gitignored, keeps every epoch) for easy
loading. Full lineage/rationale/results: `../README.md`.

All 6 share the **same architecture** — load any of them with `../config/infer_dpcrn.yaml`.

| checkpoint | pipeline stage | warm-started from | judged result (headline) |
|---|---|---|---|
| `dpcrn_curriculum_core_ep39.ckpt`   | 1/6 curriculum core   | cold | in-domain SI-SDRi median +6.98 |
| `dpcrn_curriculum_expand_ep59.ckpt` | 2/6 curriculum expand | stage 1 | in-domain +7.99, BUT real-RIR +5.46, first non-neutral real WER win (0.777<0.796) |
| `dpcrn_antisup_w1_ep19.ckpt`        | 3/6 anti-sup weight 1.0 | stage 2 | in-domain +8.21; BUT large-v3 deletion 0.308→0.291 |
| `dpcrn_antisup_w2_ep19.ckpt`        | 4/6 anti-sup weight 2.0 | stage 3 | in-domain +8.32; BUT deletion →0.276, safest bucket-by-bucket |
| `dpcrn_antisup_w3_ep19.ckpt`        | 5/6 anti-sup weight 3.0 | stage 4 | in-domain +8.45; **best in deployment-reverb domain**, worst in extreme-OOD (BUT) — the "domain split point" (see `../README.md`) |
| `dpcrn_wide_antisup_ep19.ckpt`      | 6/6 wide-domain deployment (**current best deployment candidate**) | stage 5 | widened RIR domain (rt60 0.20–0.85) + realism augs (media_voice/hpf); 4/5 judge gates passed; **streaming-verified** (see below) |

All post-wide synthetic rungs (boundary, realfar, gate-only `dpcrn_gate_synth_ep7.ckpt`, joint
sepgate `exp/dpcrn_v2_sepgate` ep19) were judged **negative on real end-to-end recordings** and are
closed — wide-ep19 stays the deployment candidate. Their artifacts are kept here / under `exp/` only
for reproducibility of the negative results (details: `EXPERIMENT_LOG.md` 2026-07-04 → 2026-07-16).

## `streaming/` — ONNX streaming export

`dpcrn_wide_antisup_ep19.{onnx,json}` — per-frame streaming export of the stage-6 checkpoint, built with
`../scripts/streaming_onnx.py export`. Carries a **30 ms (3-frame) algorithmic latency** from the
look-ahead (handled by future-buffering baked into the graph as extra state — see
`puresound/streaming/dpcrn.py`); verified bit-exact vs the offline model once aligned by that latency
(SI-SDR 88–105 dB in `scripts/streaming_onnx.py verify`). Load with
`puresound.streaming.StreamingDpcrnOrt` or the SDK's `PureSoundStreamingRuntime` (manifest-driven,
`processor: stft_frame_ort` — no runtime code changes needed for DPCRN).

## Usage

```bash
# offline enhance (gradio demo)
uv run python egs/voice_isolate/scripts/demo.py --config_path egs/voice_isolate/config/infer_dpcrn.yaml

# re-export streaming ONNX from any checkpoint here
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
    /tmp/model.onnx
```
