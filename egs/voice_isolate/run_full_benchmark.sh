#!/usr/bin/env bash
# Full Phase-1 benchmark for one checkpoint. Run from the recipe dir (metafiles
# are relative to it). Training must be stopped first (this uses the GPU).
#   bash run_full_benchmark.sh <ckpt> <tag> [device] [dry_blend]
# dry_blend is the released inference knob (1.0 = off); pass 0.9 to benchmark a
# checkpoint the way it is deployed.
# Results (one log per stage) land in $OUT; a compact summary is printed at the end.
set -u
cd "$(cd "$(dirname "$0")" && pwd)"   # recipe dir

CKPT="${1:?usage: run_full_benchmark.sh <ckpt> <tag> [device]}"
TAG="${2:?need a tag, e.g. boundary_ep39}"
DEV="${3:-cuda}"
BLEND="${4:-1.0}"
SD="${BENCH_SD:-${TMPDIR:-/tmp}}"
OUT="$SD/bench_$TAG"; mkdir -p "$OUT"
ASR=faster-whisper; ASR_MODEL=large-v3   # strong ASR reveals over-suppression whisper-small hides
say(){ echo "[bench $(date +%H:%M:%S)] $*"; }

say "1/9 real-clip scorecard (voicebot gate)"
uv run python scripts/eval_realcase.py config/infer_dpcrn.yaml \
  --ckpt "$CKPT" --cases-dir data_report/qvf22_real_cases --device cpu \
  --dry-blend "$BLEND" > "$OUT/1_scorecard.log" 2>&1

say "2/9 in-domain SI-SDRi + buckets + solo-leakage (phase1 bank)"
uv run python scripts/eval_indomain.py config/exp/eval_indomain_phase1.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 80 --by-bucket --dump-distribution > "$OUT/2_indomain.log" 2>&1

say "3/9 synthetic far-only probe (expand bank, seen distances)"
uv run python scripts/eval_indomain.py config/exp/eval_targetabsent_probe.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 60 --by-bucket > "$OUT/3_probe_expand.log" 2>&1

say "4/9 synthetic far-only probe (high bank rt60 0.85-1.5, UNSEEN reverb)"
uv run python scripts/eval_indomain.py config/exp/eval_targetabsent_probe_high.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 60 --by-bucket > "$OUT/4_probe_high.log" 2>&1

say "5/9 synthetic far-only probe (boundary held-out, UNSEEN boundary distances)"
uv run python scripts/eval_indomain.py config/exp/eval_targetabsent_probe_boundary.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 60 --by-bucket > "$OUT/5_probe_boundary.log" 2>&1

say "6/9 Dawn Chorus real WER (over-suppression deletion guardrail; $ASR/$ASR_MODEL)"
uv run python scripts/eval_dawn_chorus.py config/infer_dpcrn.yaml \
  --ckpt "$CKPT" --device "$DEV" --asr "$ASR" --asr-model "$ASR_MODEL" \
  --dry-blend "$BLEND" > "$OUT/6_dawn_wer.log" 2>&1

say "7/9 BUT-OFFICE real-RIR WER (PRIMARY deployment gate, RT30 0.56-0.69; $ASR/$ASR_MODEL)"
uv run python scripts/eval_wer.py config/exp/eval_but_real.yaml \
  --ckpt "$CKPT" --set-dir data_report/but_wer_set_office --device "$DEV" \
  --asr "$ASR" --asr-model "$ASR_MODEL" --dry-blend "$BLEND" > "$OUT/7_but_office_wer.log" 2>&1

say "8/9 BUT high/extreme-reverb WER (secondary extreme-OOD do-no-harm MONITOR, RT30 1.15-1.84; $ASR/$ASR_MODEL)"
uv run python scripts/eval_wer.py config/exp/eval_but_real.yaml \
  --ckpt "$CKPT" --set-dir data_report/but_wer_set --device "$DEV" \
  --asr "$ASR" --asr-model "$ASR_MODEL" --dry-blend "$BLEND" > "$OUT/8_but_reverb_wer.log" 2>&1

# real-RIR turn-taking keep/suppress scorecard (far-solo suppression specialty).
# Frozen set dumped with the BUT office real-RIR bank; regenerate via:
#   scripts/build_turntaking_set.py ... --rir-folder exp/but_real_rir_16k_office
TT_SET="${TT_SET:-/data/audio/eval_noisy_data/turntaking_set_realrir}"
say "9/9 real-RIR turn-taking scorecard (KEEP near / SUPPRESS far-solo; $TT_SET)"
uv run python scripts/eval_turntaking.py config/infer_dpcrn.yaml \
  --ckpt "$CKPT" --set-dir "$TT_SET" --device "$DEV" \
  --dry-blend "$BLEND" > "$OUT/9_turntaking.log" 2>&1

echo; echo "================ BENCHMARK SUMMARY [$TAG]  (dry_blend=$BLEND) ================"
echo "--- real scorecard ---";        grep -E "scenario|# (ours|gate_|reference):" "$OUT/1_scorecard.log" 2>/dev/null
echo "--- in-domain ---";             grep -E "SI-SDRi :|1N\+0F|F-only|median power|interferer-solo leakage|by turn_taking" -A1 "$OUT/2_indomain.log" 2>/dev/null | grep -vE "^--$"
echo "--- probe expand/high/boundary (F-only median) ---"
for f in 3_probe_expand 4_probe_high 5_probe_boundary; do echo -n "$f: "; grep "median power reduction" "$OUT/$f.log" 2>/dev/null | tail -1; done
echo "--- Dawn WER (deletion guardrail) ---"; grep -iE "WER|deletion|insert|substit|SI-SDR" "$OUT/6_dawn_wer.log" 2>/dev/null | tail -6
echo "--- BUT-OFFICE WER (PRIMARY deployment gate) ---"; grep -iE "WER|deletion|insert|substit|SI-SDR" "$OUT/7_but_office_wer.log" 2>/dev/null | tail -6
echo "--- BUT reverb WER (secondary extreme-OOD monitor) ---"; grep -iE "WER|deletion|insert|substit" "$OUT/8_but_reverb_wer.log" 2>/dev/null | tail -4
echo "--- real-RIR turn-taking (KEEP near / SUPPRESS far-solo) ---"; grep -E "^(system|ours|gate_)" "$OUT/9_turntaking.log" 2>/dev/null
echo "logs: $OUT"
