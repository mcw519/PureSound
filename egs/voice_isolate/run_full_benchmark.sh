#!/usr/bin/env bash
# Full Phase-1 benchmark for one checkpoint. Run from the recipe dir (metafiles
# are relative to it). Training must be stopped first (this uses the GPU).
#   bash run_full_benchmark.sh <ckpt> <tag> [device] [dry_blend] [presence_readout]
# dry_blend is the released inference knob (1.0 = off); pass 0.9 to benchmark a
# checkpoint the way it is deployed.
# presence_readout turns on the inference-only near-presence gain (EXPERIMENTAL,
# see benchmarks/probes/b_traj_README.md). Omitted, every stage is unchanged. Its
# operating point comes from the readout's sibling .json; GATE_EXTRA in the
# environment appends per-run overrides (--gate-b-hi 0.75, ...).
# Results (one log per stage) land in $OUT; a compact summary is printed at the end.
set -u
cd "$(cd "$(dirname "$0")" && pwd)"   # recipe dir

CKPT="${1:?usage: run_full_benchmark.sh <ckpt> <tag> [device]}"
TAG="${2:?need a tag, e.g. boundary_ep39}"
DEV="${3:-cuda}"
BLEND="${4:-1.0}"
READOUT="${5:-}"
GATE_ARGS=""; GATE_TAG="off"
if [ -n "$READOUT" ]; then
  GATE_ARGS="--presence-gate $READOUT ${GATE_EXTRA:-}"
  GATE_TAG="$(basename "$READOUT")${GATE_EXTRA:+ $GATE_EXTRA}"
fi
SD="${BENCH_SD:-${TMPDIR:-/tmp}}"
OUT="$SD/bench_$TAG"; mkdir -p "$OUT"

# Synthesis-chain provenance, stamped into the summary so a record says which
# chain produced it. Stages 2-5 synthesise their audio at eval time through
# puresound's device chain, so their numbers only compare against records made
# on the same chain -- 9c56e02 (2026-08-18) made the analogue path linear and
# changed ~9% of rows. Stages 1, 6, 7 and 8 read fixed audio off disk and are
# unaffected. See benchmarks/README.md.
CHAIN="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[ -z "$(git status --porcelain -- ../../puresound 2>/dev/null)" ] || CHAIN="$CHAIN+dirty"
ASR=faster-whisper; ASR_MODEL=large-v3   # strong ASR reveals over-suppression whisper-small hides
say(){ echo "[bench $(date +%H:%M:%S)] $*"; }

# ONE set since the v3 rebuild (2026-08-20): the QVF publication clips moved into
# the field benchmark, so the cross-chain reference and the STREAM/COLD-START modes
# are scored in a single pass. The 134.8 s 90D session needs expandable_segments on
# a 24 GB card. 1_scorecard.log is kept as a symlink-free copy of the same output so
# older summary greps still find their file.
say "1/9 real-clip scorecard (voicebot gate): field benchmark incl. cross-chain reference"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python scripts/eval_realcase.py config/infer_dpcrn.yaml \
  --ckpt "$CKPT" --cases-dir data_report/field_cases/test_vector_cases --device "$DEV" \
  --dry-blend "$BLEND" $GATE_ARGS > "$OUT/1_scorecard_field.log" 2>&1
cp "$OUT/1_scorecard_field.log" "$OUT/1_scorecard.log"

say "2/9 in-domain SI-SDRi + buckets + solo-leakage (phase1 bank)"
uv run python scripts/eval_indomain.py config/exp/eval_indomain_phase1.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 80 --by-bucket --dump-distribution $GATE_ARGS > "$OUT/2_indomain.log" 2>&1

say "3/9 synthetic far-only probe (expand bank, seen distances)"
uv run python scripts/eval_indomain.py config/exp/eval_targetabsent_probe.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 60 --by-bucket $GATE_ARGS > "$OUT/3_probe_expand.log" 2>&1

say "4/9 synthetic far-only probe (high bank rt60 0.85-1.5, UNSEEN reverb)"
uv run python scripts/eval_indomain.py config/exp/eval_targetabsent_probe_high.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 60 --by-bucket $GATE_ARGS > "$OUT/4_probe_high.log" 2>&1

say "5/9 synthetic far-only probe (boundary held-out, UNSEEN boundary distances)"
uv run python scripts/eval_indomain.py config/exp/eval_targetabsent_probe_boundary.yaml \
  --ckpt "$CKPT" --device "$DEV" --n-batches 60 --by-bucket $GATE_ARGS > "$OUT/5_probe_boundary.log" 2>&1

say "6/9 Dawn Chorus real WER (over-suppression deletion guardrail; $ASR/$ASR_MODEL)"
uv run python scripts/eval_dawn_chorus.py config/infer_dpcrn.yaml \
  --ckpt "$CKPT" --device "$DEV" --asr "$ASR" --asr-model "$ASR_MODEL" \
  --dry-blend "$BLEND" $GATE_ARGS > "$OUT/6_dawn_wer.log" 2>&1

# PRIMARY WER gate: the deployment reverberation range, and the only WER set here whose
# resolution matches the differences between our checkpoints -- models capture ~40% of its
# headroom, so a version difference lands outside the bootstrap interval. BUT-OFFICE below
# is kept as a monitor, not a gate: at n=200 nothing we have is distinguishable from
# doing nothing on it (see benchmarks/wer_sets/README.md).
say "7a/9 moderate-reverb WER (PRIMARY deployment gate, RT60 0.20-0.65; $ASR/$ASR_MODEL)"
uv run python scripts/eval_wer.py config/exp/eval_but_real.yaml \
  --ckpt "$CKPT" --set-dir data_report/wer_set_moderate_test --device "$DEV" \
  --asr "$ASR" --asr-model "$ASR_MODEL" --dry-blend "$BLEND" $GATE_ARGS > "$OUT/7a_moderate_wer.log" 2>&1

say "7b/9 BUT-OFFICE real-RIR WER (measured-RIR MONITOR, RT30 0.56-0.69; $ASR/$ASR_MODEL)"
uv run python scripts/eval_wer.py config/exp/eval_but_real.yaml \
  --ckpt "$CKPT" --set-dir data_report/but_wer_set_office --device "$DEV" \
  --asr "$ASR" --asr-model "$ASR_MODEL" --dry-blend "$BLEND" $GATE_ARGS > "$OUT/7_but_office_wer.log" 2>&1

say "8/9 BUT high/extreme-reverb WER (secondary extreme-OOD do-no-harm MONITOR, RT30 1.15-1.84; $ASR/$ASR_MODEL)"
uv run python scripts/eval_wer.py config/exp/eval_but_real.yaml \
  --ckpt "$CKPT" --set-dir data_report/but_wer_set --device "$DEV" \
  --asr "$ASR" --asr-model "$ASR_MODEL" --dry-blend "$BLEND" $GATE_ARGS > "$OUT/8_but_reverb_wer.log" 2>&1

# real-RIR turn-taking keep/suppress scorecard (far-solo suppression specialty).
# Frozen set dumped with the BUT office real-RIR bank; regenerate via:
#   scripts/build_turntaking_set.py ... --rir-folder exp/but_real_rir_16k_office
TT_SET="${TT_SET:-/data/audio/eval_noisy_data/turntaking_set_realrir}"
say "9/9 real-RIR turn-taking scorecard (KEEP near / SUPPRESS far-solo; $TT_SET)"
uv run python scripts/eval_turntaking.py config/infer_dpcrn.yaml \
  --ckpt "$CKPT" --set-dir "$TT_SET" --device "$DEV" \
  --dry-blend "$BLEND" $GATE_ARGS > "$OUT/9_turntaking.log" 2>&1

echo; echo "================ BENCHMARK SUMMARY [$TAG]  (dry_blend=$BLEND, presence_gate=$GATE_TAG, chain=$CHAIN) ================"
echo "--- cross-chain reference (QVF2.2 clips) ---"; grep -E "qvf_(scenario|gym|price|keep|plumb)" "$OUT/1_scorecard_field.log" 2>/dev/null
echo "--- field scorecard (sessions + verdicts) ---"; grep -E "_session|# (ours|reference):" "$OUT/1_scorecard_field.log" 2>/dev/null
echo "    cold-start far verdicts:";  awk -F'\t' '/_far/ && $3=="ours" {v=$11; c[v]++} END {for (k in c) printf "      %-40s %d\n", k, c[k]}' "$OUT/1_scorecard_field.log" 2>/dev/null
echo "--- SYNTHESISED below (in-domain + 3 probes): produced by chain=$CHAIN, compare only against records on the same chain ---"
echo "--- in-domain ---";             grep -E "SI-SDRi :|1N\+0F|F-only|median power|interferer-solo leakage|by turn_taking" -A1 "$OUT/2_indomain.log" 2>/dev/null | grep -vE "^--$"
echo "--- probe expand/high/boundary (F-only median) ---"
for f in 3_probe_expand 4_probe_high 5_probe_boundary; do echo -n "$f: "; grep "median power reduction" "$OUT/$f.log" 2>/dev/null | tail -1; done
echo "--- Dawn WER (deletion guardrail) ---"; grep -iE "WER|deletion|insert|substit|SI-SDR" "$OUT/6_dawn_wer.log" 2>/dev/null | tail -6
echo "--- moderate-reverb WER (PRIMARY deployment gate) ---"; grep -iE "WER|delta|headroom|deletion|insert|substit|SI-SDR" "$OUT/7a_moderate_wer.log" 2>/dev/null | tail -10
echo "--- BUT-OFFICE WER (measured-RIR monitor; check the CI before reading a win) ---"; grep -iE "WER|delta|headroom|deletion|insert|substit|SI-SDR" "$OUT/7_but_office_wer.log" 2>/dev/null | tail -10
echo "--- BUT reverb WER (secondary extreme-OOD monitor) ---"; grep -iE "WER|deletion|insert|substit" "$OUT/8_but_reverb_wer.log" 2>/dev/null | tail -4
echo "--- real-RIR turn-taking (KEEP near / SUPPRESS far-solo) ---"; grep -E "^(system|ours|gate_)" "$OUT/9_turntaking.log" 2>/dev/null
echo "logs: $OUT"
