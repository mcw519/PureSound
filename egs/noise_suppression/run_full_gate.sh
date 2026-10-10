#!/usr/bin/env bash
# Full gate for one checkpoint. Run from the repository root.
#
#   bash egs/noise_suppression/run_full_gate.sh <tag> [ckpt] [recipe]
#   PRECOMPUTED_DIR=/path/to/enhanced bash egs/noise_suppression/run_full_gate.sh <tag>
#
# PRECOMPUTED_DIR scores audio another system already produced -- one file per
# input, named by the input file's stem -- through the same stages, so a
# third-party model lands on the same axis as our checkpoints. Preflight and the
# RTF stage do not apply (there is no model here to load or time) and are
# skipped; the record says so.
#
# With no checkpoint every stage scores the UNPROCESSED baseline alone. That is
# not a degenerate mode -- it is the reference every stage is read against, and
# running it is how you check the gate works before there is a model to gate.
#
# What each stage is and whether it can decide a release: benchmarks/stages.md.
# This file is a list of stages. Anything that computes lives in
# `puresound.evaluation`, so both recipes get the same protocol.
set -uo pipefail

TAG="${1:?usage: run_full_gate.sh <tag> [ckpt] [recipe]}"
CKPT="${2:-}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-}"
PRECOMPUTED_NAME="${PRECOMPUTED_NAME:-}"
if [ -n "$PRECOMPUTED_DIR" ] && [ -n "$CKPT" ]; then
  echo "PRECOMPUTED_DIR and a checkpoint are two systems; score them in two runs." >&2; exit 2
fi
[ -n "$PRECOMPUTED_DIR" ] && [ -z "$PRECOMPUTED_NAME" ] && PRECOMPUTED_NAME="$(basename "$PRECOMPUTED_DIR")"
RECIPE="${3:-egs/noise_suppression/config/dpcrn.yaml}"
DEVICE="${DEVICE:-cpu}"
BLEND="${DRY_BLEND:-1.0}"
# SKIP_WER=1 drops stage 4. The record then has no WER stage and says so; it is
# for iterating on the quality stages, not for deciding a release.
SKIP_WER="${SKIP_WER:-0}"

DATA="${NS_DATA:-egs/noise_suppression/data/dns5}"
TESTSET="${NS_TESTSET:-egs/noise_suppression/data_report/ns_testset}"
VCTK="${NS_VCTK:-egs/noise_suppression/data_report/vctk_demand_test}"
# The WER stage reads stage 2's cuts unless told otherwise. NS_WERSET points it
# at another transcript-bearing set (a harder one built with
# evaluation.tools.mix_paired_set); the record names the set it was scored on.
WERSET="${NS_WERSET:-$VCTK}"
# Stage 4b: the harder WER set (held-out DEMAND noise under LibriTTS
# test-clean, -10..5 dB; built by evaluation.tools.mix_paired_set). Skipped when
# the set is not on disk or SKIP_WER_HARD=1.
WERSET_HARD="${NS_WERSET_HARD:-egs/noise_suppression/data_report/libritts_demand_hard}"
SKIP_WER_HARD="${SKIP_WER_HARD:-0}"
DEVSET="${NS_DEVSET:-$DATA/dns5_devset.jsonl}"
RTF_BUDGET="${RTF_BUDGET:-0.5}"
# Scoring is CPU-bound and per-item, so it parallelises across cores. Default to
# half of them, which leaves a concurrent training run alive; the RTF stage is
# unaffected, it measures one core on purpose.
JOBS="${GATE_JOBS:-$(( $(nproc) / 2 ))}"
ASR_MODEL="${ASR_MODEL:-large-v3}"
ASR_DEVICE="${ASR_DEVICE:-cuda}"

OUT_BASE="${GATE_OUT:-${TMPDIR:-/tmp}}"
SAFE_TAG="${TAG//[^A-Za-z0-9_.-]/_}"
mkdir -p "$OUT_BASE"
OUT="$(mktemp -d "$OUT_BASE/gate_${SAFE_TAG}.XXXXXX")"

MODEL_ARGS=()
INFERENCE_ARGS=(--inference "dry_blend=$BLEND")
if [ -n "$CKPT" ]; then
  MODEL_ARGS=(--recipe "$RECIPE" --ckpt "$CKPT" --dry-blend "$BLEND")
elif [ -n "$PRECOMPUTED_DIR" ]; then
  MODEL_ARGS=(--precomputed "$PRECOMPUTED_DIR" --precomputed-name "$PRECOMPUTED_NAME")
  INFERENCE_ARGS=(--inference "precomputed=true")
fi
# Anything that puts a second system beside the baseline.
HAVE_SYSTEM=""
[ -n "$CKPT" ] || [ -n "$PRECOMPUTED_DIR" ] && HAVE_SYSTEM=1

say() { echo; echo "[gate $(date +%H:%M:%S)] $*"; }
run() { echo "+ $*"; "$@"; }
run_logged() {
  local log="$1"
  shift
  run "$@" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

if [ -n "$CKPT" ]; then
  say "0/5 preflight: the checkpoint must load WHOLE into every recipe below"
  run_logged "$OUT/0_preflight.log" uv run python \
      -m puresound.evaluation.tools.preflight --ckpt "$CKPT" "$RECIPE"
  # A stage that scores a partly untrained model reports a regression nobody can
  # explain, so nothing downstream is worth running.
  [ "$?" -eq 0 ] || { echo "ABORTED before stage 1."; exit 1; }
fi

STAGE_FAILED=0

say "1/5 frozen synthetic set: reference metrics (GATE: pesq_wb), by SNR band"
run_logged "$OUT/1_reference.log" uv run python -m puresound.evaluation.tools.reference \
    --set-dir "$TESTSET" --device "$DEVICE" --role gate --jobs "$JOBS" \
    --stage-name frozen_testset --out "$OUT/1_reference.json" \
    "${MODEL_ARGS[@]}" || STAGE_FAILED=1

# The externally comparable one: VCTK-DEMAND is the set most of the literature
# reports PESQ on, and it is real recorded noise sitting on disk -- so unlike
# stage 1 it does not move when our own synthesis chain does.
say "2/5 VCTK-DEMAND: reference metrics (GATE: pesq_wb), by SNR band"
run_logged "$OUT/2_vctk.log" uv run python -m puresound.evaluation.tools.reference \
    --set-dir "$VCTK" --device "$DEVICE" --role gate --jobs "$JOBS" \
    --stage-name vctk_demand --out "$OUT/2_vctk.json" \
    "${MODEL_ARGS[@]}" || STAGE_FAILED=1

# Stage 3 runs the model on CPU whatever DEVICE says. The dev-set clips are long
# enough that one forward wants ~4 GiB of GPU memory, so DEVICE=cuda with any
# useful worker count runs out of memory here; the model forward is a small share
# of this stage next to DNSMOS's own ONNX sessions.
say "3/5 DNS-5 dev set: DNSMOS P.835 (MONITOR, no clean reference), by category"
run_logged "$OUT/3_dnsmos.log" uv run python -m puresound.evaluation.tools.noreference \
    --inventory "$DEVSET" --device cpu --role monitor --jobs "$JOBS" \
    --tag category --tag device \
    --stage-name dns_devset --out "$OUT/3_dnsmos.json" \
    "${MODEL_ARGS[@]}" || STAGE_FAILED=1

# The guardrail the quality stages cannot cover: a model can hold its PESQ while
# removing words. Scored against the corpus transcript with a large recogniser --
# a small one hides exactly this.
if [ "$SKIP_WER" = "1" ]; then
  say "4/5 WER: SKIPPED (SKIP_WER=1)"
else
  say "4/5 WER + deletions on $(basename "$WERSET") (GATE; $ASR_MODEL)"
  run_logged "$OUT/4_wer.log" uv run python -m puresound.evaluation.tools.wer \
      --set-dir "$WERSET" --device "$DEVICE" --role gate \
      --asr-model "$ASR_MODEL" --asr-device "$ASR_DEVICE" \
      --stage-name wer --out "$OUT/4_wer.json" \
      "${MODEL_ARGS[@]}" || STAGE_FAILED=1
fi

RUN_WER_HARD=""
if [ "$SKIP_WER" = "1" ] || [ "$SKIP_WER_HARD" = "1" ]; then
  say "4b/5 hard WER set: SKIPPED"
elif [ ! -f "$WERSET_HARD/manifest.jsonl" ]; then
  say "4b/5 hard WER set: SKIPPED ($WERSET_HARD not built)"
else
  RUN_WER_HARD=1
  # Per-noise and per-SNR bands, and every hypothesis on disk: the recogniser
  # loops on a few items, and a delta is only read after checking for them.
  say "4b/5 WER + deletions on $(basename "$WERSET_HARD") (GATE on deletions; $ASR_MODEL)"
  run_logged "$OUT/4b_wer_hard.log" uv run python -m puresound.evaluation.tools.wer \
      --set-dir "$WERSET_HARD" --device "$DEVICE" --role gate \
      --asr-model "$ASR_MODEL" --asr-device "$ASR_DEVICE" \
      --band snr_band --band noise --hypotheses "$OUT/4b_wer_hard_hyp.jsonl" \
      --stage-name wer_hard --out "$OUT/4b_wer_hard.json" \
      "${MODEL_ARGS[@]}" || STAGE_FAILED=1
fi

if [ -n "$PRECOMPUTED_DIR" ]; then
  say "5/5 CPU real-time factor: SKIPPED (precomputed output has no model to time)"
else
  say "5/5 CPU real-time factor (GATE, budget $RTF_BUDGET)"
  run_logged "$OUT/5_rtf.log" uv run python -m puresound.evaluation.tools.rtf \
      --budget "$RTF_BUDGET" --device cpu --threads 1 \
      --out "$OUT/5_rtf.json" "${MODEL_ARGS[@]}" || STAGE_FAILED=1
fi

say "collecting the record"
STAGE_FILES=()
REQUIRED_STAGE_ARGS=()
if [ -z "$PRECOMPUTED_DIR" ]; then
  STAGE_FILES+=("$OUT/5_rtf.json")
  REQUIRED_STAGE_ARGS+=(--require-stage cpu_rtf)
fi
if [ -n "$HAVE_SYSTEM" ]; then
  STAGE_FILES+=(
    "$OUT/1_reference.json" "$OUT/2_vctk.json" "$OUT/3_dnsmos.json"
  )
  REQUIRED_STAGE_ARGS+=(
    --require-stage frozen_testset.pesq_wb
    --require-stage vctk_demand.pesq_wb
    --require-stage dns_devset.dnsmos_ovr
  )
  if [ "$SKIP_WER" != "1" ]; then
    STAGE_FILES+=("$OUT/4_wer.json")
    REQUIRED_STAGE_ARGS+=(--require-stage wer.del)
  fi
  if [ -n "$RUN_WER_HARD" ]; then
    STAGE_FILES+=("$OUT/4b_wer_hard.json")
    REQUIRED_STAGE_ARGS+=(--require-stage wer_hard.del)
  fi
fi

# A precomputed system has no recipe; the record says so instead of naming the directory.
RECORD_RECIPE="$RECIPE"
[ -z "$PRECOMPUTED_DIR" ] || RECORD_RECIPE="precomputed"

run_logged "$OUT/record.log" uv run python -m puresound.evaluation.tools.collect \
    --tag "$TAG" --checkpoint "${CKPT:-${PRECOMPUTED_DIR:-unprocessed}}" --recipe "$RECORD_RECIPE" \
    "${INFERENCE_ARGS[@]}" \
    --out "egs/noise_suppression/benchmarks/records/$TAG.json" \
    "${REQUIRED_STAGE_ARGS[@]}" "${STAGE_FILES[@]}"
COLLECT_STATUS=$?

echo
echo "logs: $OUT"
if [ "$STAGE_FAILED" -ne 0 ] || [ "$COLLECT_STATUS" -ne 0 ]; then
  exit 1
fi
