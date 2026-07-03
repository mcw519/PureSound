#!/usr/bin/env bash
# One-click training-result validation for voice_isolate.
#
# Given a recipe config, this finds the latest checkpoint of its run, prints the
# loss curves, and runs the in-domain SI-SDRi probe (the real "is the model
# actually separating?" metric) -- then prints a PASS / MARGINAL / FAIL verdict.
#
# Usage:
#   bash scripts/run_valid.sh config/train_dpcrn_wide_antisup.yaml [device] [n_batches] [ckpt]
#     device     : cuda (default) | cpu
#     n_batches  : in-domain eval batches (default 40)
#     ckpt       : explicit checkpoint path (default = latest in the run's work_folder)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$HERE")"            # .../egs/voice_isolate
cd "$RECIPE_DIR"

CONFIG="${1:?usage: run_valid.sh <config.yaml> [device] [n_batches] [ckpt]}"
DEVICE="${2:-cuda}"
NB="${3:-40}"
CKPT_OVERRIDE="${4:-}"

# accept config given as bare name, relative, or absolute
[ -f "$CONFIG" ] || CONFIG="config/$(basename "$CONFIG")"
[ -f "$CONFIG" ] || { echo "config not found: $1"; exit 1; }

WF=$(grep -E '^[[:space:]]*work_folder:' "$CONFIG" | head -1 | sed -E 's/.*work_folder:[[:space:]]*//; s/[[:space:]]*#.*//')
echo "config      : $CONFIG"
echo "work_folder : $WF"

if [ -n "$CKPT_OVERRIDE" ]; then
  CKPT="$CKPT_OVERRIDE"
else
  CKPT=$(ls -t "$WF"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
fi
if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "!! no checkpoint found under $WF/lightning_logs/version_*/checkpoints/"
  echo "   (has training produced a checkpoint yet?)"
  exit 1
fi
VDIR=$(dirname "$(dirname "$CKPT")")
echo "checkpoint  : $CKPT"
echo

echo "=========== LOSS CURVES (tensorboard) ==========="
uv run python scripts/dump_tb.py "$VDIR" --keep valid_step_loss epoch_train_loss --n_samples 12 2>/dev/null \
  | grep -E "valid_step_loss \(N|epoch_train_loss \(N|step=" | awk '!seen[$0]++' || echo "(no tb scalars yet)"
echo

echo "=========== IN-DOMAIN SI-SDRi (real separation metric) ==========="
uv run python scripts/indomain_sisdri.py "$CONFIG" --ckpt "$CKPT" \
  --device "$DEVICE" --n-batches "$NB" --dump-distribution 2>&1 | tee /tmp/_run_valid_out.txt
echo

echo "=========== VERDICT ==========="
MED=$(grep -aoE 'SI-SDRi : mean [+-][0-9.]+ dB, median [+-][0-9.]+' /tmp/_run_valid_out.txt | tail -1 | grep -oE 'median [+-][0-9.]+' | grep -oE '[+-][0-9.]+')
if [ -n "$MED" ]; then
  awk -v m="$MED" 'BEGIN{
    if (m+0 > 1.0)       print "PASS  : median in-domain SI-SDRi = " m " dB (>+1) -> model IS separating.";
    else if (m+0 > -1.0) print "MARGINAL: median SI-SDRi = " m " dB (~0) -> near passthrough; check loss still dropping / train longer.";
    else                 print "FAIL  : median SI-SDRi = " m " dB (<-1) -> passthrough / over-suppression. See FAILED_NOTE.md.";
  }'
else
  echo "could not parse median SI-SDRi from output above."
fi
