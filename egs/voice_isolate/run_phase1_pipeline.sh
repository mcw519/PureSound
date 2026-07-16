#!/usr/bin/env bash
# Phase-1 auto-pipeline: waits for the boundary RIR bank to finish generating,
# then chains view-build -> phase1 merge -> sample dump -> warm-start training.
# Designed to be launched in tmux and left alone; each step logs and aborts loudly
# rather than training on incomplete data.
#
#   tmux new-window -t rirgen -n train
#   tmux send-keys -t rirgen:train 'bash egs/voice_isolate/run_phase1_pipeline.sh' Enter
set -u
RECIPE="$(cd "$(dirname "$0")" && pwd)"   # egs/voice_isolate (metafiles are relative to here)
cd "$RECIPE/../.."                        # -> repo root for the rir_generation/merge steps

WORK=/work/any_exp_link/puresound_exp
MAIN_BANK=$WORK/hybrid_rir_16k_boundary
WIDE_VIEW=$WORK/hybrid_rir_16k_levels/wide
BND_VIEW=$WORK/hybrid_rir_16k_boundary_levels/all
PHASE1=$WORK/hybrid_rir_16k_phase1
CKPT=egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt
CONFIG=egs/voice_isolate/config/train_dpcrn_boundary.yaml
TARGET_ITEMS=10000
SD=/var/tmp/claude-1001/-home-milowu-A4Audio-PureSound-egs-voice-isolate/d11cb1c7-9004-43a1-955a-84d3cd8862b7/scratchpad

say() { echo "[phase1 $(date +%H:%M:%S)] $*"; }
abort() { echo "[phase1 ABORT] $*" >&2; exit 1; }

# 1. Gate on main bank completion (count reaches target; abort if the generator
#    died early so we never train on a half-built bank).
say "waiting for main bank ($MAIN_BANK) to reach $TARGET_ITEMS items..."
while true; do
  n=$(find "$MAIN_BANK" -name '*.wav' 2>/dev/null | wc -l)
  if [ "$n" -ge "$TARGET_ITEMS" ]; then say "main bank complete: $n items"; break; fi
  if ! pgrep -f "hybrid_rir_16k_boundary --n-rooms" >/dev/null; then
    abort "generator not running and only $n/$TARGET_ITEMS items -- bank incomplete"
  fi
  sleep 120
done

# 2. Build the main bank's 'all' view (unfiltered -- low DRR gap is wanted here).
say "building main bank 'all' view..."
if [ ! -d "$WORK/hybrid_rir_16k_boundary_levels" ]; then
  uv run python egs/rir_generation/filter_rir_levels.py \
    "$MAIN_BANK" "$WORK/hybrid_rir_16k_boundary_levels" --workers 20 \
    > "$SD/boundary_view.log" 2>&1 || abort "filter_rir_levels failed (see boundary_view.log)"
fi
bn=$(ls "$BND_VIEW/items"/*.wav 2>/dev/null | wc -l)
[ "$bn" -gt 0 ] || abort "boundary 'all' view empty"
say "boundary view: $bn items"

# 3. Merge wide + boundary into the phase1 training view.
say "merging phase1 view (wide + boundary)..."
[ -d "$PHASE1" ] && abort "$PHASE1 already exists (refusing to overwrite)"
uv run python egs/rir_generation/merge_rir_views.py \
  --source wide="$WIDE_VIEW" --source bnd="$BND_VIEW" --output "$PHASE1" \
  > "$SD/phase1_merge.log" 2>&1 || abort "merge_rir_views failed (see phase1_merge.log)"
pn=$(find "$PHASE1/items" -name '*.wav' 2>/dev/null | wc -l)   # find, not ls glob (105k args)
[ "$pn" -gt 0 ] || abort "phase1 view empty after merge"
say "phase1 view: $pn items"

# 4. Dump turn-taking samples for later listening (best-effort, non-blocking).
#    main.py and the dump script resolve metafiles relative to the RECIPE dir, so
#    run them from there with recipe-relative config/ckpt paths.
say "dumping turn-taking listen samples..."
( cd "$RECIPE" && uv run python scripts/dump_turntaking_samples.py \
    config/train_dpcrn_boundary.yaml --out-dir data_report/turntaking_samples --n 6 ) \
  > "$SD/turntaking_dump.log" 2>&1 || say "sample dump failed (non-fatal, see turntaking_dump.log)"

# 5. Warm-start training from wide-ep19 (CWD = recipe dir so data/*.list resolves).
say "launching training: config/train_dpcrn_boundary.yaml warm-start $CKPT"
( cd "$RECIPE" && uv run python main.py config/train_dpcrn_boundary.yaml --training \
    --pretrained_ckpt_path pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt ) 2>&1 | tee "$SD/train_boundary.log"
say "training process exited (rc=${PIPESTATUS[0]})"
