#!/usr/bin/env bash
# Boundary-distance RIR bank: far sources at 1.10-3.5 m, filling the 0.95-2.05 m
# zone the base bank never occupied (near 0.35-0.95 / far 2.05-5.5 left it empty)
# and supplying high-DRR "near-like" far sources. Everything else matches the
# base bank (rt60 0.17-0.85 incl. dry rooms, base room dims, duration 1.6, 16k,
# 5-channel near_0/1+far_0/1/2 layout). A 0.95-1.10 m guard band is deliberate:
# <1 dB level difference across the 1 m product boundary is unlearnable noise.
# See egs/voice_isolate/NEXT_STEPS.md "2026-07-03" Phase 1.
# seeds: 2718 train / 1618 held-out (disjoint from 1337 base, 2026 held-out, 3141 high).
# CPU multi-worker: measured 18 items/min steady with 20 workers on this 24-core
# box (~= the 2-GPU pytard-cupy rate). Sized at 1000 rooms x 10 RIR = 10k items
# (~9 h): a 10k boundary fill is ~14% of the phase1 view next to wide's 62k,
# enough exposure for the boundary-coverage experiment without a 28 h 20k run.
cd "$(dirname "$0")"
set -e

uv run python generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir_16k_boundary \
  --n-rooms 1000 --rir-per-room 10 \
  --sample-rate 16000 --duration 1.6 \
  --rt60 0.17 0.85 \
  --near-dist 0.35 0.95 --far-dist 1.10 3.5 \
  --crossover-hz 1000 --low-backend pytard --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2 \
  --num-workers 20 --resume \
  --seed 2718

uv run python generate_hybrid_rir.py \
  --output-dir exp/hybrid_rir_16k_boundary_heldout \
  --n-rooms 100 --rir-per-room 10 \
  --sample-rate 16000 --duration 1.6 \
  --rt60 0.17 0.85 \
  --near-dist 0.35 0.95 --far-dist 1.10 3.5 \
  --crossover-hz 1000 --low-backend pytard --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2 \
  --num-workers 20 --resume \
  --seed 1618
