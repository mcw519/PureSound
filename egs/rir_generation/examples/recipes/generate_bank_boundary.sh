#!/usr/bin/env bash
# Boundary-distance RIR bank: far sources at 1.10-3.5 m, filling the 0.95-2.05 m
# zone the base bank never occupied (near 0.35-0.95 / far 2.05-5.5 left it empty)
# and supplying high-DRR "near-like" far sources, i.e. the hardest cases for a
# near/far decision. Everything else matches the base bank (rt60 0.17-0.85 incl.
# dry rooms, base room dims, duration 1.6, 16k, 5-channel near_0/1+far_0/1/2).
# The 0.95-1.10 m guard band is deliberate: across a 1 m decision boundary the
# level difference is under 1 dB, which is unlearnable noise rather than signal.
# seeds: 2718 train / 1618 held-out (disjoint from 1337 base, 2026 held-out, 3141 high).
# Throughput note: ~18 items/min with 20 CPU workers on a 24-core box, about the
# same as 2 GPUs on the pytard-cupy backend. 1000 rooms x 10 RIR = 10k items ~= 9 h.
repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${repo_root}"
set -e

uv run python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_boundary \
  --n-rooms 1000 --rir-per-room 10 \
  --sample-rate 16000 --duration 1.6 \
  --rt60 0.17 0.85 \
  --near-dist 0.35 0.95 --far-dist 1.10 3.5 \
  --crossover-hz 1000 --low-backend pytard --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2 \
  --num-workers 20 --resume \
  --seed 2718

uv run python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_boundary_heldout \
  --n-rooms 100 --rir-per-room 10 \
  --sample-rate 16000 --duration 1.6 \
  --rt60 0.17 0.85 \
  --near-dist 0.35 0.95 --far-dist 1.10 3.5 \
  --crossover-hz 1000 --low-backend pytard --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2 \
  --num-workers 20 --resume \
  --seed 1618
