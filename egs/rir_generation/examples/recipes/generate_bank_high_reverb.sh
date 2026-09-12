#!/usr/bin/env bash
# High-reverb (meeting-room grade) RIR bank: rt60 0.85-1.5, bigger rooms, longer tails.
# Reverb beyond the training domain, so a model can be probed outside what it saw.
# seed 3141, deliberately disjoint from the train (1337) and held-out (2026) banks.
repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${repo_root}"
exec uv run python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_high \
  --n-rooms 500 --rir-per-room 20 \
  --sample-rate 16000 --duration 2.4 \
  --rt60 0.85 1.5 \
  --room-x 6 14 --room-y 5 11 --room-z 2.8 4.2 \
  --crossover-hz 1000 --low-backend pytard-cupy --pytard-low-sample-rate 16000 \
  --pytard-spatial-samples-per-wavelength 2 --pra-max-order 12 --pra-n-rays 30000 \
  --gpu-devices 0,1 --num-workers 2 --resume \
  --seed 3141
