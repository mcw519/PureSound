# Fixed session acceptance

Run from `egs/voice_isolate`. These commands generate evaluation data or run
inference only; neither starts training. The original six-second validation
remains separate.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python \
  benchmarks/probes/v20_session_rows/session_validation.py materialize \
  config/exp/train_dpcrn_v20_r1a.yaml \
  --out benchmarks/probes/v20_session_rows/fixed_validation --rows-per-bucket 20

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python \
  benchmarks/probes/v20_session_rows/session_validation.py evaluate \
  config/exp/train_dpcrn_v20_r1a.yaml \
  --manifest benchmarks/probes/v20_session_rows/fixed_validation/manifest.json \
  --ckpt /absolute/path/to/checkpoint.ckpt --device cpu --out /tmp/session_report.json
```

`materialize` is recipe-specific. It calls the production `VoiceIsolationDataset`
on the validation corpus, checks the complete metadata speaker IDs against the
training corpus, and checks eligible validation speakers again after filtering.
An overlap fails before generation. The generator still draws both foreground and
bystanders from that validation pool. It takes a validated recipe copy with
sessions and explicit second views forced on, and legacy source-pool pairing off.
It does not mutate the training recipe. Identical device-chain waveforms are
rejected with up to 50 deterministic attempts per row: this acceptance set is
**conditional on an effective chain change**, not an estimate of training pair
frequency. Rejected draws and exact accepted seeds are recorded.

The output contains the exact recipe text, normalized evaluation recipe, source
metadata hashes, speaker lists, seed, and per-row tensors at 12 and 30 seconds.
Each row has a SHA-256 checked before inference. Existing directories are never
overwritten. Materializing twice with the same source assets, recipe, software,
and seed reproduces the tensors; subsequent evaluation loads the tensors rather
than rerendering. External audio/RIR assets must be versioned with the experiment;
the manifest's metadata hash does not hash every referenced source audio file.

## Metric definitions

Reports include individual rows and aggregates by bucket and generated shape.
Counts are retained, and unavailable rates are JSON `null`, not success values.

- **Onset context effect:** for the first foreground turn after another talker,
  and later foreground turns separated by at least one second, compare the full
  session output to a fresh forward beginning exactly at that turn's onset. The
  current input and target samples are identical. On the first second (clipped
  to the turn), compute signed target projection gain `dot(output,target) /
  dot(target,target)` and residual-to-target energy. Report contextual minus
  fresh gain and gain dB. Negative differences indicate context-related
  attenuation. This is a target-correlated amplitude proxy, not intelligibility;
  correlated interference can bias it, and residual energy is reported alongside
  it. Projection gain dB uses absolute gain; the signed gain is also retained.
- **Proximity:** truncate to the common frame prefix, exclude overlap, and pool
  by turn ID. By default compare different positive role IDs (`cross_role`);
  `all` also permits within-role comparisons. Nonpositive/unknown distances,
  padding, insufficient turn frames, and distance gaps below 0.25 m are excluded.
  Nearer means larger raw scalar. Readout ties are incorrect. Record ordering
  correct/pair counts, signed near-minus-far margin distribution, and absolute
  change of those margins under the second chain. An arbitrary common scalar
  offset therefore contributes zero consistency error.
- **Presence:** threshold logits at zero (probability 0.5), recording false
  negative/positive counts and frame denominators. Onset delay is time to the
  first detected frame in each true activity run. Runs with no detection are
  counted separately as right-censored at their duration; misses never become
  zero-delay detections. This includes energy-VAD intra-turn activity onsets.
- **Readout scale:** per-row raw scalar distributions are reported. Saturation is
  `null` for the unbounded baseline. Only pass `--bounded-limit` for an actually
  bounded readout; it measures the share reaching 99% of that explicit limit.

Offline predictions and labels use the same frame origin, matching training's
common-prefix alignment. `--presence-delay-frames` and `--waveform-delay-samples`
remove extra latency only when evaluating a path with a known additional delay;
defaults are zero, so the encoder delay is not subtracted twice.

The reusable `puresound.evaluation.session_validation.evaluate_session_manifest`
function accepts a model, manifest path, and evaluation options. It preserves the
caller's model mode and does no optimizer work. The shared module has no recipe
or voice-isolation dataset dependencies; materialization lives in this recipe.
Checkpoint CLI evaluation refuses absent or shape-incompatible proximity/presence
head weights, preventing random initialized heads from being reported as results.

## R6 new-head readout

```bash
uv run python benchmarks/probes/anchor_gate_cache.py field \
  config/exp/train_dpcrn_v20_r1a.yaml --ckpt /absolute/path/to/checkpoint.ckpt \
  --tag v20 --head proximity --device cpu --out /tmp/r6_cache
uv run python benchmarks/probes/anchor_gate_sim.py readability \
  --cache /tmp/r6_cache --tags v20 --head proximity --out /tmp/r6_proximity.json
```

The new path caches `last_proximity` directly and reports raw-scalar keep/suppress
AUC and prefix distributions. It never applies the legacy DistHead MLP or
converts scalars to metres. `--head dist` remains the default for existing cache
and readability commands; existing dist/DRR gate simulation remains unchanged.
A raw proximity scalar has no calibrated absolute gate threshold.

Tests: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest
../../test/test_session_validation.py -q` covers numerical definitions, replay,
corruption and speaker-overlap refusal, checkpoint head checks, both R6 paths,
and deterministic materialization through the actual session generator with a
small synthetic corpus and controlled chain.

## Data and memory audit before training

The following commands are read-only with respect to model weights. The first
renders 300 production rows and records the realised distance/presence/pair
distribution. The second performs a short GPU forward/backward memory smoke;
it never calls `Trainer.fit` and asserts that every parameter is unchanged.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python \
  benchmarks/probes/v20_session_rows/audit_300_rows.py \
  --config config/exp/train_dpcrn_v20_r1a.yaml \
  --out benchmarks/probes/v20_session_rows/audit_300.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python \
  benchmarks/probes/v20_session_rows/no_update_memory_smoke.py \
  --config config/exp/train_dpcrn_v20_r1a.yaml --seconds 12 --n-spk 6 \
  --steps 2 --gpu 0 --out /tmp/v20_12s_memory.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python \
  benchmarks/probes/v20_session_rows/no_update_memory_smoke.py \
  --config config/exp/train_dpcrn_v20_r1a.yaml --seconds 30 --n-spk 2 \
  --steps 2 --gpu 0 --out /tmp/v20_30s_memory.json
```

The older `memory_smoke.py` is historical and uses a Lightning fit loop; it is
not part of the fresh v20 acceptance path.

Finally, check the v16-to-v20 warm-start contract before training:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python \
  scripts/preflight_ckpt_recipe.py \
  --allow-missing-head vad_head --allow-missing-head proximity_head \
  --ckpt /work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/\
version_0/checkpoints/epoch=19-step=10000.ckpt \
  config/exp/train_dpcrn_v20_r1a.yaml
```

Only missing VAD and proximity weights are explicitly allowed in this warm-start
command. Shape mismatches always fail. For trained-checkpoint evaluation, omit
both `--allow-missing-head` flags: every head requested by the recipe must load.

### Review fixes: distribution and memory coverage

The row audit preserves the production bucket probabilities **and batch sizes**,
consumes batches until exactly the requested count, and clips the final batch's
statistics (including paired rows). Exhausting the loader early is an error.

Run memory cases in separate processes for independent peaks. Both cases use
seed 20 by default and force session/view draws to probability 1; the baseline
removes the auxiliary view before forward. These are coverage probes, not an
estimate of the production sampling distribution. Repeat for 30 s / 2 rows:

```bash
uv run python benchmarks/probes/v20_session_rows/no_update_memory_smoke.py \
  --seconds 12 --n-spk 6 --case baseline --seed 20 --out /tmp/v20_12s_baseline.json
uv run python benchmarks/probes/v20_session_rows/no_update_memory_smoke.py \
  --seconds 12 --n-spk 6 --case paired --seed 20 --out /tmp/v20_12s_paired.json
```

`paired` is the default case. Every measured paired step must log a positive
number of effective consistency pairs, otherwise the command fails without a
success report. Reports distinguish generated `paired_rows_seen`, capped
`actual_view_rows`, and loss-reported `effective_pairs`. Parameter changes also
fail the command. No optimizer update is performed. GPU peaks must still be
measured on the intended hardware; CPU regression tests do not certify capacity.
