# v14 (parallel SSM branch) -- NEGATIVE. The axis is closed.

Measured 2026-08-26/27. Round design and cost table: `config/exp/train_dpcrn_v14_mambaparallel.yaml`.

## What the round was

v13 replaced the inter LSTM with `MambaInter` and lost 8 dB of session suppression.
That was **not** an architecture verdict: a from-scratch LSTM on the same recipe
(`exp/dpcrn_allbank_scratch` ep39) scores worse still, so v13 measured the cost of
re-initialising the network's only context carrier and repaying it on the last rung of
an eight-generation curriculum. v14 removed that confound instead of paying it:

    inter = inter_rnn(x) + inter_ssm(x)        # inter_ssm.out_proj initialised to 0

At step 0 the network is **bit-identical** to its warm start (v11b ep19, `max|diff| =
0.000e+00` on a real clip), so the baseline IS the initialisation and no control run is
needed. The SSM was narrowed to `d_state 8 / expand 1` first, because the v13 width
costs 8.3 ms/frame single-threaded against a 10 ms budget.

The mechanism worked: the branch woke up (out_proj rms 0 -> 0.06/0.09 by ep12, a quarter
to a third of the LSTM projection's) and moved the model's behaviour.

## Why it is still negative

At ep19 v14 looks like a better operating point than its own start -- keep violations
6 -> 5, keep worst -16.22 -> -8.67, Dawn deletion 0.240 -> 0.175 -- for 2 dB of session
suppression (-12.90 -> -10.91). **A `dry_blend` sweep on the unchanged v11b buys more.**

| field set v3 + Dawn | v11b @ **0.85** | v14 ep19 @ 1.0 |
|---|---|---|
| sessions suppression | -10.75 | -10.91 |
| **Dawn WER / deletion** | **0.181 / 0.092** | 0.271 / 0.175 |
| cold-far device, absolute residual | **24.11** | 24.59 |
| cold-far device, median reduction | -1.73 | -3.88 |
| keep worst | -12.61 | **-8.67** |
| KEEP violations | 6 | **5** |

At matched session suppression a free knob has **half** v14's deletion. v14's two
remaining wins are keep-span energy numbers -- the metric
`span-energy-metric-conflates-four-layers` says cannot attribute -- and the ASR says the
opposite about the same axis. Its cold-far "2.2 dB deeper" is a reduction number;
on absolute residual (`absolute-residual-not-attenuation`) v11b is ahead, and no clip
changes verdict (1/2/11 vs 1/3/10).

So: +110k parameters and +2 ms/frame for something a knob does better. **Closed.**

## What to keep from it

* **The zero-init parallel branch is a good way to add a module to a trained network.**
  `inter_type: "lstm+mamba"` and `MambaInter(zero_init_out=True)` stay in the tree;
  reuse them for any future module rather than re-initialising a trained path.
* **`scripts/preflight_ckpt_recipe.py` + `scripts/make_arch_eval_configs.py`**, and the
  `CFG_*` overrides in `run_full_benchmark.sh`. Every benchmark recipe hard-coded the
  LSTM backbone; pointing one at this checkpoint loaded "successfully" while silently
  dropping the whole branch. Stage 0 now refuses that in both directions.
* **A by-product worth its own round: `v11b ep19 @ dry_blend 0.85`** -- sessions -10.75
  against v8's -9.83, Dawn 0.181/0.092 against v8's 0.180/0.094, at 6 KEEP violations
  against v8's 4. Nothing to do with the SSM; nobody had swept that knob on v11b.

## Not measured

The three WER stages (7a primary gate, 7b BUT-OFFICE, 8 extreme reverb) did not run:
`eval_wer.py` imports `whisper.normalizers` and the venv only has `faster-whisper`.
Fix with `uv pip install --no-deps openai-whisper` (`--no-deps` cannot touch torch).
Nothing in this verdict depends on them -- Dawn alone decides it -- but the v11b @ 0.85
candidate above cannot be judged without them.
