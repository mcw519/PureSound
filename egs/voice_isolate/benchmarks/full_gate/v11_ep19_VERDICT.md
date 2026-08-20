# v11-ep19 full gate -- the extreme-reverb hole closes, the suppression axis pays

`v8_v3set.txt` and `v11ep19.txt`: same day, same code, same chain, same **v3**
field set, two GPUs in parallel, `dry_blend 0.9` both. v11-ep19 is v8 warm-started
with two per-frame presence heads (EMA bank) at loss weight 0.1; **the heads are
training-only, so inference is byte-for-byte the v8 architecture** -- everything
below is the bottleneck moving under multi-task pressure, not a new inference path.

## The WER gates, paired

Aggregate differences are reported alongside a **paired bootstrap** over the same
200 utterances, which is what says whether a difference exists at all. (The
previous gate run could not do this -- both systems wrote the same transcript
filename. Different checkpoint stems fixed it.)

| set | v8 | v11 | paired dWER | 95% CI | verdict |
|---|---|---|---|---|---|
| 7a moderate **(PRIMARY)** | 0.409 | **0.394** | -0.0167 | [-0.0327, **+0.0000**] | directional, unresolved |
| 7b BUT-OFFICE | 0.538 | 0.548 | +0.0016 | [-0.0124, +0.0163] | inside noise |
| **8 BUT extreme reverb** | 0.693 | **0.658** | **-0.0303** | **[-0.0670, -0.0029]** | **v11 BETTER** |
| 6 Dawn (no paired dump) | 0.180 | **0.173** | -- | -- | both beat raw 0.184 |

**One difference resolves, and it is the one that matters for a known hole.** v8
*raises* WER on extreme reverberation (0.693 against mix 0.658, +0.035) -- the
documented reason the deployment note says use v7 above RT60 1 s. v11 lands at
**exactly mix (0.658)**: neutral instead of harmful, CI clear of zero.

The PRIMARY gate's upper bound is +0.0000. Directionally better, not established.
Note the aggregate BUT-OFFICE gap (+0.010) shrinks to +0.0016 paired -- the
aggregate was mostly noise, exactly as `but-office-lacks-resolution` predicts.

## What it costs

| synthetic far-only (same chain) | v8 | v11 | |
|---|---|---|---|
| in-domain F-only | -31.1 | -30.9 | +0.2 |
| probe expand | -29.4 | -26.5 | **+2.9** |
| probe high | -32.3 | -27.3 | **+5.0** |
| probe boundary | -26.7 | -23.6 | **+3.1** |

3-5 dB shallower far-field suppression in the synthetic domain, with in-domain
SI-SDRi flat (+8.19 -> +8.23). This is the M6 pattern -- auxiliary pressure moves
the working point conservative -- arriving through a different door.

Turn-taking (stage 9): KEEP 94/6 -> **95/5** violations, SUPPRESS 87/13 ->
**83/17** fails, SI-SDR +5.42 -> +4.89. Keep marginally better, suppress worse.

## Field v3, and a split that is the whole story

| | v8 | v11 |
|---|---|---|
| keep violations (38 clips) | 4 | **5** |
| worst keep | -11.79 | **-19.28** |
| far verdicts | FAIL 16 / PARTIAL 2 / ok 2 | FAIL 15 / PARTIAL 2 / **ok 3** |
| **180d session** suppress (held out) | -9.25 | **-14.05** |
| 270d session suppress | -11.29 | -12.92 |
| 90d session suppress | -12.35 | -11.49 |
| qvf_scenario3 session suppress | -0.80 | -0.85 |

**Every keep violation except the noisy sentinel is on the QVF chain**, in both
versions. The device recordings (90d/180d/270d) have zero. v11 improves far-field
suppression on device sessions (180D, an orientation nothing was fitted on, by
4.8 dB) and makes QVF-chain keep worse (`qvf_keep_in_touch_near1` -3.32 ->
-19.28, `qvf_gym_near1` newly failing at -9.35).

That is the same chain boundary the presence probe found
(`presence_head_v11_README.md`: 0.843 on held-out 180D, **inverted** at 0.253 on
qvf_scenario3), now visible in the scorecard. One wall, two instruments.

## Verdict

**v8 stays the default.** v11 is not a general replacement: it trades 3-5 dB of
synthetic far-field suppression and QVF-chain keep for WER that is better on three
sets and resolvably better on one.

**v11 is the candidate for the extreme-reverberation regime.** It closes v8's only
documented do-no-harm failure with the only CI in this table that clears zero. If
the deployment currently switches to v7 above RT60 1 s, v11 is the better switch
target -- and that should be tested directly against v7 before anything ships.

**The heads are the real deliverable, and they are not gated on anything.**
Inference is unchanged, so nothing above is at risk from them; what they buy is a
presence signal that is 0.953 in-domain across RT60 and 0.843 on an unseen device
orientation. Whether that can drive a gain is a v12 question, and the scenario3
inversion says the answer is "not on the QVF chain yet".

## Caveats

* v3 field set. **No v2 comparison is valid** -- `SET_V3.md`.
* Dawn has no per-item dump, so its -0.007 is unpaired.
* One training run, one warm start, one loss weight. No ablation separates the EMA
  bank from the supervision.
* Stages 2-5 are synthesised at eval time on chain `3cde9d1`; both rows here are on
  that chain, so they compare with each other and not with pre-9c56e02 records.
