# chain_readability -- verdict (2026-09-04)

Question: does compression-invariance training (v11b) fix the representation the QVF chain
inverts? Full tables: `TABLES.md`. Scripts and raw json in this directory.

## Answer 1 -- No. Not by any measure, and the effect it DID have is on the rig chain.

Sliding DistHead readout, keep spans (30-50 cm) vs suppress spans (2-3 m), W = 1 s, paired
clip-level bootstrap (B = 4000, same resampled recordings for every checkpoint):

| readout / scope | v8 | v16 ep19 | v11b ep31 | v11b - v8 (95% CI, p) |
|---|---|---|---|---|
| QVF clip, distance | 0.265 | 0.255 | 0.258 | **-0.006 [-0.161, +0.115] p 0.893** |
| QVF session, distance | 0.357 | 0.357 | 0.377 | +0.020 [-0.077, +0.219] p 0.707 |
| QVF clip, DRR | 0.799 | 0.674 | 0.841 | +0.042 [-0.017, +0.138] p 0.167 |
| QVF session, DRR | 0.517 | 0.441 | 0.528 | +0.011 [-0.129, +0.131] p 0.980 |
| device clip, distance | 0.989 | 0.977 | 0.962 | -0.027 [-0.065, +0.005] p 0.102 |
| device session, DRR | 0.757 | 0.781 | 0.903 | **+0.145 [+0.102, +0.188] p < 0.001** |
| device session, distance | 0.695 | 0.709 | 0.750 | +0.055 [-0.080, +0.142] p 0.389 |

The inversion is per-recording and total: QVF clip-scope distance AUC by group is
v8 0.191 / 0.170 / 0.047 / 0.041 (gym / plumbing / price / scenario3), v16 0.087 / 0.074 / 0.000 /
0.035, v11b 0.091 / 0.000 / 0.073 / 0.092 -- 12 of 12 below 0.2. Prefix (anchor) reads still put
QVF near speech FARTHER than QVF far speech on v11b (0.988 m vs 0.771 m), the same inversion
`anchor_gate_README` §1 recorded for v8 (0.869 vs 0.715).

What v11b actually bought: the DRR readout in the STREAMING condition on the rig chain,
0.757 -> 0.903 (p < 0.001), the one number in this table that moves. That is the deployment
condition for the AnchorGate's relative rule, so it is worth something -- on the device chain only.

## Answer 2 -- The offline compression probe does not reproduce the inversion, and the
##             probe that said it did was measuring a DC signal

`benchmarks/probes/eq_probe.py` has a defect: `compressor_gain` RETURNS a gain curve (see its
docstring; `puresound/task/device_chain.py::_compressor` multiplies it in), and eq_probe fed that
curve to the model as if it were audio. On the field clips (peak -34 to -24 dBFS, below the -28 dB
threshold) the curve is all-ones, so after RMS restore the model was shown a CONSTANT signal --
measured: `std = 2.3e-10` on 90d_far1, exactly 0 on 90d_near1. That is where
"1.009 -> 0.582 m, past QVF's 0.70 on its own" and "comp + broadcast = 0.447 m / +6.08 dB" come
from. I reproduced the artifact first (`comp_readability.py`: three different operating points
returning byte-identical medians 0.582 / 0.608 m -- the signature) and then fixed it.

Fixed probe (`comp_readability2.py`: gain multiplied in, clip normalised to the recipe's
`gain_normalized_to: -28` dBFS RMS so the -34..-22 thresholds are in range, level restored after;
device chain, W = 1 s):

| condition | GR mean / max dB | v8 AUC dist | v16 | v11b | v8 keep m | v8 supp m | v8 supp DRR |
|---|---|---|---|---|---|---|---|
| none | 0 / 0 | 0.989 | 0.977 | 0.962 | 0.591 | 1.043 | -0.55 |
| comp thr -34 r2 | 0.18 / 8.05 | 0.987 | 0.984 | 0.962 | 0.578 | 1.043 | -1.01 |
| comp thr -28 r4 (eq_probe's point) | 0.07 / 9.44 | **0.990** | 0.981 | 0.962 | 0.591 | 1.051 | -0.74 |
| comp thr -22 r6 | 0.00 / 5.91 | 0.989 | 0.977 | 0.963 | 0.591 | 1.045 | -0.47 |
| comp thr -40 r6 (hard) | 1.05 / 16.22 | 0.983 | 0.974 | 0.948 | 0.623 | 1.104 | -1.53 |
| broadcast EQ | -- | 0.974 | 0.918 | 0.949 | 0.661 | 1.014 | +1.39 |
| comp + broadcast | 0.07 / 9.44 | 0.978 | 0.927 | 0.949 | 0.657 | 1.017 | +1.03 |
| dark tilt (control) | -- | 0.987 | 0.980 | 0.972 | 0.572 | 1.096 | -1.83 |
| waveshaper \|x\|^0.8 | -- | 0.981 | 0.980 | 0.952 | 0.600 | 1.025 | +0.70 |
| waveshaper \|x\|^0.6 | -- | 0.957 | 0.957 | 0.917 | 0.679 | 0.974 | +3.42 |
| waveshaper \|x\|^0.4 | -- | **0.880** | 0.902 | 0.865 | 0.775 | 0.950 | **+5.00** |

Readings:

* **The envelope compressor v11b trains against moves this readout by nothing**: |dAUC| <= 0.013
  at every operating point on all three checkpoints, keep/suppress medians within 0.02 m. Makeup
  normalisation forces mean gain to 1.0, so mean gain reduction is 0.0-1.05 dB even when peaks are
  pulled 16 dB. Compression invariance was trained against an axis that was already flat here.
* **The |x|^p waveshaper -- a different operator, the one `compression_probe.py` used and the one
  whose causality was actually established -- does move it**, and my numbers corroborate the
  historical figure: at p = 0.4 the far talker's DRR readout walks -0.55 -> +5.00 dB (recorded:
  -0.61 -> +5.19) and its distance 1.043 -> 0.950 m. But distance AUC only falls to 0.880: it
  DEGRADES the cue, it does not invert it. Nothing I applied to a device clip reproduced 0.26.
* v11b is not more robust than v8 to either operator (waveshaper p 0.4: v8 -0.109 AUC,
  v11b -0.097, from a lower start).
* On QVF material the same stages barely move anything (0.265 -> 0.235..0.291 for v8): it is
  already-processed audio, so re-applying the suspected stage adds nothing.

So: **"compression flips the cue" is false.** `eq_probe_README.md` Verdict 2 ("the phantom near
user on the QVF chain is fully accounted for by envelope compression plus mild production EQ.
There is no missing 2/3 factor to hunt; codec and noise-gate probes drop in priority") should be
withdrawn, and with it the docstring claim in `puresound/task/device_chain.py::_compressor` /
`puresound/config/augmentation.py` insofar as it credits `compressor_gain` with the +5.19 dB (that
number belongs to the waveshaper). Codec / noise-gate / room-and-mic geometry go back on the list.

## Answer 3 -- The information is in the bottleneck. The head's mapping is what inverts.

Fresh logistic probe on the SAME cached 128-d pooled bottleneck window the DistHead reads,
holdout unit = recording group (`linear_probe.py`):

| arm | v8 | v16 | v11b |
|---|---|---|---|
| device-fit -> device-test (leave-one-group-out) | 0.999 | 0.999 | 1.000 |
| **device-fit -> QVF-test (pooled)** | **0.789** | **0.827** | 0.680 |
| device-fit -> QVF, per group (gym / plumbing / price / scenario3) | .686/1.000/.978/.951 | .672/.991/.955/.965 | .855/.970/.401/.953 |
| QVF-fit -> QVF-test (leave-one-QVF-group-out) | 0.999 | 0.998 | 0.986 |
| QVF within-group in-sample (information ceiling) | 1.000 | 1.000 | 1.000 |

A linear boundary fitted on device recordings alone reads QVF near-vs-far at 0.79 (v8) while the
model's own DistHead reads 0.26. Fitted on other QVF recordings it reads 0.99 on a held-out QVF
recording. **The QVF inversion is a property of the trained readout, not of the representation** --
which is why no amount of input-side augmentation invariance was ever going to move it, and why
the two things that did survive the chain boundary were relative / self-calibrated rules
(`presence_selfcal_README` Finding 1: within-session 0.886 / 0.911 vs pooled 0.474).
Interferer slot (slot 2, never previously read) is inverted too on QVF: 0.13-0.19 (`slot2_auc.py`).

Caveat on the probe: per-frame fitting with correlated frames inside a clip; the honest cross-chain
number is the device-fit -> QVF one (0.68-0.83), not the 0.99.

## Answer 4 -- Is chain invariance of this readout a trainable target with a fast in-loop metric?

**As a metric: yes, and it is cheap.** 47 cold-start field clips, one forward pass each, activity
mask and spans precomputed; the whole AUC costs a few seconds of GPU per validation epoch. There
is no such hook today (`puresound/system/siso.py::validation_step` logs losses only), so it is a
small addition: an `on_validation_epoch_end` that scores a frozen clip list and logs
`auc_dist_qvf`, `auc_dist_device`, `auc_drr_device_session`.

**As a target: yes, but as a HEAD/readout objective, not an augmentation objective.** Answer 3 says
the features already separate; answer 2 says the input-side stage v11b guessed at is inert. The
trainable versions are (a) train the DistHead discriminatively (near-vs-far contrast) instead of, or
beside, the metres regression; (b) a chain-consistency loss on the readout across two synthetic
chain draws of the same row -- that one needs no field data and has high resolution; (c) accept the
readout is chain-offset and use a relative rule (already the standing lesson).

**Numbers a round must hit.** Resolution first: the paired bootstrap half-width on QVF clip-scope
distance AUC is about +-0.14 with n = 9 keep / 6 suppress recordings, and the marginal CI is
[0.05, 0.67]. So this metric cannot see a move smaller than ~0.15, and QVF-session anything
(CI [0.08, 0.97], n = 3) cannot serve as a gate at all until the QVF row count grows.

* Fail-to-be-interesting: QVF clip dist AUC < 0.40 (still inverted / inside noise of 0.26).
* Minimum useful: **>= 0.50** -- not inverted, i.e. the sign of the cue is right.
* Target: **>= 0.75** clip-scope. Justification, not a wish: that is where the device-chain
  STREAMING readout already sits (0.695-0.750) and the AnchorGate demonstrably works there
  (`anchor_gate_README` §2-3, keep violations 26 -> 6); and 0.79-0.83 is the measured ceiling for a
  linear read of these same features fitted on device data only.
* Guards that must not move: device clip dist AUC >= 0.95 (v8 0.989; v11b is already 0.962),
  device session DRR AUC >= 0.75 (v8 0.757), and no group's QVF AUC below 0.35 (per-group table).
* **The AUC is not a deployment gate.** v11b improved device-session DRR AUC by +0.145 (p < 0.001)
  while losing roughly half the anchored far suppression (`v11b_VERDICT.md`). The readout metric
  gates the readout axis; the set-v3 block scorecard and the WER sets still decide a version.
