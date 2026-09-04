# P0 -- does the frozen bottleneck carry talker identity?

Stage **P0** of `benchmarks/probes/v20_self_enrolling_foreground_design.md` (§7). Probe:
`identity_probe.py` beside this note; raw tables `tables/TABLES.md`, machine-readable
`tables/results.json`, the exact synthetic render recipe `tables/synth_chains.json`.
Nothing is trained here except the leave-speakers-out linear projection of measurement B.

**The question.** The v20 requirement R4 is "key identity on voice + proximity, not on the
channel draw". Before this probe the record said: the model's foreground template is
about 30 % channel signature (`v19c_diagnostics/anchor_synthetic/TABLE_ident.md`: a
same-speaker-other-room prefix costs -0.44 dB against a different-speaker prefix's
-1.47 dB on v16 ep19), proximity is readable on the device chain (AUC 0.99) and inverted
on the QVF chain by the trained head while a device-fitted linear probe reads QVF
0.79-0.83 (`v19c_diagnostics/chain_readability/`), and identity itself had never been
measured -- `objective_landscape/LANDSCAPE.md` records it as the "unmeasured
prerequisite", because the existing per-frame cache has one talker per recording so every
same-speaker pair in it is also a same-channel pair.

**§7 decision rule.** pair AUC < 0.60 => identity is not in the features and `L_id` must be
trained first (R1b waits); 0.60-0.75 => intermediate; >= 0.75 => proceed to R1a as planned.

---

## 1. What was measured

| set | segments kept / indexed | bootstrap clusters | speakers | channels | close / far segments |
|---|---|---|---|---|---|
| DiPCo | 2220 / 2256 | 10 sessions | 32 (global `P01`-`P32`) | `CT` + `U01`-`U05` | 370 / 1850 |
| NOTSOFAR-1 | 1688 / 1688 | 19 meetings | 78 (per-meeting aliases) | `CT` + up to 3 far devices | 422 / 1266 |
| AMI | 4526 / 4568 | 60 meetings | 60 (corpus-global ids) | `HS` + `Array1-01` | 2263 / 2263 |
| synthetic | 1944 / 1944 | 36 anchor speakers | 36 (LibriTTS `test-clean`) | 2 chains x 3 rooms | 1296 near / 648 far |

3-5 talkers per real session; the synthetic set is one session by construction (36 speakers x 6
utterances x 9 chain/room/distance cells). Every pair lives inside one session, takes its two
members from **different merged turns** (no pair shares audio), and is 3 s of single-talker speech
with the other talkers' turns subtracted and dilated by 0.25 s. Segment cutting, the four feature
variants, the statistics and the exact synthetic render live in the `identity_probe.py` docstring
and `tables/synth_chains.json`; raw tables in `tables/TABLES.md`, machine-readable
`tables/results.json`. Feature variants: **raw** = pooled bottleneck at the recipe's own -28 dBFS
input level; **LN** = per-frame LayerNorm over the 128 channels, then pooled; **centred** = raw
minus the mean raw embedding of its own recording (= one `(session, channel, distance)` capture
condition). CIs are 95 % percentile intervals from a 1000-draw cluster bootstrap over the clusters
named above.

## 2. Headline -- zero-shot cosine, the primary identity arms

`A_matched`: positives = same talker through two **different far devices**, negatives = different
talkers through the **same far device** -- both members far, so proximity is held constant and only
the device differs. `A_xprox`: the corpora's own close-talk/far pairing, which moves chain **and**
proximity together. `D1` / `D3`: the synthetic same-talker-across-chains (and across rooms) arms,
both members at the near mic. `sv` = the repo's frozen ECAPA-TDNN speaker-verification backbone
(`PS-spk-v1-1.onnx`) on exactly the same pairs -- the "what a real identity embedding scores here"
row. For `sv` the LN column is a duplicate of raw by construction (LayerNorm is not defined on a
192-d utterance embedding); `sv` is fed at its own recipe's -22 dBFS.

| set | arm | feature | n pos | n neg | clusters | **v16** AUC [95 % CI] | **v8** AUC [95 % CI] | **sv** AUC [95 % CI] | EER v16 / v8 / sv |
|---|---|---|---|---|---|---|---|---|---|
| dipco | `A_matched` | raw | 4000 | 4000 | 10 | 0.487 [0.466, 0.511] | 0.500 [0.480, 0.522] | 0.937 [0.889, 0.972] | 0.511 / 0.502 / 0.132 |
| dipco | `A_matched` | LN | 4000 | 4000 | 10 | 0.488 [0.466, 0.511] | 0.500 [0.480, 0.522] | 0.937 [0.889, 0.972] | 0.510 / 0.503 / 0.132 |
| dipco | `A_matched` | centred | 4000 | 4000 | 10 | 0.556 [0.534, 0.581] | 0.561 [0.540, 0.585] | 0.960 [0.917, 0.988] | 0.465 / 0.466 / 0.102 |
| dipco | `A_xprox` | raw | 4000 | 4000 | 10 | 0.154 [0.138, 0.173] | 0.168 [0.148, 0.188] | 0.948 [0.893, 0.983] | 0.736 / 0.725 / 0.110 |
| dipco | `A_xprox` | LN | 4000 | 4000 | 10 | 0.157 [0.140, 0.175] | 0.171 [0.151, 0.190] | 0.948 [0.893, 0.983] | 0.733 / 0.722 / 0.110 |
| dipco | `A_xprox` | centred | 4000 | 4000 | 10 | 0.546 [0.527, 0.567] | 0.540 [0.521, 0.559] | 0.979 [0.951, 0.995] | 0.461 / 0.472 / 0.069 |
| notsofar | `A_matched` | raw | 6194 | 6130 | 19 | 0.454 [0.407, 0.496] | 0.457 [0.413, 0.498] | 0.997 [0.993, 0.999] | 0.531 / 0.529 / 0.025 |
| notsofar | `A_matched` | LN | 6194 | 6130 | 19 | 0.454 [0.408, 0.496] | 0.458 [0.413, 0.499] | 0.997 [0.993, 0.999] | 0.531 / 0.528 / 0.025 |
| notsofar | `A_matched` | centred | 6194 | 6130 | 19 | 0.577 [0.556, 0.600] | 0.586 [0.557, 0.616] | 0.987 [0.969, 0.997] | 0.443 / 0.438 / 0.053 |
| notsofar | `A_xprox` | raw | 6194 | 6130 | 19 | 0.352 [0.319, 0.390] | 0.327 [0.287, 0.367] | 0.999 [0.997, 0.999] | 0.589 / 0.621 / 0.016 |
| notsofar | `A_xprox` | LN | 6194 | 6130 | 19 | 0.352 [0.319, 0.389] | 0.328 [0.289, 0.368] | 0.999 [0.997, 0.999] | 0.588 / 0.621 / 0.016 |
| notsofar | `A_xprox` | centred | 6194 | 6130 | 19 | 0.570 [0.552, 0.591] | 0.581 [0.556, 0.605] | 0.990 [0.975, 0.999] | 0.446 / 0.442 / 0.046 |
| ami | `A_xprox` | raw | 19736 | 23047 | 60 | 0.383 [0.357, 0.405] | 0.355 [0.328, 0.379] | 0.963 [0.954, 0.970] | 0.567 / 0.597 / 0.091 |
| ami | `A_xprox` | LN | 19736 | 23047 | 60 | 0.386 [0.361, 0.408] | 0.360 [0.334, 0.384] | 0.963 [0.954, 0.970] | 0.564 / 0.592 / 0.091 |
| ami | `A_xprox` | centred | 19736 | 23047 | 60 | 0.573 [0.557, 0.587] | 0.579 [0.562, 0.595] | 0.989 [0.985, 0.993] | 0.447 / 0.444 / 0.047 |
| synth | `D1_chain_swap` | raw | 3204 | 6000 | 36 | 0.012 [0.008, 0.017] | 0.036 [0.027, 0.046] | 0.741 [0.707, 0.773] | 0.946 / 0.904 / 0.325 |
| synth | `D1_chain_swap` | LN | 3204 | 6000 | 36 | 0.010 [0.007, 0.014] | 0.030 [0.023, 0.040] | 0.741 [0.707, 0.773] | 0.952 / 0.912 / 0.325 |
| synth | `D1_chain_swap` | centred | 3204 | 6000 | 36 | 0.705 [0.674, 0.738] | 0.663 [0.632, 0.697] | 0.958 [0.946, 0.970] | 0.353 / 0.388 / 0.102 |
| synth | `D3_chain_and_room` | raw | 6000 | 6000 | 36 | 0.004 [0.002, 0.006] | 0.014 [0.009, 0.020] | 0.706 [0.666, 0.745] | 0.980 / 0.949 / 0.342 |
| synth | `D3_chain_and_room` | LN | 6000 | 6000 | 36 | 0.003 [0.001, 0.004] | 0.012 [0.008, 0.017] | 0.706 [0.666, 0.745] | 0.982 / 0.955 / 0.342 |
| synth | `D3_chain_and_room` | centred | 6000 | 6000 | 36 | 0.696 [0.667, 0.728] | 0.654 [0.624, 0.685] | 0.956 [0.943, 0.968] | 0.355 / 0.392 / 0.103 |

Three things to read off it.

* **raw and LN are the same measurement.** Nowhere in tables A/C/D do they differ by more than
  **0.006** AUC. Per-frame LayerNorm over the 128 channels does not touch what separates two
  recordings, because that is a per-recording *mean direction*, not a per-frame gain.
* **Zero-shot, the frozen bottleneck does not rank a talker match above a channel match.** With
  proximity held constant it sits at chance (DiPCo 0.487 / 0.500 for v16 / v8) or below it with the
  CI excluding 0.5 (NOTSOFAR 0.454 [0.407, 0.496] / 0.457 [0.413, 0.498]); let proximity move and it
  collapses (`A_xprox` raw: DiPCo 0.154, NOTSOFAR 0.352, AMI 0.383 on v16). The synthetic arms are
  the extreme case: `D1` raw v16 0.012 [0.008, 0.017] for a chain swap and `D3` **0.004** [0.002,
  0.006] for a chain *and* room swap -- changing what recorded a talker moves them further than
  replacing the talker does, and so consistently that the arm is a near-perfect *inverted*
  classifier.
* **Per-recording centring is what makes the number go the right way at all**, and it stops at
  0.540-0.586 on the real corpora while reaching 0.654-0.705 on the synthetic ones. That gap is the
  finding, not an artefact of the estimator: subtracting one vector per recording recovers most of
  the synthetic chain/room offset (it is largely additive there) and very little of the real one.

`sv` on the identical pairs reads 0.937-0.999 raw on the real corpora and 0.706-0.741 raw /
0.956-0.958 centred on the synthetic ones. The real pairs are therefore not intrinsically hard -- a
true identity embedding solves them.

## 3. The same arms with a symmetric protocol

In `A_matched` / `A_xprox` / `D1` / `D3` the positives always cross recordings while the negatives
are always taken *inside* one recording. Per-recording centring is not neutral under that asymmetry
(caveat 1 below). The `*_both` arms make **both** trial types cross the channel, which removes it.

| set | arm | feature | n pos | n neg | **v16** AUC [95 % CI] | **v8** AUC [95 % CI] | **sv** AUC [95 % CI] | EER v16 / v8 / sv |
|---|---|---|---|---|---|---|---|---|
| dipco | `A_xchan_both` | raw | 4000 | 4000 | 0.520 [0.502, 0.541] | 0.526 [0.509, 0.548] | 0.941 [0.900, 0.972] | 0.491 / 0.485 / 0.130 |
| dipco | `A_xchan_both` | LN | 4000 | 4000 | 0.520 [0.501, 0.541] | 0.526 [0.509, 0.548] | 0.941 [0.900, 0.972] | 0.492 / 0.486 / 0.130 |
| dipco | `A_xchan_both` | centred | 4000 | 4000 | 0.534 [0.510, 0.560] | 0.537 [0.515, 0.562] | 0.957 [0.914, 0.985] | 0.477 / 0.475 / 0.108 |
| dipco | `A_xprox_both` | raw | 4000 | 4000 | 0.502 [0.492, 0.510] | 0.498 [0.488, 0.508] | 0.959 [0.922, 0.986] | 0.495 / 0.498 / 0.095 |
| dipco | `A_xprox_both` | LN | 4000 | 4000 | 0.501 [0.492, 0.509] | 0.498 [0.488, 0.508] | 0.959 [0.922, 0.986] | 0.496 / 0.498 / 0.095 |
| dipco | `A_xprox_both` | centred | 4000 | 4000 | 0.539 [0.517, 0.562] | 0.547 [0.521, 0.575] | 0.982 [0.956, 0.996] | 0.467 / 0.469 / 0.065 |
| notsofar | `A_xchan_both` | raw | 6194 | 6994 | 0.522 [0.494, 0.554] | 0.532 [0.504, 0.562] | 0.997 [0.994, 0.999] | 0.479 / 0.473 / 0.022 |
| notsofar | `A_xchan_both` | LN | 6194 | 6994 | 0.522 [0.494, 0.554] | 0.532 [0.504, 0.562] | 0.997 [0.994, 0.999] | 0.479 / 0.473 / 0.022 |
| notsofar | `A_xchan_both` | centred | 6194 | 6994 | 0.545 [0.512, 0.576] | 0.562 [0.531, 0.595] | 0.984 [0.965, 0.996] | 0.464 / 0.455 / 0.057 |
| notsofar | `A_xprox_both` | raw | 6194 | 6994 | 0.501 [0.479, 0.524] | 0.507 [0.484, 0.531] | 0.999 [0.998, 1.000] | 0.500 / 0.491 / 0.013 |
| notsofar | `A_xprox_both` | LN | 6194 | 6994 | 0.501 [0.479, 0.524] | 0.507 [0.484, 0.531] | 0.999 [0.998, 1.000] | 0.499 / 0.490 / 0.013 |
| notsofar | `A_xprox_both` | centred | 6194 | 6994 | 0.559 [0.536, 0.584] | 0.585 [0.557, 0.612] | 0.989 [0.973, 0.998] | 0.459 / 0.442 / 0.050 |
| ami | `A_xprox_both` | raw | 19736 | 23968 | 0.531 [0.523, 0.540] | 0.533 [0.524, 0.542] | 0.968 [0.960, 0.975] | 0.477 / 0.475 / 0.083 |
| ami | `A_xprox_both` | LN | 19736 | 23968 | 0.532 [0.523, 0.540] | 0.534 [0.525, 0.542] | 0.968 [0.960, 0.975] | 0.476 / 0.475 / 0.083 |
| ami | `A_xprox_both` | centred | 19736 | 23968 | 0.584 [0.570, 0.598] | 0.588 [0.574, 0.602] | 0.990 [0.985, 0.993] | 0.441 / 0.439 / 0.047 |
| synth | `D5_xchain_both` | raw | 3204 | 6000 | 0.521 [0.489, 0.552] | 0.524 [0.490, 0.559] | 0.937 [0.920, 0.952] | 0.497 / 0.485 / 0.136 |
| synth | `D5_xchain_both` | LN | 3204 | 6000 | 0.524 [0.493, 0.555] | 0.528 [0.494, 0.561] | 0.937 [0.920, 0.952] | 0.493 / 0.481 / 0.136 |
| synth | `D5_xchain_both` | centred | 3204 | 6000 | 0.736 [0.703, 0.769] | 0.692 [0.658, 0.727] | 0.973 [0.960, 0.983] | 0.331 / 0.364 / 0.083 |

The asymmetry is worth at most 0.032 AUC and points both ways: DiPCo 0.556 -> 0.534 and NOTSOFAR
0.577 -> 0.545 when symmetrised (centred, v16), AMI 0.573 -> 0.584, synthetic 0.705 (`D1`) -> 0.736
(`D5`). No verdict moves. The highest real zero-shot cell anywhere in this probe is AMI
`A_xprox_both` centred v8 = 0.588 [0.574, 0.602] -- point estimate under the 0.60 line, CI upper edge
sitting on it.

## 4. Synthetic factor controls: which factor buries identity

One session, 36 held-out LibriTTS speakers, two fixed device chains (A: IIR + 100 Hz HPF; B: 8 kHz
SRC + 300 Hz HPF, both `prob=1.0`, seeds 101 / 202) and three bank rooms (RT60 0.35 / 0.70 / 0.33,
near 0.49-0.99 m, far 2.68-3.44 m; `tables/synth_chains.json`). Negatives are always the fully-fixed
cell, so each row prices exactly one nuisance factor.

| arm | what the positives cross | feature | n pos | n neg | **v16** AUC [95 % CI] | **v8** AUC [95 % CI] | **sv** AUC [95 % CI] |
|---|---|---|---|---|---|---|---|
| `D0_all_fixed` | nothing (chain+room+distance fixed) | raw | 3191 | 6000 | 0.805 [0.777, 0.837] | 0.794 [0.766, 0.825] | 0.982 [0.969, 0.991] |
| `D0_all_fixed` | nothing (chain+room+distance fixed) | LN | 3191 | 6000 | 0.801 [0.773, 0.833] | 0.790 [0.762, 0.822] | 0.982 [0.969, 0.991] |
| `D0_all_fixed` | nothing (chain+room+distance fixed) | centred | 3191 | 6000 | 0.827 [0.802, 0.852] | 0.812 [0.785, 0.838] | 0.990 [0.977, 0.997] |
| `D1_chain_swap` | device chain | raw | 3204 | 6000 | 0.012 [0.008, 0.017] | 0.036 [0.027, 0.046] | 0.741 [0.707, 0.773] |
| `D1_chain_swap` | device chain | LN | 3204 | 6000 | 0.010 [0.007, 0.014] | 0.030 [0.023, 0.040] | 0.741 [0.707, 0.773] |
| `D1_chain_swap` | device chain | centred | 3204 | 6000 | 0.705 [0.674, 0.738] | 0.663 [0.632, 0.697] | 0.958 [0.946, 0.970] |
| `D2_room_swap` | room (RIR) | raw | 6000 | 6000 | 0.270 [0.245, 0.296] | 0.299 [0.279, 0.323] | 0.963 [0.949, 0.974] |
| `D2_room_swap` | room (RIR) | LN | 6000 | 6000 | 0.265 [0.241, 0.291] | 0.294 [0.273, 0.316] | 0.963 [0.949, 0.974] |
| `D2_room_swap` | room (RIR) | centred | 6000 | 6000 | 0.783 [0.757, 0.808] | 0.766 [0.741, 0.792] | 0.989 [0.977, 0.995] |
| `D3_chain_and_room` | chain **and** room | raw | 6000 | 6000 | 0.004 [0.002, 0.006] | 0.014 [0.009, 0.020] | 0.706 [0.666, 0.745] |
| `D3_chain_and_room` | chain **and** room | LN | 6000 | 6000 | 0.003 [0.001, 0.004] | 0.012 [0.008, 0.017] | 0.706 [0.666, 0.745] |
| `D3_chain_and_room` | chain **and** room | centred | 6000 | 6000 | 0.696 [0.667, 0.728] | 0.654 [0.624, 0.685] | 0.956 [0.943, 0.968] |
| `D4_distance` | distance (near->far) | raw | 3199 | 6000 | 0.059 [0.050, 0.068] | 0.077 [0.068, 0.088] | 0.962 [0.943, 0.976] |
| `D4_distance` | distance (near->far) | LN | 3199 | 6000 | 0.057 [0.048, 0.067] | 0.076 [0.067, 0.086] | 0.962 [0.943, 0.976] |
| `D4_distance` | distance (near->far) | centred | 3199 | 6000 | 0.643 [0.619, 0.668] | 0.627 [0.604, 0.652] | 0.992 [0.978, 0.998] |
| `D5_xchain_both` | chain, on **both** trial types | raw | 3204 | 6000 | 0.521 [0.489, 0.552] | 0.524 [0.490, 0.559] | 0.937 [0.920, 0.952] |
| `D5_xchain_both` | chain, on **both** trial types | LN | 3204 | 6000 | 0.524 [0.493, 0.555] | 0.528 [0.494, 0.561] | 0.937 [0.920, 0.952] |
| `D5_xchain_both` | chain, on **both** trial types | centred | 3204 | 6000 | 0.736 [0.703, 0.769] | 0.692 [0.658, 0.727] | 0.973 [0.960, 0.983] |

`D0` is the load-bearing control: with chain, room and distance all fixed, the **raw** frozen
feature separates 36 unseen talkers at **0.805** [0.777, 0.837] (v16). The talker information is in
there. One chain swap takes that to 0.012; centring brings it back to 0.705. So on the synthetic
axis the chain signature is very largely one additive vector per recording -- and identity survives
underneath it, but only after that vector is removed.

## 5. Trained ceiling on the same frozen features (measurement B)

48-component LDA on the fit half's speaker labels, cosine in the projected space on pairs whose
**both** members are in the test half and whose speakers never appear in the fit half; 20 random
splits; nothing about the encoder changes. `held out by = session` additionally holds out the whole
recording.

| set | arm | held out by | feature | splits | fit spk | test spk | **v16** AUC [2.5, 97.5 over splits] | **v8** AUC | **sv** AUC | EER v16 / v8 / sv |
|---|---|---|---|---|---|---|---|---|---|---|
| dipco | `A_matched` | speaker | raw | 20 | 16 | 16 | 0.674 [0.621, 0.747] | 0.666 [0.601, 0.729] | 0.850 [0.803, 0.907] | 0.370 / 0.377 / 0.223 |
| dipco | `A_matched` | speaker | LN | 20 | 16 | 16 | 0.674 [0.620, 0.744] | 0.667 [0.602, 0.728] | 0.850 [0.803, 0.907] | 0.370 / 0.375 / 0.223 |
| dipco | `A_matched` | speaker | centred | 20 | 16 | 16 | 0.745 [0.699, 0.796] | 0.741 [0.694, 0.794] | 0.876 [0.813, 0.932] | 0.316 / 0.320 / 0.201 |
| dipco | `A_matched` | session | raw | 20 | 20 | 12 | 0.669 [0.626, 0.704] | 0.666 [0.622, 0.701] | 0.859 [0.812, 0.901] | 0.375 / 0.375 / 0.216 |
| dipco | `A_matched` | session | LN | 20 | 20 | 12 | 0.669 [0.628, 0.703] | 0.668 [0.624, 0.704] | 0.859 [0.812, 0.901] | 0.373 / 0.374 / 0.216 |
| dipco | `A_matched` | session | centred | 20 | 20 | 12 | 0.753 [0.717, 0.791] | 0.746 [0.713, 0.797] | 0.885 [0.835, 0.929] | 0.311 / 0.317 / 0.191 |
| dipco | `A_xprox` | speaker | raw | 20 | 16 | 16 | 0.537 [0.468, 0.653] | 0.524 [0.443, 0.628] | 0.871 [0.829, 0.922] | 0.460 / 0.470 / 0.203 |
| dipco | `A_xprox` | speaker | LN | 20 | 16 | 16 | 0.532 [0.457, 0.646] | 0.521 [0.441, 0.620] | 0.871 [0.829, 0.922] | 0.464 / 0.472 / 0.203 |
| dipco | `A_xprox` | speaker | centred | 20 | 16 | 16 | 0.756 [0.710, 0.810] | 0.749 [0.708, 0.806] | 0.909 [0.852, 0.951] | 0.304 / 0.309 / 0.168 |
| dipco | `A_xprox` | session | raw | 20 | 20 | 12 | 0.536 [0.487, 0.573] | 0.518 [0.483, 0.551] | 0.878 [0.829, 0.921] | 0.459 / 0.472 / 0.195 |
| dipco | `A_xprox` | session | LN | 20 | 20 | 12 | 0.529 [0.482, 0.563] | 0.513 [0.478, 0.548] | 0.878 [0.829, 0.921] | 0.465 / 0.475 / 0.195 |
| dipco | `A_xprox` | session | centred | 20 | 20 | 12 | 0.758 [0.721, 0.802] | 0.751 [0.705, 0.799] | 0.916 [0.873, 0.948] | 0.302 / 0.308 / 0.157 |
| notsofar | `A_matched` | speaker | raw | 20 | 39 | 37 | 0.603 [0.530, 0.641] | 0.607 [0.528, 0.660] | 0.999 [0.998, 1.000] | 0.420 / 0.420 / 0.013 |
| notsofar | `A_matched` | speaker | LN | 20 | 39 | 37 | 0.601 [0.526, 0.639] | 0.603 [0.523, 0.657] | 0.999 [0.998, 1.000] | 0.422 / 0.424 / 0.013 |
| notsofar | `A_matched` | speaker | centred | 20 | 39 | 37 | 0.740 [0.691, 0.788] | 0.743 [0.679, 0.813] | 0.994 [0.987, 0.999] | 0.317 / 0.315 / 0.037 |
| notsofar | `A_matched` | session | raw | 20 | 37 | 39 | 0.595 [0.534, 0.636] | 0.598 [0.526, 0.647] | 0.999 [0.998, 1.000] | 0.426 / 0.426 / 0.010 |
| notsofar | `A_matched` | session | LN | 20 | 37 | 39 | 0.593 [0.531, 0.634] | 0.594 [0.521, 0.643] | 0.999 [0.998, 1.000] | 0.428 / 0.430 / 0.010 |
| notsofar | `A_matched` | session | centred | 20 | 37 | 39 | 0.730 [0.689, 0.779] | 0.735 [0.679, 0.793] | 0.994 [0.989, 0.999] | 0.326 / 0.322 / 0.039 |
| notsofar | `A_xprox` | speaker | raw | 20 | 39 | 37 | 0.600 [0.530, 0.643] | 0.611 [0.541, 0.657] | 1.000 [0.999, 1.000] | 0.416 / 0.411 / 0.008 |
| notsofar | `A_xprox` | speaker | LN | 20 | 39 | 37 | 0.595 [0.524, 0.639] | 0.605 [0.534, 0.651] | 1.000 [0.999, 1.000] | 0.419 / 0.416 / 0.008 |
| notsofar | `A_xprox` | speaker | centred | 20 | 39 | 37 | 0.758 [0.709, 0.813] | 0.758 [0.702, 0.828] | 0.996 [0.990, 1.000] | 0.299 / 0.298 / 0.030 |
| notsofar | `A_xprox` | session | raw | 20 | 37 | 39 | 0.593 [0.546, 0.634] | 0.602 [0.549, 0.646] | 1.000 [0.999, 1.000] | 0.421 / 0.418 / 0.007 |
| notsofar | `A_xprox` | session | LN | 20 | 37 | 39 | 0.588 [0.541, 0.629] | 0.597 [0.544, 0.640] | 1.000 [0.999, 1.000] | 0.425 / 0.423 / 0.007 |
| notsofar | `A_xprox` | session | centred | 20 | 37 | 39 | 0.746 [0.706, 0.791] | 0.748 [0.698, 0.798] | 0.996 [0.993, 0.999] | 0.309 / 0.305 / 0.032 |
| ami | `A_xprox` | speaker | raw | 20 | 30 | 30 | 0.688 [0.632, 0.735] | 0.686 [0.628, 0.733] | 0.941 [0.925, 0.955] | 0.346 / 0.348 / 0.128 |
| ami | `A_xprox` | speaker | LN | 20 | 30 | 30 | 0.685 [0.630, 0.731] | 0.683 [0.626, 0.729] | 0.941 [0.925, 0.955] | 0.348 / 0.351 / 0.128 |
| ami | `A_xprox` | speaker | centred | 20 | 30 | 30 | 0.758 [0.699, 0.791] | 0.758 [0.702, 0.789] | 0.966 [0.957, 0.976] | 0.305 / 0.304 / 0.094 |
| ami | `A_xprox` | session | raw | 14 | 55 | 4 | 0.674 [0.601, 0.746] | 0.668 [0.601, 0.748] | 0.948 [0.905, 0.978] | 0.361 / 0.367 / 0.110 |
| ami | `A_xprox` | session | LN | 14 | 55 | 4 | 0.670 [0.596, 0.741] | 0.665 [0.601, 0.742] | 0.948 [0.905, 0.978] | 0.364 / 0.370 / 0.110 |
| ami | `A_xprox` | session | centred | 14 | 55 | 4 | 0.739 [0.608, 0.817] | 0.738 [0.595, 0.818] | 0.977 [0.953, 0.994] | 0.314 / 0.315 / 0.073 |
| synth | `D1_chain_swap` | speaker | raw | 20 | 18 | 18 | 0.677 [0.597, 0.741] | 0.649 [0.553, 0.725] | 0.893 [0.862, 0.927] | 0.355 / 0.380 / 0.186 |
| synth | `D1_chain_swap` | speaker | LN | 20 | 18 | 18 | 0.674 [0.587, 0.738] | 0.650 [0.551, 0.729] | 0.893 [0.862, 0.927] | 0.358 / 0.380 / 0.186 |
| synth | `D1_chain_swap` | speaker | centred | 20 | 18 | 18 | 0.880 [0.840, 0.921] | 0.874 [0.834, 0.918] | 0.918 [0.900, 0.940] | 0.195 / 0.201 / 0.158 |
| synth | `D3_chain_and_room` | speaker | raw | 20 | 18 | 18 | 0.568 [0.474, 0.646] | 0.536 [0.428, 0.629] | 0.888 [0.858, 0.921] | 0.436 / 0.466 / 0.190 |
| synth | `D3_chain_and_room` | speaker | LN | 20 | 18 | 18 | 0.561 [0.461, 0.638] | 0.535 [0.427, 0.632] | 0.888 [0.858, 0.921] | 0.443 / 0.467 / 0.190 |
| synth | `D3_chain_and_room` | speaker | centred | 20 | 18 | 18 | 0.868 [0.825, 0.912] | 0.862 [0.817, 0.907] | 0.915 [0.894, 0.936] | 0.206 / 0.212 / 0.161 |

This is the ceiling a *linear* read-out of the frozen bottleneck can reach: **0.730-0.758 real**
(centred; 0.518-0.688 raw) and **0.862-0.880 synthetic**, against `sv`'s 0.850-1.000 real on the
same protocol. Two consequences. First, the identity direction exists in the frozen features and is
linearly recoverable -- P0's failure is about *readability by cosine*, not about absence. Second,
that ceiling is **below the P3 pass bar of pair AUC >= 0.80 real** (design doc §6), so a
training-only head bolted onto a frozen encoder is not expected to clear P3: R1a's `L_id` has to be
allowed to reshape the shared encoder (which design §4.2 does), and if it does not move, design §7's
real-session-rows fork is the next lever, not a bigger head.

Note also the centring dependence in this table, which is the same story again: synthetic `D3` goes
**0.568 -> 0.868** and DiPCo `A_xprox` **0.537 -> 0.756** purely by subtracting each recording's own
mean before the fit. A per-recording (i.e. per-session, per-chain) mean removal is doing most of the
work that `L_id` will have to do internally -- the same "relative within the session, never an
absolute threshold" shape that design §3.1 requires of capability A and that
`presence_selfcal_README.md` already found for presence.

## 6. The geometry: 2x2 median cosine, and the proximity control

**raw**

| set | cell | n pairs | **v16** median cos [95 % CI] | **v8** median cos [95 % CI] | **sv** median cos [95 % CI] |
|---|---|---|---|---|---|
| dipco | same talker, same far device | 4000 | 0.958 [0.950, 0.963] | 0.955 [0.948, 0.961] | 0.368 [0.327, 0.401] |
| dipco | different talkers, same far device | 4000 | 0.923 [0.916, 0.930] | 0.922 [0.910, 0.932] | 0.058 [0.049, 0.068] |
| dipco | same talker, two far devices | 4000 | 0.915 [0.905, 0.922] | 0.918 [0.910, 0.925] | 0.359 [0.313, 0.398] |
| dipco | different talkers, two far devices | 4000 | 0.912 [0.899, 0.922] | 0.903 [0.891, 0.917] | 0.055 [0.044, 0.067] |
| dipco | same talker, close-talk vs far | 4000 | 0.608 [0.573, 0.643] | 0.627 [0.583, 0.659] | 0.405 [0.359, 0.439] |
| dipco | different talkers, close-talk vs far | 4000 | 0.609 [0.561, 0.651] | 0.624 [0.566, 0.681] | 0.041 [0.034, 0.050] |
| notsofar | same talker, same far device | 4128 | 0.983 [0.976, 0.986] | 0.985 [0.980, 0.987] | 0.517 [0.498, 0.533] |
| notsofar | different talkers, same far device | 6130 | 0.967 [0.956, 0.972] | 0.970 [0.963, 0.974] | 0.059 [0.050, 0.069] |
| notsofar | same talker, two far devices | 6194 | 0.959 [0.940, 0.967] | 0.964 [0.951, 0.971] | 0.482 [0.465, 0.497] |
| notsofar | different talkers, two far devices | 6994 | 0.952 [0.936, 0.960] | 0.958 [0.946, 0.964] | 0.049 [0.040, 0.058] |
| notsofar | same talker, close-talk vs far | 6194 | 0.943 [0.929, 0.951] | 0.943 [0.932, 0.950] | 0.512 [0.499, 0.523] |
| notsofar | different talkers, close-talk vs far | 6994 | 0.943 [0.932, 0.950] | 0.941 [0.931, 0.948] | 0.043 [0.031, 0.052] |
| ami | same talker, same far device | 9868 | 0.978 [0.976, 0.981] | 0.979 [0.976, 0.981] | 0.452 [0.441, 0.463] |
| ami | different talkers, same far device | 23047 | 0.966 [0.960, 0.970] | 0.966 [0.961, 0.970] | 0.094 [0.083, 0.107] |
| ami | same talker, close-talk vs far | 19736 | 0.949 [0.945, 0.953] | 0.943 [0.937, 0.947] | 0.467 [0.457, 0.477] |
| ami | different talkers, close-talk vs far | 23968 | 0.944 [0.939, 0.948] | 0.936 [0.931, 0.941] | 0.084 [0.073, 0.095] |
| synth | same talker, everything fixed | 3191 | 0.997 [0.997, 0.997] | 0.996 [0.996, 0.997] | 0.597 [0.580, 0.613] |
| synth | different talkers, everything fixed | 6000 | 0.992 [0.992, 0.993] | 0.991 [0.991, 0.992] | 0.183 [0.173, 0.192] |
| synth | same talker, chain swapped | 3204 | 0.952 [0.949, 0.954] | 0.958 [0.955, 0.961] | 0.305 [0.285, 0.323] |
| synth | different talkers, chain swapped | 6000 | 0.951 [0.950, 0.952] | 0.957 [0.955, 0.958] | 0.069 [0.062, 0.075] |
| synth | same talker, room swapped | 6000 | 0.986 [0.985, 0.986] | 0.985 [0.983, 0.985] | 0.527 [0.510, 0.542] |
| synth | different talkers, room swapped | 6000 | 0.982 [0.981, 0.983] | 0.980 [0.979, 0.981] | 0.150 [0.138, 0.160] |
| synth | same talker, near vs far | 3199 | 0.894 [0.866, 0.913] | 0.901 [0.866, 0.921] | 0.527 [0.509, 0.543] |
| synth | different talkers, near vs far | 6000 | 0.886 [0.870, 0.900] | 0.903 [0.882, 0.914] | 0.079 [0.071, 0.087] |

**centred**

| set | cell | n pairs | **v16** median cos [95 % CI] | **v8** median cos [95 % CI] | **sv** median cos [95 % CI] |
|---|---|---|---|---|---|
| dipco | same talker, same far device | 4000 | 0.498 [0.344, 0.610] | 0.497 [0.371, 0.614] | 0.250 [0.205, 0.289] |
| dipco | different talkers, same far device | 4000 | -0.187 [-0.273, -0.111] | -0.123 [-0.185, -0.067] | -0.117 [-0.131, -0.103] |
| dipco | same talker, two far devices | 4000 | 0.136 [0.029, 0.301] | 0.168 [0.053, 0.300] | 0.249 [0.198, 0.293] |
| dipco | different talkers, two far devices | 4000 | -0.064 [-0.203, 0.042] | -0.084 [-0.185, -0.022] | -0.105 [-0.119, -0.088] |
| dipco | same talker, close-talk vs far | 4000 | 0.020 [-0.005, 0.040] | 0.019 [-0.005, 0.044] | 0.310 [0.266, 0.347] |
| dipco | different talkers, close-talk vs far | 4000 | -0.041 [-0.055, -0.029] | -0.038 [-0.052, -0.017] | -0.117 [-0.133, -0.099] |
| notsofar | same talker, same far device | 4128 | 0.387 [0.309, 0.487] | 0.366 [0.284, 0.460] | 0.338 [0.301, 0.367] |
| notsofar | different talkers, same far device | 6130 | -0.135 [-0.195, -0.076] | -0.130 [-0.179, -0.083] | -0.194 [-0.207, -0.184] |
| notsofar | same talker, two far devices | 6194 | 0.142 [0.047, 0.251] | 0.143 [0.068, 0.226] | 0.321 [0.280, 0.349] |
| notsofar | different talkers, two far devices | 6994 | -0.003 [-0.040, 0.028] | -0.025 [-0.056, 0.016] | -0.180 [-0.194, -0.169] |
| notsofar | same talker, close-talk vs far | 6194 | 0.083 [0.051, 0.111] | 0.085 [0.048, 0.121] | 0.363 [0.324, 0.390] |
| notsofar | different talkers, close-talk vs far | 6994 | -0.027 [-0.048, -0.006] | -0.045 [-0.062, -0.029] | -0.197 [-0.211, -0.185] |
| ami | same talker, same far device | 9868 | 0.400 [0.352, 0.453] | 0.396 [0.348, 0.450] | 0.313 [0.298, 0.325] |
| ami | different talkers, same far device | 23047 | -0.029 [-0.057, 0.005] | -0.039 [-0.066, -0.006] | -0.132 [-0.136, -0.127] |
| ami | same talker, close-talk vs far | 19736 | 0.172 [0.146, 0.197] | 0.166 [0.138, 0.194] | 0.352 [0.339, 0.364] |
| ami | different talkers, close-talk vs far | 23968 | -0.005 [-0.021, 0.009] | -0.014 [-0.028, -0.001] | -0.133 [-0.138, -0.128] |
| synth | same talker, everything fixed | 3191 | 0.592 [0.540, 0.643] | 0.607 [0.558, 0.659] | 0.492 [0.475, 0.510] |
| synth | different talkers, everything fixed | 6000 | -0.016 [-0.045, 0.011] | -0.016 [-0.054, 0.016] | -0.025 [-0.029, -0.021] |
| synth | same talker, chain swapped | 3204 | 0.326 [0.266, 0.398] | 0.281 [0.223, 0.358] | 0.290 [0.273, 0.307] |
| synth | different talkers, chain swapped | 6000 | -0.021 [-0.043, -0.006] | -0.028 [-0.046, -0.011] | -0.013 [-0.018, -0.009] |
| synth | same talker, room swapped | 6000 | 0.493 [0.444, 0.542] | 0.503 [0.459, 0.550] | 0.446 [0.431, 0.461] |
| synth | different talkers, room swapped | 6000 | -0.016 [-0.053, 0.016] | -0.017 [-0.058, 0.017] | -0.021 [-0.031, -0.012] |
| synth | same talker, near vs far | 3199 | 0.165 [0.129, 0.204] | 0.152 [0.119, 0.189] | 0.492 [0.474, 0.507] |
| synth | different talkers, near vs far | 6000 | -0.012 [-0.024, -0.001] | -0.011 [-0.023, 0.000] | -0.020 [-0.023, -0.014] |

Every raw cell for v16 / v8 sits between 0.90 and 1.00: the pooled bottleneck is dominated by a
common mean direction, and all the structure lives in the last two decimals. Differencing against
the fully-fixed cell gives the factor ladder -- the proximity control that stage P0 was asked for:

| set | factor changed between the two 3 s windows | v16 median cos | Δ vs the fixed cell | v8 Δ |
|---|---|---|---|---|
| synth | *(reference: sameT_sameChain_sameRoom)* | 0.997 | -- | -- |
| synth | talker | 0.992 | -0.004 | -0.005 |
| synth | room (RIR) | 0.986 | -0.011 | -0.012 |
| synth | device chain | 0.952 | -0.045 | -0.038 |
| synth | distance (near->far) | 0.894 | -0.103 | -0.095 |
| dipco | *(reference: sameT_sameC)* | 0.958 | -- | -- |
| dipco | talker | 0.923 | -0.034 | -0.033 |
| dipco | far device | 0.915 | -0.043 | -0.037 |
| dipco | proximity (close-talk vs far) | 0.608 | -0.350 | -0.328 |
| notsofar | *(reference: sameT_sameC)* | 0.983 | -- | -- |
| notsofar | talker | 0.967 | -0.016 | -0.014 |
| notsofar | far device | 0.959 | -0.024 | -0.020 |
| notsofar | proximity (close-talk vs far) | 0.943 | -0.040 | -0.042 |
| ami | *(reference: sameT_sameC)* | 0.978 | -- | -- |
| ami | talker | 0.966 | -0.013 | -0.013 |
| ami | proximity (headset vs Array1-01) | 0.949 | -0.029 | -0.036 |

On the synthetic axis, where every factor can be varied one at a time, the ordering is **distance
(-0.103) > chain (-0.045) > room (-0.011) > talker (-0.004)**: a distance change moves the
representation ~26x further than a talker change, a chain change ~11x further. On the real corpora
proximity is again the dominant axis (DiPCo -0.350 vs -0.034 for the talker) and the device costs
slightly more than the talker does (-0.043 vs -0.034). This is the same conclusion
`dist_cue_anatomy_README.md` and `presence_probe_README.md` reached about the distance and presence
cues, now for identity: the bottleneck's geometry is organised by *where the talker is and what
recorded them*, and talker identity is a low-order perturbation on top of that.

The centred panel adds the control that matters for caveat 1: on DiPCo two **different** talkers on
the **same** far device read -0.187 [-0.273, -0.111], while two different talkers on **two** far
devices read -0.064 [-0.203, 0.042]; on NOTSOFAR the same contrast is -0.135 [-0.195, -0.076]
against -0.003 [-0.040, 0.028]. Same comparison, one within-recording and one across; the difference
is what centring itself puts there.

For completeness, the two confound controls that state the same result most directly:

| set | arm | feature | n pos | n neg | **v16** AUC [95 % CI] | **v8** AUC [95 % CI] | **sv** AUC [95 % CI] |
|---|---|---|---|---|---|---|---|
| dipco | `C1_chan_fixed` | raw | 4000 | 4000 | 0.609 [0.582, 0.636] | 0.607 [0.582, 0.634] | 0.945 [0.907, 0.972] |
| dipco | `C1_chan_fixed` | centred | 4000 | 4000 | 0.625 [0.592, 0.657] | 0.621 [0.591, 0.651] | 0.961 [0.918, 0.987] |
| notsofar | `C1_chan_fixed` | raw | 4128 | 6130 | 0.634 [0.596, 0.674] | 0.646 [0.609, 0.687] | 0.998 [0.995, 0.999] |
| notsofar | `C1_chan_fixed` | centred | 4128 | 6130 | 0.646 [0.610, 0.689] | 0.653 [0.611, 0.699] | 0.988 [0.972, 0.997] |
| ami | `C1_chan_fixed` | raw | 9868 | 23047 | 0.605 [0.594, 0.616] | 0.610 [0.598, 0.622] | 0.958 [0.950, 0.966] |
| ami | `C1_chan_fixed` | centred | 9868 | 23047 | 0.630 [0.614, 0.645] | 0.636 [0.620, 0.654] | 0.981 [0.974, 0.986] |
| dipco | `C3_channel_vs_talker` | raw | 4000 | 4000 | 0.503 [0.486, 0.517] | 0.505 [0.487, 0.520] | 0.066 [0.032, 0.109] |
| dipco | `C3_channel_vs_talker` | centred | 4000 | 4000 | 0.441 [0.424, 0.455] | 0.442 [0.424, 0.457] | 0.040 [0.014, 0.079] |
| notsofar | `C3_channel_vs_talker` | raw | 6130 | 6194 | 0.543 [0.500, 0.594] | 0.538 [0.496, 0.585] | 0.003 [0.001, 0.006] |
| notsofar | `C3_channel_vs_talker` | centred | 6130 | 6194 | 0.420 [0.396, 0.443] | 0.411 [0.382, 0.438] | 0.013 [0.003, 0.031] |

`C1` (device *and* proximity held fixed) is the real-corpus analogue of `D0`: 0.605-0.646 raw. So
the real features do separate talkers -- as long as the channel never changes, which is precisely
what R4 forbids relying on. `C3` prices channel against talker head-on: > 0.5 means "two different
talkers on one device look more alike than one talker on two devices", and v16 / v8 read 0.503 /
0.505 (DiPCo) and 0.543 / 0.538 (NOTSOFAR) while `sv` reads 0.066 and 0.003.

## 7. The P0 answer

**No, not separably, and not zero-shot -- so the design's §7 rule fires on its `< 0.60` branch for
the real corpora and on the intermediate branch for the synthetic controls.** On DiPCo / AMI /
NOTSOFAR the frozen v16 and v8 bottleneck ranks a *device* match at or above a *talker* match:
proximity-matched zero-shot pair AUC is 0.487 / 0.500 (DiPCo, v16 / v8) and 0.454 / 0.457 (NOTSOFAR,
CI excluding 0.5), the corpora's own close-talk/far pairing inverts outright (0.154-0.383),
per-recording centring lifts it only to 0.540-0.586 (primary arms) / 0.534-0.588 (symmetric arms;
0.498-0.533 raw), the best real cell any feature variant reaches is 0.588 [0.574, 0.602] (AMI
`A_xprox_both`, centred, v8) -- point estimate under the line, CI upper edge on it -- the best EER
on any real identity arm for v16 / v8 is 0.438 (NOTSOFAR `A_matched`, centred, v8), and the same
pairs give the `sv` reference 0.94-1.00 -- so the pairs are easy and the features are the problem;
**R1a's `L_id` is the first thing to train and R1b waits**. On the synthetic controls the same
features land in the 0.60-0.75 intermediate band, at its top (`D1` centred 0.705 [0.674, 0.738],
`D3` 0.696 [0.667, 0.728], symmetric `D5` 0.736 [0.703, 0.769], all v16, all near/near) and the
controls say why the two verdicts differ: identity *is* in the raw features when nothing else moves
(`D0` 0.805 [0.777, 0.837] on 36 unseen speakers), a single chain swap destroys it (`D1` raw 0.012),
and subtracting one vector per recording restores most of it synthetically (0.012 -> 0.705) but
almost none of it on real chains (0.454 -> 0.577) -- i.e. the synthetic chain offset is largely
additive in the pooled bottleneck and the real one is not, which is the same synthetic-to-real chain
wall this programme has hit at every previous rung. The caveat to hold onto: the primary arms pair
cross-recording positives against within-recording negatives, and per-recording centring is not
neutral under that asymmetry (the centred 2x2 panel above), so the `centred` column of `A_matched` /
`A_xprox` / `D1` / `D3` is partly mechanical -- the symmetric `*_both` / `D5` arms are the honest
read, they differ by at most ~0.03 AUC, and they move neither verdict.

| | zero-shot best (any variant) | design §7 band | decision |
|---|---|---|---|
| DiPCo / AMI / NOTSOFAR | 0.588 [0.574, 0.602] | **< 0.60** | train `L_id` first; **R1b waits** |
| synthetic controls | 0.736 [0.703, 0.769] | 0.60-0.75 (intermediate) | proceed only as R1a's *loss* target, not as a ready feature |
| `sv` reference, same pairs | 0.937-0.999 | -- | the pairs are solvable; the bottleneck is what fails |

`v19c_diagnostics/objective_landscape/LANDSCAPE.md` listed identity as this programme's unmeasured
prerequisite. It is now measured, and the answer is the pessimistic branch: the frozen bottleneck is
a proximity-and-channel representation with talker identity as a residual.

Corollary for R1a's gate: the *trained* linear ceiling on these frozen features is 0.730-0.758 real
(§5), i.e. below P3's `pair AUC >= 0.80 real`. Heads-only on a frozen encoder is not enough on
paper; P3 needs the encoder to move.

## 8. Caveats

1. **Pairing asymmetry x centring (the one that matters).** As above: primary-arm positives cross
   recordings, primary-arm negatives do not, and per-recording mean removal pushes within-recording
   pair cosines toward zero-mean. Quantified in the centred 2x2 panel (DiPCo -0.187 within vs -0.064
   across, the same different-talker contrast) and neutralised by the `*_both` / `D5` arms.
2. **`raw_rawlvl` is not an independent variant for v16 / v8.** `DPCRN` inherits `Unet`'s
   instance-LayerNorm on the input spectrogram (`puresound/nnet/unet.py:96` used at
   `puresound/nnet/dpcrn.py:296`), so absolute input level is removed before the encoder: over all
   A/C/D cells `|AUC(raw) - AUC(raw_rawlvl)| <= 2e-5`, and on the embeddings `max |cos(raw,
   raw_rawlvl) - 1| = 3e-6` with a norm ratio of 1.000. The check is therefore vacuous for the
   bottleneck (and it also means the channel/proximity offsets measured here are *spectral*, not
   level). It is not vacuous for `sv`, which loses up to 0.459 AUC at file-native level (NOTSOFAR
   `A_xprox` 0.999 -> 0.540) -- the -28 dBFS / -22 dBFS normalisation was load-bearing.
3. **n per cell.** Pairs are plentiful (4000-23 968 real, 3191-6000 synthetic) but the *independent*
   units are few: 10 DiPCo sessions, 19 NOTSOFAR meetings, 60 AMI meetings, 36 synthetic anchor
   speakers. Every v16 / v8 CI half-width in the A/D tables is <= 0.044 (widest: NOTSOFAR
   `A_matched` raw), so this protocol does satisfy P3's "CI half-width <= 0.05"; the only wider
   cells in the whole probe are `sv` at file-native level (up to 0.160, DiPCo `A_xprox`
   `raw_rawlvl`).
4. **AMI has one far device.** The download carries no second array, so `A_matched`, `A_xchan_both`,
   `C2` and `C3` have zero positives or zero negatives there and are `--` (see the anomalies
   section). AMI therefore only ever speaks about the close/far (`xprox`) regime.
5. **AMI's `held out by = session` Table B row degenerates**: 14 of 20 splits usable, median 4 test
   speakers (against 30 under the `speaker` split). Read the `speaker` row.
6. **`CT` / `HS` is a device *class*, not one file.** Each participant wears their own unit, so the
   centring group for the close channel pools that session's headsets. Harmless for these arms
   because no real arm ever pairs close with close, but it means the close side of `A_xprox` has its
   session-level class mean removed rather than a per-headset mean.
7. **The synthetic identity arms are near/near.** `D1` / `D3` / `D5` compare two near captures
   (0.49-0.99 m) -- the favourable proximity regime. With distance varied and chain fixed the same
   features read `D4` 0.643 centred / 0.059 raw. Nothing here measures identity across a distance
   change *and* a chain change at once.
8. **Measurement B is a ceiling, not a readout.** The LDA sees every segment of the fit half, both
   proximities and all channels, so it can find a channel-invariant direction with the channel
   variation available to it; it is speaker-disjoint but not chain-disjoint.
9. **Pair sampling is not reproducible across processes.** `make_pairs` seeds from `abs(hash((sess,
   seed)))` and Python salts `str.__hash__` per process, so re-running `report` draws a different
   pair sample. Measured jitter between the two full runs of the same command: see the anomalies
   section.
10. **What this probe does *not* test**: overlapped speech (excluded by construction), segments
    shorter than 3 s or with < 1 s of speech (dropped), any frame-level or causal read-out (all
    pooling is over the whole 3 s), and any checkpoint other than v16 ep19 / v8.

## 9. Anomalies, verbatim from the logs

From `extract_gpu.log` (once per DPCRN checkpoint load; `load_loss_func=False`, so the loss buffers
are expected to be absent):

```
loss_func_list.1.stft_losses.0.window is not in the model.
loss_func_list.1.stft_losses.1.window is not in the model.
loss_func_list.1.stft_losses.2.window is not in the model.
loss_func_list.2.hann_window is not in the model.
Loaded params is ok.
```

Segments dropped by the `< 100 active frames` (< 1 s of speech) rule, from `extract_all.log`:

```
dipco/v16: kept 2220/2256 segments
notsofar/v16: kept 1688/1688 segments
ami/v16: kept 4526/4568 segments
synth/v16: kept 1944/1944 segments
```

An interrupted first extraction pass, also in `extract_all.log` -- `synth/v8` started at 09:56:25
and never reported `DONE`, and `synth/v16` then started again at 09:58:06 instead of being skipped:

```
=== START synth / v8 09:56:25 ===
=== START synth / v16 09:58:06 ===
```

This was the synthetic set being re-indexed in between (`segments_synth.jsonl` and
`tables/synth_chains.json` are stamped 09:57 / 09:58, after the first `synth/v16` pass finished at
09:56:25), so the re-extraction was required, not duplicated work. Checked, so that it is on the
record: for all four sets the `v16`, `v8` and `sv` `.npz` files hold **exactly the same kept-segment
list in the same order**, and every kept `sid` is present in the current `segments_*.jsonl`. No
embedding in `tables/results.json` comes from a stale index.

Empty cells in `report_full.log` (identical in the re-run log), all of them AMI's single far device
(`A_matched` / `A_xchan_both` have no two-far-device pairs; `C2` / `C3` have no
same-talker-different-far-device negatives):

```
ami       v16   A_matched              raw             -- [--,--]     --      0  23047
ami       v16   A_xchan_both           raw             -- [--,--]     --      0      0
ami       v16   C2_channel_move        raw             -- [--,--]     --   9868      0
ami       v16   C3_channel_vs_talker   raw             -- [--,--]     --  23047      0
ami       v16   A_matched              speaker  raw           --    0
```

**Re-run reproducibility.** `report` was run a second time on the identical `.npz` inputs to check
that `tables/` is complete and current. It reproduced 738 rows with identical keys, no cell crossed
either decision line (0.60 or 0.75) on any v16 / v8 arm, and `A_matched` / `A_xchan_both` / `C2` / `C3`
stayed empty on AMI -- but it did not reproduce bit-identically, for the reason in caveat 9. Measured
over all 1056 point statistics: median `|delta|` 0.0018, p90 0.0068, max 0.101. Over the 276
headline-arm AUC cells: median 0.0018, max 0.0135 (DiPCo `A_xchan_both` centred v8, 0.550 -> 0.537).
The unstable end is the `centred` 2x2 medians on DiPCo -- 10 sessions, 4 talkers each, 400 pairs per
session per cell -- where `sameT_diffC` moved 0.236 -> 0.136 and `diffT_sameC` -0.247 -> -0.187
between draws. Synthetic `n pos` counts also move by +-9 (`D1` 3195 -> 3204, `D0` 3199 -> 3191)
because that set takes the sampling path and de-duplicates unique pairs across draws. **`tables/`
holds the second run and every number in this note comes from it**; treat the third decimal of any
single cell as noise and the DiPCo `centred` medians as +-0.10.

`tables/TABLES.md` describes its own CIs as a "1000-draw cluster bootstrap" (the code's default,
`cluster_bootstrap(..., b=1000)` and `median_ci(..., b=1000)`); the method notes drafted alongside
the run say 2000. **1000 is what ran.**
