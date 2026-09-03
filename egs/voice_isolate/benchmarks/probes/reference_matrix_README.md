# Reference-provenance matrix — the "room tone" that unlocked cold start was speech

2026-09-03. Script beside this note: `reference_matrix_probe.py` (blocks of 5 checkpoints,
`FIELD_BLOCK.md` protocol, dry_blend 1.0, all 58 cold-start clips of set v3). Exploratory
scripts that led here (sliding window over the 90D gap, anchor-type probe) live in the session
scratchpad; their decisive numbers are reproduced in §0–§1. Companion: the warm-context WER probe
(`--context` flag in `eval_dawn_chorus.py` / `eval_wer.py`), §5.

## 0. What this corrects

`v17_round_design.md` §2b and `field_test_vector/COLDSTART_V2.md` state that **3 s of real room
ambience (no voice) unlocks cold-start suppression (−1.2 → −16 dB)**, and build on it the v17
training knob, the "ambient = deployment steady state" column, and the deployment advice "prime
new streams with buffered room audio". The pads behind that number were drawn at random from the
only ≥ 3 s un-annotated gap in set v3, 90D 98.4–107.0 s. That gap is not room tone:

| 90D gap segment | level | content (250 ms level profile + whisper) |
|---|---|---|
| 98.5–101.0 | −62 dBFS (frames down to −78) | floor |
| **101.4–103.9** | **−47 dBFS** | **an unlabelled utterance** (Mandarin "對隨便隨便隨便唸", f0 ≈ 100 Hz — an aside between takes) |
| 104.9–107.0 | −61 dBFS (frames down to −80) | floor |

Sliding a 2 s window across the gap and priming the six 90D cold-far clips with it
(v8 ep19 / v16 ep19, median suppression, dB):

| window | level | v8 far | v16 far |
|---|---|---|---|
| 98.5–100.5 (floor) | −69 | −0.15 | −0.51 |
| 100.0–102.0 | −52 | −11.86 | −15.28 |
| **101.5–103.5 (utterance)** | −47 | **−25.95** | **−21.17** |
| 104.0–106.0 (floor) | −61 | −2.51 | −4.16 |
| 105.0–107.0 (floor) | −62 | −3.53 | −2.00 |

Every pad that "unlocked" contained the utterance. The 180D gap (62.9–64.9 s) is true floor at
−79 dBFS and unlocks nothing; qvf_gym's tail 29.7–34.1 contains speech too ("Maybe I should work
for you…"), and qvf_keep_in_touch's 2.2 s lead-in is floor. So the v2 `ambient` column was a
lottery over whether the draw caught the utterance (hence the ±2.5 dB per-checkpoint spread and
the non-monotone pad-length sweep). **Room tone does not unlock cold start; speech through the same
room/chain does.** The v17 reading "the model self-calibrates against the room's true signature,
not against energy" is withdrawn. The mechanism is the one `near-anchor-dependence` (2026-08-14)
named — an anchor — with one new fact: the anchor need not be the user. Any talker through the
same chain, even at −47 dBFS, is enough.

## 1. Anchor type (90D cold-far clips, single ckpt, 3 s pads unless noted)

| pad | level | v8 far median | v16 far median | v8 near median |
|---|---|---|---|---|
| none | | −0.22 | −1.21 | −0.10 |
| white noise at floor (−78) | −78 | −0.13 | −0.28 | −0.05 |
| white noise at the utterance's level | −47 | −0.14 | −0.24 | −0.08 |
| the 90D-gap utterance (2.5 s) | −47 | **−24.16** | **−21.76** | −0.16 |
| same utterance time-reversed | −47 | −13.41 | −12.26 | −0.79 |
| near speech, same room, other recording (180d_near1) | −39 | **−21.56** | **−27.58** | −1.20 |
| same, scaled to −47 | −47 | −21.61 | −27.92 | −1.36 |
| far speech, same room, other recording (180d_far1) | −54 | −11.84 | −6.56 | −2.35 |
| far speech, other chain (qvf_gym_far1) | −41 | −0.16 | −0.18 | −0.08 |

Level is irrelevant (scale-equivariance, v17 §1). Noise is irrelevant. Speech through the same
chain is the whole effect; time-reversal keeps half (spectral envelope carries part, temporal
structure the rest). Speech through another chain carries nothing to the device clips.

## 2. Block results — far side (20 cold-far clips, block means per clip, paired vs `none`)

| pad (3 s) | v8 block median | ≤ −6 dB | Δ vs none, p | v16 block median | ≤ −6 dB | Δ vs none, p |
|---|---|---|---|---|---|---|
| none | −1.06 | 4/20 | | −1.71 | 4/20 | |
| true floor, same chain | −1.79 | 6/20 | −0.04, p=0.08 | −1.56 | 5/20 | +0.03, p=0.76 |
| digital silence | −5.62 | 9/20 | −2.58, p<0.001 | −5.73 | 9/20 | −1.92, p<0.001 |
| **near speech, same room / chain, other recording** | **−17.17** | **18/20** | **−15.12, p<0.001** | **−15.67** | **16/20** | **−12.03, p<0.001** |
| far speech, same room / chain, other recording | −6.35 | 11/20 | −2.75, p=0.001 | −4.15 | 7/20 | −1.09, p=0.03 |
| the 90D utterance (device clips, n=14) | −20.26 | 14/14 | −17.73 | −19.66 | 13/14 | −16.90 |
| near speech, OTHER chain | −6.47 | 11/20 | −4.49 | −4.35 | 8/20 | −1.37 |
| `stream` (true preceding 3 s, n=19) | −15.90 | 15/19 | −11.74 | −13.01 | 15/19 | −7.83 |

The floor row is the null the whole v17 round was built on. "Near speech, other chain" looks
useful in the median but is a different phenomenon: on device clips it does nothing
(qvf_gym_near1 → 90D/180D/270D far: ≈ 0), on QVF clips 180d_near1 kills everything, keep included
(§4). Digital silence is a mild suppress-bias on the device chain and a catastrophe on the QVF
chain — it is not a neutral prefix and must never be used as one.

## 3. How much anchor, how long does it last (device clips, 3 s near speech from the other
recording of the same room; 14 far / 17 near clips)

| anchor length | v8 far | v16 far | v8 near | v8 near violations (< −3 dB) |
|---|---|---|---|---|
| none | −0.96 | −1.61 | −0.18 | 0/17 |
| 0.25 s | −0.53 | −1.11 | −0.14 | 0 |
| 0.5 s | −0.81 | −1.52 | −0.19 | 0 |
| 1 s | −13.14 | −9.38 | −0.50 | 0 |
| **2 s** | **−19.04** | **−16.93** | **−2.01** | **4** |
| 3 s | −19.25 | −17.25 | −1.56 | 6 |
| 5 s | −20.02 | −18.46 | −0.59 | 3 |
| 8 s | −15.67 | −13.65 | −0.43 | 1 |

| 3 s anchor, then N s of true floor | v8 far | v16 far | v8 near |
|---|---|---|---|
| 0 s | −19.25 | −17.25 | −1.56 |
| 2 s | −14.56 | −12.38 | −0.48 |
| 5 s | −9.83 | −6.60 | −0.20 |
| **10 s** | **−0.92** | **−1.06** | −0.08 |
| 20 s | −0.25 | −0.28 | −0.08 |

Three facts, replicated on both model families:

1. **The anchor needs ~1 s of speech and saturates at 2 s.** Half a second does nothing.
2. **It is forgotten within 10 s of quiet.** Half the effect is gone after 5 s of true floor. The
   streaming state does not hold a room reference; it holds a recent-speech context with a
   time constant of a few seconds. "Never reset state" buys nothing across a normal pause.
3. **The anchor is not free on the keep side.** The same 2–3 s near-speech prefix that unlocks far
   suppression deletes the *next* near talker: v8 device near clips go from 0 to 4–6 violations
   (2 s / 3 s anchor), v16 from 0 to 3. The talker in the anchor is not the talker in the clip
   (other recording, same room, same 30–50 cm) — so the model is not keeping "near"; it is keeping
   "whoever/whatever it just heard" and treating a change as a foreground change. This is the
   interjection deletion of `presence_selfcal_README` and the 90d_near4 `stream` row (−3.5 dB after
   3 s of far speech), generalised.

## 4. QVF chain (per-clip block means, dB)

| clip | side | none v8 / v16 | floor (QVF) | near speech (QVF, other recording) | stream (own preceding 3 s) | silence |
|---|---|---|---|---|---|---|
| qvf_price_far1 | far | −3.64 / −3.62 | −16.60 / −11.89 | −29.86 / −5.52 | −2.35 / −2.09 | −54.98 / −62.82 |
| qvf_plumbing_far1 | far | −1.53 / −1.07 | −1.91 / −1.15 | −7.26 / −3.01 | −19.32 / −8.04 | −20.84 / −3.41 |
| qvf_scenario2_far1 | far | −1.34 / −1.25 | −1.34 / −1.25 | −1.31 / −1.25 | — | −1.36 / −1.41 |
| qvf_scenario3_far1 | far | −0.94 / −0.99 | −0.94 / −0.98 | −0.94 / −0.94 | −0.92 / −0.95 | −0.93 / −1.13 |
| qvf_price_near1 | near | −7.97 / −8.34 | −4.18 / −4.80 | −2.64 / −1.22 | — | −58.66 / −40.95 |
| qvf_keep_in_touch_near1 | near | −7.68 / −6.68 | **−26.73 / −12.44** | −11.47 / −2.33 | **−20.71 / −12.22** | −70.39 / −75.69 |
| qvf_keep_in_touch_dt1 | dt | −9.71 / −8.94 | −7.36 / −7.87 | −5.58 / −4.54 | −7.23 / −5.71 | −55.82 / −56.55 |
| qvf_gym_near1 | near | −3.38 / −1.76 | −2.72 / −2.15 | −0.91 / −0.61 | −2.99 / −1.30 | −57.12 / −44.43 |

* **The scenario clips are context-insensitive** — every prefix, silence included, leaves
  scenario1/2/3 within ±0.05 dB of cold start, on both models, while every other QVF clip swings by
  tens of dB. The scenario3 wall (ours −0.9 vs QVF2.2's −29.7) is not a missing-reference problem;
  nothing the model hears before the clip changes its decision. That leaves the chain
  (compression + EQ, `eq_probe_README`) as the only live explanation, which is what v11b acted on.
* **QVF floor is a suppress-bias, not a reference.** 2 s of the recording's own quiet lead-in
  deletes qvf_keep_in_touch_near1 by 12–27 dB (this is what happens when the full file is
  streamed — the `stream` row says the same), and pushes price_far1 down without any talker
  information. On this chain the model reads "quiet, then sound" as "suppress".
* Same-chain near speech heals the QVF keep violations (price_near1 −8 → −1..−3, gym_near1 → ≈0)
  and unlocks price_far1 on v8 (−30) but not v16 (−5.5): not a mechanism we can build on.

## 5. Keep side / WER with warm context (`--context` in `eval_dawn_chorus.py`, `eval_wer.py`)

Same utterances, same recogniser (faster-whisper large-v3), dry_blend 1.0, single ckpt (ep19)
per family; the comparison is paired per utterance within a checkpoint, so the single-checkpoint
caveat of `FIELD_BLOCK.md` does not apply to the *differences*. `self` = the utterance's own mix
prepended (the model has heard the scene including the user); `background` = ~3 s of the mix's
foreground-inactive frames prepended (room + bystanders, never the user; 74/200 moderate
utterances had < 0.3 s of such frames and fell back to cold). Only the utterance itself is scored.

| | Dawn WER / del, v8 | Dawn WER / del, v16 | moderate WER / del, v8 | moderate WER / del, v16 |
|---|---|---|---|---|
| raw mix | 0.184 / 0.082 | 0.184 / 0.082 | 0.582 / 0.212 | 0.582 / 0.212 |
| cold (`none`) | 0.333 / 0.229 | 0.296 / 0.207 | 0.434 / 0.215 | 0.420 / 0.231 |
| `self` | 0.320 / 0.210 | 0.276 / 0.173 | 0.392 / 0.194 | 0.386 / 0.195 |
| `background` | 0.374 / 0.258 | 0.334 / 0.235 | 0.456 / 0.219 | 0.452 / 0.216 |

Paired per-utterance deletion, warm − cold (Wilcoxon signed-rank):

| | Dawn v8 | Dawn v16 | moderate v8 | moderate v16 |
|---|---|---|---|---|
| `self` | −0.019, 173↓/92↑, p=4e-4 | −0.034, 167↓/80↑, p=4e-7 | −0.021, 55↓/34↑, p=0.004 | −0.037, 58↓/33↑, p=0.002 |
| `background` | **+0.028**, 98↓/160↑, p=3e-4 | **+0.029**, 79↓/148↑, p=8e-6 | +0.004, p=0.43 | −0.016, p=0.45 |

Both families, same reading as §3.3: having heard the *user* helps keep a little (−0.02..−0.04
deletion, WER −0.013..−0.042); having heard only *bystanders* through the same real chain hurts
keep by about as much on Dawn (real recordings), and does nothing on the synthetic moderate set
(where "background" is RIR-convolved interferers + noise, not a real chain). The streaming state is
a recent-talker template: the right talker in it protects the user, the wrong talker in it deletes
the user. Neither direction is large enough to change a gate verdict, and the harmful direction is
the one deployment produces at every cold start (bystander speaks first).

## 6. Consequences for the decision problem

1. **Cold start has no reference-free solution and no "room calibration" solution either.**
   The only thing that moves it is another talker through the same chain within the last few
   seconds. That is the deployment truth: the model suppresses a lone far talker only if someone
   has just spoken. The v17 deployment advice ("prime with buffered room audio") is withdrawn;
   buffered *speech* would work for ~5–10 s and is not available at stream start. And §5 shows
   the sign of the context matters: bystander speech before the user raises Dawn deletion by
   +0.03 on both families.
2. **What the state actually holds is a short-memory "recent foreground" template, not a room or
   distance reference.** It unlocks suppression of others AND deletes a new near talker; it decays
   in seconds. The interjection deletion and the cold-start passthrough are two faces of the same
   mechanism. Any fix to one on this architecture moves the other.
3. **The QVF wall is chain, full stop.** Context-insensitivity of the scenario clips closes the
   "QVF needs its room's reference" hypothesis; `eq_probe` already accounts for the phantom.
4. **Protocol.** The v2/`FIELD_BLOCK` `ambient` column must be re-specified as an explicit
   `anchor` condition (3 s near speech from another recording of the same room) and a true
   `floor` condition; the 90D gap must be re-annotated (101.4–103.9 s is speech). Until then the
   `ambient` numbers in `COLDSTART_V2.md` / `FIELD_BLOCK.md` are read as "anchor, sometimes".
5. **Axis 5 of `v12_candidate_axes.md` (explicit reference architecture) stands, but with a
   different target:** the reference to learn is not "this room's tone" (the model already ignores
   it) — it is a *persistent* representation of "the user" that survives 10 s of silence and
   does not transfer to the next talker. Whether that is buildable without enrollment is the
   question the next design round has to answer before any training.
