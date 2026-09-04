# anchor_synthetic — does the recent-talker template exist on SYNTHETIC data?

2026-09-04. Script: `anchor_synthetic_probe.py` (+ `strat.py`). Data:
`data_report/wer_set_moderate_test` (200 utts, `mix = near + interferers + noise`,
`ref = near`, verified additive in `scripts/build_wer_set.py:116`). Models: v8
(`config/train_dpcrn.yaml`, `pretrained_ckpt/dpcrn_v8.ckpt`) and v16 ep19
(`config/exp/train_dpcrn_v16_lengthmix.yaml`). dry_blend 1.0, no gate, no guard, cuda:0.

Every condition is a **prefix** prepended to a signal; the model runs once over
prefix+signal and only the signal part is scored, **paired per utterance against `none`**
(so the single-checkpoint caveat of `FIELD_BLOCK.md` applies to the absolute columns but
not to the differences — same argument as `reference_matrix_README` §5).

## Answer in one line

**All three behaviours reproduce on synthetic data, on both checkpoint families, with the
same ~1 s onset / 2 s saturation / few-second decay constants as the real recordings.** The
anchor mechanism is manufactured by the training distribution, not by the real chain.

| | verdict | synthetic | real (reference_matrix §2–§3) |
|---|---|---|---|
| (i) needs ~1 s of a talker before it suppresses others | **reproduced** | 0.25 s −0.5, 0.5 s −3.7, **1 s −12.6**, 2 s −13.9, 3 s −13.6 (v8 Δon1) | 0.25/0.5 s ≈ 0, **1 s −13.1**, 2 s −19.0, 3 s −19.3 |
| (ii) forgets the anchor after ~5–10 s of quiet | **reproduced** (constant ~2× longer) | gap-corrected: 2 s −8.2, 5 s −4.0, 10 s −2.8, 20 s −0.9 | 2 s −14.6, 5 s −9.8, **10 s −0.9**, 20 s −0.25 |
| (iii) deletes a NEW near talker who follows a different anchor | **reproduced**, confined to the onset | Δon1 (first 1 s of the user): 1 s anchor −1.07, 2 s −1.71, 3 s −1.47; Δlate −0.05..−0.10 | near-clip median −0.18 → −2.01 (2 s anchor); violations 0 → 4–6 / 17 |

Full tables: `TABLE_bgonly.md` (i, ii), `TABLE_utt.md` (iii), `TABLE_ident.md`,
`TABLE_strat.md`. Rows: `out2/rows_{arm}_{tag}.jsonl`; a first pass without onset windows
or gap controls is in `out_v1/` (agrees row-for-row).

## Arms and metrics

* **`bgonly`** — the scored signal is `mix − ref`: interferers + noise with the **user
  absent**. This is the synthetic analogue of a *lone-far* field clip and the only honest
  test of (i), because inside a normal utterance the user's own voice is already an anchor.
  `supp_db` = output/input energy over the interferer-active frames.
* **`utt`** — the scored signal is the user's `mix`. `del_db` = output/mix over ref-active
  frames. Baseline `none` sits at −3.6..−3.7 dB because the model correctly removes the
  interferers *inside* those frames; only the **paired Δ** is a deletion measure.
* **`ident`** — separates speaker identity from channel signature.
* **Onset windows** (`db_on1`, `db_on2`, `db_late`) — active frames within 1 s / 2 s of the
  talker's first active frame, and after 2 s. This is what makes (iii) legible: the
  whole-span Δ is only −0.40 dB, but Δon1 is −1.5..−1.7 dB and Δlate is ~0.

## (i) Anchor onset — `bgonly`, Δon1 vs `none`, dB

| prefix (3 s unless noted) | v8 | v16 ep19 |
|---|---|---|
| `none` (cold, absolute) | −6.40 | −6.25 |
| `quiet3` — quietest 30 % of `mix−ref`, the "true floor / same chain" analogue | −1.39 | −0.55 |
| `bg3` — ref-inactive mix frames (bystanders + room), n=126 | −0.09 (p 0.92) | −0.04 (p 0.75) |
| `bgsub3` — 3 s of the interferers themselves (a FAR talker) | **+0.85** | **+1.65** |
| `silence3` — digital zeros | −5.90 | −19.22 |
| anchor 0.25 s | −0.48 | −0.70 |
| anchor 0.5 s | −3.67 | −4.14 |
| **anchor 1 s** | **−12.61** | **−9.72** |
| anchor 2 s | −13.85 | −9.50 |
| anchor 3 s | −13.62 | −10.75 |

Same knee, same saturation, and the same three nulls as the field: **room tone is a null,
noise/bystanders are a null, and a FAR talker is a null (here slightly protective) — only a
near talker arms suppression.** Digital silence is again *not* a neutral prefix but a
suppress bias (field: "a mild suppress-bias on the device chain, a catastrophe on QVF").
The `self` sweep (the utterance's own near talker) is the same curve: −0.57 / −2.43 /
−8.04 / −13.82 at 0.25 / 0.5 / 1 / 2 s.

## (ii) Decay — `bgonly`, gap-corrected

A long quiet or silent prefix is itself a suppress bias, so each decay condition has a
**gap-only control** with no anchor in front (`gapq{N}`, `gapbg{N}`); the anchor's surviving
effect is Δdecay − Δgap, paired per utterance.

| gap after a 3 s anchor | v8 Δdecay | v8 Δgap | v8 corrected | v16 corrected | p (v8) |
|---|---|---|---|---|---|
| 2 s quiet | −8.72 | −0.82 | **−8.23** | −7.93 | 2e-24 |
| 5 s quiet | −5.93 | −0.92 | **−4.00** | −4.57 | 7e-23 |
| 10 s quiet | −2.87 | −0.52 | **−2.81** | −2.01 | 2e-20 |
| 20 s quiet | −1.57 | −0.51 | **−0.89** | −0.70 | 7e-17 |
| 2 s background | −7.62 | −0.19 | **−7.10** | −7.76 | 1e-20 |
| 20 s background | −0.80 | −0.27 | **−0.48** | −0.59 | 1e-11 |

Half-life ≈ 4–5 s, ~90 % gone by 20 s. The field says "half gone at 5 s, gone by 10 s";
synthetic still retains ~1/3 at 10 s and is gone by 20 s — the **same mechanism with a
roughly 2× longer time constant**. The gap material barely matters (quiet vs background
agree), which matches the field's finding that the *content* of the gap is irrelevant.

## (iii) Wrong-anchor deletion — `utt`, Δ vs `none`, dB

| prefix | v8 whole-span | v8 Δon1 | v8 Δlate | v16 Δon1 | p (v8, on1) |
|---|---|---|---|---|---|
| `self3` (same talker, same room) | +0.01 | **+0.01** | +0.01 | +0.04 | 0.82 (null) |
| `bg3` (bystanders only) | −0.03 | −0.14 | +0.01 | −0.17 | 0.04 |
| `bgsub3` (the far interferers) | +0.07 | **+0.18** | +0.06 | +0.14 | 3e-4 |
| `other` 0.5 s | −0.09 | −0.38 | −0.02 | −0.31 | 8e-16 |
| `other` 1 s | −0.25 | −1.07 | −0.05 | −0.84 | 5e-27 |
| `other` 2 s | −0.40 | **−1.71** | −0.10 | −1.25 | 4e-26 |
| `other` 3 s | −0.40 | −1.47 | −0.10 | −1.33 | 3e-25 |
| `silence3` | −0.12 | −0.28 | −0.04 | **−2.33** | 5e-9 |

The sign contrast is the mechanism, and it is clean: **the same talker costs nothing, a
different talker costs −1.5 dB at the onset, a far talker or bystander noise costs nothing.**
`del < −3 dB` counts go 126 → 143 / 200 with a 3 s wrong anchor. SI-SDR: `self3` +0.43,
`other3` −0.60. The deletion also decays (v8 gap-corrected whole-span: 2 s −0.15, 5 s −0.07,
10 s −0.04, 20 s −0.01 p=0.24; v16 −0.13 / −0.08 / −0.04 p=0.09 / −0.01 p=0.93 — closed at
20 s on both), i.e. (ii) and (iii) share one clock, exactly as `reference_matrix_README`
§6.2 claims.

Note `silence3` on v16: Δon1 = **−2.33 dB**, larger than any real anchor. A digital-silence
prefix is the single most destructive onset condition measured here.

This reconciles the apparent magnitude gap with the field. The field's near clips are only a
few seconds long, so its whole-span −1.8 dB **is** an onset number; the synthetic Δon1 of
−1.5..−1.7 dB is the same size. The synthetic whole-span figure is small only because the
utterances run ~9 s and the damage is over after 1–2 s.

## Identity vs channel (`TABLE_ident.md`) — the new fact

`other` changes both the talker and the room draw, so it cannot say which the template keys
on. Splitting them (Δon1, dB):

| anchor | v8 1 s / 3 s | v16 1 s / 3 s |
|---|---|---|
| `self` — same talker, same utterance, same room | +0.02 / +0.01 | +0.04 / +0.04 |
| `selflate` — same talker, same room, a LATER excerpt | −0.05 / −0.04 | −0.05 / −0.01 |
| `samespk` — **same talker, different room / distance / level** | **−0.52 / −0.44** | **−0.45 / −0.40** |
| `diffspk` — different talker, different room | −1.07 / −1.47 | −0.84 / −1.33 |

`selflate` being null rules out byte-identity memorisation. **Changing only the channel
(same speaker) already buys ~30 % of the wrong-anchor deletion; changing the talker too
roughly triples it.** So the template is a mixture — part talker identity, part acoustic /
channel signature — and neither half is dispensable. The relative-distance component is
real but weak and does not explain it either (`TABLE_strat.md`: Spearman ρ = +0.15 / +0.14,
p 0.035 / 0.05; a nearer anchor deletes more, −1.72 vs −1.02, **but the deletion is
significant even when the anchor was farther than the user**).

## The one place synthetic and real diverge

Cold start. On synthetic `bgonly` the model already suppresses a lone interferer by
**−6.3..−6.4 dB with no anchor at all** (103–106 / 200 clips already pass ≤ −6 dB). On real
recordings cold far is **−1.06 / −1.71 dB, 4/20 passing**. The anchor mechanism sits on top
of an already-working baseline in the training distribution and on top of near-passthrough
in the field. So the training data reproduces the *mechanism* faithfully but not its
*severity*: nothing in this set asks the model to suppress a lone far talker it has no
context for, because the synthetic version of that case is already easy.

## What this licenses for v19

1. **The objective can be changed, because the behaviour is in-distribution.** Onset
   deletion and wrong-anchor deletion are measurable on 200 synthetic utterances in ~10 min
   per checkpoint with no real recordings and no ASR — usable as a training-time metric or
   an early-stopping signal, unlike the field set.
2. **Weight the loss on the first 1–2 s of each talker turn.** Δlate is ~0 in every
   condition; all the damage is in `on1`. A curriculum row that concatenates
   `talker A (3 s) → talker B` and scores B's first second is exactly the failing case, and
   it is buildable from this set today.
3. **A "persistent user" target cannot be built on channel cues** — `samespk` across rooms
   already loses 30 % of the protection, and that is the deployment case (same user, moving,
   or a new session). Whatever representation v19 trains has to be invariant to the room
   draw while staying sensitive to the talker.
4. **Do not use digital silence as the neutral prefix** in any training or eval row: it is a
   −5.9 dB (v8) / −19.2 dB (v16) suppress bias here, reproducing the field warning.
5. **Cold-start severity must come from somewhere else.** Fixing the anchor on this set will
   not by itself fix the real-chain cold start (−1 dB, the QVF wall), which
   `reference_matrix_README` §6.3 and `eq_probe_README` already attribute to the chain.
