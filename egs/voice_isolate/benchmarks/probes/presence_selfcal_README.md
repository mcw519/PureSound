# Layer-1 self-calibration: a session-relative threshold for the presence head

Run 2026-08-21/22. Scripts beside this note: `presence_selfcal_sim.py`
(causal calibrator + per-frame scoring), `presence_selfcal_gate_replay.py`
(the same streams through the real `PresenceGate` math),
`presence_head_judgment.py` (the checkpoint-judgment protocol both of these
feed on). Frame caches are derived data and are not committed.

## Why this exists

Every failure on record is an ABSOLUTE judgment failing for want of a
reference: cold start (no anchor), high reverb (DRR threshold decays), the QVF
chain (boundary in the wrong place). The evidence that a RELATIVE judgment
survives: moving the threshold alone recovered 0.500 -> 0.883 on real clips
(`presence_probe_README.md`), and the b readout separates 99.1% within a
recording. Layer 1 = exploit that at inference, no retraining: collect the
head's own logits over the session, split them into two clusters (Otsu), act
only when the split is trustworthy, place the boundary keep-biased between the
clusters. While inactive the gate is forced open -- **inactive IS today's
behaviour, by arithmetic**, so the scheme can only add risk while active.

## Finding 1 -- the chain wall is largely a per-recording offset

Within-session ordering survives the QVF chain that kills cross-clip pooling:

| scope | AUC (v11a head) |
|---|---|
| QVF cold-start clips pooled across recordings | 0.474 (dead) |
| within qvf_gym_session | 0.886 |
| within qvf_price_session | 0.911 |

Different recordings sit at different offsets; pooling mixes the offsets and
reads as overlap. Comparisons that stay inside one recording survive. This is
the measured basis for the whole relative-judgment program.

## Finding 2 -- the safety/effect trade is a triangle, and evidence-rate is the currency

Per-frame scoring (`presence_selfcal_sim.py`, v11a cache), three guard settings:

| config | wrong activations (58 one-talker clips) | qvf_price keep-damage | qvf_price bystander caught |
|---|---|---|---|
| loose (1 s/cluster) | **24, of which 13 keep-side** | 0.221 -> 0.079 | 83% |
| medium (3 s/cluster) | 7 | -> 0.000 | 45% |
| strict (3 s + 20 s hold) | **1, sentinel refused** | -> 0.000 | 45% |

The guards that make it safe also delay activation past the useful window on
short sessions. Same law as the measured onset lag (~1 s of audible speech to
decide): the evidence and the harm run on the same clock.

## Finding 3 -- through the real actuator, damage AND benefit shrink

`presence_selfcal_gate_replay.py`, PresenceGate defaults (b_hi .5, tau_dn 1 s,
floor -26 dB), span dB in the scorecard's units. On the v11a head: the
per-frame keep-damage numbers (11-15%) become **zero violations** -- transient
low frames are exactly what the integrator absorbs -- but the per-frame QVF
wins shrink to -2..-5.7 dB medians, because the gain needs SUSTAINED
sub-threshold evidence and scattered correct frames don't move it. One real
find: the factory arm (head + threshold 0) deletes `qvf_keep_in_touch_near1`
at **-3.88 dB** on cold start; the strict calibrator removes that by refusing.

## Finding 4 -- checkpoint drift kills the fixed threshold; the calibrator survives (v11b ep31)

Twelve epochs later (same head, more training), on the SAME sessions:

| | fixed threshold 0 | self-calibrated (strict) |
|---|---|---|
| rig-session balanced acc | 0.74 -> **0.62-0.70 (collapsed)** | 0.77-0.82 |
| rig-session suppression through gate | -3.2 dB -> **~-0.5 (dead)** | **-2.5..-4.2 dB** |

The head's logit distribution drifted; the absolute boundary died with it, the
relative one followed it. Second independent validation of the thesis -- first
across recording chains, now across checkpoints. **A shipped fixed threshold
would silently die on every retrain.**

## The open cost -- one real violation, and it is the honest kind

On ep31 the calibrated gate (both configs) deletes one 90d keep span at
-5.25 dB: a **1.8 s user interjection right after 9 s of far speech** -- the
integrator has fully discharged and the span's logits (median +1.89 vs
threshold 1.62) are too weak to recover it in time. keep_bias 0.3 -> 0.15
softens it to -3.19 dB but halves suppression everywhere: a continuous trade,
not a bug. Operating-point selection must run on the field set's own held-out
protocol (fit on 90d/270d, read 180d once) -- not by iterating on all six
sessions, which is what this note must not become.

Also on ep31: first-ever nonzero actuation on scenario3 (-1.9 dB, loose
config, zero keep damage). QVF2.2 does -29.7 there. From zero to one.

## Verdict

Self-calibration is a **safety belt, not a weapon**: it removes the absolute
threshold's failure modes (chain offset, checkpoint drift, one measured
cold-start deletion) and never acts without evidence, but through the deployed
actuator it adds little suppression that the fixed threshold could not -- the
binding constraint is sustained evidence quality, i.e. the head itself.
Adopt for any head-driven gate; do not expect it to move the walls. The wall
levers stay training-side (see `v11b_VERDICT.md`) and architectural (a room /
session reference the model is TRAINED to use).

## Caveats

* Six sessions, two of them shorter than the strict 20 s hold -- the strict
  config can never activate there by construction.
* The sim's audibility is per-clip floor+12 dB; the judgment protocol's is a
  fixed -60 dBFS. Comparable within a table, not across tables.
* The calibrator is cumulative (no forgetting); a deployed one needs a slow
  forgetting horizon and a mid-session chain-change is untested.
* Knobs (d' >= 2, mass, keep_bias 0.3, T = 1) are first-cut, chosen before
  seeing results; only keep_bias 0.15 was probed since, and only on ep31.
