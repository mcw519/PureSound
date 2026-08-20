# v11 presence heads at ep19 -- the confound broke, the chain wall did not

`dpcrn_v11_ep19.ckpt` (warm-started from v8, two per-frame heads with a
50 ms / 250 ms / 1 s / 4 s EMA bank, BCE at 0.1 each, heads training-only so
inference is unchanged). Judged on the two criteria set before training in
`presence_stratified_README.md`. ep16 was scored alongside because ep19 sits on a
cosine up-swing; they agree, so nothing here is a phase artefact.

## Criterion 1 -- in-domain, no inversion across RT60: PASS

Near-presence head, same probe and material as v8's "before":

| rt60 band | v8 readout AUC | **v11 head AUC** | v11 med(pres) |
|---|---|---|---|
| 0.0-0.3 | 0.571 | **0.951** | 6.76 |
| 0.3-0.5 | 0.523 | **0.959** | 6.20 |
| 0.5-0.7 | 0.533 | **0.945** | 5.35 |
| **0.7-0.9** | **0.472 (inverted)** | **0.963** | 5.67 |
| 0.9+ (n=706) | 0.558 | 0.821 | 1.39 |
| ALL | **0.529** | **0.953** | 5.94 |

The confound is gone in-domain. v8's present-frame score fell monotonically with
reverberation (-1.02 -> -3.04); v11's is flat (6.76 -> 5.35) and the 0.7-0.9 band
is now its **best**, not its inverted one. That was the pass criterion and it is
met with room to spare.

**This alone proves nothing.** The head trained on these rows, and the 2026-07-10
gate head scored 0.751 in-domain before failing completely on real audio.

## Criterion 2 -- real recordings, nothing refitted: PARTIAL

`presence_head_real_transfer.py` on the v3 field set. Truth = the clip's role,
audible frames only, no threshold or weight touched.

| scope | ep19 AUC | ep16 AUC |
|---|---|---|
| **180d (HELD OUT -- nothing fitted on it)** | **0.843** | 0.854 |
| 270d | 0.895 | 0.909 |
| 90d | 0.804 | 0.807 |
| qvf_plumbing | 0.846 | 0.901 |
| qvf_gym | 0.763 | 0.739 |
| qvf_price | 0.712 | 0.735 |
| **qvf_scenario3** | **0.253 (inverted)** | 0.281 |
| pooled fitted-on rooms | 0.680 | 0.695 |

**The held-out orientation reads 0.843.** That is the number that speaks to
generalisation: 180D is a device orientation no fitting has seen, and the head
separates a talking user from a lone bystander on it at 0.84. Set against
2026-07-10 -- where the gate read ~0.9 for near *and* far and never closed -- this
is the first presence signal that survives contact with an unseen real recording.
The pooled 0.680 is not the headline: it averages device captures with a
commercial publication chain, and the per-group split is what the design decisions
rest on.

### The failure is the canonical clip, and that matters

`qvf_scenario3_far1` is **t = 6.63-11.65 s of QVF clip 3** -- the isolated far
talker that QVF2.2 gates to -44 dB and every version of ours passes through at
about -1 dB. It has been the reference failure since 2026-07-03. On it the head is
**inverted**: absent frames score 8.10 against 6.23 for present ones. It reads the
distant talker as *more* of a user than the actual user.

So the wall moved but did not fall. Presence is now readable across reverberation
and across device orientation; it is still not readable on the one recording chain
the product is benchmarked against.

## The background head does not transfer at all

| scope | in-domain | real |
|---|---|---|
| pooled | 0.707 | **0.475** |
| qvf_scenario3 | -- | 0.058 |
| qvf_plumbing | -- | 0.123 |
| qvf_gym | -- | 0.136 |
| 180d (held out) | -- | 0.595 |

Chance or inverted nearly everywhere. It also degrades with reverberation
in-domain (0.745 at RT60 < 0.3 down to 0.657 at 0.7-0.9) -- the confound the near
head escaped, still present in its twin. Two heads reading the same bottleneck
with the same architecture and opposite outcomes says the near head's success came
from its supervision, not from the EMA bank alone.

**Do not gate on the background head.** Its intended job -- distinguishing
(near=0, bg=1) cold-start bystander from (near=1, bg=1) double-talk -- is not
supported by these numbers.

## What this licenses and what it does not

* **Licensed**: the near-presence head as a *measured quantity* on device-chain
  audio, including unseen orientations. The reverberation confound that killed the
  b-trajectory gate is broken in-domain and 0.80-0.90 on real device captures.
* **Not licensed**: gating anything. Criterion 2 passed on device recordings and
  failed on the QVF chain, and `full_gate/v8_gate050_VERDICT.md` is the record of
  what happens when a gate ships on a presence signal that holds in one condition.
  A gate needs the full nine gates first, and the scenario3 inversion predicts it
  would leak exactly where it always has.
* **Open**: whether the separator moved. Heads are training-only, but multi-task
  pressure changes the bottleneck, so v11's nine-gate run is still required before
  any claim about v11 as a checkpoint.

## Caveats

* n=30 batches for the in-domain table; per-group real counts range 180-6,734
  audible frames, and the three thin groups (0d, qvf_scenario1/2,
  qvf_keep_in_touch) carry only one class and are unscorable by AUC.
* v3 field set (2026-08-20). v2 numbers do not compare -- see `SET_V3.md`.
* One training run, one warm start, one loss weight (0.1). No ablation says the
  EMA bank rather than the supervision produced the in-domain result, though the
  background head failing under an identical architecture is evidence for
  supervision.
