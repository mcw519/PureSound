> ⚠️ **Measured on field set v2 (27 clips, 90D/270D only).** The field benchmark was
> rebuilt as v3 on 2026-08-20 (64 clips, four orientations + seven QVF clips, new
> hand labelling) — see `benchmarks/field_test_vector/SET_V3.md`. **Numbers below do
> not compare against v3 runs.** Synthetic and WER stages are unaffected.

# One continuous near-presence quantity, run over the real recordings

Run 2026-08-19 on `dpcrn_v8`. Scripts beside this note (`b_traj_extract.py`,
`b_traj.py`, `b_traj_settled.py`, `b_traj_spans.py`, `b_traj_ceiling.py`);
derived features are not committed, `PROBE_WORK` says where they go.

Offline only -- no training, no product code touched, nothing exported. This
measures whether a proposed inference-side mechanism would behave, before any of
it is built.

## The mechanism being measured

A per-frame linear readout `s_t` on the frozen bottleneck (the same tensor
`presence_probe_frames.py` reads), then one continuous state:

```
b_t = b_{t-1} + a * (s_t - b_{t-1})      a = a_up if s_t > b_{t-1} else a_dn
```

fast up, slow down, started at 1.0 -- biased toward "someone is there", which is
the safe direction and the one the time course says the evidence already leans.
`b` then sets `dry_blend`, flat at today's 0.9 above `b_hi` so that segment is
bit-identical to what ships now.

## Material

All 27 `field_cases/test_vector_cases` clips on one frame grid (~99.8 fps). The
per-frame probe had only extracted the two sessions; the cold-start clips were
extracted utterance-pooled, and this question needs both on the same grid.

Frames below **-60 dBFS are excluded from every number here**. A frame with
nothing audible in it has no user to protect and no bystander to suppress, and
including them measures room tone.

## M0 -- the readout transfers between the two recordings

Fitted on one recording's frames, applied to the other. No session adaptation, no
threshold picked on the test points.

| | cold-start frames | balanced acc | AUC | mean p present / absent |
|---|---|---|---|---|
| train 90d -> test 270d | 7,166 | 0.958 | 0.995 | 0.956 / 0.153 |
| train 270d -> test 90d | 9,452 | 0.969 | 0.997 | 0.924 / 0.096 |

Against the shipped synthetic-fitted boundary, which scores 0.500 with mean
p(present) 1.000 against 0.999. So the calibration defect is fixable from a small
amount of real material, and the fix survives being held out.

**But 90D and 270D are the same room and the same rig at two mic orientations.**
This is cross-orientation, not cross-room and not cross-device. It does not show
that a boundary fitted in one deployment transfers to another, and that is the
question a shipped fixed offset would actually face.

## M1 -- GO. The two distributions separate.

Audible frames, at least 1.5 s past audible onset so `b` is reporting a decision
rather than its prior. Pooled: 8,689 keep frames, 4,870 far frames.

| | keep | far | gap |
|---|---|---|---|
| p1 / p99 | 0.736 | 0.578 | +0.158 |
| p5 / p95 | 0.909 | 0.491 | +0.419 |
| median | 0.984 | 0.198 | +0.786 |

99.1% of keep frames sit above the worst far frame, and 99.1% of far frames below
the worst keep frame.

The dead zone the design needs exists:

| `b_hi` | keep frames >= | far frames >= |
|---|---|---|
| 0.70 | 99.3% | 0.4% |
| **0.75** | **98.9%** | **0.0%** |
| 0.85 | 97.7% | 0.0% |
| 0.95 | 89.6% | 0.0% |

Settling, median `b` by time since audible onset -- keep is flat from the first
window, far falls through it:

| | 0-0.25 s | 0.5-1 s | 1-2 s | 3-5 s | >5 s |
|---|---|---|---|---|---|
| keep | 0.963 | 0.975 | 0.985 | 0.983 | 0.985 |
| far | 0.837 | 0.611 | 0.492 | 0.318 | 0.159 |

Consistent with the ~1 s evidence-accumulation figure in
`presence_probe_README.md`: the groups part company in the 1-2 s bin.

## M2 -- it falls in time, on every clip

Time from first audible frame to `b` crossing each threshold, 10 cold-start lone
bystander clips:

| | fastest | slowest | never |
|---|---|---|---|
| b < 0.9 | before onset | 0.18 s | 0 clips |
| b < 0.7 | 0.07 s | 0.90 s | 0 clips |
| b < 0.5 | 0.64 s | 2.28 s | 0 clips |
| b < 0.3 | 1.28 s | 4.85 s | 0 clips |

Shortest clip in the set is 2.8 s, and it reaches b<0.3 at 1.89 s. Nothing runs
out of clip.

## M3 -- and here is the failure: short near spans inside a session

Session far spans behave -- median `b` 0.05-0.51, every span below `b_hi`, so the
mechanism acts in the anchored case too rather than only at cold start.

Session **near** spans do not, and it is entirely a length effect:

| span length | spans | frames below b_hi=0.75, mean | worst span |
|---|---|---|---|
| < 2.5 s | 7 | **31.2%** | **83.9%** |
| > 8 s | 8 | 4.2% | 14.3% |

6.3% of all in-session near frames, but concentrated: `90d near 1` (1.8 s) has
median `b` 0.403, `270d near 4` (1.8 s) has 0.594. The user is talking and the
quantity reads "absent".

**It is the readout, not the integrator.** On `90d near 1` the raw `s` median is
0.412 against `b` 0.403; on `270d near 4`, 0.668 against 0.594. `b` is faithfully
reporting a readout that is already wrong.

Which means no time constant fixes it -- the sweep confirms it directly:

| tau_up | tau_dn | cold keep >= b_hi | cold far >= b_hi | sess near >= b_hi | sess far >= b_hi |
|---|---|---|---|---|---|
| 0.05 | 1.0 | 99.2% | 0.2% | 96.3% | 9.2% |
| 0.05 | 4.0 | 100.0% | 3.9% | 98.5% | 21.3% |
| 0.30 | 1.0 | 98.7% | 0.0% | 90.1% | 8.3% |

Buying the short spans back with a slower fall costs the far side more than it
gains: `tau_dn` 4.0 lifts in-session near from 96.3% to 98.5% and lets 21.3% of
in-session far frames into the dead zone. A faster rise is free and strictly
better -- `tau_up` 0.05 beats 0.15 and 0.30 on every column.

This is the same short-span deficit `presence_probe_README.md` already recorded
(<2.5 s spans 77.7% mean correct against >8 s 90.8%), now expressed in the
quantity that would drive the action.

## What it buys, in dB

`dry_blend` 0.9 -> 0.995 over `b_lo` 0.35 -> `b_hi` 0.75, against today's flat
-20.0 dB ceiling on every frame of every clip:

| | frames bit-identical | ceiling, median |
|---|---|---|
| cold-start user | **99.2%** | -20.00 dB |
| in-session user | 96.3% | -20.00 dB |
| cold-start bystander | 0.2% | **-46.02 dB** |
| in-session bystander | 9.2% | -46.02 dB |

26 dB of headroom released where suppression should be deep, with the user's
frames essentially untouched. Worst single frame: cold-start user -23.73 dB,
in-session user **-46.02 dB** -- that last one is the M3 exposure, a short near
span where the blend is fully released while the user is talking.

**The ceiling is a bound, not an outcome.** Releasing it lets the model suppress
deeper; it does not make it. What it removes is the arithmetic floor every
far-field residual on record was measured against.

## What this does not establish

* **Whether the M3 exposure does any damage.** Releasing the blend only matters if
  the mask is *also* wrong on those frames. The field scorecard has 0/15 KEEP
  violations across four versions on these sessions, which suggests the mask is
  fine there -- but that was measured *with* the blend at 0.9, so it is not
  evidence about what happens without it. Measuring this needs the enhancement
  actually run at the varying blend, and it is the next thing to do.
* **Cross-room transfer**, per M0.
* **`b_lo` and `b_hi` were read off this material.** 0.35/0.75 is where the gap is
  on 27 clips from one room; it is not a held-out setting.
* **No WER.** The mechanism's own worst case is deleting the user, and only Dawn
  and the moderate set can see that.
* n=27 clips, two recordings, one checkpoint (v8).

---

# Follow-up (same day): the exposure, and what it turned up instead

`b_traj_exposure.py`, `b_traj_cold.py`, `b_traj_gain.py`, `b_traj_sweep.py`. One
forward pass of `dpcrn_v8` per clip; the mask output is identical across every
system below, only what is done with it differs.

## The exposure is nil -- and for a reason that condemns the actuator

`pure` (no blend at all) is the worst case a blend release can reach, and
`b`-driven is bounded between it and `fixed`. On the in-session near spans:

| span | len | median b | fixed | b-driven | pure |
|---|---|---|---|---|---|
| 90d near 1 | 1.8 s | 0.609 | -0.16 | -0.16 | **-0.17** |
| 90d near 5 | 15.6 s | 0.986 | -1.95 | -1.95 | -2.03 |
| 270d near 6 | 11.6 s | 0.994 | -2.28 | -2.28 | -2.34 |

Worst difference on any span, anywhere: **-0.08 dB**. The insurance was never
being claimed -- the mask keeps the user on those spans, so removing the floor
changes nothing.

The same fact read the other way is fatal to the design. **On the cold-start
clips -- the actual defect -- the `b`-driven blend gains a median of 0.03 dB.**

| | fixed | b-driven | pure |
|---|---|---|---|
| cold-start far, median | -0.39 dB | **-0.42 dB** | -0.43 dB |

Not because `b` is wrong: it reads 0.05-0.48 on those clips, it has them right.
Because **the -20 dB ceiling was never what limited suppression there.** The mask
reduces those clips by 0.39 dB. Releasing a ceiling 20 dB below where the signal
actually sits does nothing.

Where the blend release *did* help was in-session far spans the mask had already
pushed to -15 to -18 dB, i.e. up against the ceiling: +1.5 to +5 dB. Those spans
were already succeeding.

**So `b` is a working detector wired to an actuator that cannot move the defect.**
`dry_blend` was chosen because it is inference-side, needs no training and is
instantly verifiable. It is all of those, and it is also incapable of the job.

## The same detector on an actuator that can move it

`out = g(b) * (0.9*enh + 0.1*mix)`, `g` flat at 1.0 above `b_hi` so that segment
stays bit-identical, falling to -26 dB below `b_lo`. This is the 2026-07-10
gate-only architecture, whose readout could not tell near from far on real
recordings (both ~0.9); this readout can.

| | fixed | gain, b_hi 0.75 |
|---|---|---|
| **cold-start far** (the defect) | -0.39 dB | **-10.01 dB** |
| cold-start keep (near + dt) | -0.18 dB | **-0.18 dB** (identical) |
| in-session far | ~-12 dB | -22.68 dB |

Cold-start keep is untouched to two decimals -- `b` reads 0.96-0.995 on every near
and double-talk clip, so `g` is exactly 1.0 there.

**And it costs exactly the span M3 predicted.** `90d near 1`, 1.8 s, median `b`
0.609: **-0.16 -> -7.20 dB, a KEEP-VIOLATION.** The user is talking, the readout
says absent, and unlike a blend release a gain can act on that.

## The operating curve

`b_hi` is the only decision parameter (`b_lo` tracked at `b_hi - 0.40`), against
all 30 keep spans in the set:

| b_hi | cold far | cold keep | sess far | worst keep span | KEEP-VIOLATIONs |
|---|---|---|---|---|---|
| off (today) | -0.39 | -0.18 | ~-12 | -2.28 | 0 |
| **0.50** | **-4.97** | -0.18 | -16.92 | **-2.28** | **0** |
| 0.60 | -6.64 | -0.18 | -18.45 | -3.18 | 1 |
| 0.70 | -9.16 | -0.18 | -21.34 | -6.34 | 1 |
| 0.75 | -10.01 | -0.18 | -22.68 | -7.20 | 1 |
| 0.85 | -12.45 | -0.18 | -26.17 | -9.95 | 1 |

**There is a violation-free operating point.** At `b_hi` 0.50 the defect moves
4.6 dB and the worst keep span is -2.28 dB, which is the untouched baseline --
that span is `270d_dt2` and it scores -2.26 with the mechanism off.

Everything between 0.60 and 0.85 buys more suppression with the same single
violation, so the curve is a genuine choice rather than a cliff.

## What this does and does not settle

**Settled:** the blend actuator cannot address cold start, whatever drives it.
That is arithmetic, not a tuning question. A gain actuator can, and the readout is
good enough to drive one without touching cold-start keep at all.

**Not settled, and it is most of what matters:**

* **This is not suppression, it is attenuation.** At `b_hi` 0.75 the cold-start
  residual still sits **6.4 to 22.6 dB above the recording's own noise floor**,
  above 15 dB on 6 of 10 clips. QVF2.2 gates those to -44 dB. The bystander gets
  quieter; it does not go away. **The wall is not down.**
* **The one violation is the readout's short-span deficit**, and no time constant
  fixes it (see the sweep above). Either the readout improves on short turns or
  the mechanism needs a hold that protects a recently-active near speaker.
* **No WER.** A -7.20 dB gate on a 1.8 s turn is exactly the shape that drops a
  turn, and only Dawn and the moderate set can see it. Nothing here is a
  deployment claim.
* `b_hi`, `b_lo`, both time constants and the -26 dB floor were all read off this
  same 27-clip set from one room. None is held out.
* One checkpoint (v8), two recordings, one room at two mic orientations.
