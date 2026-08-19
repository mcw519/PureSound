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
