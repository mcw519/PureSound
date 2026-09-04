# EQ probe -- and the compressor-model revision it forced

Run 2026-08-22, `eq_probe.py` beside this note. Protocol identical to the one
that convicted compression: the 14 device lone-far clips, every manipulation
RMS-restored, foreground slot read from dpcrn_v8's DistHead. Baseline
reproduces the record exactly (1.009 m / -0.62 dB vs the recorded ~1.00/-0.61).

| condition | fg_dist med (m) | vs none | DRR readout (dB) |
|---|---|---|---|
| none | 1.009 | -- | -0.62 |
| hp100 | 1.005 | -0.004 | -0.30 |
| podcast (hp + presence +4) | 0.972 | -0.037 | -0.20 |
| broadcast (hp + presence +6, bass -4) | 0.960 | -0.049 | +0.90 |
| bright tilt | 0.950 | -0.059 | +0.21 |
| **dark tilt (directional control)** | **1.021** | **+0.012** | -1.45 |
| **compressor_gain (thr -28, ratio 4)** | **0.582** | **-0.427** | -3.45 |
| **comp + broadcast** | **0.447** | **-0.562** | **+6.08** |

References: QVF publication clips read ~0.70 m / +5.33 dB.

## Verdict 1: EQ is a minor accomplice, causally confirmed

Brightening moves the estimate nearer (-0.04..-0.06 m) and darkening moves it
farther (+0.012) -- the directional control passes, so spectral tilt IS a live
cue at the whole-model readout even though the first-principles window work
found it is not independent inside the direct window. But the magnitude is a
sixth of the 0.30 m phantom gap: EQ alone does not need its own training knob
beyond the `augmentation_ir_response` filter diversity already in every recipe.

## Verdict 2 -- the bigger one: the "compression explains only 1/3" figure was
an artifact of probing with the wrong compressor model

The original conviction (`compression_probe.py`, 2026-08-21) used the
`|x|^p` waveshaper from `apply_media_coloring` and got 1.00 -> 0.90 m -- a
third of the way. This probe uses the REAL envelope compressor
(`compressor_gain`, the exact stage v11b trains against, at thr -28 / ratio 4,
inside v11b's training range): **1.009 -> 0.582 m, past QVF's 0.70 on its
own.** Stacked with broadcast EQ it lands 0.447 m / +6.08 dB -- overshooting
the full QVF signature on BOTH readout slots. A milder operating point on the
same two stages passes straight through QVF's measured values.

So the phantom near user on the QVF chain is **fully accounted for by
envelope compression plus mild production EQ**. There is no missing "2/3
factor" to hunt; codec and noise-gate probes drop in priority accordingly.
It also explains why v11b outperformed the offline prediction (scenario3
closed ~half the gap by ep19 and kept improving to ep31): the recipe trains
against `compressor_gain`, the stage that actually matters, not the waveshaper
the estimate came from.

Curious and recorded, not yet explained: the envelope compressor alone drives
the DISTANCE slot hard while pushing the DRR slot the other way (-3.45); EQ
alone does the reverse. The two slots decouple under single stages and only
the stack reproduces QVF's joint (near, high-DRR) signature -- consistent with
the standing finding that the model reads cues jointly, as comparisons.

## Caveats

* 14 clips, one recording session family, medians only, v8 only.
* One compressor operating point; no expander-direction control (compression
  causality was already established by the original probe -- this revises its
  magnitude, not its direction).
* `dark_tilt`'s DRR drop (-1.45) is larger than `bright_tilt`'s rise: the tilt
  cue is asymmetric at this readout, unquantified further.

---

## ERRATUM (2026-09-04) — Verdict 2 is withdrawn

`eq_probe.py`'s `comp()` returns `compressor_gain(...)` — the GAIN CURVE — instead of `x * gain`, so the "comp" and
"comp+broadcast" conditions fed the model a near-constant signal (measured std 2.3e-10), not compressed audio. The
"1.009 → 0.582 m" reading and the conclusion "envelope compression plus mild EQ fully accounts for the QVF phantom near
user; codec and noise-gate probes drop in priority" are artifacts of that bug. Re-run with the gain actually applied at
the recipe's operating point, the envelope compressor moves the v8/v16/v11b distance readout by |ΔAUC| ≤ 0.013 (inert);
the `|x|^p` waveshaper (the operator the ORIGINAL 2026-08-21 conviction used) does move it but degrades AUC to 0.880,
never inverts it. Nothing applied to a device clip reproduces the QVF chain's 0.26. Verdict 1 (EQ is a minor
accomplice) stands. Codec, noise gate and room/mic geometry go back onto the candidate list; the `+5.19 dB` DRR shift
belongs to the waveshaper. See `v19c_diagnosis.md` (chain_readability) and `v19c_round_design.md` §4.
