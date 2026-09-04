# SNR-aware re-read of the onset-deletion diagnosis (2026-09-05)

Prompted by the user's objection that dB deltas were being compared across sets without accounting for the
noise/interferer level. Two scripts, both re-run by hand: `onset_snr.py` (stratify the existing onset measure
by the local SNR of the onset window, real vs synthetic) and `onset_snr_sweep.py` (rescale the background of
120 synthetic utterances per set to target SNRs −15…+10 dB and re-measure).

## 1. Dawn Chorus has no usable waveform reference

`speech` (the "clean foreground") is not waveform-consistent with `mix`: over 30 sampled utterances the
mix/speech correlation is median |0.30| at the best lag, negative on most (polarity), lags range from 0 to
thousands of samples, and 3/30 show no correlation within 4 s at all. Projection of the mix onto `speech`
has median −0.83 over the 450-utterance cache. Consequences:

* every SI-SDR number ever printed for Dawn by `eval_dawn_chorus.py` (always ≈ −9 dB for the raw mix) is
  meaningless, and so is the foreground-projection gain `P` on Dawn — 64 % of onset windows have a
  NEGATIVE projection, which the diagnosis's `10log10(g²)` turned into a positive number.
* the diagnosis's headline "Dawn onset excess −3.85 / −3.90 dB @0.5 s (fg-projection)" is **withdrawn**.
  The energy-based Dawn variant (−1.15 / −1.39 dB) survives arithmetically but is noise-confounded.
* the only valid Dawn instruments are ASR (deletion / insertion / WER, paired per utterance) and
  reference-free listening; `ref` may still be used to locate speech activity (lags are mostly < 10 ms).

## 2. Real vs synthetic onset comparison was not SNR-matched

Local SNR of the onset window (ref vs mix−ref): synthetic sets median +0.6 dB; Dawn's ref-based value
(−10.8 dB) is itself invalid (§1), so the two worlds cannot be compared on this axis today. The claim
"cold onset deletion is real-only (11–23×)" therefore has no valid measurement behind it and is
**downgraded to unresolved**.

## 3. Onset excess vs controlled SNR on synthetic data (n = 120 per cell; excess = g_on − g_rest, dB)

| target SNR | v8 moderate med / p10 | v16 moderate med / p10 | v8 indomain med / p10 | v16 indomain med / p10 | g_on median (v8 / v16, moderate) |
|---|---|---|---|---|---|
| −15 | +0.15 / −18.7 | −0.71 / −22.6 | +0.45 / −10.8 | −0.88 / −13.1 | −11.8 / −11.5 |
| −10 | −0.77 / −10.0 | −1.44 / −14.1 | +1.37 / −8.0 | +0.71 / −8.5 | −7.9 / −9.1 |
| −5 | −0.84 / −7.6 | −1.29 / −8.9 | +0.75 / −6.8 | +0.58 / −7.1 | −4.9 / −5.2 |
| 0 | −0.29 / −5.4 | −0.65 / −6.0 | +0.34 / −3.6 | +0.20 / −5.3 | −2.3 / −2.3 |
| +5 | −0.04 / −2.3 | −0.11 / −2.7 | +0.19 / −1.8 | +0.23 / −2.2 | −1.2 / −1.3 |
| +10 | +0.00 / −0.6 | +0.01 / −0.8 | +0.22 / −0.4 | +0.18 / −0.5 | −0.7 / −0.8 |

Reading, plainly:

1. **At low SNR the model deletes the user everywhere, not specially at the onset**: g_on median −5 dB at
   −5 dB SNR, −8..−9 dB at −10 dB, −12 dB at −15 dB, and the rest of the utterance moves with it. The
   onset-specific excess stays within ±1.4 dB in the median; only its p10 tail widens (−8 to −23 dB).
2. **The energy-based excess is biased the other way** at low SNR (+1.3 dB at −5 dB: the onset window looks
   *better* because noise removal dominates the number) — the bias the user pointed at.
3. The training distribution reaches these SNRs only in its tail (training-row realized SIR: median +0.5 dB,
   p25 −2.8, p5 −8.1; noise SNR median +9.6 — `../training_data_audit`).

## 4. What survives, what changes in the plan

* **Survives**: the synthetic wrong-anchor result (Δon1 −1.25/−1.71 dB) — it is a *paired within-utterance*
  difference between prefixes at identical SNR, so the noise term cancels. The v19c axis stands.
* **Changes**: plan §0 row 1 ("cold onset deletion is real-only → not a loss axis") becomes *unresolved*;
  gate P1 (Dawn fg-projection onset excess) is replaced by ASR-based paired deletion in `--context none`
  and `--context background`, plus "the guard's benefit shrinks" (unguarded-vs-guarded ASR delta) as the
  behavioural read; every onset/keep dB claim must be reported as a paired within-utterance delta and
  stratified by local SNR; no cross-set absolute dB comparison; Dawn SI-SDR is not to be quoted.
* **New standing fact**: low-SIR global deletion (g ≈ −5 dB at −5 dB SIR) is in-distribution at the tail;
  it is the classic over-suppression the moderate-set ASR gate already measures.
