# puresound.nnet.loss.active_bins

繁體中文版本：[active_bins.zh-TW.md](active_bins.zh-TW.md)

Per-bin log-magnitude error counted only where the clean target has speech.
Silence and noise-only bins contribute nothing.

## Class: `ActiveBinLogMagLoss`

### What it computes

On a Hann-window STFT of both signals (`center=True`), in dB:

```
ref_db, enh_db = 10*log10(|STFT(ref)|^2 + 1e-10), 10*log10(|STFT(enh)|^2 + 1e-10)
peak           = max over (freq, frame) of ref_db, per row
active         = (ref_db >= peak - dynamic_range_db) & (ref_db > silence_floor_db)
diff           = enh_db - ref_db
w              = under_weight where diff < 0, else 1
per_row        = sum(|diff| * w * active) / max(sum(active), 1)
loss           = mean of per_row over rows that have at least one active bin
```

Rows with no active bin (a silent target) are excluded from the mean. If the
signals are shorter than `n_fft`, the loss is `enh.sum() * 0.0`, a zero that
keeps a graph edge.

### Constructor

```python
ActiveBinLogMagLoss(
    n_fft: int = 512,                 # analysis grid; defaults match the DPCRN recipes'
    hop: int = 160,                   #   encoder, so a bin here is a bin the mask acts on
    dynamic_range_db: float = 50.0,   # active = within this many dB of the row's peak bin
    under_weight: float = 1.0,        # multiplier where the output is below the target
    silence_floor_db: float = -80.0,  # bins below this absolute level are never active
)
```

`dynamic_range_db` and `under_weight` must be positive (`ValueError` otherwise).

### Inputs

`forward(enh, ref) -> Tensor`, waveforms `[N, T]` (a singleton channel axis
`[N, 1, T]` is squeezed; the two are cut to the shorter length).
`required_inputs = ("enhanced", "target")`.

### Config usage

```yaml
loss_func:
  - type: ActiveBinLogMagLoss
    weighted: 0.5
    args:
      n_fft: 512
      hop: 160
      dynamic_range_db: 30.0
      under_weight: 1.0
```

The noise-suppression recipe adds it in a short fine-tune at low learning rate
on top of the last curriculum stage (`train_dpcrn_mamba_activebin_ft.yaml`), with
the other losses unchanged.

### Design notes

- Why only active bins. The waveform and spectral losses in this package put
  most of their gradient on noise and silence bins (MR-STFT, ASR features) or
  weight by energy without per-bin structure (SDR). An output can improve on all
  of them while mid-level harmonic peaks inside speech are shaved. Restricting a
  log-magnitude L1 to the target's speech bins asks for that structure directly,
  and keeps the large, noisy log errors of silent regions from outvoting it.
- `dynamic_range_db` is relative to the whole row's loudest bin, so it selects
  "speech" per utterance regardless of level. Narrowing it also concentrates the
  term: the same `weighted` over fewer bins is more gradient per bin, so a range
  change is not only a scope change.
- `under_weight > 1` leans against undershoot (shaved peaks). Its effect depends
  on the range: a wide range includes the valleys between harmonics, and
  penalising undershoot there tells the model to leave noise in them.
- `silence_floor_db` protects target-absent rows. For a fully silent target
  every bin sits at the log floor (−100 dB) and equals the peak, so the relative
  test alone would mark every bin active and the row would cost about 100 — two
  orders above the rest of the objective, pushing such rows to digital zero.
  −80 dB is far below any speech bin of a normally levelled row and above the
  log floor.
- Intended use is a short calibration pass after the curriculum ladder. In a
  full-length stage the other terms outweigh it.

Tests: `test/nnet/test_active_bins.py` pins the speech-bin scope, the level
invariance, the undershoot weighting and the handling of silent rows.
