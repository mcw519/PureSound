# puresound.nnet.loss.vad

繁體中文版本：[vad.zh-TW.md](vad.zh-TW.md)

Frame-activity supervision. `VADActivityLoss` scores the enhanced waveform's
own frame energy against an activity target; `VADHeadBCELoss` and
`BackgroundVADHeadBCELoss` train explicit backbone heads; `F1_loss` is a soft F1
on probabilities.

## Class: `VADActivityLoss`

Differentiable BCE on the frame energy of the enhanced waveform. Framing is in
the time domain (`unfold` over samples, mean square per frame); there is no STFT.

### What it computes

```
P_enh[t] = mean(enh[t*hop : t*hop + frame_length]^2)          # rows shorter than one frame are padded
logit[t] = (10*log10(P_enh[t] / P_ref) - activity_threshold_db) * logit_scale
loss     = BCEWithLogits(logit, target, weight = FN weight on target frames, FP weight on the rest)
```

`target` and `P_ref` depend on whether a `vad_target` is supplied:

| | `target` | `P_ref` |
| --- | --- | --- |
| no `vad_target` | `1` where the clean reference's frame power is within `activity_threshold_db` of its own loudest frame; all zeros on a fully silent reference | the reference's loudest frame power (`1` on a fully silent reference) |
| `vad_target` given | the labels, truncated or zero-padded to the frame count | `1` (0 dBFS), so the threshold is an absolute dBFS level |

### Constructor

```python
VADActivityLoss(
    frame_length: int = 400,              # samples per frame
    hop_length: int = 160,                # samples between frames
    activity_threshold_db: float = -40.0, # decision point of the soft activity
    logit_scale: float = 0.25,            # sharpness of the soft decision (per dB)
    false_positive_weight: float = 1.0,   # BCE weight on inactive frames
    false_negative_weight: float = 1.0,   # BCE weight on active frames
    require_vad_target: bool = False,     # raise instead of deriving the target from the reference
    eps: float = 1e-8,                    # power floor
)
```

### Inputs

`forward(enh, ref, vad_target=None) -> Tensor`. Waveforms `[B, T]`, `[B, C, T]`
(averaged over channels) or `[T]`; cut to the shorter length.
`required_inputs = ("enhanced", "target", "vad_target")`. `vad_target` is
`batch["vad_target"]`, present when the dataset config has a `vad_label` block
(`backend: energy` or `silero`, see `puresound/audio/vad.py`); otherwise it is
`None`.

### Config usage

```yaml
loss_func:
  - type: VADActivityLoss
    weighted: 0.1
    args:
      frame_length: 800
      hop_length: 320
      activity_threshold_db: -40
      false_positive_weight: 2.0
      false_negative_weight: 1.0
```

### Design notes

- The loss acts on the output waveform itself, so it penalises output energy in
  frames that should be silent (background talkers, noise) and missing energy in
  frames that should be active, without adding any module to the model.
- Without external labels the target is relative to each reference's own peak,
  so it adapts to the clip's level. With external labels the output is compared
  with an absolute 0 dBFS anchor instead: the threshold then does not move with
  volume or clipping perturbations applied to the reference, and since the model
  output is clamped to `[-1, 1]` its level is always at or below 0 dBFS.
- `false_positive_weight > 1` favours silence over residual activity.

## Class: `VADHeadBCELoss`

BCE-with-logits on a backbone VAD head's frame logits against the frame labels.
Unlike `VADActivityLoss`, which reads the output waveform, this supervises an
explicit head ([`VADHead`](../models/lobe/heads.md)).

```python
VADHeadBCELoss(
    false_positive_weight: float = 1.0,   # BCE weight on label-inactive frames
    false_negative_weight: float = 1.0,   # BCE weight on label-active frames
    balance_per_batch: bool = False,      # equalise total BCE mass of the two classes
)
```

`forward(vad_logits, vad_target) -> Tensor`.
`required_inputs = ("vad_logits", "vad_target")`: the backbone's
`last_vad_logits` `[N, T]` and `batch["vad_target"]`. Frame counts are aligned
by truncating to the shorter (backbone framing and label framing can differ by
one frame). A missing head or missing labels raises `ValueError` naming the
config knob to enable (`vad_head`, `vad_label`).

With `balance_per_batch`, positive frames are scaled by `n / (2 * n_pos)` and
negative frames by `n / (2 * n_neg)` on top of the static weights, so an
imbalanced batch cannot reward an all-speech or all-silence constant. A batch
containing only one class keeps the static weights. The deployment threshold on
the head is calibrated separately.

```yaml
model:
  backbone:
    backbone_args:
      vad_head: {enabled: True, hidden: 64, kernel_t: 5}
loss_func:
  - type: VADHeadBCELoss
    weighted: 0.1
    args: {false_positive_weight: 1.0, false_negative_weight: 1.0, balance_per_batch: False}
```

## Class: `BackgroundVADHeadBCELoss`

The same BCE as `VADHeadBCELoss` (it inherits `forward` and the constructor),
aimed at a background-speech head: "is non-target speech present now" instead
of "should the recogniser listen now". It gives the bottleneck an explicit
representation of background talkers without making background speech part of
the output.

`required_inputs = ("background_vad_logits", "background_vad_target")`: the
backbone's `last_background_vad_logits`, populated when DPCRN is built with
`backbone_args.background_vad_head`, and `batch["background_vad_target"]`. The
declaration replaces the parent's, so the subclass never reads the foreground
head. Error messages name `background_vad_head` and `background_vad_target`.

When no row in a batch carries background speech, the collate emits no
`background_vad_target`. The training module's input provider then supplies an
all-zero target shaped like the logits, so the loss runs its ordinary BCE on a
valid "no background speech" signal instead of failing on `None`.

## Class: `F1_loss`

Soft F1 on probabilities, after
[asteroid's `soft_f1`](https://github.com/asteroid-team/asteroid/blob/fc0967a2eaf42f9446b17f7d039598deffd46f91/asteroid/losses/soft_f1.py).

```python
F1_loss(eps: float = 1e-10)
```

`forward(estimates, targets) -> Tensor`: soft TP/FP/FN summed over all
elements, `precision = tp / (tp + fp)`, `recall = tp / (tp + fn)`, returns
`1 - F1`. No thresholding, so it is differentiable. It declares no
`required_inputs`, so through the training module it would receive
`("enhanced", "target")`; it is meant for callers that pass probabilities.
