# puresound.nnet.loss.vad

繁體中文版本：[vad.zh-TW.md](vad.zh-TW.md)

Frame-activity supervision: one loss derives an activity target from the
waveform itself (`VADActivityLoss`), two supervise an explicit backbone head
(`VADHeadBCELoss`, `BackgroundVADHeadBCELoss`), and one is a generic
differentiable F1 metric (`F1_loss`).

## Class: `VADActivityLoss`

Differentiable BCE on frame-level speech activity. **Pure time-domain**: it
frames the waveform with `Tensor.unfold` and takes per-frame mean-square
power directly — there is no STFT/FFT anywhere in this class (a previous
version of this doc described STFT-based framing with `n_fft` / `win_length`
arguments that do not exist on this class; the example below reflects the
real constructor).

### Constructor

```python
VADActivityLoss(
    frame_length: int = 400,
    hop_length: int = 160,
    activity_threshold_db: float = -40.0,
    logit_scale: float = 0.25,
    false_positive_weight: float = 1.0,
    false_negative_weight: float = 1.0,
    require_vad_target: bool = False,
    eps: float = 1e-8,
)
```

**Parameters:**
- `frame_length` / `hop_length` – window/hop for the raw-sample `unfold`
  framing (samples, not FFT bins).
- `activity_threshold_db` – dB threshold that separates "active" from
  "inactive" frames. What it's measured relative to depends on which of the
  two modes below is in effect.
- `logit_scale` – scales `(enh_db - activity_threshold_db)` into a BCE logit;
  controls how sharp the soft decision is around the threshold.
- `false_positive_weight` / `false_negative_weight` – BCE class weights.
- `require_vad_target` – if `True`, `forward` raises `ValueError` instead of
  falling back to the reference-derived target when no `vad_target` is given.
- `eps` – division/log guard.

### `forward(enh, ref, vad_target=None) -> Tensor`

Both waveforms are coerced to `[B, T]` and length-aligned, then framed via
`_frame_power` (pad up to one frame if shorter, `unfold(-1, frame_length,
hop_length)`, `frames.square().mean(dim=-1)` — no window function, no FFT).
There are two modes, selected by whether `vad_target` is passed:

- **No external target** (the default; raises instead if
  `require_vad_target=True`): the activity label comes from `ref`'s *own*
  power, framed the same way, relative to that utterance's own peak frame
  power (`ref_db = 10*log10(ref_power / ref_power.amax())`) — an
  utterance-relative threshold that adapts to each clip's loudness. A
  utterance whose reference is entirely silent is forced to an all-inactive
  target explicitly (rather than relying on the epsilon-guarded division to
  land below threshold on its own).
- **External `vad_target`** (from a labeler upstream, e.g. `vad_label:
  {backend: silero}` or `{backend: energy}` in the recipe config): the given
  labels are used directly (padded/truncated to the frame count), and — this
  is the part most easily missed — the enhanced signal's power is measured
  against an **absolute 0 dBFS anchor** (`reference_power = ones`) instead of
  the reference's own peak. The source comment explains why: this makes
  `activity_threshold_db` an absolute dBFS threshold, invariant to
  volume/clipping perturbations on the reference signal, since the model's
  output is clamped to `[-1, 1]` and so `enh_db` always lies in `(-inf, 0]`.

Either way, `enh`'s framed power is converted to dB against the mode's
`reference_power`, turned into a BCE logit
(`(enh_db - activity_threshold_db) * logit_scale`), and scored against
`target_activity` with `false_positive_weight` / `false_negative_weight`
class weights via `binary_cross_entropy_with_logits`.

### Config usage

```yaml
# egs/noise_suppression/config/dpcrn.yaml (active recipe)
# VAD-driven objective: reduce background-speaker false activity while
# preserving clean foreground-active frames.
- type: VADActivityLoss
  weighted: 0.1
  args:
    frame_length: 800
    hop_length: 320
    activity_threshold_db: -40
    false_positive_weight: 2.0
    false_negative_weight: 1.0
```

### Example

```python
from puresound.nnet.loss.vad import VADActivityLoss

vad_loss = VADActivityLoss(frame_length=400, hop_length=160, activity_threshold_db=-40.0)
loss = vad_loss(enhanced_wav, clean_wav)
```

## Class: `VADHeadBCELoss`

Frame-level BCE-with-logits against the dataset's `vad_target`, for training a
backbone `VADHead` (see [nnet.lobe.heads](../lobe/heads.md)) — as opposed to
`VADActivityLoss` above, which derives activity from waveform energy, this
supervises an explicit per-frame logit head exposed by the backbone. Sets
`uses_vad_logits = True` so `EncDecMaskBase.compute_loss`
(`puresound/system/siso.py`) routes `backbone.last_vad_logits` in.

```python
VADHeadBCELoss(false_positive_weight: float = 1.0, false_negative_weight: float = 1.0, balance_per_batch: bool = False)
```

`false_positive_weight` up-weights silence/far frames wrongly scored active.
`balance_per_batch` reweights the two classes to equal total BCE mass so an
imbalanced batch can't reward an all-speech or all-silence constant (a batch
with only one class present falls back to the plain static weights, since
there is no class balance to speak of). Frame counts between `vad_logits` and
`vad_target` are aligned by truncating to the shorter (a comment in the source
notes the backbone's internal STFT framing and the label framing can differ
by one frame). Not part of the shipped `train_dpcrn.yaml` recipe; used by the
gated-bottleneck experiment `egs/voice_isolate/config/exp/train_dpcrn_gate.yaml`
together with `backbone_args.vad_head: {enabled: True, hidden: 128, kernel_t: 5}`.

## Class: `BackgroundVADHeadBCELoss`

```python
class BackgroundVADHeadBCELoss(VADHeadBCELoss):
    uses_vad_logits = False
    uses_background_vad_logits = True
    _logits_attr = "last_background_vad_logits"
    _head_config_key = "background_vad_head"
    _target_key = "background_vad_target"
```

The three `_*` attributes only feed the error messages `VADHeadBCELoss.forward`
raises when a required input is missing, so a misconfigured background head is
told to enable `background_vad_head`, not the foreground `vad_head`.

It **does not override `forward`** — it is the exact same strict,
backpropagating BCE computation as `VADHeadBCELoss`, just re-pointed (via the
dispatch flags above) at `background_vad_target` /
`backbone.last_background_vad_logits` (interferer-speech activity) instead of
the foreground labels. The companion head gives the bottleneck an explicit
representation for background talkers without making background speech part
of the enhanced output.

Because `forward` is inherited unchanged, this class has **no built-in
behavior for a missing/`None` target** — it still raises, exactly like
`VADHeadBCELoss` does. A previous version of this doc attributed a
"synthesizes a zero loss with a graph edge" mechanism to the class itself;
that is not where it lives. The actual guard is one layer up, in
`EncDecMaskBase.compute_loss` (`puresound/system/siso.py`):

```python
elif getattr(loss_func, "uses_background_vad_logits", False):
    bg_target = None if batch is None else batch.get("background_vad_target")
    # When no sample in the batch carries background speech, the dataset/
    # collate pipeline emits no `background_vad_target` key at all (it only
    # zero-fills missing rows when *some* row has background speech). An
    # all-silent batch is still a valid signal -- the target is simply
    # all-zeros -- so synthesize it rather than crashing on a None target.
    if bg_target is None and background_vad_logits is not None:
        bg_target = torch.zeros_like(background_vad_logits)
    weighted_loss = weighted * loss_func(background_vad_logits, bg_target)
```

`compute_loss` synthesizes the all-zero **target** before calling the loss —
the loss then runs its ordinary (non-zero-valued, real) BCE against that
target; it never actually sees a `None`. This is a real fix for a real past
failure (an all-background-silent batch previously crashed the whole DDP job
on a `None` target inside the loss); see
`test/test_system/test_siso_compute_loss.py::test_background_vad_loss_handles_all_silent_batch`
and `::test_background_vad_loss_matches_explicit_zero_target` for the
regression coverage, the latter confirming the synthesized path produces
exactly the same loss as passing the zero target explicitly.

One more thing worth knowing if you go looking for it: the current `DPCRN`
backbone (`puresound/nnet/dpcrn.py`) wires up `vad_head` and `dist_head` but
has no `background_vad_head` — `backbone.last_background_vad_logits` is a
hook `compute_loss` reads with a `None` default (`getattr(..., None)`), not
something the shipped backbone currently populates. No active recipe currently
selects `BackgroundVADHeadBCELoss`.

## Class: `F1_loss`

Differentiable soft-F1 loss for binary frame/utterance classification, from
[asteroid's `soft_f1`](https://github.com/asteroid-team/asteroid/blob/fc0967a2eaf42f9446b17f7d039598deffd46f91/asteroid/losses/soft_f1.py).

```python
F1_loss(eps: float = 1e-10)
```

`forward(estimates, targets) -> Tensor` computes soft precision/recall from
`estimates`/`targets` directly (no thresholding) and returns `1 - f1.mean()`.
Not currently used by any recipe in this repo.
