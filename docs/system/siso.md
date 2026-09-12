# puresound.system.siso

繁體中文版本：[`siso.zh-TW.md`](siso.zh-TW.md)

Single-Input Single-Output (SISO) PyTorch Lightning training modules. "Single input" counts *acoustic* observations: one noisy waveform in, one enhanced waveform out. MISO ([`miso.EncDecCondMaskBase`](miso.md)) is reserved for the case where a second input is itself an *audio stream* that needs its own front-end (e.g. an enrollment utterance for target-speaker extraction).

Use cases: `EncDecMaskBase` — mask-based or mapping-based speech enhancement. `EncPredClassBase` — speaker embedding (**legacy**).

## Class: `EncDecMaskBase`

Mask-based (or mapping-based) enhancement trainer.

```
Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Restore Features -> Decoder -> Wav
```

### Constructor

```python
EncDecMaskBase(
    encoder: nn.Module,
    feats: nn.Module,
    backbone: nn.Module,
    mask_type: str = "complex",
    encoder_lr_factor: float = 1.0,
    feats_lr_factor: float = 1.0,
    backbone_lr_factor: float = 1.0,
    train_vad_head_only: bool = False,
    gate_head_lr_factor: float = 1.0,
    channel_consistency: Optional[dict] = None,
    verbose: bool = False,
)
```

**Core args:**
- `encoder` — STFT/Conv1D-based encode/decode structure (e.g. `ConvEncDec`, `FreeEncDec`).
- `feats` — feature transform between encoder and backbone (e.g. `FeatureEncoder`); returns `(features, features_for_enhanced)`.
- `backbone` — model backbone that predicts the mask.
- `mask_type` — lowercased, not validated eagerly. `forward()` only handles `"complex"`, `"deepfilter"`, `"wiener"`, `"mvdr"`, `"mapping"`; any other value (e.g. `"real"`, `"polar"`) falls through to a bare `raise NameError` the first time `forward()` runs, not at construction time.
- `*_lr_factor` — per-module learning-rate multipliers, consumed by `get_total_param_groups()`.

**Optional training features** (both off by default, each a config knob):

- **`train_vad_head_only` + `gate_head_lr_factor`** — gate-only training. Freezes `encoder` / `feats` / `backbone` (`requires_grad_(False)`, and kept in `.eval()` so BatchNorm running stats and dropout stay fixed even across Lightning's per-epoch `.train()`) and trains only `backbone.vad_head`. Requires the backbone to already expose an enabled `vad_head` (e.g. DPCRN's `vad_head={"enabled": True, ...}` ctor arg) — otherwise `__init__` raises `ValueError("train_vad_head_only=True requires an enabled backbone vad_head")`. `get_total_param_groups()` then returns a single `"gate_head"` group instead of the usual three. Real recipe: a gate-only recipe (`backbone_lr_factor: 0.0`, `train_vad_head_only: True`, warm-started from a frozen separator checkpoint). Covered by `test/test_system/test_dpcrn_gate_training.py` and `test_param_groups_optim.py`.
- **`channel_consistency`** — with probability `prob` per training step, re-runs `forward()` on a channel-perturbed copy of (a sub-batch of) the mixture and penalizes the mask for changing. The perturbation (`_random_channel_perturb`) is a smooth random cosine-series EQ plus a random gain applied to the *whole* signal — the physical form of a device recording chain, which scales every source in the mix by the same real positive `H(f)`, so the near/far level contrast (and the ideal complex ratio mask) is unchanged; any mask change under it is pure channel sensitivity being regularized away. Config keys: `{enabled, prob, weight, eq_db, gain_db, eq_orders, period, max_rows}`; `None` / absent / `{"enabled": False}` is a strict no-op. Real recipe: `egs/voice_isolate/config/train_dpcrn.yaml` (`{enabled: True, prob: 0.3, weight: 1.0, eq_db: 6.0, gain_db: 4.0, max_rows: 4}`). Covered by `test/test_system/test_channel_consistency.py`.

**Optional inference knobs** (`forward()` args, no effect on training — training callers never pass non-default values):
- `dry_blend`, `spec_floor` — over-suppression relief; see `forward()` below.

### `train(mode: bool = True)`

Overridden so gate-only training keeps the frozen separator deterministic: `requires_grad=False` alone doesn't stop BatchNorm/dropout from drifting, so whenever `train_vad_head_only` is set, this override always forces `encoder` / `feats` / `backbone` back to `.eval()` regardless of `mode`, while letting `backbone.vad_head` follow the requested mode. Without `train_vad_head_only`, it behaves exactly like `nn.Module.train`.

### `forward(wav, dry_blend=1.0, spec_floor=0.0) -> Tensor`

- `wav` — `[N, T]` (a leading singleton channel is squeezed away).
- `dry_blend` — inference-only over-suppression relief in `(0, 1]`. Output becomes `dry_blend * enh + (1 - dry_blend) * input` over their overlapping length, then clamped to `[-1, 1]`; `1.0` (default) is a no-op. This is exactly the release blend `egs/voice_isolate`'s deployed checkpoints ship with — `dpcrn_v8`/`dpcrn_curriculum_v1` are released with `dry_blend 0.9` (`out = 0.9*enhanced + 0.1*input`), bounding worst-case attenuation to about −20 dB and trading a little interferer leakage for fewer deletions (see `egs/voice_isolate/README.md`). Mask-type-agnostic — it's applied in the waveform domain, after decoding, regardless of `mask_type`.
- `spec_floor` — inference-only spectral floor in `[0, 1)`, **complex-mask models only**. Clamps each enhanced T-F bin's magnitude to at least `spec_floor * |mix bin|` while preserving the enhanced phase (`_apply_spec_floor`); `0.0` (default) is a no-op.
- Returns the enhanced waveform clamped to `[-1, 1]`.
- Side effect: `self.last_mask` is reassigned on every call — the channel-consistency regularizer in `training_step` reads it immediately afterward; it carries no meaning across unrelated calls.

### `compute_loss(enhanced, target, vad_target=None, batch=None) -> (Tensor, List[float])`

Aligns `enhanced`/`target` to the shorter length, then iterates `self.loss_func_list` / `self.loss_func_list_w` and dispatches each term by attribute flags the loss module sets on itself:

| flag on `loss_func` | call |
|---|---|
| `uses_vad_logits` | `loss_func(backbone.last_vad_logits, vad_target)` |
| `uses_background_vad_logits` | `loss_func(backbone.last_background_vad_logits, bg_target)` — see note below |
| `uses_dist_preds` | `loss_func(backbone.last_dist_preds, batch or {})` |
| `uses_batch` | `loss_func(enhanced, target, batch or {})` |
| `uses_vad_target` | `loss_func(enhanced, target, vad_target=vad_target)` |
| `uses_inactive_labels` | `loss_func(enhanced, target, inactive_labels=inactive_labels)` |
| *(none of the above)* | `loss_func(enhanced, target)` |

`inactive_labels` marks rows whose reference is fully silent (`target.abs().amax(dim=-1) == 0`, i.e. target-absent training rows) — losses that opt in (e.g. the SDR/STFT losses) route those rows through a variant that avoids the `10*log10(0/X)` blow-up of vanilla SDR on an all-zero reference. `last_vad_logits` / `last_background_vad_logits` / `last_dist_preds` are backbone side outputs from the *preceding* `forward()` call, read via `getattr(self.backbone, ..., None)` — not every backbone defines all three. DPCRN currently exposes `last_vad_logits` / `last_dist_preds` through its optional `vad_head` / `dist_head` ctor args (`puresound/nnet/dpcrn.py`, heads in `puresound/nnet/lobe/heads.py`); no in-repo backbone defines `last_background_vad_logits` yet, so that flag is forward-compatible plumbing already exercised by `BackgroundVADHeadBCELoss` and its tests, not by any deployed head.

`bg_target` for `uses_background_vad_logits` is `batch.get("background_vad_target")` if present, but if that key is absent **and** `background_vad_logits is not None`, it is synthesized as `torch.zeros_like(background_vad_logits)` rather than passed through as `None`. This matters because an all-silent-background batch produces no `background_vad_reference`/`_target` at all from the dataset/collate (which only zero-fills missing rows when *some* row in the batch has background speech) — a real, valid batch state, not an error — so `compute_loss` treats it as "no background activity" instead of crashing `BackgroundVADHeadBCELoss` on `None` (a regression covered by `test/test_system/test_siso_compute_loss.py`; this exact gap once took down a DDP job). The analogous foreground case is **not** rescued the same way: a missing `vad_target` for a loss with `uses_vad_target=True` still surfaces as a `ValueError` (raised by `VADHeadBCELoss` itself, not by `compute_loss`) — missing foreground labeling is treated as a configuration error, not a valid silent state.

### `training_step` / `validation_step` / `test_step` / `predict_step`

All four start with `batch = self.ensure_vad_targets(batch)` ([base.md](base.md)).

- **`training_step`** — `forward(noisy_speech)` → `compute_loss(..., batch=batch)` → optional channel-consistency term → logs `train_step_loss` (`sync_dist=False`; a synced progress-bar metric would deadlock DDP, since the bar's refresh — and the collective it triggers — isn't rank-lockstep) [+ per-term logs if `verbose`] → accumulates `epoch_train_loss` → returns `{"loss": total_loss}`.

  The channel-consistency term, when configured, fires on a **rank-synchronized schedule**: `(batch_idx % period) < round(prob * period)` — a pure function of `batch_idx`, never a per-rank random draw. This is a hard DDP-safety requirement, not a style choice: the extra forward triggers SyncBatchNorm all-gathers, so if ranks fired on independent random draws their collective sequences would desync and the job would hang until the NCCL watchdog kills it. When it fires, the added loss is `weight * L1(mask(perturbed_subbatch), mask(clean_subbatch).detach())`, logged as `train_step_cons_loss`; `max_rows` caps the sub-batch so the extra forward's activations (kept alive on top of the outstanding main-forward graph) don't double peak memory.

- **`validation_step`** — `forward` → `compute_loss(..., batch=batch)` → logs `valid_step_loss(_i)` (`sync_dist=True`). No channel-consistency term at validation.
- **`test_step`** — for each registered `_metrics_func`, resamples `clean_speech`/`enhanced_speech` to that metric's declared sample rate (`wav_resampling(..., backend="sox")`) if it differs from `batch["sr"]`, then scores and accumulates into `puresound_logging`.
- **`predict_step`** — `forward` → `AudioIO.save` the enhanced wav to `{eval_output_folder_path}/{batch['name'][0]}.wav`. No embedding export (contrast `EncPredClassBase.predict_step` below, and `miso.EncDecCondMaskBase.predict_step`, both of which also write a `.txt` embedding).

### `get_total_param_groups()`

If `train_vad_head_only`: returns `{"gate_head": {"params": backbone.vad_head.parameters(), "lr_factor": gate_head_lr_factor}}` only. Otherwise: `{"encoder": ..., "feats": ..., "backbone": ...}`, each `{"params": <module>.parameters(), "lr_factor": <module>_lr_factor}`. Feeds directly into [`system.optim.create_optimizer_and_scheduler`](optim.md).

---

## Class: `EncPredClassBase`

> **Status: legacy.**

SISO classification pipeline for speaker-embedding extraction — no mask, no decoder.

```
Wav -> Encoder -> Features -> Backbone -> Predict classes/embedding
```

### Constructor

```python
EncPredClassBase(
    encoder: nn.Module,
    feats: nn.Module,
    backbone: nn.Module,
    encoder_lr_factor: float = 1.0,
    feats_lr_factor: float = 1.0,
    backbone_lr_factor: float = 1.0,
    verbose: bool = False,
)
```

### `forward(wav) -> Tensor`

`encoder` → `feats` (keeps only the `features` half of its `(features, features_for_enhanced)` return) → squeeze channel dim → `backbone(features)`. No mask application, no decoder.

### `compute_loss(pred, target)`

Plain weighted sum over `loss_func_list` / `loss_func_list_w` — no attribute-flag dispatch (unlike `EncDecMaskBase.compute_loss`).

### `training_step` / `validation_step` / `test_step` / `predict_step`

Same total-loss logging pattern as `EncDecMaskBase` (`train_step_loss` with `sync_dist=False`, `valid_step_loss(_i)` with `sync_dist=True`). `test_step` calls `_metrics_func[name]["func"](pred, target)` directly — no per-metric resampling (there's no waveform output to resample). `predict_step` L2-normalizes the embedding, mean-pools across rows if the batch has more than one, and writes it to `{eval_output_folder_path}/{batch['name'][0]}.txt` via `np.savetxt` — no wav output.

### `get_total_param_groups()`

`encoder` / `feats` / `backbone` groups, **plus one `loss{i}` group per entry in `loss_func_list`** (`lr_factor: 1.0` each) — parametrized losses (e.g. a margin-based classification loss with learned class centers) carry trainable weights of their own that need an optimizer group too. `EncDecMaskBase.get_total_param_groups()` never does this.

## Example

```python
import torch
from puresound.system.siso import EncDecMaskBase
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.nnet import FeatureEncoder, DPCRN
from puresound.nnet.loss.sdr import SDRLoss

encoder  = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, trainable=False)
feats    = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False)
backbone = DPCRN(input_dim=256, channels=(2, 32, 64, 128), rnn_hidden=96)

model = EncDecMaskBase(encoder=encoder, feats=feats, backbone=backbone, mask_type="complex")
model.register_loss_func(torch.nn.ModuleList([SDRLoss()]), [1.0])
```

Any backbone exposing the same `forward(features) -> mask` contract plugs in the same way — e.g. `DPRNN` (`puresound/nnet/dprnn.py`; real kwargs are `input_size, hidden_size, output_size, n_blocks=2, seg_size=20, seg_overlap=False, causal=True, embed_dim=0, ...`, **not** `in_channel`/`hid_channel`/`out_channel`/`num_layers`):

```python
from puresound.nnet import DPRNN

backbone = DPRNN(input_size=256, hidden_size=64, output_size=256, n_blocks=6)
```
