# puresound.system.siso

繁體中文版本：[`siso.zh-TW.md`](siso.zh-TW.md)

Single-Input Single-Output (SISO) PyTorch Lightning training modules. "Single input" counts *acoustic* observations: one noisy waveform in, one enhanced waveform out. MISO ([`miso.EncDecCondMaskBase`](miso.md)) is reserved for the case where a second input is itself an *audio stream* that needs its own front-end (e.g. an enrollment utterance for target-speaker extraction).

Use cases: `EncDecMaskBase` — mask-based or mapping-based speech enhancement (the noise-suppression and voice-isolation recipes). `EncPredClassBase` — speaker embedding (**legacy**).

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
    paired_view_consistency: Optional[dict] = None,
    metric_gan: Optional[dict] = None,
    verbose: bool = False,
)
```

**Core args:**
- `encoder` — STFT/Conv1D-based encode/decode structure (e.g. `ConvEncDec`, `FreeEncDec`).
- `feats` — feature transform between encoder and backbone (e.g. `FeatureEncoder`); returns `(features, features_for_enhanced)`.
- `backbone` — model backbone that predicts the mask.
- `mask_type` — lowercased, not validated at construction. `forward()` handles `"complex"`, `"deepfilter"`, `"wiener"`, `"mvdr"`, `"mapping"`; any other value raises `ValueError` the first time `forward()` runs.
- `*_lr_factor` — per-module learning-rate multipliers, consumed by `get_total_param_groups()`.

**Optional training features** (all off by default, each a config knob under the recipe's `lightning_module.module_args`):

- **`train_vad_head_only` + `gate_head_lr_factor`** — gate-only training. Freezes `encoder` / `feats` / `backbone` (`requires_grad_(False)`, and kept in `.eval()` so BatchNorm running statistics and dropout stay fixed even across Lightning's per-epoch `.train()`) and trains only `backbone.vad_head`. Requires the backbone to expose an enabled `vad_head` (e.g. DPCRN's `vad_head` constructor argument) — otherwise `__init__` raises `ValueError("train_vad_head_only=True requires an enabled backbone vad_head")`. `get_total_param_groups()` then returns a single `"gate_head"` group. Covered by `test/system/test_siso.py`.
- **`channel_consistency`** — with average rate `prob` per training step, re-runs `forward()` on a channel-perturbed copy of (a sub-batch of) the mixture and penalizes the mask for changing. The perturbation (`_random_channel_perturb`) is a smooth, zero-phase random cosine-series EQ plus a random per-row gain applied to the *whole* mixture — the physical form of a device recording chain, which scales every source in the mix by the same real positive `H(f)`, so the near/far level contrast (and the ideal complex ratio mask) is unchanged; any mask change under it is channel sensitivity, which the term regularizes away. Keys: `{enabled, prob, weight, eq_db, gain_db, eq_orders, period, max_rows}` (defaults `eq_db 6.0`, `gain_db 4.0`, `eq_orders 4`, `period 10`, `weight 1.0`); `None`, absent or `{"enabled": False}` is a strict no-op. The FFT pair behind the EQ retries once after releasing cached device memory and then falls back to the CPU if the device cannot plan the transform — the result is identical, only that step's latency differs, and each fallback is logged. Covered by `test/system/test_siso.py`.
- **`paired_view_consistency`** — `{enabled, max_rows}` ([`paired_views.PairedViewConsistencyConfig`](index.md)). When a batch carries a `paired_view` (an auxiliary rendering of some rows), runs at most one extra forward on it and hands primary and auxiliary outputs to every registered loss that declares `paired_output` / `paired_consistency`. The extra forward's batch size is the maximum across ranks, so every rank runs the same collectives; the primary forward's side outputs (`last_mask`, `backbone.last_*`) are restored afterwards. Raises if enabled with no loss declaring `paired_output`. Covered by `test/system/test_paired_views.py`.
- **`metric_gan`** — a learned PESQ critic ([`metric_gan.MetricGanConfig`](index.md): `enabled, weight, warmup_steps, n_fft, hop, ndf, lr_factor, pesq_workers, rows_per_step, buffer_rows, d_rows, sample_rate`). Builds a `MetricDiscriminator` as `self.metric_disc`; each training step adds a discriminator term on replayed rows whose PESQ-WB labels a background process pool computes, plus, after `warmup_steps`, a generator term `weight * (D(clean, enhanced) - 1)^2`. Target-absent rows (silent clean) are left out of both. The critic's weights ride in the checkpoint; nothing at inference or in the streaming export reads them. Covered by `test/system/test_metric_gan.py`.

### `train(mode: bool = True)`

Overridden so gate-only training keeps the frozen separator deterministic: `requires_grad=False` alone does not stop BatchNorm/dropout from drifting, so whenever `train_vad_head_only` is set, this override forces `encoder` / `feats` / `backbone` back to `.eval()` regardless of `mode`, while `backbone.vad_head` follows the requested mode. Without `train_vad_head_only` it behaves like `nn.Module.train`.

### `forward(wav, dry_blend=1.0, spec_floor=0.0, postprocess=None, presence_gate=None, onset_guard=None) -> Tensor`

- `wav` — `[N, T]` (a leading singleton channel is squeezed away).
- `dry_blend`, `spec_floor` — inference-only over-suppression relief, resolved into a [`Postprocessor`](index.md) (`system.postprocess`). `spec_floor` in `[0, 1)` clamps each enhanced T-F bin's magnitude to at least `spec_floor * |mix bin|`, keeping the enhanced phase; it applies to the complex-mask path only, and the other mask types raise rather than ignore it. `dry_blend` in `(0, 1]` mixes the untouched input back in on the finished waveform, `dry_blend * enh + (1 - dry_blend) * input` over their overlapping length, clamped to `[-1, 1]`; it is mask-type-agnostic. It puts a hard ceiling under suppression of `20*log10(1 - dry_blend)` (0.9 caps attenuation at −20 dB). Both defaults are no-ops and the training path never sets them.
- `postprocess` — a built `Postprocessor`, as an alternative to the two keywords above; passing both raises `ValueError`.
- `presence_gate` — a built [`PresenceGate`](index.md) (`system.presence_gate`): an inference-only gain applied after the blend, driven by a linear readout on the backbone's bottleneck. Requires a backbone with `last_bottleneck` (`ValueError` otherwise); the backbone's `stash_bottleneck` is switched on for the duration of the call.
- `onset_guard` — a built [`OnsetGuard`](index.md) (`system.onset_guard`), applied last, after the blend and the presence gate, because its job is to restore the dry input whatever the earlier stages did. It reads only the input waveform. `None` is bit-identical to not having it.
- Returns the enhanced waveform clamped to `[-1, 1]`.

For `mask_type="complex"` the mask is applied by `Masker.apply_complex_mask_with_df`, which also takes the backbone's `last_df_coefs` when it has a deep-filter residual head (`None` otherwise, which makes it the plain complex mask). Side effect: `self.last_mask` is reassigned on every call — the channel-consistency term reads it right after the call; it carries no meaning across unrelated calls.

None of the post-mask stages is part of the exported graph. The streaming export records the `Postprocessor` and `OnsetGuard` settings in its manifest for the ONNX runtime to apply; `PresenceGate` runs only here (see [streaming](../../usage/streaming/index.md)).

### `compute_loss(enhanced, target, vad_target=None, batch=None) -> (Tensor, List[float])`

Aligns `enhanced`/`target` to the shorter length, computes `inactive_labels` (rows whose reference is fully silent, `target.abs().amax(dim=-1) == 0`), and reduces the registered losses with `reduce_losses`, calling each through [`invoke_loss`](base.md) against the provider table `_loss_providers` builds. A loss asks for inputs by declaring `required_inputs`; this module can provide:

| provider name | value |
|---|---|
| `enhanced`, `target` | the aligned waveform pair |
| `batch` | the batch dict (`{}` if none) |
| `inactive_labels` | the per-row target-absent mask |
| `vad_target` | the foreground VAD target |
| `vad_logits` | `backbone.last_vad_logits` |
| `background_vad_logits` | `backbone.last_background_vad_logits` |
| `background_vad_target` | `batch["background_vad_target"]`, or zeros shaped like the logits (see below) |
| `dist_preds` | `backbone.last_dist_preds` |
| `bottleneck` | `backbone.last_bottleneck_graph` — the frequency-pooled bottleneck with its graph |
| `identity_emb` | `backbone.last_identity_emb` |
| `identity_head` | the `backbone.identity_head` module itself |
| `proximity` | `backbone.last_proximity` |

Backbone entries are side outputs of the *preceding* `forward()`, read with `getattr(self.backbone, name, None)`; not every backbone defines them. DPCRN produces them from its optional heads (`vad_head`, `background_vad_head`, `dist_head`, `identity_head`, `proximity_head`; see `puresound/nnet/dpcrn.py` and `puresound/nnet/lobe/heads.py`), and `last_bottleneck_graph` only when built with `expose_bottleneck`. That graph-carrying bottleneck is deliberately a different attribute and switch from `last_bottleneck`, the detached inference stash the presence gate reads. `identity_head` is the one entry that hands over a module rather than a tensor: the identity loss keeps a stop-gradient EMA copy of the head's weights inside itself, which keeps untrained teacher weights out of the checkpoint. `test/system/test_base.py` checks every shipped loss's `required_inputs` against this table.

`inactive_labels` exists for losses that opt in (the SDR/STFT losses): they route target-absent rows through a variant that avoids the `10*log10(0/X)` blow-up of vanilla SDR on an all-zero reference.

`background_vad_target` is synthesized as `torch.zeros_like(background_vad_logits)` when the batch has no `background_vad_target` but the backbone produced logits. An all-silent-background batch produces no background reference at all from the dataset/collate (which zero-fills missing rows only when *some* row has background speech); that is a valid batch state meaning "no background activity", so it is treated as all-zeros rather than crashing `BackgroundVADHeadBCELoss` on `None` (`test/system/test_siso.py`). The foreground case is not rescued the same way: a missing `vad_target` for a loss that asks for it surfaces as a `ValueError` from the loss itself — missing foreground labeling is a configuration error, not a valid silent state.

### `training_step` / `validation_step` / `test_step` / `predict_step`

All four start with `batch = self.ensure_vad_targets(batch)` ([base.md](base.md)).

- **`training_step`** — `forward(noisy_speech)` → `compute_loss(..., batch=batch)` → optional paired-view term → optional MetricGAN term → optional channel-consistency term → logs `train_step_loss` (`sync_dist=False`: a synced progress-bar metric deadlocks DDP, because the bar's refresh — and the collective it triggers — is not in lockstep across ranks) [+ per-term logs if `verbose`] → accumulates `epoch_train_loss` → returns `{"loss": total_loss}`.

  The channel-consistency term fires on a **rank-synchronized schedule**: `(batch_idx % period) < round(prob * period)` — a pure function of `batch_idx`, never a per-rank random draw. This is a DDP-safety requirement, not a style choice: the extra forward runs SyncBatchNorm all-gathers, so if ranks fired on independent draws their collective sequences would desync and the job would hang until the NCCL watchdog kills it. When it fires, the added loss is `weight * L1(mask(perturbed_subbatch), mask(clean_subbatch).detach())`, logged as `train_step_cons_loss`; `max_rows` caps the sub-batch (the same leading rows on every rank) so the extra forward's activations, kept alive on top of the main forward's, do not double peak memory.

- **`validation_step`** — `forward` → `compute_loss(..., batch=batch)` → logs `valid_step_loss` and, with more than one loss, `valid_step_loss_{i}` (`sync_dist=True`, per epoch). No auxiliary terms at validation.
- **`test_step`** — for each registered metric, resamples copies of `clean_speech`/`enhanced_speech` to that metric's declared sample rate (`wav_resampling(..., backend="sox")`) if it differs from `batch["sr"]`, then scores and accumulates into `puresound_logging` (a metric returning a dict contributes each of its keys).
- **`predict_step`** — `forward` → `AudioIO.save` the enhanced wav to `{eval_output_folder_path}/{batch['name'][0]}.wav`. No embedding export (contrast `EncPredClassBase.predict_step` below).

`on_train_end` closes the MetricGAN PESQ worker pool when one was started.

### `get_total_param_groups()`

If `train_vad_head_only`: returns `{"gate_head": {"params": backbone.vad_head.parameters(), "lr_factor": gate_head_lr_factor}}` only. Otherwise `{"encoder": ..., "feats": ..., "backbone": ...}`, each `{"params": <module>.parameters(), "lr_factor": <module>_lr_factor}`, plus `"metric_disc"` (`lr_factor` from the MetricGAN config) when MetricGAN is on. Feeds [`system.optim.create_optimizer_and_scheduler`](optim.md).

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

`reduce_losses` with every loss called as `loss_func(pred, target)` — no provider dispatch (unlike `EncDecMaskBase.compute_loss`).

### `training_step` / `validation_step` / `test_step` / `predict_step`

Same total-loss logging pattern as `EncDecMaskBase` (`train_step_loss` with `sync_dist=False`, `valid_step_loss(_i)` with `sync_dist=True`); no `ensure_vad_targets` call. `test_step` calls `_metrics_func[name]["func"](pred, target)` directly — no per-metric resampling (there is no waveform output to resample). `predict_step` L2-normalizes the embedding, mean-pools across rows if the batch has more than one, and writes it to `{eval_output_folder_path}/{batch['name'][0]}.txt` via `np.savetxt` — no wav output.

### `get_total_param_groups()`

`encoder` / `feats` / `backbone` groups, **plus one `loss{i}` group per entry in `loss_func_list`** (`lr_factor: 1.0` each) — parametrized losses (e.g. a margin-based classification loss with learned class centers) carry trainable weights of their own that need an optimizer group too.

## Example

```python
import torch
from puresound.system.siso import EncDecMaskBase
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.nnet import FeatureEncoder, DPCRN
from puresound.nnet.loss.sdr import SDRLoss

encoder  = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, trainable=False)
feats    = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False)
# Every per-layer tuple has len(channels) - 1 entries.
backbone = DPCRN(
    input_dim=256,
    channels=(2, 16, 32, 64),
    kernel_t=(2, 2, 2), stride_t=(1, 1, 1), dilation_t=(1, 1, 1),
    kernel_f=(5, 3, 3), stride_f=(2, 2, 1), dilation_f=(1, 1, 1),
    delay=(0, 0, 0),
    rnn_hidden=96,
)

model = EncDecMaskBase(encoder=encoder, feats=feats, backbone=backbone, mask_type="complex")
model.register_loss_func(torch.nn.ModuleList([SDRLoss()]), [1.0])

enhanced = model(torch.randn(2, 16000) * 0.1)                  # training-time forward
relieved = model(torch.randn(2, 16000) * 0.1, dry_blend=0.9)   # inference relief
```

Any backbone exposing the same `forward(features) -> mask` contract plugs in the same way — e.g. `DPRNN` (`puresound/nnet/dprnn.py`; constructor `input_size, hidden_size, output_size, n_blocks=2, seg_size=20, seg_overlap=False, causal=True, embed_dim=0, ...`):

```python
from puresound.nnet import DPRNN

backbone = DPRNN(input_size=256, hidden_size=64, output_size=256, n_blocks=6)
```
