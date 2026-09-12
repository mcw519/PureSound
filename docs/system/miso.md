# puresound.system.miso

繁體中文版本：[`miso.zh-TW.md`](miso.zh-TW.md)

> **Status: legacy** — kept working and frozen: no new features, no rewrites.

Multi-Input Single-Output (MISO) PyTorch Lightning training module. "Multiple input" means a second *acoustic* input alongside the noisy mixture: a conditioning waveform (e.g. an enrollment utterance) run through its own front-end (`c_encoder` / `c_feats` / `c_backbone`) to produce an embedding that steers the mask. A model conditioned on a *non-acoustic* scalar (e.g. a distance query) is **not** MISO — that belongs in [`siso.EncDecMaskBase`](siso.md) instead, which forwards the scalar to the backbone as a FiLM bias without any second front-end.

Use case: Personalized Speech Enhancement / Target Speaker Extraction (TSE).

## Class: `EncDecCondMaskBase`

> Exported from `puresound.system` as `EncDecCondMaskBase`.

### Architecture

```
Noisy Mixture Waveform                 Enrollment (Conditioning) Waveform
  └─ encoder                             └─ c_encoder (siamese copy of encoder, or separate module)
       └─ feats                               └─ c_feats (siamese copy of feats, or separate module)
            └─ backbone(features, c_features) ◄──── c_backbone
                 └─ Mask Estimation
                      └─ Mask Application (complex/deepfilter/wiener/mvdr/mapping)
                           └─ Decoder
                                └─ (Enhanced Waveform, c_features)
```

### Constructor

```python
EncDecCondMaskBase(
    encoder: nn.Module,
    feats: nn.Module,
    backbone: nn.Module,
    c_backbone: nn.Module,
    jointed_trained: bool = True,
    siamese_encoder: bool = True,
    siamese_feats: bool = True,
    c_encoder: Optional[nn.Module] = None,
    c_feats: Optional[nn.Module] = None,
    mask_type: str = "complex",
    encoder_lr_factor: float = 1.0,
    feats_lr_factor: float = 1.0,
    backbone_lr_factor: float = 1.0,
    c_encoder_lr_factor: float = 1.0,
    c_feats_lr_factor: float = 1.0,
    c_backbone_lr_factor: float = 1.0,
    verbose: bool = False,
)
```

**Parameters:**
- `encoder` / `feats` / `backbone` — main enhancement path (identical roles to `siso.EncDecMaskBase`).
- `c_backbone` — conditioning-path backbone that turns encoded conditioning features into an embedding (e.g. an ECAPA-TDNN speaker extractor); always required and always used regardless of `siamese_*`.
- `siamese_encoder` / `siamese_feats` — independently control whether `c_encoder`/`c_feats` is a `deepcopy` of `encoder`/`feats` (`True`, the default) or the module explicitly passed via `c_encoder=`/`c_feats=` (`False` — that argument is then required). There is no single combined `siamese` flag.
- `c_encoder` / `c_feats` — only consulted when the matching `siamese_*` flag is `False`.
- `jointed_trained` (note the exact spelling — not `joint_training`) — if `False`, `c_backbone`/`c_encoder`/`c_feats` are put into `.eval()` in `__init__`, and `forward()` runs them under `torch.no_grad()`. The `torch.no_grad()` guard is what actually blocks gradients unconditionally; the `.eval()` call alone would be reverted by Lightning's normal per-epoch `.train()` call, since — unlike `siso.EncDecMaskBase`'s gate-only mode — this class does not override `train()` to keep those submodules pinned to eval.
- `mask_type` — lowercased, not validated eagerly. `forward()` only handles `"complex"`, `"deepfilter"`, `"wiener"`, `"mvdr"`, `"mapping"`; any other value (including `"real"`/`"magnitude"`, which earlier docs implied were valid) falls through to a bare `raise NameError` the first time `forward()` runs.
- `*_lr_factor` — six independent per-module learning-rate multipliers, one per component group in `get_total_param_groups()`.

### `register_loss_func` (override)

```python
register_loss_func(
    loss_func_list: nn.ModuleList,
    loss_func_list_weights: List,
    c_loss_func_list: Optional[nn.ModuleList] = None,
    c_loss_func_list_weights: Optional[List] = None,
)
```

Extends the base registry ([base.md](base.md)) with an optional second loss pair for the conditioning embedding, consumed by `compute_loss2`.

### `forward(wav, conditional_wav) -> Tuple[Tensor, Tensor]`

Returns **`(enh, c_features)`** — not a single tensor. `enh` is the enhanced waveform (clamped to `[-1, 1]`); `c_features` is the post-`c_backbone` conditioning embedding, passed through so callers (`training_step`, `compute_loss2`, `predict_step`) can supervise or export it.

### `compute_loss(enhanced, target, vad_target=None)`

Same length-alignment convention as `siso.EncDecMaskBase.compute_loss`, but a narrower dispatch: each registered loss is called as `loss_func(enhanced, target, vad_target=vad_target)` if it sets `uses_vad_target = True`, otherwise plainly `loss_func(enhanced, target)`. It does **not** have the `uses_vad_logits` / `uses_background_vad_logits` / `uses_dist_preds` / `uses_batch` / `uses_inactive_labels` dispatch branches that `siso.EncDecMaskBase.compute_loss` has — those are newer, SISO-only additions.

### `compute_loss2(pred, target)`

`assert self.c_loss_func_list is not None`, then a plain weighted sum over `c_loss_func_list` / `c_loss_func_list_w` (no attribute-flag dispatch at all). Supervises the conditioning embedding (`pred`) against a target (`conditional_target` from the batch, e.g. a speaker-ID label).

### `training_step` / `validation_step` / `test_step` / `predict_step`

- **`training_step`** — reads `noisy_speech`, `clean_speech`, `conditional_speech`, `conditional_target`; `forward` → `compute_loss`; **if** `self.jointed_trained and self.c_loss_func_list is not None`, also runs `compute_loss2(embedding, conditional_target)` and adds it into the total. Logs `train_step_loss` (`sync_dist=False` — a synced progress-bar metric would deadlock DDP, since the bar refresh that triggers the read is not rank-lockstep).
- **`validation_step`** — same `forward` + `compute_loss`, but **the `compute_loss2` addition is commented out in source**, so validation loss never includes the conditioning-embedding term even when `jointed_trained=True`. Logs `valid_step_loss(_i)` with `sync_dist=True`.
- **`test_step`** — for each registered `_metrics_func`, resamples `clean_speech`/`enhanced_speech` to that metric's declared sample rate (`wav_resampling(..., backend="sox")`) before scoring — identical pattern to `siso.EncDecMaskBase.test_step`.
- **`predict_step`** — saves the enhanced wav via `AudioIO.save`, **and** L2-normalizes the embedding, mean-pools it across rows if the batch has more than one, and writes it to a sibling `.txt` file via `np.savetxt` — mirroring `EncPredClassBase.predict_step` (siso.md), not `EncDecMaskBase.predict_step` (which saves only a wav).

### `get_total_param_groups()`

Always returns `encoder` / `feats` / `backbone` groups; adds `c_encoder` / `c_feats` / `c_backbone` groups **only if `self.jointed_trained`** — in frozen-conditioning mode the conditioning path's parameters never enter the optimizer at all, consistent with `jointed_trained=False` also blocking their gradients in `forward()`.

## Example

Adapted from the real recipe, `egs/target_speaker_extraction/config/default_config.yaml`:

```python
from puresound.system.miso import EncDecCondMaskBase
from puresound.nnet import ConvEncDec, DPCRN, EcapaTdnnExtractor, FeatureEncoder

encoder    = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, trainable=False)
c_encoder  = ConvEncDec(fft_length=512, win_length=512, hop_length=160, fmin=0, fmax=8000, sr=16000, preemphasis=0.97, trainable=False)
feats      = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False)
c_feats    = FeatureEncoder(feats_type="fbank80_16k", normalized_mode=None, trainable=False)  # see nnet/features.md
backbone   = DPCRN(input_dim=256, dvec_dim=192, channels=(2, 32, 32, 32, 64, 128))
c_backbone = EcapaTdnnExtractor(input_size=80, embedding_size=192, model_scale=8, ndim=1024, att_size=1536)

model = EncDecCondMaskBase(
    encoder=encoder,
    feats=feats,
    backbone=backbone,
    c_backbone=c_backbone,
    c_encoder=c_encoder,
    c_feats=c_feats,
    siamese_encoder=False,
    siamese_feats=False,
    jointed_trained=True,
    mask_type="complex",
)
```
