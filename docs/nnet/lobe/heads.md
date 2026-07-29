# puresound.nnet.lobe.heads

Auxiliary prediction heads that read a backbone's bottleneck features. These are
backbone-agnostic: any model whose bottleneck is a `[N, C, F, T]` feature map
can attach them. Checkpoint keys bind to the attribute name the backbone stores
the head under (e.g. `backbone.vad_head.*`), not this module's path, so
relocating or reusing a head never invalidates a checkpoint.

## Class: `VADHead`

Frame-level speech-activity logits from the bottleneck; **causal in time**
(left-padded depthwise Conv1d), so streaming needs only `kernel_t - 1` frames of
state.

```python
VADHead(enc_channels: int, hidden: int, kernel_t: int)
# forward: [N, C, F, T] -> mean-pool F -> Linear -> causal Conv1d -> [N, T] logits
```

Backbones expose the result as `backbone.last_vad_logits`; trained with
[`VADHeadBCELoss`](../loss/vad.md), applied at inference as a multiplicative
gate on the enhanced output.

## Class: `DistHead`

Utterance-level distance/DRR regression from the bottleneck. Auxiliary
multi-task pressure that makes the bottleneck encode physical proximity cues
(DRR / source distance) rather than the capture-chain signature of the training
near-field rows.

```python
DistHead(enc_channels: int, hidden: int = 128, n_out: int = 3)
# forward: [N, C, F, T] -> global pool -> MLP -> [N, 3]
# outputs: [fg_drr_db / drr_scale, log10(fg_dist_m), log10(nearest_itf_dist_m)]
```

Backbones expose the result as `backbone.last_dist_preds`; trained with
[`DistHeadRegressionLoss`](../loss/dist.md). Training-only: inference never
reads it and the streaming export is untouched.

## Attaching to a backbone (DPCRN example)

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      vad_head:  {enabled: True, hidden: 128, kernel_t: 5}
      dist_head: {enabled: True, hidden: 128}
```

Both default to disabled; configs without the keys build byte-identical models.
