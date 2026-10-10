# puresound.nnet.loss.identity

繁體中文版本：[identity.zh-TW.md](identity.zh-TW.md)

Speaker-identity contrastive loss on the per-frame `IdentityHead`
([nnet.lobe.heads](../models/lobe/heads.md)): turn embeddings are contrasted
against a stop-gradient EMA teacher's turn embeddings across the whole batch.
The module also holds the turn-pooling helpers that
[`RelativeProximityLoss`](proximity.md) reuses.

## Class: `IdentityContrastiveLoss`

InfoNCE over single-talker turns. For turn `k`, `e_k` is the L2-normalised mean
of the head's per-frame embedding over the turn's non-overlap frames; `ē_j` is
the same quantity from the EMA teacher. Candidates are every other eligible turn
in the batch (all rows); positives `P(k)` are those with the same
`turn_speaker`:

```
L_id = mean_k  -log  Σ_{p ∈ P(k)} exp(cos(e_k, ē_p) / τ)
                     -----------------------------------
                     Σ_{j ≠ k}    exp(cos(e_k, ē_j) / τ)
```

The mean runs over anchors that have at least one positive. A positive can be a
later turn of the same row or the same speaker in another row, rendered through
a different capture chain and possibly in the bystander role.

### Constructor

```python
IdentityContrastiveLoss(
    temperature: float = 0.1,   # τ; must be > 0
    momentum: float = 0.99,     # EMA momentum m of the teacher; in [0, 1)
    min_turn_frames: int = 1,   # a turn needs at least this many frames to be eligible
)
```

### Inputs

`required_inputs = ("identity_emb", "identity_head", "bottleneck", "batch")`,
resolved by `puresound.system.base.invoke_loss` against the loss providers of
`puresound/system/siso.py`:

| input | source | shape |
| --- | --- | --- |
| `identity_emb` | `backbone.last_identity_emb`; needs `backbone_args.identity_head.enabled: true` | `[B, T, D]`, unit-norm per frame |
| `identity_head` | the `backbone.identity_head` module itself (the teacher copies it) | — |
| `bottleneck` | `backbone.last_bottleneck_graph`, the frequency-pooled bottleneck with its graph; needs `backbone_args.expose_bottleneck: true` | `[B, C, T]` |
| `batch` | the training batch | dict |

Batch keys, emitted by the session-row generator (`puresound/task/session_rows.py`):

| key | shape | meaning |
| --- | --- | --- |
| `turn_id` | long `[B, T]` | `1..K` for the turn a frame belongs to, `0` = no turn or overlap |
| `turn_speaker` | long `[B, K]` | global speaker index, `-1` = pad |
| `turn_role` | long `[B, K]` | `1` user, `2` bystander, `0` pad; pad turns are ineligible |
| `user_active`, `bystander_active` | float `[B, T]` | optional; frames where both are 1 are excluded |

Missing heads raise `ValueError` naming the config switch. A batch without
`turn_id` or `turn_speaker`, with fewer than two eligible turns, or where no
speaker appears twice returns the graph-carrying zero `identity_emb.sum() * 0`.

### The teacher

A deep copy of the head, created on the first call and advanced on every
training-mode call as `p_t ← m·p_t + (1 − m)·p_s` (buffers are copied). It
runs under `no_grad` on the detached bottleneck, its parameters have
`requires_grad = False`, and it follows the head across device or dtype moves.
It is held in a plain list, not registered as a submodule, so it never enters a
checkpoint; after a resume it restarts as a copy of the student. Validation
calls (`loss.eval()`) do not advance it.

### Config usage

No shipped recipe enables this loss.

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      identity_head: {enabled: True, dim: 64, kernel_t: 5}
      expose_bottleneck: True
loss_func:
  - type: IdentityContrastiveLoss
    weighted: 0.1
    args: {temperature: 0.1, momentum: 0.99, min_turn_frames: 1}
```

## Turn-pooling helpers

Shared with `RelativeProximityLoss`, so both losses pool turns on one grid.

```python
NO_TURN = 0
align_frames(*tensors) -> tuple                       # truncate every tensor to the shortest axis 1
align_turn_frames(values, batch, device)              # -> (values, turn_id, exclude) or None
pool_turn_means(values, turn_id, n_turns, exclude=None)  # [B,T,D] -> means [B,K,D], counts [B,K]
eligible_turns(counts, batch, device, min_frames=1)   # [B,K] bool: enough frames and non-pad role
```

`exclude` is `user_active AND bystander_active`. `pool_turn_means` raises
`ValueError` when `turn_id` exceeds `K`; an empty turn has count 0 and mean 0.

## Design notes

- **Positives cross capture chains.** A turn's positives include the same
  speaker rendered through another chain, and the generator also renders
  bystanders at the user's own distance. With positives all near and negatives
  all far, a near/far scalar would satisfy the objective and no identity would
  be learned.
- **Stop-gradient teacher.** With both sides trainable the two embeddings
  co-adapt onto each other and the loss falls without learning identity.
- **Teacher outside the state dict.** A lazily created submodule would change
  the checkpoint layout after the first step and break its own resume.
- **Unit-norm frames, re-normalised turns.** Cosine is a plain dot product and
  no turn can win by growing its norm.
- **Label grid versus head grid.** Both are nominally 100 fps at 16 kHz, but
  the labeler's window is 400 samples and the encoder's 512, so the label grid
  runs one frame longer; everything truncates to the common prefix, as
  `VADHeadBCELoss` does.
- **float32 pooling with autocast off.** bf16 cannot represent a long turn's
  frame count exactly, which would mis-scale the turn mean; the `[M, M]`
  similarity matrix is also computed in float32.
- **Graph-carrying zero.** Every rank reaches the head's parameters, so DDP
  never sees an unused parameter.
