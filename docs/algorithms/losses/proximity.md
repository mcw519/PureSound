# puresound.nnet.loss.proximity

繁體中文版本：[proximity.zh-TW.md](proximity.zh-TW.md)

Relative-proximity ordering on the per-frame `ProximityHead`
([nnet.lobe.heads](../models/lobe/heads.md)), supervised by the rendered
distance of each turn rather than by speaker role. The readout has no absolute
scale: only the ordering of turn pairs and its stability across capture chains
are trained.

## Class: `RelativeProximityLoss`

For each turn `k`, `m_k` is the mean of the head's readout `r_t` (or
`tanh(r_t)` when `scale_free`) over the turn's non-overlap frames, pooled with
the [identity helpers](identity.md#turn-pooling-helpers). Within one row, an
unordered pair of eligible turns `(i, j)` is selected when both distances
`d_i, d_j` are finite and positive, `|d_j − d_i| ≥ min_distance_gap_m`, and,
with `pair_selection: cross_role`, the two turns have different roles. Its
signed gap is positive when the nearer turn reads higher:

```
g_ij    = (m_i − m_j) · sign(d_j − d_i)
L_order = mean_pairs  softplus(margin − g_ij)                     # default
        = mean_pairs  τ · softplus((margin − g_ij) / τ)           # scale_free: True
```

**Within-batch consistency.** Two rows sharing a `row_source_id ≥ 0` are the
same source material; when their `turn_chain` differs on at least one involved
turn, the matched pair gaps are compared one by one:

```
L_cons = mean_row-pairs  mean_valid-pairs |g^left_ij − g^right_ij|
L      = L_order + consistency_weight · L_cons
```

**Paired-view consistency.** `paired_consistency(proximity, second, batch, view)`
scores the same `|g^a − g^b|` between a primary row and its second capture-chain
rendering (`batch["paired_view"]`, one extra forward). It is called by
`puresound/system/paired_views.py`, not by `forward`, and returns
`(mean, n_views, n_turn_pairs)`; the dispatcher weights it by the loss weight
times `paired_weight` (= `consistency_weight`). A view whose `row_source_id` or
`turn_chain` provenance is missing or inconsistent raises `ValueError`.

### Constructor

```python
RelativeProximityLoss(
    margin: float = 1.0,               # ordering margin in head units; > 0
    consistency_weight: float = 1.0,   # weight of L_cons and of the paired view; >= 0
    min_turn_frames: int = 1,          # frames a turn needs to be eligible; >= 1
    min_distance_gap_m: float = 0.25,  # minimum distance difference for a pair; > 0
    scale_free: bool = False,          # tanh-bound the readout, use temperature τ
    temperature: float = 1.0,          # τ, used only when scale_free; > 0
    pair_selection: str = "cross_role",  # "cross_role" | "all"
)
```

### Inputs

`required_inputs = ("proximity", "batch")` and `paired_output = "proximity"`.
`proximity` is `backbone.last_proximity` `[B, T]`, present when
`backbone_args.proximity_head.enabled: true`; `None` raises `ValueError`.

| batch key | shape | meaning |
| --- | --- | --- |
| `turn_id` | long `[B, T]` | `1..K` per single-talker turn, `0` = none or overlap |
| `turn_role` | long `[B, K]` | `1` user, `2` bystander, `0` pad |
| `turn_distance` | float `[B, K]` | rendered RIR distance in metres, NaN = unknown or pad; same shape as `turn_role` |
| `turn_chain` | long `[B, K]` | the row's capture-chain draw id (consistency only) |
| `row_source_id` | long `[B]` | source identity, `-1` = unpaired (consistency only) |
| `user_active`, `bystander_active` | float `[B, T]` | optional; overlap frames are excluded |

A batch without `turn_id`, `turn_role` or `turn_distance`, or with no turns,
returns the graph-carrying zero `proximity.sum() * 0`. After each call
`last_stats` holds detached `ordering_pairs`, `ordering_correct`,
`ordering_loss`, `within_batch_pairs` and `consistency_loss` for logging.

### Config usage

From `egs/voice_isolate/config/train_dpcrn_curriculum_v1.yaml` (abridged):

```yaml
augmentation_session_rows:
  paired_view_prob: 0.50          # second chain view, consistency only
curriculum:
  tracks:
    - path: "loss:RelativeProximityLoss"   # weight ramps in with the session rows
      interp: linear
      points: [[4, 0.0], [12, 0.1]]
loss_func:
  - type: RelativeProximityLoss
    weighted: 0.1
    args:
      margin: 1.0
      consistency_weight: 1.0
      min_turn_frames: 1
      min_distance_gap_m: 0.25
      scale_free: False
      temperature: 1.0
      pair_selection: cross_role
model:
  lightning_module:
    module_args:
      paired_view_consistency: {enabled: True, max_rows: 1}
  backbone:
    backbone_args:
      proximity_head: {enabled: True, hidden: 64}
```

## Design notes

- **Ordering in head units, eligibility in metres.** Distances decide which
  pairs count and in which direction; the readout itself is never compared to
  a fixed value, because an absolute threshold on a readout does not carry
  across capture chains or checkpoints.
- **Distance, not role.** Either role may be the nearer one, and a missing
  distance never falls back to "the user is nearer".
- **Minimum gap.** Pairs closer than `min_distance_gap_m` have no reliable
  order and are left out.
- **Pair-by-pair consistency.** Matched contrasts are compared individually so
  opposite errors on different pairs cannot cancel.
- **Paired view scored separately.** The second view contributes only
  consistency, so no ordering example is counted twice.
- **`scale_free` is opt-in.** The default objective is the unbounded readout
  with a plain softplus hinge and does not change implicitly.
