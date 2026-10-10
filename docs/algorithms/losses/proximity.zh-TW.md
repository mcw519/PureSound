# puresound.nnet.loss.proximity

English version: [proximity.md](proximity.md)

作用在逐幀 `ProximityHead`（[nnet.lobe.heads](../models/lobe/heads.zh-TW.md)）上的相對
距離排序 loss，監督訊號是每個 turn 渲染時的距離，而不是講者角色。讀數沒有絕對尺度：
只訓練 turn pair 的排序，以及這個排序在收音鏈變換下的穩定性。

## Class: `RelativeProximityLoss`

對每個 turn `k`，`m_k` 是 head 讀數 `r_t`（`scale_free` 時為 `tanh(r_t)`）在該
turn 非重疊幀上的平均，以 [identity 輔助函式](identity.zh-TW.md)
pooling。在同一個 row 內，兩個 eligible turn 組成的無序 pair `(i, j)` 被選中的條件：
兩個距離 `d_i, d_j` 皆為有限正值、`|d_j − d_i| ≥ min_distance_gap_m`，且在
`pair_selection: cross_role` 時兩個 turn 角色不同。帶號差距在較近的 turn 讀數較高時
為正：

```
g_ij    = (m_i − m_j) · sign(d_j − d_i)
L_order = mean_pairs  softplus(margin − g_ij)                     # 預設
        = mean_pairs  τ · softplus((margin − g_ij) / τ)           # scale_free: True
```

**Batch 內一致性。** 兩個 `row_source_id ≥ 0` 相同的 row 是同一份來源素材；若它們的
`turn_chain` 在至少一個相關 turn 上不同，就把對應的 pair 差距逐一比較：

```
L_cons = mean_row-pairs  mean_valid-pairs |g^left_ij − g^right_ij|
L      = L_order + consistency_weight · L_cons
```

**配對視角一致性。** `paired_consistency(proximity, second, batch, view)` 在主 row
與它的第二條收音鏈渲染（`batch["paired_view"]`，多一次 forward）之間計算同樣的
`|g^a − g^b|`。它由 `puresound/system/paired_views.py` 呼叫，而不是 `forward`，
回傳 `(mean, n_views, n_turn_pairs)`；dispatcher 以 loss 權重乘上 `paired_weight`
（= `consistency_weight`）加權。視角的 `row_source_id` 或 `turn_chain` 來源資訊缺漏或
不一致時丟出 `ValueError`。

### Constructor

```python
RelativeProximityLoss(
    margin: float = 1.0,               # 排序 margin，head 單位；> 0
    consistency_weight: float = 1.0,   # L_cons 與配對視角的權重；>= 0
    min_turn_frames: int = 1,          # turn 要 eligible 所需的幀數；>= 1
    min_distance_gap_m: float = 0.25,  # 一個 pair 的最小距離差；> 0
    scale_free: bool = False,          # 用 tanh 限制讀數範圍，並使用溫度 τ
    temperature: float = 1.0,          # τ，只在 scale_free 時使用；> 0
    pair_selection: str = "cross_role",  # "cross_role" | "all"
)
```

### 輸入

`required_inputs = ("proximity", "batch")`，`paired_output = "proximity"`。
`proximity` 是 `backbone.last_proximity` `[B, T]`，在
`backbone_args.proximity_head.enabled: true` 時存在；為 `None` 時丟出 `ValueError`。

| batch key | shape | 意義 |
| --- | --- | --- |
| `turn_id` | long `[B, T]` | 每個單一講者 turn 的 `1..K`，`0` = 無或重疊 |
| `turn_role` | long `[B, K]` | `1` 使用者、`2` 旁人、`0` padding |
| `turn_distance` | float `[B, K]` | 渲染用的 RIR 距離（公尺），NaN = 未知或 padding；shape 與 `turn_role` 相同 |
| `turn_chain` | long `[B, K]` | 該 row 的收音鏈抽樣 id（僅一致性使用） |
| `row_source_id` | long `[B]` | 來源身分，`-1` = 未配對（僅一致性使用） |
| `user_active`、`bystander_active` | float `[B, T]` | 可選；重疊幀會被排除 |

batch 沒有 `turn_id`、`turn_role` 或 `turn_distance`，或沒有任何 turn 時，回傳帶計算
圖的零 `proximity.sum() * 0`。每次呼叫後 `last_stats` 保存 detach 過的
`ordering_pairs`、`ordering_correct`、`ordering_loss`、`within_batch_pairs` 與
`consistency_loss` 供 log 使用。

### Config 用法

節錄自 `egs/voice_isolate/config/train_dpcrn_curriculum_v1.yaml`：

```yaml
augmentation_session_rows:
  paired_view_prob: 0.50          # 第二條鏈的視角，只用於一致性
curriculum:
  tracks:
    - path: "loss:RelativeProximityLoss"   # 權重隨 session row 一起爬升
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

## 設計說明

- **排序用 head 單位、eligibility 用公尺。** 距離決定哪些 pair 算數以及方向；讀數本身
  從不和固定值比較，因為讀數上的絕對門檻無法跨收音鏈或跨 checkpoint 成立。
- **看距離，不看角色。** 兩種角色都可能是較近的一方，距離缺漏時也不會退回「使用者比較
  近」的假設。
- **最小差距。** 距離差小於 `min_distance_gap_m` 的 pair 沒有可靠的順序，不納入。
- **逐 pair 一致性。** 對應的差距逐一比較，不同 pair 上方向相反的誤差不會互相抵消。
- **配對視角分開計分。** 第二個視角只貢獻一致性，沒有排序樣本被算兩次。
- **`scale_free` 需明確開啟。** 預設目標是無界讀數加上一般的 softplus hinge，不會被
  隱性改變。
