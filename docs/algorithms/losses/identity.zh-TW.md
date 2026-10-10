# puresound.nnet.loss.identity

English version: [identity.md](identity.md)

作用在逐幀 `IdentityHead`（[nnet.lobe.heads](../models/lobe/heads.zh-TW.md)）上的語者身分
contrastive loss：把 turn embedding 與一個 stop-gradient EMA teacher 的 turn
embedding 在整個 batch 內做對比。這個模組也放了
[`RelativeProximityLoss`](proximity.zh-TW.md) 共用的 turn pooling 輔助函式。

## Class: `IdentityContrastiveLoss`

在單一講者 turn 上做 InfoNCE。對 turn `k`，`e_k` 是 head 的逐幀 embedding 在該
turn 非重疊幀上的平均再做 L2 正規化；`ē_j` 是 EMA teacher 算出的同一個量。候選是
batch 內（所有 row）其他每個 eligible turn；正例 `P(k)` 是 `turn_speaker` 相同者：

```
L_id = mean_k  -log  Σ_{p ∈ P(k)} exp(cos(e_k, ē_p) / τ)
                     -----------------------------------
                     Σ_{j ≠ k}    exp(cos(e_k, ē_j) / τ)
```

平均只取至少有一個正例的 anchor。正例可以是同一 row 後面的 turn，也可以是其他
row 裡的同一位講者——經過不同的收音鏈渲染，也可能是旁人（bystander）角色。

### Constructor

```python
IdentityContrastiveLoss(
    temperature: float = 0.1,   # τ；必須 > 0
    momentum: float = 0.99,     # teacher 的 EMA momentum m；範圍 [0, 1)
    min_turn_frames: int = 1,   # turn 至少要有這麼多幀才 eligible
)
```

### 輸入

`required_inputs = ("identity_emb", "identity_head", "bottleneck", "batch")`，由
`puresound.system.base.invoke_loss` 對照 `puresound/system/siso.py` 的 loss
providers 解析：

| 輸入 | 來源 | shape |
| --- | --- | --- |
| `identity_emb` | `backbone.last_identity_emb`；需要 `backbone_args.identity_head.enabled: true` | `[B, T, D]`，每幀單位長度 |
| `identity_head` | `backbone.identity_head` 這個 module 本身（teacher 複製它） | — |
| `bottleneck` | `backbone.last_bottleneck_graph`，沿頻率 pooling、帶計算圖的 bottleneck；需要 `backbone_args.expose_bottleneck: true` | `[B, C, T]` |
| `batch` | 訓練 batch | dict |

Batch key 由 session-row 產生器（`puresound/task/session_rows.py`）產出：

| key | shape | 意義 |
| --- | --- | --- |
| `turn_id` | long `[B, T]` | 該幀所屬 turn 的 `1..K`，`0` = 無 turn 或重疊 |
| `turn_speaker` | long `[B, K]` | 全域講者 index，`-1` = padding |
| `turn_role` | long `[B, K]` | `1` 使用者、`2` 旁人、`0` padding；padding turn 不 eligible |
| `user_active`、`bystander_active` | float `[B, T]` | 可選；兩者皆為 1 的幀會被排除 |

head 缺席時丟出 `ValueError` 並指出對應的 config 開關。batch 沒有 `turn_id` 或
`turn_speaker`、eligible turn 少於兩個、或沒有任何講者出現兩次時，回傳帶計算圖的零
`identity_emb.sum() * 0`。

### Teacher

head 的 deep copy，在第一次呼叫時建立，之後每次 training 模式的呼叫依
`p_t ← m·p_t + (1 − m)·p_s` 更新（buffer 直接複製）。它在 `no_grad` 下對 detach 過的
bottleneck 執行，參數 `requires_grad = False`，並跟著 head 的 device 或 dtype 移動。
它放在一般 list 裡、不註冊成 submodule，所以永遠不會進 checkpoint；resume 後重新
從 student 複製一份。驗證時的呼叫（`loss.eval()`）不會更新它。

### Config 用法

沒有任何已出貨的 recipe 啟用這個 loss。

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

## Turn pooling 輔助函式

與 `RelativeProximityLoss` 共用，讓兩個 loss 在同一個網格上 pool turn。

```python
NO_TURN = 0
align_frames(*tensors) -> tuple                       # 把每個 tensor 截到 axis 1 最短者
align_turn_frames(values, batch, device)              # -> (values, turn_id, exclude) 或 None
pool_turn_means(values, turn_id, n_turns, exclude=None)  # [B,T,D] -> means [B,K,D], counts [B,K]
eligible_turns(counts, batch, device, min_frames=1)   # [B,K] bool：幀數足夠且 role 非 padding
```

`exclude` 是 `user_active AND bystander_active`。`turn_id` 超過 `K` 時
`pool_turn_means` 丟出 `ValueError`；空 turn 的 count 為 0、mean 為 0。

## 設計說明

- **正例跨收音鏈。** 一個 turn 的正例包含經另一條鏈渲染的同一位講者，而產生器也會把
  旁人渲染在使用者自己的距離上。若正例全是近場、負例全是遠場，一個 near/far 純量就能
  滿足目標，學不到身分。
- **Stop-gradient teacher。** 兩邊都可訓練時，兩個 embedding 會互相遷就而塌縮，loss
  下降卻沒有學到身分。
- **Teacher 不在 state dict 裡。** lazy 建立的 submodule 會在第一步之後改變
  checkpoint 佈局，讓它自己無法 resume。
- **幀單位長度、turn 再正規化。** cosine 就是單純的內積，沒有 turn 能靠放大 norm 取勝。
- **標籤網格與 head 網格。** 在 16 kHz 下兩者名義上都是 100 fps，但 labeler 的窗是
  400 samples、encoder 的是 512，所以標籤網格多一幀；一律截到共同前綴，與
  `VADHeadBCELoss` 相同。
- **float32 pooling、關閉 autocast。** bf16 無法精確表示長 turn 的幀數，會讓 turn
  平均的尺度出錯；`[M, M]` 相似度矩陣也以 float32 計算。
- **帶計算圖的零。** 每個 rank 都會碰到 head 的參數，DDP 不會看到未使用的參數。
