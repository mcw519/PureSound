# puresound.nnet.lobe.heads

English version: [heads.md](heads.md)

從 backbone 的 bottleneck feature map `[N, C, F, T]` 讀出的輔助讀出（readout）。
它們不改變 enhanced 輸出：backbone 在 forward 之後把每個結果存在一個 attribute 上，
loss 或評估程式從那裡讀取。任何 bottleneck 為 `[N, C, F, T]` 的 backbone 都能接上；
`DPCRN` 讀的是 dual-path 區塊之後的 feature map。

| head | `backbone_args` key | 輸出 attribute | shape | 訓練它的 loss |
| --- | --- | --- | --- | --- |
| `VADHead` | `vad_head` | `last_vad_logits` | `[N, T]` | [`VADHeadBCELoss`](../../losses/vad.zh-TW.md) |
| `VADHead` | `background_vad_head` | `last_background_vad_logits` | `[N, T]` | [`BackgroundVADHeadBCELoss`](../../losses/vad.zh-TW.md) |
| `DistHead` | `dist_head` | `last_dist_preds` | `[N, 3]` | [`DistHeadRegressionLoss`](../../losses/dist.zh-TW.md) |
| `IdentityHead` | `identity_head` | `last_identity_emb` | `[N, T, dim]` | [`IdentityContrastiveLoss`](../../losses/identity.zh-TW.md) |
| `ProximityHead` | `proximity_head` | `last_proximity` | `[N, T]` | [`RelativeProximityLoss`](../../losses/proximity.zh-TW.md) |

## 設定

每個 head 都有一個 config model（`VADHeadConfig`、`DistHeadConfig`、
`IdentityHeadConfig`、`ProximityHeadConfig`）與一個
`from_config(config, *, enc_channels)` classmethod；區塊不存在或 `enabled` 為 false
時回傳 `None`。停用的 head 不增加任何參數與運算。config model 拒絕未知的 key，
拼錯的 key 會直接報錯，而不是默默套用預設值。

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      vad_head:
        enabled: True
        hidden: 64
        kernel_t: 5
        ema_taus_s: [0.05, 0.25, 1.0, 4.0]   # 可選；省略即為一般的 head
      background_vad_head: {enabled: True}
      dist_head: {enabled: True, hidden: 128}
      identity_head: {enabled: True, dim: 64, kernel_t: 5}
      proximity_head: {enabled: True, hidden: 64}
      expose_bottleneck: True                 # IdentityContrastiveLoss 需要
```

Checkpoint 的 key 跟著 backbone 存放 head 的 attribute 名稱（例如
`backbone.vad_head.*`），而不是這個 module 的路徑。

## Class: `VADHead`

逐幀的語音活性 logits，時間上是 causal 的。

```python
VADHead(
    enc_channels: int,                           # bottleneck 的 channel 數 C
    hidden: int,
    kernel_t: int,                               # causal conv 的窗長（幀）
    ema_taus_s: Optional[Sequence[float]] = None,  # EMA 時間常數（秒）
    frame_rate: float = 100.0,                   # bottleneck 每秒幀數
)
```

config key：`enabled`（False）、`hidden`（None = `enc_channels`）、`kernel_t`（5）、
`ema_taus_s`（None；非空、全為正值的 list）、`frame_rate`（100.0 = 16000 / hop 160）。

```
h = mean over F of x                                   # [N, C, T]
h = [h, ema_1(h), ..., ema_K(h)]                       # if ema_taus_s: [N, (K+1)C, T]
h = Linear(-> hidden)(h)
h = SiLU(Conv1d(hidden, hidden, kernel_t)(left_pad(h, kernel_t - 1)))
logit = Conv1d(hidden, 1, 1)(h)                        # [N, T]
```

每個 EMA 使用 `a = 1 - exp(-1 / (tau * frame_rate))`，從 `s_{-1} = 0` 起
`s_t = s_{t-1} + a (h_t - s_{t-1})`，再除以 `1 - (1 - a)^(t+1)`，因此從第一幀起就是
「目前為止所見幀」的平均。

串流：`initial_stream_state(batch_size=1, device="cpu", dtype=None)` 回傳
`(ema [N, K, C], conv_cache [N, hidden, kernel_t - 1], count [N, 1])`，
`step(x [N, C, F, 1], state) -> (logit [N, 1], new_state)` 一次一幀重現 `forward`。
串流版 DPCRN 匯出時把 `vad_head` 與 `background_vad_head` 發佈為逐幀輸出
`vad_logit` 與 `background_vad_logit`，狀態 port 為 `<name>_ema`、
`<name>_conv_cache`、`<name>_count`。

模型本身不會把輸出乘上這個 head。要以它做 gate 的使用端自行套 `sigmoid` 並使用自己
校準的門檻。

## Class: `DistHead`

utterance-level 的接近程度標籤回歸。

```python
DistHead(enc_channels: int, hidden: int = 128, n_out: int = 3)
# [N, C, F, T] -> mean over F and T -> Linear -> SiLU -> Linear -> [N, n_out]
```

`n_out=3` 時輸出為
`[foreground_drr / drr_scale, log10(foreground_distance), log10(nearest_interferer_distance)]`；
`drr_scale` 是 loss 的參數。僅限訓練：它對整段 utterance 池化，所以串流匯出不包含它。

## Class: `IdentityHead`

逐幀的語者身分 embedding，時間上是 causal 的。

```python
IdentityHead(enc_channels: int, dim: int = 64, kernel_t: int = 5)
```

```
h = mean over F of x                                  # x 已是 [N, C, T] 時略過
h = LayerNorm(DepthwiseConv1d(kernel_t)(left_pad(h, kernel_t - 1)))   # [N, T, C]
e = L2-normalise(Linear(C -> dim)(h))                 # [N, T, dim]
```

`IdentityContrastiveLoss` 把 embedding 在每個單一說話者 turn 內取平均，再與 head 的
stop-gradient EMA 副本所產生的 turn embedding 做對比。副本跑在帶計算圖的 bottleneck
上，所以 backbone 需要 `expose_bottleneck: true`。僅限訓練。

## Class: `ProximityHead`

逐幀的相對接近程度純量，時間上是 causal 的。

```python
ProximityHead(enc_channels: int, hidden: int = 64, kernel_t: int = 5)
```

主幹與 `IdentityHead` 相同（causal depthwise conv 與 LayerNorm），接著
`Linear(C -> hidden) -> SiLU -> Linear(hidden -> 1)`，得到 `[N, T]`。除了 `enabled`
之外唯一的 config key 是 `hidden`；`kernel_t` 固定為 5，兩個逐幀 head 共用同一個
時間尺度。僅限訓練。

`RelativeProximityLoss` 只監督差值：各 turn 依其渲染距離排序並要求 margin，且同一批
資料經另一條錄音鏈渲染的第二份版本上，這個順序也必須成立。

## 相關

[`multiframe.DeepFilterResidualHead`](multiframe.zh-TW.md) 也接在 `DPCRN` 上
（`df_head`），但它會改變輸出，且讀的是 decoder 而非 bottleneck。

## 設計說明

- `VADHead` 的一般窗長是 `kernel_t` 幀（預設 50 ms），而判斷「有沒有人在說話」需要
  大約一秒的語音。EMA 組讓 head 同時讀多個時間尺度，而快、慢平均的比值量的是
  envelope 調變深度——在逐幀 feature 分不出來時，這個線索能分開近講與遠講者。
  衰減率是固定的：可學的衰減率可能漂到單幀或常數，固定值讓時間尺度成為明確的超參數。
- EMA 遞迴在 float32 下計算並關閉 autocast：`tau = 4 s` 時步長 `a = 0.0025`，
  bf16 狀態會把這個增量完全吃掉。`torchaudio.functional.lfilter` 以 `clamp=False`
  呼叫，因為 bottleneck feature 不受 1 的界限。
- 去偏移消除了零初始化平均值的起始暫態，串流時帶著 `(state, count)` 就有相同語意。
- `DistHead` 促使 bottleneck 編碼物理上的接近程度線索（DRR、聲源距離），而不是訓練
  資料的錄音鏈特徵。
- `IdentityHead` 把輸出正規化，對比用的 cosine 就是內積，沒有 turn 能靠放大 norm
  取勝。它不需要逐幀可區分，只需要逐 turn 可區分。
- `ProximityHead` 沒有絕對尺度，因為對一個會隨錄音鏈與 checkpoint 漂移的讀出設固定
  門檻是站不住的；只有同一段錄音內的差值有意義。
