# puresound.nnet.loss.vad

English version: [vad.md](vad.md)

frame 活動度的監督。`VADActivityLoss` 拿 enhanced waveform 自己的 frame 能量
對活動度 target 評分；`VADHeadBCELoss` 與 `BackgroundVADHeadBCELoss` 訓練
backbone 上明確的 head；`F1_loss` 是作用在機率上的 soft F1。

## Class：`VADActivityLoss`

對 enhanced waveform 的 frame 能量做可微分的 BCE。分 frame 在時域進行
（沿樣本 `unfold`，每個 frame 取均方）；完全沒有 STFT。

### 計算內容

```
P_enh[t] = mean(enh[t*hop : t*hop + frame_length]^2)          # rows shorter than one frame are padded
logit[t] = (10*log10(P_enh[t] / P_ref) - activity_threshold_db) * logit_scale
loss     = BCEWithLogits(logit, target, weight = FN weight on target frames, FP weight on the rest)
```

`target` 與 `P_ref` 取決於是否有提供 `vad_target`：

| | `target` | `P_ref` |
| --- | --- | --- |
| 沒有 `vad_target` | 乾淨 reference 的 frame 功率與它自己最大 frame 相差在 `activity_threshold_db` 以內處為 `1`；reference 全靜音時全為零 | reference 最大的 frame 功率（reference 全靜音時為 `1`） |
| 有 `vad_target` | 該標籤，截斷或補零到 frame 數 | `1`（0 dBFS），threshold 因此是絕對的 dBFS 位準 |

### Constructor

```python
VADActivityLoss(
    frame_length: int = 400,              # 每個 frame 的樣本數
    hop_length: int = 160,                # frame 之間的樣本數
    activity_threshold_db: float = -40.0, # soft 活動度的判斷點
    logit_scale: float = 0.25,            # soft 判斷的銳利度（每 dB）
    false_positive_weight: float = 1.0,   # inactive frame 的 BCE 權重
    false_negative_weight: float = 1.0,   # active frame 的 BCE 權重
    require_vad_target: bool = False,     # 不從 reference 推導 target，改為 raise
    eps: float = 1e-8,                    # 功率 floor
)
```

### 輸入

`forward(enh, ref, vad_target=None) -> Tensor`。waveform 可為 `[B, T]`、
`[B, C, T]`（對 channel 取平均）或 `[T]`；截到較短的長度。
`required_inputs = ("enhanced", "target", "vad_target")`。`vad_target` 即
`batch["vad_target"]`，dataset config 有 `vad_label` 區塊時才存在
（`backend: energy` 或 `silero`，見 `puresound/audio/vad.py`）；否則為 `None`。

### Config usage

```yaml
loss_func:
  - type: VADActivityLoss
    weighted: 0.1
    args:
      frame_length: 800
      hop_length: 320
      activity_threshold_db: -40
      false_positive_weight: 2.0
      false_negative_weight: 1.0
```

### 設計說明

- 這個 loss 直接作用在輸出 waveform 上，所以不需要替模型加任何 module，就能
  懲罰應該靜音的 frame（背景說話者、噪音）裡的輸出能量，以及應該 active 的
  frame 裡缺少的能量。
- 沒有外部標籤時，target 相對於每個 reference 自己的峰值，會隨 clip 的音量
  調整。有外部標籤時，輸出改為對絕對的 0 dBFS 錨點比較：threshold 因此不會
  隨施加在 reference 上的音量或削波擾動而移動；且模型輸出被 clamp 在
  `[-1, 1]`，其位準一定不超過 0 dBFS。
- `false_positive_weight > 1` 偏向靜音而非殘留的活動。

## Class：`VADHeadBCELoss`

對 backbone VAD head 的 frame logit 與 frame 標籤做 BCE-with-logits。與讀取
輸出 waveform 的 `VADActivityLoss` 不同，這個 loss 監督的是一個明確的 head
（[`VADHead`](../models/lobe/heads.zh-TW.md)）。

```python
VADHeadBCELoss(
    false_positive_weight: float = 1.0,   # 標籤為 inactive 的 frame 的 BCE 權重
    false_negative_weight: float = 1.0,   # 標籤為 active 的 frame 的 BCE 權重
    balance_per_batch: bool = False,      # 讓兩個類別的 BCE 總量相等
)
```

`forward(vad_logits, vad_target) -> Tensor`。
`required_inputs = ("vad_logits", "vad_target")`：backbone 的
`last_vad_logits` `[N, T]` 與 `batch["vad_target"]`。frame 數截到較短的一方
對齊（backbone 的分 frame 與標籤的分 frame 可能差一個 frame）。缺少 head 或
標籤時 raise `ValueError`，並指出要開啟的設定（`vad_head`、`vad_label`）。

開啟 `balance_per_batch` 時，在靜態權重之外，正類 frame 再乘上
`n / (2 * n_pos)`、負類 frame 乘上 `n / (2 * n_neg)`，讓不平衡的 batch 無法
獎勵全語音或全靜音的常數輸出。只含單一類別的 batch 維持靜態權重。部署時
head 的 threshold 另外校準。

```yaml
model:
  backbone:
    backbone_args:
      vad_head: {enabled: True, hidden: 64, kernel_t: 5}
loss_func:
  - type: VADHeadBCELoss
    weighted: 0.1
    args: {false_positive_weight: 1.0, false_negative_weight: 1.0, balance_per_batch: False}
```

## Class：`BackgroundVADHeadBCELoss`

與 `VADHeadBCELoss` 相同的 BCE（繼承 `forward` 與 constructor），但對準背景
語音的 head：問的是「現在有沒有非 target 的語音」，而不是「辨識器現在該不該
聽」。它讓 bottleneck 對背景說話者有明確的表徵，而不必讓背景語音成為輸出的
一部分。

`required_inputs = ("background_vad_logits", "background_vad_target")`：
backbone 的 `last_background_vad_logits`（DPCRN 以
`backbone_args.background_vad_head` 建立時才會產生）與
`batch["background_vad_target"]`。這個宣告取代父類別的宣告，所以子類別永遠
不會讀到前景 head。錯誤訊息會指出 `background_vad_head` 與
`background_vad_target`。

當 batch 裡沒有任何 row 含背景語音時，collate 不會產生
`background_vad_target`。此時訓練 module 的 input provider 會提供一個與 logit
同形狀的全零 target，loss 便對一個合法的「沒有背景語音」訊號跑一般的 BCE，
而不是因 `None` 而失敗。

## Class：`F1_loss`

作用在機率上的 soft F1，參考
[asteroid 的 `soft_f1`](https://github.com/asteroid-team/asteroid/blob/fc0967a2eaf42f9446b17f7d039598deffd46f91/asteroid/losses/soft_f1.py)。

```python
F1_loss(eps: float = 1e-10)
```

`forward(estimates, targets) -> Tensor`：對所有元素加總 soft TP/FP/FN，
`precision = tp / (tp + fp)`、`recall = tp / (tp + fn)`，回傳 `1 - F1`。
沒有做 threshold，因此可微分。它沒有宣告 `required_inputs`，經由訓練 module
呼叫時會拿到 `("enhanced", "target")`；它是給傳入機率的呼叫端使用的。
