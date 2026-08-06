# puresound.nnet.lobe.heads

English version: `heads.md`

讀取 backbone bottleneck feature 的輔助預測 head。這些是
backbone-agnostic 的:任何 bottleneck 是 `[N, C, F, T]` feature map 的
model 都能接上它們。Checkpoint 的 key 綁定的是 backbone 把該 head
存放的 attribute 名稱(例如 `backbone.vad_head.*`),而不是這個 module
的路徑,所以搬動或重複使用某個 head 都不會讓 checkpoint 失效。

## Class: `VADHead`

從 bottleneck 產生 frame-level 語音活性(speech-activity)logits;
**在時間軸上是 causal 的**(左側 padding 的 depthwise Conv1d),所以
streaming 只需要 `kernel_t - 1` 個 frame 的狀態。

```python
VADHead(enc_channels: int, hidden: int, kernel_t: int)
# forward: [N, C, F, T] -> mean-pool F -> Linear -> causal Conv1d -> [N, T] logits
```

Backbone 會把結果放在 `backbone.last_vad_logits`;搭配
[`VADHeadBCELoss`](../loss/vad.md) 訓練,inference 時作為對 enhanced
輸出的乘法式 gate。

## Class: `DistHead`

從 bottleneck 做 utterance-level 的距離/DRR 回歸。這是一個輔助的
multi-task 壓力,促使 bottleneck 編碼真實的物理接近程度線索
(DRR / source distance),而不是訓練用近場資料的錄音鏈特徵。

```python
DistHead(enc_channels: int, hidden: int = 128, n_out: int = 3)
# forward: [N, C, F, T] -> global pool -> MLP -> [N, 3]
# outputs: [fg_drr_db / drr_scale, log10(fg_dist_m), log10(nearest_itf_dist_m)]
```

Backbone 會把結果放在 `backbone.last_dist_preds`;搭配
[`DistHeadRegressionLoss`](../loss/dist.md) 訓練。僅限訓練使用:
inference 完全不會讀取它,streaming 匯出也不受影響。

## Attaching to a backbone (DPCRN example)

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      vad_head:  {enabled: True, hidden: 128, kernel_t: 5}
      dist_head: {enabled: True, hidden: 128}
```

兩者預設都是 disabled;設定檔沒有這些 key 時,建出來的 model 完全
byte-identical。
