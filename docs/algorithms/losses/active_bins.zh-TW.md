# puresound.nnet.loss.active_bins

English version: [active_bins.md](active_bins.md)

只在乾淨 target 有語音的地方計算的逐 bin log-magnitude 誤差。靜音與只有
噪音的 bin 完全不貢獻。

## Class：`ActiveBinLogMagLoss`

### 計算內容

對兩個訊號做 Hann window STFT（`center=True`），以 dB 表示：

```
ref_db, enh_db = 10*log10(|STFT(ref)|^2 + 1e-10), 10*log10(|STFT(enh)|^2 + 1e-10)
peak           = max over (freq, frame) of ref_db, per row
active         = (ref_db >= peak - dynamic_range_db) & (ref_db > silence_floor_db)
diff           = enh_db - ref_db
w              = under_weight where diff < 0, else 1
per_row        = sum(|diff| * w * active) / max(sum(active), 1)
loss           = mean of per_row over rows that have at least one active bin
```

沒有任何 active bin 的 row（靜音 target）不列入平均。訊號短於 `n_fft` 時，
loss 為 `enh.sum() * 0.0`，一個保留 graph edge 的零。

### Constructor

```python
ActiveBinLogMagLoss(
    n_fft: int = 512,                 # 分析網格；預設值對齊 DPCRN recipe 的
    hop: int = 160,                   #   encoder，這裡的 bin 就是 mask 作用的 bin
    dynamic_range_db: float = 50.0,   # active = 與該 row 最大 bin 相差不超過此 dB
    under_weight: float = 1.0,        # 輸出低於 target 處的乘數
    silence_floor_db: float = -80.0,  # 低於此絕對位準的 bin 一律不算 active
)
```

`dynamic_range_db` 與 `under_weight` 必須為正（否則 `ValueError`）。

### 輸入

`forward(enh, ref) -> Tensor`，waveform `[N, T]`（單一 channel 軸 `[N, 1, T]`
會被 squeeze；兩者截到較短的長度）。
`required_inputs = ("enhanced", "target")`。

### Config usage

```yaml
loss_func:
  - type: ActiveBinLogMagLoss
    weighted: 0.5
    args:
      n_fft: 512
      hop: 160
      dynamic_range_db: 30.0
      under_weight: 1.0
```

noise-suppression recipe 在最後一個 curriculum 階段之上，以低 learning rate 做
短暫微調時加入它（`train_dpcrn_mamba_activebin_ft.yaml`），其他 loss 不變。

### 設計說明

- 為什麼只看 active bin。本 package 的波形與頻譜 loss，要不是把大部分梯度放在
  噪音與靜音 bin 上（MR-STFT、ASR feature），就是以能量加權而看不到逐 bin
  結構（SDR）。輸出可以在這些 loss 上全部進步，同時語音內部中等位準的諧波峰
  卻被削平。把 log-magnitude L1 限制在 target 的語音 bin 上，就是直接要求這個
  結構，也避免靜音區域又大又雜的 log 誤差蓋過它。
- `dynamic_range_db` 是相對於整個 row 最大的 bin，所以不論音量都以每句話為準
  選出「語音」。把範圍縮窄也會讓這一項集中：同樣的 `weighted` 分到更少的
  bin，每個 bin 的梯度就更大，因此改範圍不只是改涵蓋範圍。
- `under_weight > 1` 偏向懲罰低估（被削平的峰）。它的效果取決於範圍：範圍寬時
  會包含諧波之間的谷，在那裡懲罰低估等於叫模型把噪音留在谷裡。
- `silence_floor_db` 保護 target-absent row。完全靜音的 target 每個 bin 都落在
  log floor（−100 dB）且等於峰值，只靠相對判準會把每個 bin 都標成 active，
  該 row 的代價約為 100——比目標函數其他部分高兩個數量級，會把這類 row 推到
  數位零。−80 dB 遠低於正常位準 row 的任何語音 bin，又高於 log floor。
- 預期用法是在 curriculum 階梯之後做一次短的校準。在完整長度的階段裡，其他
  項的份量會壓過它。

測試：`test/nnet/test_active_bins.py` 固定住語音 bin 的範圍、不受音量影響、
對低估的加權，以及靜音 row 的處理。
