# puresound.nnet.loss.sdr

English version: [sdr.md](sdr.md)

時域 SDR 系列 loss。單一 class `SDRLoss` 以四個 flag 涵蓋 SI-SNR、SD-SDR、
一般 SDR、SA-SDR 與 t-SDR，並對 target-absent row（reference 為靜音）改用
另一個能量項評分，而不是去算沒有定義的比值。

## Class：`SDRLoss`

### 計算內容

每個 active row，在（可選的）去均值之後：

```
s_target    = <s1, s2> / (<s2, s2> + eps) * s2      if scaled          else s2
e_noise     = s1 - s2                               if scale_dependent else s1 - s_target
target_norm = <s_target, s_target>
noise_norm  = <e_noise, e_noise>   (+ tau * target_norm when sdr_max is set,
                                      tau = 10 ** (-sdr_max / 10))
loss        = -10 * log10(target_norm / (noise_norm + eps) + eps)
```

`<a, b>` 是 `sum(a * b, dim=-1)`。`source_aggregated=True` 時，先沿 source 軸
把能量加總再取比值。SI-SNR 的情形（`scaled=True, scale_dependent=False`）：

$$\mathcal{L} = -10 \log_{10} \frac{\|\alpha s\|^2}{\|\hat{s} - \alpha s\|^2}, \qquad \alpha = \frac{\langle \hat{s}, s \rangle}{\langle s, s \rangle}$$

其中 $\hat s$ = `s1`（enhanced）、$s$ = `s2`（reference）。

### Constructor

```python
SDRLoss(
    scaled: bool = True,                # 把 reference 投影到 estimate 上（scale-invariant）
    scale_dependent: bool = False,      # 殘差改對原始 reference 計算；只在 scaled=True 時有作用
    zero_mean: bool = True,             # 先去掉每個 row 的 DC
    source_aggregated: bool = False,    # SA-SDR；輸入必須是 [N, S, L]
    sdr_max: int = None,                # soft ceiling，單位 dB（t-SDR）
    eps: float = 1e-8,
    reduction: bool = True,             # bool：True -> scalar 平均，False -> 逐 row [rows, 1]
    threshold: Optional[float] = None,  # 丟掉 loss <= threshold 的 active row
    inactive_mode: str = "absolute",    # "absolute" | "mean" | "relative"，見下
)
```

- `scaled=False` 時 `scale_dependent` 沒有作用：此時 `s_target` 就是 `s2`，
  `e_noise` 兩種設定下都是 `s1 - s2`。
- `sdr_max` 讓 loss 的下界為 `-sdr_max`。因此 `threshold` 等於 `-sdr_max`
  時什麼都不會丟；`threshold` 設得更高才會丟掉已經很容易的 row。若所有
  active row 都會被丟掉，就跳過這個篩選。
- `source_aggregated=True` 會 assert 輸入是 3-D `[N, S, L]`；否則輸入必須是
  2-D `[N, L]`。
- 其他的 `inactive_mode` 值會 raise `ValueError`。

### 輸入

`forward(s1, s2, inactive_labels=None, batch=None) -> Tensor`

`required_inputs` 是 `("enhanced", "target", "inactive_labels")`，
`inactive_mode="relative"` 時再加上 `"batch"`。訓練 module 透過 `invoke_loss`
解析這些名稱（見 [system/base](../../architecture/system/base.zh-TW.md)）；
`inactive_labels` 是 `target.abs().amax(-1) == 0`，每個 row 一個 bool。

### Target-absent row：`inactive_sdr_loss`

reference 為靜音時比值沒有定義。只要 `inactive_labels` 標記了任何 row，
`forward` 就把 batch 拆開：active row 走上面的公式，inactive row 走
`inactive_sdr_loss(s1, s2, mode=inactive_mode)`，兩邊的逐 row 結果串接後再做
reduction（所以 `reduction=False` 時回傳順序是先 active row、再 inactive row，不是 batch
順序）。inactive row 是被評分，不是被排除。兩個訊號都會先去均值。

| `inactive_mode` | 每個 row 的值 |
| --- | --- |
| `absolute`（預設） | `10*log10(sum(s1^2) + 0.01*sum(s2^2) + 1e-8)` |
| `mean` | `10*log10(mean(s1^2) + 0.01*mean(s2^2) + 1e-8)` |
| `relative` | `10*log10(mean(s1^2) + 1e-10) - 10*log10(mean(noisy^2) + 1e-10)`，其中 `noisy = batch["noisy_speech"]` |

三者都是「越小越好」，都把輸出推向靜音。差別在獎勵的是什麼：

- `absolute` 的 floor 加在總和上，換算成平均功率後 floor 會隨 row 長度移動
  （差 `10*log10(L)`），而且輸入較小聲的 row 不必做任何壓制分數就比較低。
- `mean` 是同一個目標但用平均功率，floor 與長度無關。
- `relative` 評的是相對於該 row 自己的 mixture 實際達到的壓制量（dB），與
  音量和長度無關。batch 裡沒有 `noisy_speech` 時會 raise `ValueError`。

### 預設組合：`init_mode`

`SDRLoss.init_mode(loss_func="sisnr", reduction=True, threshold=None)` 依名稱
建出對應變體（`zero_mean=True`、`eps=1e-8`）。它是 Python 端的 constructor；
recipe 直接把 flag 傳給 `SDRLoss`。其他名稱會 raise `NameError`。

| 名稱 | `scaled` | `scale_dependent` | `source_aggregated` | `sdr_max` |
|---|---|---|---|---|
| `sisnr` | True | False | False | None |
| `sdsdr` | True | True | False | None |
| `sdr` | False | False | False | None |
| `tsdr` | False | False | False | 30 |
| `tsdr50` | False | False | False | 50 |
| `sasdr` | False | False | True | None |
| `sasisnr` | True | False | True | None |
| `satsdr` | False | False | True | 30 |

### Config usage

```yaml
loss_func:
  - type: SDRLoss
    weighted: 1.
    args:
      scaled: False
      scale_dependent: False
      zero_mean: True
      source_aggregated: False
      sdr_max: 50
      threshold: -50
```

### 設計說明

- 音量監督。`scaled=True` 時 loss 看不到輸出增益，輸出音量不受約束。
  出貨的 recipe 用 `scaled: False`（一般 SDR），讓音量也受到監督；
  `scaled` 與 `scale_dependent` 要當成一組、一起改。
- soft ceiling（`sdr_max`）避免已經接近完美的 row 在殘差趨近零時主導梯度。
- target-absent row 有自己的項，因為訓練資料裡有正確輸出就是靜音的 row，
  而對零 reference 取比值沒有有限值。

## Module-level helpers

無法從 `loss_func[].type` 取用。

- `si_snr(s1, s2, eps=1e-8, reduction=True)` —— 作為 metric 的 SI-SNR（正的
  dB 值，未取負號）。`puresound/metrics.py` 與評測工具使用。
- `inactive_sdr_loss(s1, s2, reduction=True, mode="absolute", noisy=None)` ——
  上述的 target-absent 項。
- `l2_norm(s1, s2)` —— `sum(s1 * s2, dim=-1, keepdim=True)`。
- `attenuation_ratio(s1, s2, mask, reduction=True)` —— 每個 row 在 `mask == 0`
  的樣本上，未處理訊號（`s2`）能量對輸出（`s1`）能量的 dB 比：輸出在應該
  靜音處被壓了多少。
