# puresound.metrics

English version: [metrics.md](metrics.md)

給語音增強與 source separation 用的音訊品質評估 metrics。

## Class: `Metrics`

一組用於客觀音訊品質評估的 static methods。雖然有些 method 自己的型別標注
寫的是 `np.array`，但每個 method 最終都會經過 `check_shape`，而它會呼叫
`.detach()`——所以實務上**每個輸入都必須是 `torch.Tensor`**，不能是單純的
NumPy array。

### `check_shape(clean, enhanced, retun_as_tensor=False)`

```python
check_shape(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    retun_as_tensor: bool = False,
) -> Tuple[np.ndarray, np.ndarray]   # retun_as_tensor=True 時是 (Tensor, Tensor)
```

下面其他每個 method 內部都會用到它——所以它的行為會影響到全部的 metric：

- 只要哪個 tensor 的第一維不是 `1`，就只保留 channel `0`
  （`clean[0, ...]`）——對真正的多聲道輸入來說，這會靜默地丟掉第一個
  channel 以外的所有 channel。
- Squeeze 成 1-D。
- 從開頭對齊長度：把兩者之中**比較長**的那個裁到跟比較短的一樣長（不做
  cross-correlation 對齊）。
- 轉成 NumPy，接著**各自獨立地對兩個訊號做 peak-normalize**——各自除以
  自己的 `abs().max()`（`clean = clean / abs(clean).max()`，`enhanced`
  也是一樣）——兩個訊號並*不是*用同一個共用係數縮放的。這代表下面每個
  metric 算的都是各自獨立做過 peak-normalize 之後的訊號，而不是原始的
  絕對音量——`noise_reduction` 也不例外，所以它算出來的其實是一個
  normalize 過的功率比，而不是字面上「處理前後」的能量比較。
- `retun_as_tensor=True`（這就是真正的參數名稱，不是打錯字、也不需要
  修正）會在回傳前轉回 `torch.Tensor`，而不是留著 NumPy array。

全零的輸入在這裡會除以零（`abs().max() == 0`），可能會讓下游某個 metric
冒出 NaN——例如 `f1_score` 碰到一整段完全沒有正樣本 frame 的標籤 tensor。

### `pesq_wb(clean, enhanced) -> float`

Wide-band PESQ。取樣率**寫死在函式內部是 16000**——沒有 `sr` 參數可以
覆寫。分數範圍：-0.5 到 4.5（越高越好）。

### `pesq_nb(clean, enhanced) -> float`

Narrow-band PESQ。取樣率**寫死是 8000**——這裡同樣沒有 `sr` 參數。分數
範圍：1.0 到 4.5。

### `stoi(clean, enhanced, sr=16000) -> float`

Short-Time Objective Intelligibility（STOI）。分數範圍：0 到 1（越高越
好）。跟上面兩個 PESQ wrapper 不同，這裡的 `sr` **是**一個真正的參數。

### `estoi(clean, enhanced, sr=16000) -> float`

Extended STOI（`stoi(..., extended=True)`），更適合非常低 SNR 的情境。
分數範圍：0 到 1。

### `bss_sdr(clean, enhanced) -> float`

BSS Eval 的 Signal-to-Distortion Ratio，透過
`mir_eval.separation.bss_eval_sources(clean, enhanced, False)[0][0]`
算出（`compute_permutation=False`，假設只有單一個 source）。回傳 SDR，
單位 dB。

### `sisnr(clean, enhanced) -> float`

Scale-Invariant Signal-to-Noise Ratio，實際上是算
`si_snr(enhanced, clean)`（`puresound.nnet.loss.sdr.si_snr`）。數值越高
越好，單位 dB。

### `sisnr_imp(clean, enhanced, noisy) -> float`

相對於 noisy baseline 的 SI-SNR 改善量：

```
SI-SNRi = SI-SNR(enhanced, clean) - SI-SNR(noisy, clean)
```

`clean` 分別跟 `enhanced`、跟 `noisy` 各做了一次獨立的 `check_shape`
呼叫——當這三個訊號長度都一樣時（通常都是如此）沒有影響，但如果你餵進去
的長度不一致，這點值得留意。

### `dnsmos_p835(clean, enhanced, sr=16000, personalized=False) -> Dict[str, float]`

Non-intrusive 的 DNSMOS P.835 分數，透過
`torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore` 算出。
**`clean` 雖然是參數之一，但馬上就被丟棄**（`del clean`）——留著它只是為了
讓呼叫方式跟其他 metric 一致；DNSMOS 是 reference-free 的，只會對
`enhanced` 打分數。

`enhanced` 會先經過內部的 `_mono_audio_tensor` 輔助函式處理：detach、
搬到 CPU、轉成 float、反覆把最前面 size 為 1 的 batch 維度 squeeze 掉，
如果還是多聲道就只留 channel 0，最後 clamp 到 `[-1, 1]`。注意這裡**不會**
經過 `check_shape`，所以不會像其他 metric 那樣做 peak-normalize。

回傳的是 dict，**不是 tuple**：

```python
{"dnsmos_p808": ..., "dnsmos_sig": ..., "dnsmos_bak": ..., "dnsmos_ovr": ...}
```

底層的 `torchmetrics` metric instance 會依 `(sr, personalized)` 這組組合
在 module 層級快取起來，所以用同樣設定重複呼叫不會重新建立 instance。
如果沒裝 `torchmetrics` 的 audio 相關依賴，會丟出 `ModuleNotFoundError`，
並附上安裝提示（`uv sync`，或手動裝 `librosa` + `onnxruntime` +
`requests`）。

### `f1_score(y_true, y_pred) -> Dict[str, float]`

二元分類 metrics（例如 frame 層級的 VAD 標籤）——注意這裡的參數名稱是
`y_true`/`y_pred`，不是像上面音訊 metric 那樣的 `clean`/`enhanced`。
兩者一樣會經過 `check_shape`（所以一樣適用相同的 shape/對齊/peak-normalize
規則——對 0/1 標籤來說，只要每個 tensor 裡至少有一個正樣本 frame，
normalize 就是個 no-op）。

回傳的是 dict，**不是 tuple**：

```python
{"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}
```

### `noise_reduction(noisy, enhanced) -> Tensor`

注意第一個參數是 `noisy`，不是 `clean`。算的是（經過 `check_shape`
peak-normalize 過的）enhanced 跟 noisy 訊號之間的功率比，單位 dB：

```
10 * log10(sum(enhanced ** 2) / sum(noisy ** 2))
```

## Example

```python
from puresound.metrics import Metrics

sisnr_val = Metrics.sisnr(clean_wav, enhanced_wav)
pesq_val  = Metrics.pesq_wb(clean_wav, enhanced_wav)
stoi_val  = Metrics.stoi(clean_wav, enhanced_wav, sr=16000)
dnsmos    = Metrics.dnsmos_p835(clean_wav, enhanced_wav)  # clean_wav 會被忽略
```
