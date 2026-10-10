# puresound.audio.noise

English version: [noise.md](noise.md)

兩個加性混合的基本元件。`add_bg_noise` 調整噪音床的大小，讓混音達到指定的
SNR；同一個運算子也用來以 SIR 混入干擾說話者。`add_bg_white_noise` 以指定 SNR
加上高斯噪音。兩者都接受比值的**串列**並回傳**串列**，每個比值一個結果。

[`AudioEffectAugmentor`](augmentation.zh-TW.md) 在外面包了噪音池與重新取樣；
task dataset（`puresound/task/ns.py`、`voice_isolation.py`）與
`puresound.evaluation.tools.mix_paired_set` 在第二個訊號已在手上時直接呼叫
`add_bg_noise`。

## `add_bg_noise(wav, noise, snr_list) -> (noisy_list, noise_list)`

輸入 `wav: [1, L]`（或 `[L]`）與一串噪音 tensor：

1. 每個噪音 tensor 取第一個聲道、做 RMS 正規化，整串沿時間接成一條噪音床，
   再做一次 RMS 正規化。傳兩段 clip 就得到由兩段首尾相接的一條床。
2. 把床對齊到 `L`：較長時隨機裁切（起點來自 `torch.randint`），較短或等長時
   重複鋪滿再裁切。
3. 對每個 `snr_db`：

   ```
   g        = rms(wav) / 10^(snr_db / 20)
   noisy    = wav + g · bed
   noise    = g · bed
   ```

回傳 `(noisy_list, noise_list)`，長度都是 `len(snr_list)`。

這個比值是整段 clip 的能量比：`rms(wav)` 包含 `wav` 中的停頓，床在全長上是單位
RMS。它不是主動語音位準（ITU-T P.56）；靜音很長的 clip，其語音有效 SNR 會低於
名目值。當 `wav` 是前景說話者、`noise` 是干擾說話者時，`snr_db` 就是 SIR。

## `add_bg_white_noise(wav, snr_list) -> (noisy_list, noise_list)`

對每個 `snr_db`，由 torch generator 抽形狀 `[1, L]` 的 `n ~ N(0, σ²)`，
`σ = rms(wav) / 10^(snr_db/20)`，回傳 `wav + n` 與 `n`。只支援單聲道（`σ` 會被
轉成 Python float）。沒有噪音池、沒有檔案 I/O。有色噪音尚未實作。

## 範例

```python
from puresound.audio.noise import add_bg_noise, add_bg_white_noise

(noisy_0db, noisy_10db), _ = add_bg_noise(wav=speech, noise=[noise_wav], snr_list=[0.0, 10.0])
(mix,), (itf_scaled,) = add_bg_noise(wav=near, noise=[far_talker], snr_list=[5.0])   # SIR 5 dB
(noisy_5db,), _ = add_bg_white_noise(wav=speech, snr_list=[5.0])
```
