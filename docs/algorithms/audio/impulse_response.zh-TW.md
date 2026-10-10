# puresound.audio.impulse_response

English version: [impulse_response.md](impulse_response.md)

把房間脈衝響應（RIR）套到波形上，以及兩個作用在 RIR 或訊號上的通道操作：隨機的
二階染色與直達聲抹除（smear）。一般入口是 `AudioEffectAugmentor.apply_rir`
（見 [augmentation.zh-TW.md](augmentation.zh-TW.md)）；RIR 軸的物理見
[房間聲學](../augmentation/room_acoustics.zh-TW.md)。

## `wav_apply_rir(wav, impaulse, sample_rate, rir_mode="full") -> Tensor`

把 `wav: [C, T]` 與 `impaulse: [C_rir, T_rir]` 做 convolution（兩者都是 2-D；
RIR 參數拼作 `impaulse`）。

1. **取窗。** `rir_mode` 決定保留多少 RIR，以峰值索引 `p`（攤平後 RIR 最大樣本的
   索引）為基準：

   | `rir_mode` | 保留的 RIR | 用途 |
   | --- | --- | --- |
   | `"full"` | 全部 | 混音 |
   | `"early"` | `h[:, : p + 50 ms]` | 含早期反射的目標 |
   | `"direct"` | `h[:, : p + 6 ms]` | 接近乾聲的目標 |

   取窗模式假設 RIR 是單聲道、且直達聲是其最大樣本。
2. **峰值正規化。** `h ← h / max|h|`，對所有聲道取一個最大值（低於 `1e-12` 時
   略過）。窗一定包含峰值，所以三種模式對同一個 RIR 套用相同縮放：由同一個 RIR
   產生的 `"early"` 目標與 `"full"` 混音，直達聲位準相同，只差在反射能量。
3. **Convolution 與對齊。** `fftconvolve(..., mode="full")`，再把輸出切成
   `[d, d + T)`，`d = argmax|h[0]|` 為聲道 0 的峰值。結果長度與輸入相同、從直達聲
   開始，因此傳播延遲被移除，混音與目標都與乾聲樣本對齊。

聲道配置：單聲道 RIR 會分別作用於 `wav` 的每個聲道（`[C, T] → [C, T]`）；多聲道
RIR 要求單聲道 `wav`，每個 RIR 聲道產生一個輸出（`[1, T] → [C_rir, T]`），全部
對齊到聲道 0 的峰值，因此聲道間延遲得以保留。

**位準設計。** 以峰值正規化會移除 bank RIR 在檔案上攜帶的絕對 `1/r` 位準（聲道間
比例保留，因為所有聲道用同一個縮放）。殘響後的聲源因此不會隨距離變小聲；位準由
recipe 透過 SNR/SIR 明確設定，近場與遠場只差在 DRR、衰減形狀與頻譜傾斜。
voice-isolation 的 `mix_mode` 區塊以此為前提：它的 `physical` 模式不重新縮放、
直接相加 RIR 後的聲源（比值因此接近 0 dB），`distance_level` 模式則以
`SIR = 20·log10(d_itf / d_fg)` 加上抖動，明確放回位準線索。

## `compute_drr_db(rir, sample_rate, direct_window_ms=2.5) -> float`

單聲道的 `puresound.audio.rir.metrics.compute_drr_db` 轉出：

```
DRR = 10·log10( Σ_{n=p}^{p+W−1} h[n]²  /  Σ_{n≥p+W} h[n]² ),   W = round(direct_window_ms · fs / 1000)
```

其中 `p = argmax|h|`。`p` 之前的能量忽略；尾段沒有能量時回傳 `+inf`。2.5 ms 窗與
room simulator、bank loader、DRR-contrast 增強使用的相同，所以 pipeline 中每個
DRR 都是同一個量（見 [RIR metrics](rir_metrics.zh-TW.md)）。

## `rand_add_2nd_filter_response(wav, a=None, b=None) -> (wav, a, b)`

以隨機極零點 biquad 作為換能器頻率響應的廉價模型（J.-M. Valin, *A Hybrid
DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement*,
MMSP 2018）：

```
H(z) = (1 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²),   a1, a2, b1, b2 ~ U(−3/8, 3/8)
```

未給 `a` 或 `b` 時由 torch generator 抽。因為 `|a1| + |a2| ≤ 3/4 < 1`，兩個極點都在
單位圓內，每次抽樣都穩定。濾波使用 `torchaudio.functional.lfilter(..., clamp=False)`：
頻率響應是線性的，預設的 clamp 會截斷大聲的混音、卻放過較小聲的目標。係數會回傳，
讓同一個響應可以套到配對的目標上。

## `smear_direct_arrival(impaulse, sample_rate, *, smear_ms, generator=None) -> Tensor`

逐聲道打亂直達聲之後 `smear_ms` 內的細部時間結構，RIR 其他部分不變：

1. 取窗 `h[p : p + n]`，`n = round(smear_ms · fs / 1000)`；`smear_ms ≤ 0` 或
   `n < 2` 時不做事。
2. 把窗與一個隨機、單位能量的高斯 kernel 做因果 convolution（`torch.randn`，可給
   `generator`），所以沒有任何能量早於 `p` 到達。
3. 讓第一個樣本保持最大（`1.001 × max`），再把窗縮放回原本的能量。

保留峰值索引，是因為 `wav_apply_rir` 取窗與對齊都讀它；保留窗能量，讓這個操作不是
位準變化；晚期尾段不動。Augmentor 在 RIR 進快取之前套用它，所以 `"full"` 混音與
`"early"` 目標看到的是同一個被抹除的脈衝。設定位置為
`augmentation_reverb.direct_smear`（`used`、`prob`、`smear_ms_range`）；recipe 沒設
就不啟用。它移除的是什麼，見 [距離線索](../augmentation/distance_cues.zh-TW.md)。

## 範例

```python
from puresound.audio.io import AudioIO
from puresound.audio.impulse_response import rand_add_2nd_filter_response, wav_apply_rir

rir, _ = AudioIO.open("rir.wav", resample_to=16000)
clean, sr = AudioIO.open("clean.wav", resample_to=16000)

mixture = wav_apply_rir(clean, rir, sr, rir_mode="full")
target = wav_apply_rir(clean, rir, sr, rir_mode="early")
colored, a, b = rand_add_2nd_filter_response(mixture)
target_colored, _, _ = rand_add_2nd_filter_response(target, a=a, b=b)
```
