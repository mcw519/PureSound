# Multiband FDN — `puresound.audio.rir.render.multiband_fdn`

English version: [multiband_fdn.md](multiband_fdn.md)

一個決定性、被動的 feedback delay network（FDN），依每個 octave 頻帶指定的 RT60
渲染晚期殘響尾巴。它是 `path-events-m4` backend（透過
[晚場耦合](rir_late_coupling.zh-TW.md)）與[空間 renderer](spatial_rir.zh-TW.md) 的
晚場引擎。它只渲染尾巴；把尾巴接進 RIR 是耦合模組的工作。Policy 字串：
`MULTIBAND_FDN_POLICY = "puresound.multiband_fdn.v2"`。

## 為什麼用 FDN

相干 image-source renderer 在最高階數就停止產生路徑，遠早於一段長衰減結束，而且
它的晚期路徑很稀疏。FDN 以固定成本產生密集、指數衰減的尾巴，且每個頻帶的衰減
可以由目標 RT60 精確設定。

## 設計

```python
design = design_multiband_fdn(
    sample_rate, target_rt60_s_by_hz,     # {octave 中心頻率 Hz: RT60 s}
    target_mixing_time_s=0.024, delay_line_count=16,
    delay_range_ms=None, seed=0,
)  # -> MultibandFDNDesign
```

- **Delay**（`select_prime_delay_lengths`）：`delay_line_count` 個相異質數，
  各自最接近在 `max(1, 0.125·t_mix)` 到 `0.75·t_mix` ms 間等比分布的目標值
  （預設混合時間 24 ms 時為 3–18 ms），每個目標依 seed 抖動 ±2 %。互質的長度
  避免共同週期，回音不會排成可聽見的週期性。
- **回授矩陣**（`randomized_hadamard_matrix`）：以 seed 決定列／行排列與正負號
  的正規化 Hadamard 矩陣。它是正交的，所以不含增益的迴路保持能量；它是稠密的，
  每條 line 都饋入其他所有 line，回音密度很快建立。`delay_line_count` 必須是
  2 的冪次。
- **迴路增益**（`delay_proportional_loop_gains`）：頻帶 `b`、delay 為 `d_i`
  sample 時，

  ```text
  g_{b,i} = 10^(−3 · d_i / (fs · RT60_b))
  ```

  在第 `i` 條 line 中循環的訊號每經過一次損失 `60 · d_i / (fs · RT60_b)` dB，
  也就是不論 line 長度，每 `RT60_b` 秒損失 60 dB。每個增益都在 (0, 1) 之間，
  所以每個頻帶都是收縮、被動的迴路。
- **輸入與輸出**向量為依 seed 抽取的 ±1/√N；頻帶權重相等且平方範數為 1。

每個 octave 中心頻率都必須低於 Nyquist（`valid_octave_centers`）。

## 渲染

`render_multiband_fdn(design, excitation, output_gain=1.0, filter_order=4)` 對單
聲道激發訊號為每個頻帶跑一個 FDN 遞迴（共用 delay 與矩陣、各頻帶迴路增益不同），
再把每個頻帶的輸出限頻後相加。`render_multiband_fdn_impulse(design, duration_s)`
對單位脈衝做同樣的事。結果保留全頻帶 `rir`、濾波後的 `band_rirs` 與各頻帶的原始
輸出。

遞迴以不長於最短 delay 的區塊計算，這與逐 sample 計算完全等價：寫入的值不可能
早於那個 delay 回到輸出。

**濾波器組。** 頻帶邊界是相鄰中心頻率的幾何平均。第 `k` 個頻帶是
Butterworth 高通與一個低通的串接 `H_0 ⋯ H_{k−1} L_k`（最低頻帶為 `L_0`，最高為
`H_0 ⋯ H_{n−1}`）。這種二元切分對 Butterworth 對是功率互補的，且從 DC 延伸到
Nyquist，所以晚場能量在頻帶邊界或 `fs/2` 以下都沒有頻譜空洞。
`fdn_filterbank_power_response` 回傳總功率響應；渲染時會記錄其漣波。

一切都有 seed：相同輸入渲染出逐位元組相同的尾巴。

## 診斷

`analyze_fdn_coloration(rir, sample_rate, centers_hz)` 逐 octave 回報頻譜平坦度
（功率的幾何平均除以算術平均）與第 95 百分位對中位數的功率比（dB）——這是模態
染色的量度。`puresound.audio.rir.metrics.analyze_multiband_late_field` 把渲染出的
尾巴與各 octave 目標比對。

## 限制

FDN 會實現它被給定的目標。在 renderer 中這些目標是依 scene 材質做的 Sabine 預測
並經空氣吸收修正，所以 Sabine 估計的任何誤差都會原封不動地出現在輸出中。
