# RIR metrics — `puresound.audio.rir.metrics`

English version: [rir_metrics.md](rir_metrics.md)

對脈衝響應做的房間聲學量測。用途是記錄 renderer 實際產出了什麼（item metadata
中的 `realized_acoustics`）、在 QC 中篩選 bank item，以及比較不同 bank。每個函式
都接受一個陣列與明確的分析設定；沒有任何函式知道訊號是哪個 renderer 產生的。
衰減、清晰度與噪音底的定義依循 ISO 3382-1。

## 報告入口

```python
from puresound.audio.rir.metrics import analyze_rir, DEFAULT_OCTAVE_CENTERS_HZ

report = analyze_rir(
    rir_channel, 16000,                 # 單聲道 RIR、取樣率
    direct_window_ms=2.5,               # DRR 的直達窗
    direct_index=None,                  # 預設：絕對峰值
    octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    noise_compensation=True,            # 經 Lundeby 修正的 Schroeder 曲線
    echo_density=True,
)
report["t20"]["rt60_s"], report["drr_db"], report["octave_bands"]["1000"]["t20_s"]
```

寬頻欄位：`direct_sample`、`direct_delay_ms`、`peak_abs`、`energy`、`drr_db`、
`c50_db`、`c80_db`、`spectral_tilt_db_per_octave`、`noise_floor`、
`edt`／`t20`／`t30`（完整擬合紀錄）與 `edt_s`／`t20_s`／`t30_s`。只有給了
`octave_centers_hz` 才會出現 `octave_bands`；每個頻帶含 DRR、C50、EDT/T20/T30、
各自的擬合 R² 與該頻帶的噪音底。只有 `echo_density=True` 才會出現
`echo_density`。無限大的值（例如沒有殘響能量的響應）回報為 `None`。

## 定義

除非給了 `direct_index`，直達 sample `n_d` 取絕對峰值。Hybrid renderer 的
metadata 會傳入幾何到達時間 `round(d / c · fs)`。

| 指標 | 函式 | 定義 |
|---|---|---|
| DRR | `compute_drr_db` | `10 log10(Σ h² over [n_d, n_d + W) / 之後的 Σ h²)`，`W = direct_window_ms`；`n_d` 之前的能量忽略 |
| C50、C80 | `clarity_db` | 同樣形式，分界在 `n_d` 後 50 或 80 ms |
| Schroeder 衰減 | `schroeder_decay_db` | 從 `n_d` 起算 `10 log10(∫_t^∞ h² / ∫_0^∞ h²)`，下限 −120 dB |
| 噪音補償衰減 | `noise_compensated_schroeder_decay_db` | 在 Lundeby 交點截斷、扣除預期噪音能量並保持單調的 Schroeder 積分；找不到可靠噪音底時退回原始曲線 |
| EDT／T20／T30 | `estimate_decay_time` → `DecayEstimate` | 在衰減曲線 0…−10、−5…−25、−5…−35 dB 區間做最小平方直線；`RT60 = −60 / slope`；需 ≥ 8 點且跨 ≥ 10 ms |
| 噪音底 | `estimate_noise_floor_lundeby` → `NoiseFloorEstimate` | 以 10 ms 區塊能量做 Lundeby 迭代：尾端噪音底、其上的衰減擬合、交點、再精修 |
| 頻譜斜率 | `spectral_tilt_db_per_octave` | `20 log10 |H(f)|` 對 `log2 f` 的最小平方斜率，200 Hz – 4 kHz |
| Octave 頻帶 | `octave_band_rir`、`valid_octave_centers` | `[f_c/√2, f_c·√2]` 的因果四階 Butterworth 帶通；上緣超過 0.99 × Nyquist 的中心頻率會被剔除 |
| 回音密度 | `abel_normalized_echo_density_profile`、`analyze_echo_density` | Abel–Huang：20 ms 窗內 `|h| > σ` 的 sample 比例，除以高斯比例 `erfc(1/√2)`（擴散聲場約為 1） |
| 混合時間 | `estimate_abel_mixing_time` | `n_d` 之後 profile 持續 ≥ 0.9 達 10 ms 的第一個時間點 |
| Multiband 晚場 | `analyze_multiband_late_field` | 逐 octave、考慮噪音的衰減與回音密度，以寬頻直達 sample 為時間錨點 |
| IACC | `interaural_cross_correlation`、`analyze_binaural_iacc` | ±1 ms lag 內正規化互相關的最大絕對值，分早期（0–80 ms）與晚期窗，寬頻與 500 Hz – 4 kHz octave |
| 陣列相干性 | `analyze_array_spatial_coherence`、`diffuse_field_coherence` | 同步麥克風對在 80 ms 之後的複數相干 `S₁₂ / √(S₁₁ S₂₂)`（Welch），逐 octave 與擴散聲場目標 `sinc(2 f d / c)` 比較 |

`DEFAULT_OCTAVE_CENTERS_HZ` 為 63 Hz – 8 kHz。

## 解讀規則

- 衰減估計只和它的擬合一樣好。信任 `rt60_s` 前先看 `DecayEstimate.r_squared`；
  bank QC 要求 `RIRBankQCPolicy.minimum_decay_r_squared`（0.70）。
- `estimate_noise_floor_lundeby` 只在衰減高出噪音底至少 `min_dynamic_range_db`
  （15 dB）時才修正。否則回報 `correction_applied=False` 並附 `reason`，例如
  `insufficient_dynamic_range`、`insufficient_decay_above_noise` 或
  `tail_is_not_stationary_noise`。尾端在發佈前被淡出的量測語料會因為與量測
  噪音無關的原因落到這裡。
- IACC 與相干性假設各 channel 是同一聲源的同步 receiver。Channel 彼此是獨立
  聲源到麥克風路徑的 bank item 不能這樣評估；bank QC 會標為 `not_applicable`。

## Policy 字串

`ABEL_ECHO_DENSITY_POLICY`、`MULTIBAND_LATE_FIELD_POLICY`、`IACC_POLICY` 與
`DIFFUSE_FIELD_COHERENCE_POLICY` 會蓋在各自函式的結果裡。其中只有回音密度的
policy 會進入 `analyze_rir` 的報告（在 `echo_density` 內）。指標計算方式一旦
改變就換新的 policy 字串，讓已儲存的報告仍可解讀。

## 使用處

- `generate_hybrid_rir(..., record_realized_metrics=True)` 逐 channel 儲存
  `analyze_rir`。
- Bank QC（`puresound.audio.rir.bank.qc`）對每個 item 執行它。
- `puresound.audio.rir.bank.evaluation.compare_release_distributions` 比較
  release variant；`egs/rir_generation/compare_bank_acoustics.py` 比較不同 bank
  資料夾的 DRR、清晰度、衰減與頻譜統計。

為稽核 renderer 而把響應拆成直達、早期與晚期部分，是另一個模組：
[attribution](rir_attribution.zh-TW.md)。
