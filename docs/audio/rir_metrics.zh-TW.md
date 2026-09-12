# RIR metrics — `puresound.audio.rir.metrics`

English version: `rir_metrics.md`

針對 impulse response 的聲學量測。單一進入點
`analyze_rir(signal, sample_rate, ...)` 會回傳完整報告；每個 metric 也都
可以單獨呼叫。Algorithm-to-code 總覽見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §7。

| Metric | Function |
|---|---|
| Schroeder decay | `schroeder_decay_db`、`noise_compensated_schroeder_decay_db` |
| T20/T30 | `estimate_decay_time` → `DecayEstimate`（數值＋擬合 R²） |
| Clarity C50/C80 | `clarity_db` |
| Direct-to-reverberant ratio | `compute_drr_db`、`direct_sample` |
| Spectral tilt | `spectral_tilt_db_per_octave` |
| Noise floor（Lundeby） | `estimate_noise_floor_lundeby` → `NoiseFloorEstimate` |
| Echo density／mixing time | `analyze_echo_density`、`abel_normalized_echo_density_profile`、`estimate_abel_mixing_time` |
| Octave banding | `octave_band_rir`、`valid_octave_centers`、`DEFAULT_OCTAVE_CENTERS_HZ` |
| Multiband late field | `analyze_multiband_late_field` |
| Spatial（僅限同步 receiver） | `analyze_binaural_iacc`、`analyze_array_spatial_coherence`、`diffuse_field_coherence`、`interaural_cross_correlation` |

報告裡會帶上版本化的 policy 字串：`ABEL_ECHO_DENSITY_POLICY`、
`MULTIBAND_LATE_FIELD_POLICY`、`IACC_POLICY`、
`DIFFUSE_FIELD_COHERENCE_POLICY`——每一個都是各自函式自己回傳字典裡的
戳記，不見得會出現在 `analyze_rir` 的合併報告中。只有
`ABEL_ECHO_DENSITY_POLICY` 會出現在那份合併報告裡，而且只在
`echo_density=True` 時才會出現；另外三個只會出現在各自專屬函式的報告裡。

## 判讀規則

- 一個 decay 估計值，好壞取決於它擬合得好不好：在信任 `rt60_s` 這個秒數
  之前，要先檢查 `DecayEstimate.r_squared`（M6 QC 的 physical policy
  要求 ≥ 0.70，見 `bank/qc.py`）。
- `estimate_noise_floor_lundeby` 需要至少 15 dB 的動態範圍
  （`min_dynamic_range_db=15.0`）；低於這個門檻時，會回報
  `reason="insufficient_dynamic_range"`——尾段已經淡出（faded）的語料庫
  會因為這個理由而觸發，跟量測雜訊本身沒有關係。
- 空間類 metric 假設各 channel 是同一個聲源、彼此同步的 receiver。Bank
  裡那些 channel 各自是獨立 source-to-mic 路徑的項目，沒辦法用這種方式
  評估（M6 QC 會把它們標成 `not_applicable`）。

## Example

```python
from puresound.audio.rir.metrics import analyze_rir, DEFAULT_OCTAVE_CENTERS_HZ

report = analyze_rir(
    rir_channel, 16000,
    octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    echo_density=True,
)
report["t20"]["rt60_s"], report["drr_db"], report["octave_bands"]["1000"]["t20_s"]
```

`octave_bands` 只有在傳入 `octave_centers_hz` 時才會出現——如果保持預設值
`None`，這個 key 會直接不存在（不是空字典）。

Bank 層級的分布比較，位於
`puresound.audio.rir.bank.evaluation.compare_release_distributions`；
獨立的 CLI 是 `egs/rir_generation/compare_bank_acoustics.py`。
