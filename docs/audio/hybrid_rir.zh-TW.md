# Hybrid RIR 生成 — `puresound.audio.rir.render`

English version: `hybrid_rir.md`

一個公開的進入點，組合起整條 render chain：

```python
from puresound.audio.rir.render.hybrid import generate_hybrid_rir
# 也從 puresound.audio.rir.api re-export
```

整條 chain（細節與 code map 見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §3–§5）：

1. **低頻頻段（20–1000 Hz）** —— ARD／DCT 波動方程求解，搭配 per-mode
   材質阻尼（`render/low_frequency/pytard.py`；GPU 版本是
   `GpuARDPytARDCuPyBackend`）。
2. **高頻頻段（1000 Hz–Nyquist）** —— 三種 backend 之一
   （`render/high_frequency/`）：`pyroomacoustics`（ISM ＋ ray tracing）、
   `path-events-m3`（coherent PathEvents）、`path-events-m4`（PathEvents
   早場 ＋ multiband FDN 晚場）。
3. **Crossover** —— 在 `HybridRIRConfig.crossover_hz` 上做 causal
   4 階 Linkwitz–Riley，並在 crossover 頻率的 0.7×–1.3× 範圍內做 RMS
   能量匹配（預設 1000 Hz 時即 700–1300 Hz；`render/crossover.py`）。
   匹配增益會被夾在 `HybridRIRConfig.crossover_match_gain_range` 範圍內，
   並逐 channel 記錄下來。當低頻 backend 已經回報 source convention 已經
   匹配（`direct_path_source_convention_matched`），且
   `HybridRIRConfig.preserve_source_convention_at_crossover`（預設
   `True`）有設定時，匹配這一步會被跳過——crossover 的 metadata 會同時
   記錄「是否有要求匹配」與「是否真的有匹配」這兩件事。
4. **Output calibration** —— `output_mode`：`calibrated`（物理 SPL
   語意，peak 可能超過 1.0）或 `peak_normalized`。

## Causality contract（因果性契約）

每一階段都會保留前導零，直到
`floor(distance / sound_speed * sample_rate)` 這個 sample 為止：低頻
頻段在做 causal low-pass 之前先 clip；高頻頻段在做 direct alignment 之後、
causal high-pass 之前 clip；PathEvents 使用單邊（one-sided）的
fractional-delay kernel；FDN coupling 在其 transition 之前是逐 sample
精確的（sample-exact）。M6 QC 把這件事實作成 `prearrival_energy` 這個
gate。

## Backend selection（Backend 選擇）

`render/backend.py` 只定義了 `RIRBackend` 這個 protocol。它的搭檔
`BackendCapabilities` 宣告——也就是某個具體 backend 承諾了什麼，包括
determinism（pyroomacoustics 的 ray tracing 並不是每個 seed 都能逐位元組
重現）——則放在低一層的 `contracts.py` 裡。
M6 wrapper `egs/rir_generation/generate_m6_bank.py` 預設用
`path-events-m4`；低階的 generator
`egs/rir_generation/generate_hybrid_rir.py` 則保留 `pyroomacoustics`
作為自己的預設值，因為 M4／M5 的 exit gate 是釘住（pin）在那一層上的。

## 套用一支 RIR

把乾淨語音跟 bank item 的某個 channel 卷積；JSON sidecar 是資料契約的一
部分（channel map、距離、level policy）——絕對不要把 WAV 和 metadata
分開。訓練端的整合要走 bank loader
（[`rir_bank.zh-TW.md`](rir_bank.zh-TW.md)），而不是隨手卷積。
