# PathEvent–FDN 晚場耦合 — `puresound.audio.rir.render.coupling`

English version: [rir_late_coupling.md](rir_late_coupling.md)

把單一 channel 的相干 PathEvent 早場接到 [multiband FDN](multiband_fdn.zh-TW.md)
晚場。它是 `path-events-m4` backend（`render/high_frequency/fdn.py`）的核心；
[空間 renderer](spatial_rir.zh-TW.md) 對同步陣列使用相同的元件。Policy 字串：
`PATH_EVENT_FDN_COUPLING_POLICY = "puresound.path_event_fdn_coupling.v1"`。

## 為什麼要耦合

PathEvent 提供精確、可檢視的直達與早期到達，但 image-source 列舉在響應後段變得
稀疏，並停在最高階數。FDN 提供衰減正確的密集尾巴，但沒有特定的早期結構。耦合
完整保留早場，之後的部分改用 FDN，且能量等於路徑場本應帶有的能量。

## 契約

```python
couple_path_event_rir_with_fdn(
    path_event_rir, sample_rate, direct_sample, target_rt60_s_by_hz,
    *, mixing_time_s=0.024, transition_duration_s=0.016,
    delay_line_count=16, seed=0, filter_order=4,
) -> PathEventFDNCouplingResult
# .rir, .coherent_component, .diffuse_component, .early_weight, .late_weight,
# .design, .metadata
```

`direct_sample` 是幾何到達 sample；`target_rt60_s_by_hz` 把 octave 中心頻率（Hz）
對應到 RT60（s）。

## 演算法

1. **Transition 窗**（`transition_samples`）。中心
   `n_c = n_d + round(t_mix · fs)`，寬度 `W = max(2, round(t_trans · fs))`，起點
   `max(n_d + 1, n_c − W/2)`。若窗會超出尾端就往左移；若 RIR 短到放不下直達之後的
   transition，呼叫會 raise。
2. **權重**（`equal_power_transition_weights`）。窗內 `w_e = cos φ`、`w_l = sin φ`，
   φ 由 0 到 π/2，所以 `w_e² + w_l² = 1`；窗之前 `w_e = 1`，之後為 0。
3. **FDN 激發。** FDN 依目標值設計（相同混合時間、給定 seed），以 `h_pe · w_e`
   驅動：直達與早期 event 成為晚場狀態的種子，而 transition 之後的路徑無法持續
   注入稀疏的相干圖樣。
4. **能量目標**（`extrapolated_path_tail_energy_target`）。從 transition 起點到
   RIR 結尾的 PathEvent 能量，加上路徑場在最後一個 event 之後本應還帶有的能量：
   最後 50 ms event 的每 sample 平均能量，以目標 RT60 的中位數衰減到渲染結尾。這讓
   尾巴錨定在衰減律上而不是截斷後的總和，前提是材質衰減在外推區間內成立。
5. **增益**（`energy_preserving_diffuse_gain`）。令相干部分 `c = h_pe · w_e`、擴散
   部分 `f = FDN · w_l`，`g` 是在 transition 之後 sample 上
   `‖c + g f‖² = E_target` 的正根：

   ```text
   g = (−⟨c, f⟩ + √(⟨c, f⟩² + ‖f‖² (E_target − ‖c‖²))) / ‖f‖²
   ```

6. **輸出** `h = c + g f`。直到 transition 起點的每個 sample 都與 PathEvent 輸入
   完全相同；metadata 記錄該區段的最大偏差、transition sample、各能量、外推與
   FDN 設計。

## 在 backend 中的用法

`PathEventFDNHighFrequencyBackend`（PathEvent backend 的子類別）先渲染相干場，
再以下列設定耦合每個聲源 channel：

- 目標取自 `RoomSceneV2.predicted_octave_rt60_s()` 並經空氣吸收修正
  （`air_adjusted_rt60_s`：`60 / (60 / RT60 + a(f) · c)`，`a` 為 ISO 9613-1 的衰減
  量，單位 dB/m），保留低於 Nyquist 且不低於
  `max(minimum_fdn_center_hz, 0.5 × crossover_hz)`（預設 500 Hz）的 octave；
- `direct_sample = round(d / c · fs)`；
- 每個 channel 的 seed 由 FDN seed、scene id 與聲源索引的 BLAKE2b 雜湊導出，
  所以同一房間的各 channel 有獨立的尾巴，而同一個 scene 永遠渲染出相同的尾巴。

`generate_hybrid_rir.py` 提供 `--fdn-mixing-time-ms`、`--fdn-transition-ms`、
`--fdn-delay-lines` 與 `--fdn-seed`。

## 測試

`test/rir/test_rir_late_coupling.py` 固定住早場完全保留、transition 之後的
能量、seed 決定性與長 RT60 的外推；`test/rir/test_hybrid_rir.py` 檢查 backend
保留相干早期路徑並序列化耦合紀錄。
