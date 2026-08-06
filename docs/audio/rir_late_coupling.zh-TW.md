# PathEvent–FDN 晚場耦合（late-field coupling）— `puresound.audio.rir.render.coupling`

English version: `rir_late_coupling.md`

把一個 coherent 的 PathEvent 早場（early field），耦合到一個 deterministic
multiband FDN 尾段。由 M4 backend 使用（`render/high_frequency/fdn.py`）；
設計對照詳見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §4.3。

## Coupling contract（耦合契約）

```python
couple_path_event_rir_with_fdn(
    path_event_rir, sample_rate, direct_sample, target_rt60_s_by_hz,
    *, mixing_time_s=0.024, transition_duration_s=0.016,
    delay_line_count=16, seed=0, filter_order=4,
) -> PathEventFDNCouplingResult  # `.rir`, `.metadata`
```

- 早場在 transition 窗之前都會被**逐 sample 精確保留**（sample-exact）。
- Crossfade 會*置中*在 direct sample 之後 `mixing_time_s` 的位置，並跨越
  `transition_duration_s`（會被夾住，確保仍在 direct sample 之後、且落在
  RIR 範圍內）；它使用 `equal_power_transition_weights`，作用在
  `transition_samples` 回傳的那個窗上。
- Policy 字串：`PATH_EVENT_FDN_COUPLING_POLICY`。

## Energy matching（能量匹配）

`extrapolated_path_tail_energy_target` 會用材質推導出的 RT60 衰減律，把
PathEvent 那段被截斷的尾段，從 render 邊界往外積分回去，這樣 FDN 的增益
就能錨定在「這段尾段*原本應該*帶有的能量」上，而不是錨定在那段被截斷的
能量總和上。`energy_preserving_diffuse_gain` 則把這個目標值換算成 diffuse
增益。前提假設：材質衰減律在這段外推範圍內仍然成立——如果 scene 的 RT60
超過大約 1.5 秒，這個前提就該重新檢視。

## Validation（驗證）

Sample-exactness、transition 前後的能量守恆，以及各 channel 各自不同的
seed，這幾件事都由
`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_path_event_fdn_coupling.py`
與 `test/test_hybrid_rir.py` 驗證。
