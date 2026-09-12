# RIR 生成各 phase

這個目錄把各里程碑專屬的實驗，跟 `egs/rir_generation/` 裡七個穩定的
公開指令分開存放。每個 phase 都遵循同一套慣例：

English version: [`README.md`](README.md)

這七個公開指令是 `generate_hybrid_rir.py`、`generate_m6_bank.py`、
`render_spatial_rir.py`、`plot_rir.py`、`inspect_bank.py`、
`compare_bank_acoustics.py`、以及 `compare_modal_acoustics.py`。其餘的都放
在這裡，或放在輔助用的 `tools/` 與 `examples/` 目錄下。

- `scripts/`：該里程碑的 validator、校準執行程式、探索性指令；
- `config/`：輸入設定與範本；
- `reports/`：某次執行產出的凍結版 JSON 證據；
- `fixtures/`：測試用的小型非正式 fixture；
- `measurements/`：原始／化簡後的量測資產（僅 M2 有）。

| Phase | 範圍 |
|---|---|
| `m0_baseline` | 凍結的 baseline 設定 |
| `m1_material` | 材質先行場景與 priors |
| `m2_impedance` | 複數阻抗、modal residue、阻抗管量測 |
| `m3_wave_path` | 同調 PathEvents 與低頻 wave-path 驗證 |
| `m4_spatial_late_field` | 空間渲染與多帶晚場 |
| `m5_calibration` | 實測房間反演校準與約束殘差 |
| `m6_bank` | 確定性 bank、QC、release、評估、產線證據、實測參考目標有效性 |

report 刻意留在產出它的 phase 旁邊。report 與文件中的路徑應該採用這套
佈局，而非已移除的扁平設定目錄。
