# 複數聲學阻抗

English: [impedance_measurements.md](impedance_measurements.md)

這套流程把保留 phase 的阻抗量測轉成被動邊界模型，供 modal 與 time-domain
solver 使用。目前屬於研究與驗證工具；量測結果不會自動寫入 production
material catalog。

材料名稱相同，不代表厚度、backing、air gap、安裝方式或入射條件相同。

## 為何需要複數阻抗

令 characteristic impedance (Z_0=ho c)：

[
Gamma(f)=rac{Z(f)-Z_0}{Z(f)+Z_0},
qquad
alpha(f)=1-|Gamma(f)|^2.
]

吸音係數只保留反射 magnitude，會失去影響 reflection delay、modal frequency
與 Q 的 phase。因此本流程要求 real 與 imaginary impedance，不從 diffuse-field
absorption 猜測 phase。

## SI 量測契約

`ComplexImpedanceMeasurement.from_csv_and_metadata()` 同時讀取 CSV 與 JSON
sidecar。

CSV 必要欄位：

```csv
frequency_hz,impedance_real_pa_s_m,impedance_imag_pa_s_m
60.0,421.2,-880.4
80.0,398.1,-631.7
120.0,376.5,-402.8
```

Real 與 imaginary uncertainty 欄位若使用，必須成對出現。

Loader 會拒絕非遞增或非正 frequency、少於三點、non-finite values、負的 real
impedance、不完整 uncertainty，以及推導出反射 magnitude 大於一的資料。

最小 metadata：

```json
{
  "schema_version": "puresound.complex_impedance_measurement.v1",
  "measurement_id": "lab_sample_001",
  "method": "ISO 10534-2 two-microphone transfer function",
  "incidence": "normal",
  "environment": {
    "air_density_kg_m3": 1.204,
    "sound_speed_m_s": 343.0
  },
  "sample": {
    "material_label": "100 mm glass wool",
    "thickness_m": 0.1,
    "backing": "rigid",
    "air_gap_m": 0.0
  },
  "provenance": {
    "source_url": "https://example.org/measurement/001",
    "license": "CC-BY-4.0"
  }
}
```

Production measurement 還應記錄 tube geometry、microphone spacing、sample
batch 與 tolerance、mounting、溫濕度、calibration 與 repeatability。

ISO 10534-2 normal-incidence tube 資料不能直接視為 reverberation-room 的
diffuse-incidence absorption。

## Normalized impedance 契約

部分文獻只提供 (z=Z/(ho c))，沒有保留當時的 normalization values。此時
使用 `puresound.normalized_complex_impedance_measurement.v1` 與欄位：

```csv
frequency_hz,normalized_impedance_real,normalized_impedance_imag
500,0.785,-3.596
600,0.474,-3.213
```

Metadata 必須說明 field geometry、phasor convention、flow Mach number、
source SPL、sample、provenance、license 與 applicability limits。

有界的 fitting coordinate 為：

[
mathcal C(z)=rac{z-1}{z+1}.
]

只有在適用的 locally reacting、normal-incidence 條件下，它才等於壓力反射係數。
對 grazing-duct data，它主要是穩定的 Cayley transform；不能把這類資料重新標成
impedance-tube measurement。

## 被動邊界 fitting

離散量測要先轉成 causal model，才能供 time-domain solver 使用。Normalized
admittance 使用非負的 parallel branches：

[
y(s)=g_s+
sum_krac{g_{L,k}}{1+s	au_k}+
sum_k g_{H,k}rac{s	au_k}{1+s	au_k},
qquad
	au_k=(2pi f_{p,k})^{-1}.
]

[
Z(s)=rac{ho c}{y(s)},
qquad
Gamma(s)=rac{1-y(s)}{1+y(s)}.
]

所有 gain 都限制為非負，因此 parallel sum 在結構上就是 passive。
`fit_passive_multi_pole_admittance()` 使用 fixed poles，同時 fitting real 與
imaginary reflection，並報告 complex、magnitude 與 phase errors。

Fixed poles 是刻意保守的選擇。Acceptance gate 失敗時，應調整 model order、
pole placement 或 measurement band，不能移除 passivity constraint。

Resonant material 可能需要 passive conjugate-pole 或 series-RLC branch。令
(u=s/omega_0)：

[
y_k(s)=rac{g_{mathrm{peak},k}}{Q_k}
       rac{u}{u^2+u/Q_k+1}.
]

只有 measured reactance 支持該 resonance，而且 fit 沒有超出宣告適用範圍時，
才應使用此類 branch。

## Solver 使用

Fitted model 會一致地供應 boundary reflection 或 admittance 給：

- path-event boundary filters；
- impedance-modal rectangular-room solver；
- FDTD reference solver。

矩形空間尺寸為 (L_x,L_y,L_z) 時，rigid reference frequencies 為：

[
f_{n_xn_yn_z}=rac{c}{2}
sqrt{left(rac{n_x}{L_x}ight)^2+
      left(rac{n_y}{L_y}ight)^2+
      left(rac{n_z}{L_z}ight)^2}.
]

Complex impedance 會改變 frequency 並加入 decay。比較 modal 與 FDTD 時，
geometry、environment、boundary orientation 與 phasor convention 必須一致。

這些 solver 是驗證工具。單一 normal-incidence sample fitting 良好，不代表已
證明 diffuse、angle-dependent 的房間邊界模型。

## 量測流程

Two-microphone impedance-tube template：

```text
egs/rir_generation/phases/m2_impedance/measurements/
  impedance_tube_template/
```

轉換 raw transfer data：

```bash
python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py --help
```

驗證 measurement 與 fit：

```bash
python egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py --help
```

`measurements/zenodo_15195587/` 下的公開 liner examples 用於驗證 normalized
grazing-duct ingestion 與 resonant fitting，不是 production wall-material
measurement。

## Release 規則

正式採用 fitted boundary 前：

1. 驗證 source files、provenance 與 redistribution license。
2. 保留 measurement geometry 與 phasor convention。
3. 在 fit band 內外檢查 passivity。
4. 報告 held-out complex、magnitude 與 phase error。
5. 交叉檢查 modal 與 FDTD behavior。
6. 保存 thickness、backing、air gap、mounting 與環境限制。
7. 只有經過明確 review 的 mapping，才能加入 material catalog。
