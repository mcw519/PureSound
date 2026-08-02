# 複數聲學阻抗：真實量測、被動共振擬合、FDTD 與模態

本文件說明 M2.4–M2.5 的垂直切片：如何把保留相位的 complex
impedance 量測，轉成一個可追溯、被動、因果，而且能同時供時域 FDTD
與複數模態方程使用的邊界模型。

這條路徑目前是研究與驗證基礎設施，不會把量測結果自動套用到 production
材料 catalog。材料名稱相同，不代表厚度、背板、空氣層、安裝方式或入射
條件相同。

## 1. 為什麼不能只讀 absorption

法向入射的壓力反射係數與表面阻抗為：

\[
\Gamma(f)=\frac{Z(f)-Z_0}{Z(f)+Z_0},
\qquad
Z_0=\rho c
\]

\[
\alpha(f)=1-|\Gamma(f)|^2
\]

吸收率只保留 \(|\Gamma|\)，沒有 \(\angle\Gamma\)。因此同一個
\(\alpha\) 可以對應許多不同的複數阻抗，而這些邊界會產生不同的反射
延遲、模態頻率和 Q。匯入格式所以直接要求
\(\operatorname{Re}Z\) 與 \(\operatorname{Im}Z\)，不接受用
diffuse-field absorption 補猜相位。

SI contract 支援法向入射、兩麥克風 transfer-function 類型的 complex
impedance 資料。ISO 10534-2 的適用範圍正是由此類量測取得法向入射
吸收率與複數表面阻抗；它和 reverberation-room 的 diffuse-incidence
吸收率不是可直接互換的量。M2.5 另加入 normalized \(Z/(\rho c)\)
contract，保留公開 liner eduction 資料原本的無因次表示與量測幾何，
不把 grazing duct 假稱為 normal-incidence tube。

- [ISO 10534-2:2023 overview](https://www.iso.org/standard/81294.html)

## 2. 檔案契約

`ComplexImpedanceMeasurement.from_csv_and_metadata()` 同時讀取：

1. 一個逐頻率 CSV；
2. 一個記錄環境、樣品與來源的 JSON sidecar。

### 2.1 CSV

必要欄位與單位：

```csv
frequency_hz,impedance_real_pa_s_m,impedance_imag_pa_s_m
60.0,421.2,-880.4
80.0,398.1,-631.7
120.0,376.5,-402.8
```

可選的不確定度欄位必須成對出現：

```text
impedance_real_std_pa_s_m
impedance_imag_std_pa_s_m
```

loader 會拒絕：

- 少於三個頻率點；
- 非有限、非正或未嚴格遞增的頻率；
- NaN/Inf impedance；
- \(\operatorname{Re}Z<0\)；
- 只有一個不確定度欄位；
- 由資料算出的 \(|\Gamma|>1\)。

### 2.2 JSON metadata

最小 sidecar：

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

目前 schema 強制要求：

- 精確的 schema version；
- `measurement_id`、`method` 與 `incidence: "normal"`；
- 正值的 air density 與 sound speed；
- `sample.material_label`；
- `provenance.source_url` 與 `provenance.license`。

production 資料還應記錄 tube 直徑、麥克風間距、樣品批次、厚度公差、
背板、air gap、溫濕度、校正方式與重複量測統計。這些欄位尚未全部設為
schema 的硬性要求，但材料映射時不可省略。

### 2.3 Normalized complex impedance contract

有些文獻直接發表：

\[
z(f)=\frac{Z(f)}{\rho c}
\]

卻沒有在資料檔逐筆保存 normalization 使用的精確 \(\rho,c\)。這種資料
使用另一個 schema：

```text
puresound.normalized_complex_impedance_measurement.v1
```

CSV 必須包含：

```csv
frequency_hz,normalized_impedance_real,normalized_impedance_imag
500,0.785,-3.596
600,0.474,-3.213
```

JSON 必須明列：

- `acoustic_field_geometry`：`normal_incidence_tube` 或 `grazing_duct`；
- `phasor_convention`：目前固定為 `exp(+i*omega*t)`；
- mean-flow Mach 與 source SPL；
- 樣品、來源、license 與用途限制。

loader 使用
\[
\mathcal C(z)=\frac{z-1}{z+1}
\]
作為 bounded fitting coordinate，並檢查
\(\operatorname{Re}z\ge0\) 與 \(|\mathcal C(z)|\le1\)。
只有 locally reacting、normal-incidence 解讀成立時，
\(\mathcal C(z)\) 才等於實際的 normal-incidence pressure reflection；
對 grazing-duct 資料，它首先是一個數值穩定的 Cayley transform。

## 3. 被動 rational admittance

匯入的離散頻率資料不能直接放進 time-domain solver。M2.4 使用固定實數
pole 的 rational normalized admittance：

\[
y(s)=g_s
+\sum_{k=1}^{K}\frac{g_{L,k}}{1+s\tau_k}
+\sum_{k=1}^{K}g_{H,k}\frac{s\tau_k}{1+s\tau_k}
\]

\[
\tau_k=\frac{1}{2\pi f_{p,k}},
\qquad
Z(s)=\frac{\rho c}{y(s)},
\qquad
\Gamma(s)=\frac{1-y(s)}{1+y(s)}
\]

其中所有係數都限制為：

\[
g_s\ge0,\qquad g_{L,k}\ge0,\qquad g_{H,k}\ge0
\]

每個 low-pass 與 high-pass branch 在右半 \(s\)-plane 都是
positive-real；非負平行和仍為 positive-real。因此模型由結構保證被動，
不是在擬合後才抽樣幾個頻率檢查 \(|\Gamma|\)。

低、高頻端點為：

\[
y(0)=g_s+\sum_k g_{L,k},
\qquad
y(\infty)=g_s+\sum_k g_{H,k}
\]

`fit_passive_multi_pole_admittance()` 固定 \(f_{p,k}\)，以有界
least-squares 在複數壓力反射域同時最小化實部與虛部誤差。輸出會記錄：

- RMS 與最大 complex reflection error；
- 最大 magnitude error；
- 最大 phase error；
- fit band、pole strategy、passivity strategy；
- 來源 measurement id 與 acceptance gate。

這不是完整的 vector fitting。固定 pole 可避免 pole relocation、unstable
pole 翻轉與擬合後 passivity repair，代價是可能需要較多 pole，或在 pole
位置不合適時無法達到誤差門檻。若資料通不過 gate，正確處理方式是調整
模型階數、pole placement 或量測範圍，而不是關掉 passivity constraint。

### 3.1 為什麼真實 liner 需要共軛 pole

Zenodo liner 的 reactance 在約 1.6 kHz 由負穿越零再變正，這是
Helmholtz／series-RLC resonance。只有實數 relaxation pole 的模型無法
表示這個符號翻轉；初次嘗試的 held-out complex error 接近 0.9，明確
不合格。

M2.5 因此加入被動 series-RLC admittance branch。令
\(u=s/\omega_0\)：

\[
y_k(s)=
\frac{g_{\mathrm{peak},k}}{Q_k}
\frac{u}{u^2+u/Q_k+1}
\]

實作等價地把上式兩個因子相乘：

\[
y_k(s)=
\frac{g_{\mathrm{peak},k}}{Q_k}
\cdot
\frac{s/\omega_0}
{(s/\omega_0)^2+s/(Q_k\omega_0)+1}
\]

其中 \(f_0>0,Q>0,g_{\mathrm{peak}}\ge0\)。它就是被動 series RLC 的
admittance；在 \(f_0\) 的 conductance 為 \(g_{\mathrm{peak}}\)，pole
必定在左半平面。再和平行的非負 static conductance 相加，仍是
positive-real。

`fit_passive_single_resonance_admittance()` 同時擬合 static、\(f_0\)、
\(Q\) 與 peak admittance。預設以偶數位置的頻率 bin 訓練、奇數位置的
bin held out，避免只報告 training interpolation。

## 4. 多極點如何進入 FDTD

每個 pole 都維護一個 bilinear-transform low-pass state \(u_k[n]\)：

\[
u_k[n]
=b_kp[n]+b_kp[n-1]-a_{1,k}u_k[n-1]
\]

\[
b_k=\frac{1}{1+2\tau_k/\Delta t},
\qquad
a_{1,k}=\frac{1-2\tau_k/\Delta t}
              {1+2\tau_k/\Delta t}
\]

因為

\[
\frac{s\tau_k}{1+s\tau_k}
=1-\frac{1}{1+s\tau_k},
\]

high-pass branch 不需要第二組濾波器 state，直接使用 \(p[n]-u_k[n]\)。
牆面法向速度為：

\[
v_n[n]=\frac{
g_sp[n]
+\sum_k g_{L,k}u_k[n]
+\sum_k g_{H,k}\left(p[n]-u_k[n]\right)
}{\rho c}
\]

`simulate_fdtd_reference()` 在每一面牆、每一個 wall cell、每一個 pole
保存獨立 state。舊的一階 relaxation 是這個表示法的特例；靜態實數
admittance 仍會退化到原本的 boundary update。

RLC branch 則由 bilinear transform 變成二階 biquad：

\[
w_k[n]=b_0p[n]+b_1p[n-1]+b_2p[n-2]
-a_1w_k[n-1]-a_2w_k[n-2]
\]

每個 branch 都在自己的 \(f_0\) prewarp，讓 analog 與 digital resonance
對齊。solver 的 boundary time-step gate 不再只看 DC/無限頻端點；它使用
`static + sum(g_peak)` 的 conservative all-frequency admittance bound，
避免窄頻高 admittance resonance 躲過穩定檢查。

顯式求解器除了 3D interior CFL，也使用
\(\max(y(0),y(\infty))\) 估計 edge/corner 累積的 boundary time-step
上限。完整 metadata 同時序列化 interior limit、boundary limit、實際
sample rate 與每個 pole 的數位係數。

## 5. 複數模態最小連接點

在把 boundary 接入完整 3D pytARD eigenproblem 之前，M2.4 先建立一個
兩端使用相同 locally reacting boundary 的 1D cavity。往返自洽條件為：

\[
F(s)=1-\Gamma(s)^2e^{-2sL/c}=0
\]

求得的 pole 定義為：

\[
s_n=-\gamma_n+j\omega_n
\]

因此：

\[
f_n=\frac{\omega_n}{2\pi},
\qquad
Q_n=\frac{\omega_n}{2\gamma_n}
\]

`solve_1d_impedance_cavity_modes()` 直接在複數 \(s\)-plane 解
\(\operatorname{Re}F=\operatorname{Im}F=0\)。靜態實數反射的回歸測試會
對照解析解：

\[
f_n=\frac{nc}{2L},
\qquad
\gamma_n=-\frac{c}{L}\ln|\Gamma|
\]

另一個測試則固定參考頻率上的 \(|\Gamma|\)，比較 phase-aware 多極點
邊界與 zero-phase 實數邊界，確認兩者產生不同的複數模態頻率。

這個 1D solver 已證明同一份 rational boundary 可以供 FDTD 與非線性
eigenvalue equation 共用，但它不是完整房間求解器。3D 六面材料、
斜入射、非矩形幾何、mode coupling 與 production renderer 的替換仍待
後續完成。

## 6. M2.5 真實資料結果

第一份接受資料是
[Zenodo 15195587](https://zenodo.org/records/15195587) 的 NASA GFIT
無流、130 dB、KT eduction；UFSC nominally identical sample 作獨立
cross-rig 比較。來源 HDF5 是 CC BY 4.0，轉換保留 500–2500 Hz 的原始
normalized resistance/reactance，不做插值。

NASA 的 alternating-frequency fit 得到：

| 參數／gate | 結果 |
|-------------|------|
| resonance \(f_0\) | 1646.66 Hz |
| branch \(Q\) | 11.69 |
| peak normalized admittance | 7.69 |
| training RMS / max complex error | 0.0437 / 0.0815 |
| held-out RMS / max complex error | 0.0385 / 0.0590 |
| dense 4096-point max \(|\mathcal C|\) | 0.9178 |
| NASA–UFSC cross-rig RMS difference | 0.1182 |

held-out error 小於 NASA–UFSC 兩個 nominally identical samples／rigs 的
差異，且 4096 點 dense sweep 全部保持 positive-real 與
\(|\mathcal C|\le1\)。在以 resonance 設定第一軸模態的 1D diagnostic
中，phase-aware RLC boundary 的 \(Q=17.73\)，參考頻率匹配
\(|\mathcal C|\) 的實數 boundary 為 \(Q=6.04\)，比值 2.94。這再次顯示
只匹配 absorption magnitude 不足以預測 modal decay。

可重現報告：

```bash
python egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py \
  --csv egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/nasa_gfit_noflow_130db_kt.csv \
  --metadata egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/nasa_gfit_noflow_130db_kt.json \
  --comparison-csv egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/ufsc_noflow_130db_kt.csv \
  --comparison-metadata egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/ufsc_noflow_130db_kt.json \
  --output egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/impedance_zenodo_15195587_validation.json
```

完整來源接受／拒絕理由見
[`complex_impedance_source_audit.md`](complex_impedance_source_audit.md)。

這份資料是高聲壓穿孔 aircraft liner 的 grazing-duct eduction，只用來
驗證 phase-aware pipeline、共振模型、FDTD 與 eigenproblem。sidecar
明確設為 `automatic_scene_catalog_mapping: false`；它不會被當成一般牆面、
地毯或天花板。

## 7. M2.6 正入射阻抗管 acquisition

第二輪公開來源審核沒有找到可直接接受的 room-finish complex dataset。
因此新增：

- `puresound.impedance_tube_transfer_measurement.v1` 原始 H12 契約；
- repeated \(H_{12}=P(x_2)/P(x_1)\) long-form CSV；
- microphone-switch complex channel calibration；
- 圓管第一 transverse mode cutoff 與
  \(|\sin(ks)|\) microphone-spacing conditioning gate；
- mean coherence、被動性與跨安裝 repeatability uncertainty；
- 阻抗不確定度經
  \(\partial\Gamma/\partial Z=2Z_0/(Z+Z_0)^2\) 傳播後的
  inverse-uncertainty fit weighting；
- 可直接讀回 `ComplexImpedanceMeasurement` 的 reduction CLI。

完整推導、CSV/JSON 格式和實驗 checklist 見
[`impedance_tube_protocol_zh-TW.md`](impedance_tube_protocol_zh-TW.md)。
目前 synthetic H12／複數通道失配／換麥校正 round-trip 已通過，但
room-finish 實體 specimen 尚待量測。

## 8. M2.7 六面 rational impedance 的 3D 模態

矩形房間每一面都可以在 explicit boundary JSON 中指定
`FirstOrderRelaxationAdmittance`、`PassiveMultiPoleAdmittance` 或
`PassiveResonantAdmittance`。程式不從 scene absorption 自動產生這份
JSON。

令 \(s=-\gamma+j\omega\)，每個牆面的 normalized admittance 為
\(y(s)\)。由 momentum equation 得到 Robin boundary：

\[
\partial_n p+\frac{s}{c}y(s)p=0.
\]

沿一個長度為 \(L\) 的軸，令 \(h_\pm=(s/c)y_\pm(s)\)，其 complex
wavenumber \(k\) 必須滿足：

\[
(h_-h_+-k^2)\sin(kL)+k(h_-+h_+)\cos(kL)=0.
\]

三軸還必須共用同一個 temporal pole：

\[
k_x^2+k_y^2+k_z^2+\left(\frac{s}{c}\right)^2=0.
\]

因此不是先算 rigid frequency 再附加 Q，而是同時解四個 complex
equations。這會一起改變 modal frequency、decay 和 complex spatial
eigenfunction。

驗證結果：

- 只有 x 軸兩面有 static admittance 時，3D 解和 exact 1D
  frequency／decay／Q 一致；
- 均勻立方體的三個 axial modes 保持 permutation degeneracy；
- 2.0 × 1.2 × 1.0 m phase-aware controlled case：
  3D 解為 63.8846 Hz、Q 17.32；獨立 FDTD 為 64.0049 Hz、Q 15.78；
- generator 的 `analytic-impedance` backend 已通過完整 CLI smoke。
- fixed-pole modal residue 已用兩個 controlled rooms、四組 training
  positions 與兩組 position holdout 的 independent FDTD 校正；
- position holdout mean correlation 0.9734、NRMSE 0.2304、energy ratio
  1.0039，通過預先固定的 0.90／0.35 gate；
- 校正 JSON 已接入 `analytic-impedance` backend，且完整 generator CLI
  smoke 通過。
- M2.9 加入同一 measured flow resistivity 的 50/100 mm rigid-backed
  thickness variants，並把 position、unseen-room、fine-grid holdout 分開；
- 兩個 thickness variants 的三種 holdout 全部通過 correlation ≥0.90、
  NRMSE ≤0.35 gate；跨兩個 variants 的 shared residue 也通過；
- 這只支持 tested thickness scope，report 仍明確設定
  `general_boundary_invariance_established=false`。

範例 boundary：

`egs/rir_generation/phases/m2_impedance/config/impedance_reference_glass_wool_14kgm3_100mm.json`

它只在 60–300 Hz 有效，且是 measured flow resistivity + Miki model +
one-pole fit；不是直接 complex measurement，也不是一般房間六面全鋪
玻璃棉的 production recipe。

## 9. 驗證與使用邊界

目前自動測試涵蓋：

- CSV/JSON round-trip、uncertainty 與 provenance；
- active、diffuse-incidence 或缺 provenance 資料的拒絕；
- 已知二 pole 合成量測的係數與 complex reflection 回復；
- 兩份真實 normalized impedance 的 provenance 與被動性；
- alternating-frequency train/held-out RLC fitting；
- dense frequency sweep 的 passivity；
- FDTD relaxation、multi-pole、RLC biquad per-wall-cell state；
- 1D 靜態解析模態與 phase-induced frequency shift。
- two-microphone H12 forward/inverse round-trip；
- microphone-switch complex mismatch correction；
- transverse-mode、spacing conditioning 與 coherence rejection；
- repeated-H12 CLI 到 strict complex-impedance contract。
- 3D static-to-1D limit、cube degeneracy 與 dynamic phase-aware root；
- 3D complex mode 對 independent FDTD frequency/Q；
- explicit six-wall config 到 hybrid generator metadata。
- fixed-pole complex residue recovery 與 position／room／grid split gate；
- 50/100 mm controlled thickness protocol 與 shared-parameter diagnostic。

尚未完成：

- train/development/held-out material split；
- 一般室內材料的 normal-incidence、正常聲壓直接量測；
- 不同安裝、air gap 與斜入射的模型；
- 不同實測材料、air gap／mounting 的 residue FDTD 校正與真實房間
  transfer measurement 校正；
- accepted room-finish boundary 的 production mapping；
- room-disjoint measured-bank 與固定下游任務驗證。

## 10. 程式入口

| 檔案 | 職責 |
|------|------|
| `puresound/audio/impedance_measurements.py` | SI 與 normalized CSV/JSON 量測契約 |
| `puresound/audio/impedance_tube.py` | H12、換麥校正、管徑／間距 gate、coherence 與 uncertainty reduction |
| `puresound/audio/acoustic_impedance.py` | 被動 relaxation 與 series-RLC admittance |
| `puresound/audio/impedance_fitting.py` | 固定實 pole 與共軛 RLC pole fitting |
| `puresound/audio/fdtd_reference.py` | 每牆面 cell 的一階／biquad boundary state |
| `puresound/audio/impedance_modes.py` | 1D cavity 與 separable 3D 六面 rational impedance eigenproblem |
| `puresound/audio/impedance_residues.py` | fixed-pole complex residue fitting、position holdout 與版本化 calibration |
| `puresound/audio/hybrid_rir.py` | experimental `ImpedanceModalLowFrequencyBackend` renderer |
| `egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py` | held-out、passivity、cross-rig 與 modal report |
| `egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py` | raw repeated H12 到 strict complex impedance |
| `egs/rir_generation/phases/m2_impedance/scripts/calibrate_impedance_modal_residues.py` | 多房間 FDTD residue calibration 與 holdout report |
| `egs/rir_generation/phases/m2_impedance/scripts/calibrate_impedance_residue_protocol.py` | M2.9 multi-boundary、unseen-room、grid-holdout protocol |
| `test/test_impedance_measurements.py` | 匯入、passivity 與 fitting 測試 |
| `test/test_impedance_tube.py` | H12 round-trip、calibration、gate 與 CLI 測試 |
| `test/test_impedance_modes.py` | 解析模態與 phase shift 測試 |
