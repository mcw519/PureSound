# 正入射複數阻抗量測與匯入流程

狀態：M2.6 的量測 reduction 已完成；尚待第一批實體 room-finish 樣品資料。

## 1. 我們為什麼要量 H12，而不是只抄吸收率

房間低頻邊界不只會減少能量，也會改變反射相位。只有

\[
\alpha(f)=1-|\Gamma(f)|^2
\]

時，我們只知道反射係數的大小，不知道其相位，因此無法唯一決定表面
阻抗、模態頻率偏移或因果 time-domain boundary。

阻抗管中的兩支固定麥克風會量到複數傳遞函數

\[
H_{12}(f)=\frac{P(x_2,f)}{P(x_1,f)}
\]

它同時保留振幅與相位。ISO 10534-2 的正式適用範圍正是 normal-incidence
absorption 與 normal surface impedance；它和混響室的 diffuse-incidence
吸收率不是同一個量。

## 2. 座標、波動分解與反射係數

程式採用以下明確約定：

- 樣品表面為 \(x=0\)；
- \(x\) 正方向由樣品指向聲源；
- \(x_1,x_2>0\) 是兩支麥克風到樣品的距離；
- 原始傳遞函數固定為 \(H_{12}=P(x_2)/P(x_1)\)；
- phasor convention 固定為 \(\exp(+i\omega t)\)。

管內只有平面波時，

\[
p(x)=A e^{ikx}+B e^{-ikx},
\qquad
k=\frac{2\pi f}{c}.
\]

\(A\) 是往樣品傳播的波，\(B\) 是反射波，樣品表面的複數壓力反射係數
為 \(\Gamma=B/A\)。由兩點壓力消去 \(A,B\)：

\[
\Gamma=
\frac{e^{ikx_2}-H_{12}e^{ikx_1}}
{H_{12}e^{-ikx_1}-e^{-ikx_2}}.
\]

接著由空氣特性阻抗 \(Z_0=\rho c\) 得到：

\[
Z_s=Z_0\frac{1+\Gamma}{1-\Gamma}.
\]

這個 \(Z_s(f)\) 才是後續被動 rational fit、FDTD boundary 與複數模態
eigenproblem 的共同輸入。

## 3. 麥克風交換校正

兩個麥克風／量測通道不會有完全相同的複數靈敏度。令其失配為 \(C(f)\)，
同一校正聲場在正常位置與交換麥克風後分別量到：

\[
H_\mathrm{I}=C H,\qquad H_\mathrm{II}=C/H.
\]

因此：

\[
C(f)=\sqrt{H_\mathrm{I}(f)H_\mathrm{II}(f)},
\qquad
H_{12,\mathrm{corrected}}=H_{12,\mathrm{raw}}/C.
\]

實作會先 unwrap \(H_\mathrm{I}H_\mathrm{II}\) 的 phase，再取連續的複數
平方根，避免頻率間突然切換正負 branch。production metadata 預設要求
提供這份交換校正；不能把「兩支同型號麥克風」當成已校正。

## 4. 有效頻帶 gate

不是 FFT 中每個 bin 都可用。程式同時檢查：

### 4.1 圓管第一橫向模態

直徑 \(D\) 的剛性圓管，其第一個非平面模態 cutoff 約為：

\[
f_\mathrm{plane,max}
=\frac{1.841c}{\pi D}.
\]

選定頻帶必須完全低於這個頻率。

### 4.2 麥克風間距病態點

若麥克風間距 \(s=|x_1-x_2|\)，當

\[
|\sin(ks)|
\]

太小時，兩點幾乎無法區分入射波與反射波，反算會放大量測誤差。目前
預設要求整個選定頻帶的 \(|\sin(ks)|\ge0.05\)。這是數值 conditioning
gate，不是拿來取代實驗室依標準決定頻帶的程序。

## 5. 品質、不確定度與被動性

每個樣品至少做多次「拆下、重裝、再量」；只重播訊號而不重新安裝，
無法量到 sealing、壓縮量與邊緣漏氣的不確定度。

目前 reduction：

1. 要求所有 repeat 使用相同的 frequency grid；
2. 預設每個頻率的平均 magnitude-squared coherence 不得低於 0.95；
3. 對每次 sweep 分別計算 \(\Gamma\) 與 \(Z_s\)；
4. 輸出 \(Z_s\) 實部、虛部的跨 repeat sample standard deviation；
5. 要求平均 \(\operatorname{Re}Z_s\ge0\) 且
   \(|\Gamma|\le1\)；超出容差就拒收，不悄悄 clipping；
6. rational fit 會把阻抗標準差經
   \[
   \frac{\partial\Gamma}{\partial Z}
   =\frac{2Z_0}{(Z+Z_0)^2}
   \]
   傳播到 reflection domain，再做 inverse-uncertainty weighting。

若只有一次 sweep，仍可輸出阻抗，但不宣稱具有 repeatability uncertainty，
也不應通過 production material gate。

## 6. 原始資料契約

原始長格式 CSV：

```csv
repeat_id,frequency_hz,h12_real,h12_imag,coherence
install_01,200,0.91,-0.13,0.997
install_01,250,0.88,-0.17,0.998
install_02,200,0.90,-0.14,0.996
install_02,250,0.87,-0.18,0.997
```

麥克風交換校正 CSV：

```csv
frequency_hz,h12_original_real,h12_original_imag,h12_swapped_real,h12_swapped_imag
200,1.02,0.03,1.01,0.02
250,1.02,0.03,1.01,0.02
```

JSON sidecar 的 schema 為：

```text
puresound.impedance_tube_transfer_measurement.v1
```

必填內容包括：

- 空氣密度、聲速、溫度、相對濕度與氣壓；
- 圓管直徑和兩個麥克風到樣品表面的距離；
- 分析頻帶、最低 coherence、source SPL 與校正要求；
- sample id、材料名稱、厚度、mounting、backing 與 air gap；
- source URL／實驗記錄位置和 license；
- `automatic_scene_catalog_mapping`，在驗收完成前必須為 `false`。

可複製的範本位於
`egs/rir_generation/phases/m2_impedance/measurements/impedance_tube_template/`。

## 7. 執行 reduction

```bash
.venv/bin/python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py \
  --transfer-csv path/to/raw_h12.csv \
  --metadata path/to/raw_h12.json \
  --microphone-switch-csv path/to/microphone_switch.csv \
  --output-csv path/to/complex_impedance.csv \
  --output-metadata path/to/complex_impedance.json
```

輸出符合既有
`puresound.complex_impedance_measurement.v1`，可以直接交給
`fit_complex_impedance_measurement()`。原始檔 SHA-256、校正檔 SHA-256、
頻帶 gate、repeat ids、coherence、座標與 transformation 都會保留在
sidecar。

## 8. 第一批實驗建議

第一批不要同時測十幾種未知材料。先選一個安裝方式清楚、可重複切樣的
常見多孔吸音材，例如 50 mm 或 100 mm 厚玻璃棉／岩棉，剛性背板、無
air gap：

1. 至少三個獨立 specimen；
2. 每個 specimen 至少三次重新安裝；
3. 量測 75 與 85 dB 兩個線性聲壓級，檢查 level dependence；
4. 保存厚度公差、密度／面密度、批次、裁切直徑與 perimeter sealing；
5. 先以 microphone-spacing 和 tube cutoff 的交集選定頻帶；
6. fit 後做 held-out specimen，而不是只做 held-out frequency；
7. 通過 reflection magnitude/phase、被動性、FDTD 和 1D mode gate 後，
   才能把完全相同的 installed configuration 映射到 scene catalog。

目前還沒有這批實體量測，因此 M2.6 完成的是「可信地取得下一份資料的
算法與介面」，不是宣稱已經取得一般房間材料的 ground truth。

## 9. 目前 reduction 的已知邊界

- 管內傳播目前使用實數 \(k=2\pi f/c\)，尚未加入窄管中的 thermoviscous
  propagation loss correction；
- uncertainty 目前來自 specimen／重新安裝 repeats，尚未把麥克風位置、
  空氣參數和 calibration spectrum 的誤差做完整 Monte Carlo propagation；
- 高 coherence 只表示線性頻譜估計穩定，不等於沒有邊緣漏氣、樣品壓縮、
  lateral constraint 或 sample-to-sample variability；
- 現在只支援圓管與 normal plane incidence；
- normal-incidence locally reacting impedance 進入房間模型後，仍需另外
  處理 oblique incidence、finite patch 與非局部反應。

因此第一批實測還要保留 raw spectra；未來加入 tube-loss 或幾何修正時，
必須從 raw H12 重新 reduction，不能只留下最後的 absorption curve。
