# 低頻模態驗證（Low-frequency modal validation）

English version: `modal_validation.md`。完整背景、算法與目前實驗判斷見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md)。

M2 有兩個驗證元件，答案都不是直接沿用 scene 的 metadata：一個獨立的 3D
有限差分（finite-difference）參考解，以及一個壓力響應模態估計器。

## 具相位資訊的阻抗基礎元件（Phase-aware impedance primitives）

`puresound.audio.rir.physics.impedance.admittance` 建立了第一層複數邊界
（complex-boundary）驗證層，單位是 SI 制的 Pa·s/m：

```text
Z0 = rho c
Gamma = (Z - Z0) / (Z + Z0)
alpha = 1 - |Gamma|^2
Z = Z0 (1 + Gamma) / (1 - Gamma)
```

它會強制反射量值（magnitude）必須是被動的（passive），阻抗實部必須非負。
從吸音係數反推阻抗的路徑，需要一個明確給定的反射相位：
`|Gamma| = sqrt(1 - alpha)`。測試驗證了複數反射／阻抗之間的往返轉換、
量值、相位、matched／rigid／pressure-release 極限情況，以及「相同吸音
係數但不同相位會得到不同阻抗」這件事。

Scene schema 現在把 `impedance_real` 與 `impedance_imag` 定義為 Pa·s/m
單位，會驗證頻譜是否被動，經過 JSON 之後仍會保留，並且用面積加權的導納
（admittance）方式，把完整指定的表面 patch 組合起來。這份材質目錄刻意
不去為那些只有吸音係數證據的材質，「發明」出一個阻抗值。

## 被動因果鬆弛邊界（Passive causal relaxation boundary）

第一個時域（time-domain）複數邊界，使用的是一個正規化過的 positive-real
導納：

```text
y(s) = rho*c*Y(s) = g_infinite + g_relaxation / (1 + s*tau)
tau = 1 / (2*pi*f_relaxation)
Z(s) = rho*c / y(s)
Gamma(s) = (1 - y(s)) / (1 + y(s))
```

無限頻率與零頻率的導納都被限制為非負；兩者的差可正可負。這個模型用
bilinear transform 離散化，會把穩定的類比極點（analog pole）映射到數位
單位圓內側。一個單一牆面的 impulse-response 測試，拿數位複數反射跟類比
目標值互相比較。在 16 kHz 取樣率下，整個 0–300 Hz 驗證頻帶（包含鬆弛
頻率本身）的絕對複數誤差，都維持在 0.003 以下。

`simulate_fdtd_reference(..., boundary_admittance=model)` 會在每一個
牆面 cell 上，各自獨立維護鬆弛壓力狀態（relaxation pressure state），
並寫下完整的模型／離散化 metadata。把鬆弛電導（relaxation conductance）
設成零，就會退化回舊版的實數邊界；一個自動化的回歸測試驗證了這種情況
下，兩份 FDTD RIR 會逐 sample 完全一致。非零鬆弛的情境，也會檢查輸出
是否有限、狀態是否穩定，以及 metadata 是否為嚴格合法的 JSON。

這個模型證明了時域邊界機制是可行的。下面提到的參考先驗（reference
priors）都還只是 validation 用途，並沒有接上 production 的 scene 材質，
也沒有接上 pytARD 模態 backend。

## 直接量測、multi-pole 與共振路徑（Direct-measurement, multi-pole, and resonant path）

M2.4 加入一條嚴格的法向入射（normal-incidence）複數阻抗匯入路徑，而不是
把吸音係數當成一種帶相位的量測值來處理。一筆量測資料由頻率、阻抗實部、
阻抗虛部（單位 Pa·s/m）三個欄位組成，外加一份 JSON sidecar，內含量測
方法、環境、樣品設定、來源與授權。選填的實部／虛部標準差，若要提供就
必須同時提供兩者。頻率非遞增、電阻為負、違反被動反射限制、diffuse
incidence（擴散入射），以及缺少 provenance 資訊，都會被拒絕。

匯入的響應資料，會用以下這個 positive-real 的 multi-pole 正規化導納去
擬合：

```text
y(s) = g_static
     + sum(g_low[k]  / (1 + s*tau[k]))
     + sum(g_high[k] * s*tau[k] / (1 + s*tau[k]))
```

所有分支強度都是非負值，所有極點也都是穩定的實數鬆弛極點。因此被動性
（passivity）是由模型結構本身保證的，不需要額外檢查。目前的 fitter 會
固定住這些極點，只用 bounded least squares 去最小化壓力反射的實部與
虛部誤差。它刻意不宣稱自己是會重新配置極點、修復被動性的完整
vector fitting。

FDTD 參考解現在會為每個牆面 cell、每個極點各自維護一個 low-pass 狀態。
High-pass 分支則是重複利用同一個狀態，算成 `p - lowpass(p)`。當已知
極點被直接給定時，一筆合成的 two-pole 量測資料，可以在最大複數反射誤差
`1e-7` 以下被還原；而在 3D 參考解的情境下，一個非零的 multi-pole 邊界
仍然維持被動且輸出有限。

完整的 schema、方程式、注意事項與繁體中文說明，都在
[`impedance_measurements.zh-TW.md`](impedance_measurements.zh-TW.md)。

M2.5 加入第一組有授權（licensed）的直接複數資料集：CC BY 4.0 授權、
針對名義上相同的 NASA 與 UFSC 穿孔襯板（perforated liner），在無氣流、
130 dB grazing-duct 條件下量得的正規化電阻／電抗（resistance/
reactance）。來源數值仍維持無因次的 `Z/(rho*c)` 形式，因為 HDF5 檔案
並沒有交代確切的正規化大氣條件。Metadata 保留了量測幾何資訊，並把兩份
樣品都標為僅供 validation 使用。

量得的 Helmholtz 電抗過零點（reactance crossing），沒辦法用實數鬆弛
極點模型表示，所以加入了一個被動的 series-RLC 導納分支。它擬合出來的
共軛極點，在建構方式上就保證穩定，其 Tustin biquad 也在共振點做了
prewarp。NASA 這組資料的擬合結果，held-out RMS 為 0.0385，held-out
最大複數 Cayley 誤差為 0.0590（跨越交錯的頻率 bin 計算）。一次 4096 點
的掃頻仍然維持被動，共振 FDTD 狀態維持有限，1D 共振／僅量值
（magnitude-only）模態 Q 值比為 2.94。資料來源的取捨與限制，記錄在
[`complex_impedance_source_audit.zh-TW.md`](complex_impedance_source_audit.zh-TW.md)。

## 第一批附有 provenance 的多孔材質參考資料（First provenance-bearing porous references）

`puresound.audio.rir.physics.impedance.priors` 實作了 Miki 在 1990 年
提出、給 rigid-backed 材質層用的 positive-real 多孔材質模型。前兩筆
參考資料，用的是 Tarnow 量測的法向（normal-direction）流阻（flow
resistivity），對象是 100 mm 厚、密度分別為 14 kg/m³ 與 30 kg/m³ 的
玻璃棉。它們的 evidence tier 被明確標為
`measured_flow_resistivity_plus_miki_model`：阻抗相位是用模型算出來的，
並非直接量測而得。

這個單極點（one-pole）FDTD 邊界，是在複數反射空間中擬合出來的。
14 kg/m³ 那組參考資料，在 60–300 Hz 範圍內的最大複數誤差是 0.0571；
30 kg/m³ 那組在 155–300 Hz 範圍內的最大複數誤差是 0.0160。兩者都通過
診斷用的 0.08 門檻。低於實驗有效範圍的外推（extrapolation）會被拒絕。

一個受控的 2.0 × 1.2 × 1.0 m FDTD 案例，拿 14 kg/m³ 這組具相位資訊的
擬合結果，跟一個在 80 Hz 有相同反射量值的實數邊界互相比較。主要峰值從
85.7 Hz、Q 值 10.5，變成 64.0 Hz、Q 值 15.8。這證明了為什麼吸音係數的
量值沒辦法取代相位資訊；但這不是一次真實房間的驗證。

擬合出來的高頻導納，也暴露出一個超出內部 CFL 準則之外的邊緣／角落時間
步（time-step）限制。FDTD 現在會強制一個總和正規化邊界 Courant 限制為
0.9，並且把兩個限制值都序列化下來。完整推導、資料來源與注意事項，都在
[`impedance_priors.md`](impedance_priors.md)。

## 複數模態特徵值參考解（Complex modal eigenvalue reference）

`puresound.audio.rir.physics.impedance.modes` 是 FDTD 所用的同一個
rational boundary，與複數模態特徵值問題之間，最精簡的一個連結。對一個
長度 `L` 的 1D cavity，兩端都是相同的 locally reacting boundary，它
求解的是：

```text
1 - Gamma(s)^2 exp(-2 s L / c) = 0
s = -decay_rate + j*angular_frequency
Q = angular_frequency / (2*decay_rate)
```

一個頻率無關的實數邊界，會在數值精度範圍內，與封閉形式（closed-form）
的 `f_n = n*c/(2*L)` 及 `decay_rate = -(c/L) log(|Gamma|)` 相符。一個
具相位資訊的 two-pole 邊界，相較於一個反射量值相同的實數邊界，也確實
會使模態頻率位移。這驗證了這套非線性求根（nonlinear root）公式的
正確性。

## 可分離的 3D rational-impedance 特徵值問題（Separable 3D rational-impedance eigenproblem）

M2.7 把同一個 rational \(y(s)\) 延伸到一個矩形的 3D 房間。在每一面牆
上：

```text
normal derivative(p) + (s/c) y(s) p = 0
```

對於一個長度為 \(L\) 的軸，定義 \(h_\pm=(s/c)y_\pm(s)\)。它的複數空間
波數（wavenumber）必須滿足：

```text
(h_minus*h_plus - k^2) sin(kL)
    + k(h_minus + h_plus) cos(kL) = 0
```

三個軸共用同一個時間極點（temporal pole），並且還要額外滿足：

```text
kx^2 + ky^2 + kz^2 + (s/c)^2 = 0
```

這個實作會把這四條複數方程式一起求解，並使用邊界強度延續法
（boundary-strength continuation），確保解一直落在所要求的 rigid-mode
分支上。它也會用 bilinear 體積正規化，在聲源與接收端上求出對應的複數
可分離特徵函數值。

驗證結果：

| Gate | 結果 |
|------|--------|
| 僅 x 軸為靜態邊界 | 3D 的頻率、衰減與 Q 皆與精確 1D 解相符 |
| 均勻立方體邊界 | (1,0,0)、(0,1,0)、(0,0,1) 仍維持簡併（degenerate） |
| 受控、具相位資訊的房間 | 特徵值問題解出 63.8846 Hz／Q 17.32 |
| 同一房間、獨立 FDTD | 64.0049 Hz／Q 15.78 |

頻率差異為 0.19%；Q 值差異為 9.8%，落在既有 15% 的受控參考 gate 之內。

`ImpedanceModalLowFrequencyBackend` 把這些極點與特徵函數，接到 hybrid
generator 上。六面牆的邊界設定，來自一份明確版本化的 JSON，絕不是從
diffuse absorption 推斷出來的。這些特徵值就是物理上真正的非線性解。
M2.8 現在可以選擇性載入一份版本化的 fixed-pole FDTD 留數（residue）
校正。它使用精確的 Ricker 激發訊號、FDTD cell 的體積／中心慣例
（convention），在兩個房間裡各取四個訓練位置，並留兩個完全沒碰過的
位置作為 holdout。Holdout 的平均相關係數是 0.9734，NRMSE 是 0.2304；
相較之下，用 fitted-gain 的 legacy `1/f` sine 留數，相關係數只有
0.2467、NRMSE 是 0.9703。

有這份 JSON 時，metadata 會設定 `modal_residue_fdtd_validated: true`；
沒有的話，工程上的 fallback 值就會被明確標示為尚未驗證。不管走哪條
路徑，`modal_residue_production_validated: false` 與
`production_material_mapping_enabled: false` 都會保持不變，直到真實
房間的 transfer function 通過驗證為止。

M2.9 把評估拆成三個獨立的軸，而不是把所有 holdout 混在一起算。針對模型
推導出來、rigid-backed 厚度分別為 50 mm 與 100 mm 的兩個變體：

| Split | 50 mm 相關係數／NRMSE | 100 mm 相關係數／NRMSE |
|-------|----------------------------|-----------------------------|
| 未見過的位置 | 0.9904 / 0.1366 | 0.9734 / 0.2304 |
| 完全未見過的房間 C | 0.9929 / 0.1187 | 0.9933 / 0.1177 |
| 0.12 m 訓練後改用 0.08 m 網格 | 0.9961 / 0.0919 | 0.9929 / 0.1193 |

一次同時橫跨兩種厚度的共用診斷擬合，也通過了全部三個 gate。這證實的是
「同一個量測流阻，在這兩種厚度變體之間可以遷移」，而不是「邊界具有
一般性的不變性（general invariance）」。因此機器產生的報告，仍然保持
`general_boundary_invariance_established: false`。

## 獨立的 FDTD 參考解（Independent FDTD reference）

`puresound.audio.rir.physics.wave.fdtd` 實作了一個小型、僅供
validation 使用的求解器。它在 cell 中心存壓力，在交錯（staggered）
排列的 cell 表面存質點速度（particle velocity）：

```text
rho dv/dt = -grad(p)
dp/dt = -rho c^2 div(v)
```

時間步（time step）遵循三維 CFL 限制。原始的參考案例，給每一面牆一個
locally reacting、頻率無關的實數阻抗：

```text
R = sqrt(1 - alpha)
Z = rho c (1 + R) / (1 - R)
v_normal = p / Z
```

這是刻意跟 `hybrid_rir.py` 裡的 cosine-mode 遞迴式分開的。它是給小型
矩形 scene 用的參考解，不是一個資料集用的 renderer：當空間與時間解析度
一起提高時，它的成本會隨網格體積、時長增加，並以網格間距的四次方反比
成長。上面提到的 `alpha -> Z` 相容路徑，選的是零相位、高阻抗的實數
分支。不應該把它解讀成「從吸音係數還原出唯一阻抗值」。

自動化的參考案例，使用一個 3.0 × 2.5 × 2.0 m 的房間、0.15 m 的標稱
網格間距、吸音係數 0.08，以及一個 band-limited 的 Ricker 訊號源。結果
如下：

| 模態 | Rigid-room 頻率 | FDTD 估計值 | 頻率誤差 |
|------|----------------------|---------------|-----------------|
| (1, 0, 0) | 57.17 Hz | 57.33 Hz | 0.3% |
| (0, 1, 0) | 68.60 Hz | 68.35 Hz | 0.4% |

量得的 half-power Q 值，跟指定牆面反射係數所預期的衰減率，誤差也在
20% 以內。這個容許誤差涵蓋了網格頻散（grid dispersion）、離散邊界
近似、有限觀測時間，以及頻譜重疊等因素。

## RIR 模態峰值與 Q 值估計器（RIR modal peak and Q estimator）

`puresound.audio.rir.physics.wave.low_frequency` 會對一支量測或合成
的 RIR 做時間窗（gate），找出低頻響應中顯著的峰值，並回報：

- 峰值頻率與 prominence（顯著度）；
- half-power 的上下交越點；
- `bandwidth = upper - lower`；
- `Q = peak frequency / bandwidth`；
- 原生的時間窗解析度，會跟做過 zero-padding 之後的 FFT bin 間距分開
  回報。

Zero padding 只是讓交越點的內插更準，並不代表多出額外的物理解析度。
如果某個峰值的 half-power 交越點沒辦法被解析出來，它還是會留在結果裡，
只是 `Q: null`。

一個 80 Hz 的解析（analytic）正弦波，振幅衰減率為 10/s，理論上
`Q = 25.13`；這個估計器實際算出來大約是 25.10。

## Bank 比較（Bank comparison）

對生成出來的 bank 跟量測到的 bank，使用同一套 gate 與估計器：

```bash
python egs/rir_generation/compare_modal_acoustics.py \
  v0=/path/to/v0 \
  m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  m2=egs/rir_generation/exp/rir_realism/m2/rir_m2_probe100 \
  measured=/path/to/measured \
  --per-bank 100 \
  --analysis-duration-s 0.8 \
  --reference-tag measured \
  --json-output egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics.json
```

這份報告包含逐 channel 的峰值資訊，以及整個 bank 在峰值數量、頻率、
prominence、間距、half-power bandwidth 與 Q 值上的分布。Direct
arrival 之後的一段固定延遲，會把激發脈衝（excitation pulse）本身濾掉；
固定的最大 gate 時長，則讓不同 bank 之間的頻譜解析度可以互相比較。

這些都是 response 層級的工程指標，不是自動化的房間模態標籤。彼此相隔
很近的模態、聲源／接收端剛好落在節點上、量測雜訊、非矩形幾何，以及
有限的錄音長度，都可能讓峰值合併、消失或變寬。因此 bank 的驗收標準是
比較分布，而不是要求跟真實房間的解析模態一一對應。

最初那版 64-mode 的 probe 已經被淘汰，因為它漏掉了幾乎所有合成的
200–300 Hz 模態。有了 reciprocal 特徵函數端點耦合，以及完整的預設模態
涵蓋範圍之後，修正後的 M1 bridge 是目前最強的診斷 baseline。在完全
沒有重疊的 heldout 量測項目上，它正規化後的 Wasserstein 距離，Q 值是
0.087、間距是 0.080、bandwidth 是 0.183。目前的 material-modal loss，
則分別是 0.505、0.614、0.364。

一個 scalar 的 material-loss 校正值，能還原 median Q，卻會產生太多
過窄、清晰可見的模態，整體來看還是輸給修正後的 M1 bridge。新的 3D
複數阻抗特徵值，在受控實驗中已經取代了那套阻尼律。這個 renderer 已經
通過了受控的數值 position-holdout 留數驗證，但在有多個安裝好的邊界、
相容的真實房間表面材質、量測到的 transfer function，以及真正
room-disjoint 的比較出現之前，都還算是實驗性質。
