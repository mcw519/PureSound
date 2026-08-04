# 更逼真的 RIR：PureSound 的物理建模與驗證算法

狀態：M1、M3、M4.1–M4.6、M5.1–M5.6 與 M6.1–M6.6 implementation 已完成；M6 production promotion 因 empirical evidence 而 BLOCKED，M5 受控 empirical exit、M4 empirical exit 與 M2 phase-aware production mapping 仍在研究中
最後更新：2026-08-01
適用對象：聲學、語音資料、模型訓練與 RIR 生成程式的開發者

這份文件用繁體中文說明 PureSound 正在解決什麼問題、目前的 RIR
生成器如何運作、各項公式代表什麼，以及現有實驗支持或否決了哪些
假設。完整里程碑與驗收條件見
[`RIR_REALISM_PLAN.md`](../../RIR_EXP_LOG.md)。

一句話版本：我們正在打造一個 **material-first、位置相關、保持因果、
可被量測反駁** 的混合 RIR 生成器。M1 已建立場景與材料基礎；M2 已修正
模態耦合和數值阻尼，但第一版「吸收係數直接轉模態損失」的假設未通過
真實資料比較。現在已建立低頻複數阻抗契約、納入第一份真實
normalized complex liner 資料、以被動 RLC 共振模型通過 held-out
frequency gate，並接上 FDTD 與 1D／3D 複數模態。M3 已完成可序列化
PathEvent、因果 angle-aware reflection、家具 visibility、穿透、edge
diffraction、受控散射與 measured C50／early-timing exit。M4.1–M4.2 已建立
echo density／mixing time、noise-aware octave target 與 spatial contract；
M4.3 已完成獨立 deterministic multiband FDN core；M4.4 已以明確 opt-in
方式將它與 PathEvent direct/early response 因果耦合；M4.5 已讓同一聲場
同步投影到 multi-receiver 與 ACN/SN3D Ambisonics；M4.6 已加入接收器
一階指向性與可注入 HRTF FIR 的 BRIR decoder。M4 的程式與 model-derived
gate 已完成，但 measured multi-receiver、licensed HRTF calibration 與受控
聽測仍是獨立的 empirical／production exit，不能由合成 fixture 代替。
M5.1 已凍結 repeated-ESS 受控量測 schema、room-disjoint split 與
multi-objective inverse-calibration loss。現有 measured bank 仍可作 acoustic
distribution reference，但因缺 raw sweep、transducer calibration、geometry／
pose／environment 與同步 receiver provenance，尚不能宣稱已開始真實房間
inverse fitting。
M5.2 已通過 noise-free parameter recovery；M5.2b 再加入已知 gain／latency
nuisance、32–38 dB SNR 與未知 early／late mismatch，並以 noise-aware
M4-proxy objective 通過 held-out-position gate。M5.2c 已把 mixing time、
aggregate coherent-path gain 與 octave RT60 接到真正的 PathEvent＋multiband
FDN coupling；逐牆材料／scattering 識別與 measured-room fit 仍未完成。
M5.2d 已找出六面 effective reflection 可辨識、mono absorption/scattering
不可分；M5.3–M5.6 的 readiness-gated measured runner、同步 spatial profile、
constrained residual 與 aggregate exit 皆已完成。因現有資料不符合 repeated-ESS
campaign 契約，M5 implementation PASS 與 empirical／production OPEN 必須分開。
M6.1 已進一步凍結 production bank manifest、deterministic acoustic-space split、
asset／scene／manifest hashes 與 renderer evidence tier；M6.2 已接上實際
serial／parallel／resume generator，M6.3 再加入 per-item QC/quarantine；M6.4
凍結 distribution／variant／recipe，M6.5 則凍結 acoustic、listening 與 downstream
證據契約；M6.6 再用 immutable certificate 做最終 production decision。

## 1. 我們正在做什麼

RIR（Room Impulse Response，房間脈衝響應）可以視為一個聲音空間的
「指紋」。若乾聲為 \(x(t)\)，RIR 為 \(h(t)\)，在此空間中收到的聲音
可寫成：

\[
y(t) = x(t) * h(t)
\]

其中 \(*\) 是卷積。RIR 不只是一條具有殘響尾巴的波形；它同時描述：

- 聲音從音源到麥克風的傳播時間與距離衰減；
- 牆面、地板、天花板和物件造成的早期反射；
- 房間尺寸造成的低頻共振模態；
- 材料吸收、散射和空氣吸收造成的頻率相依衰減；
- 後期殘響的時間、頻譜和空間特性；
- 音源與接收器所在位置如何改變上述結果。

我們的目標不是讓合成 RIR「看起來更亂」，也不是先抽一個 RT60 再把
任意波形乘上一條衰減包絡。目標是讓可觀察到的 RT60、DRR、C50/C80、
低頻峰值和 Q 值，盡可能由下列物理原因自然產生：

```text
房間幾何 + 六個邊界的材料 + 環境 + 音源/接收器位置
    -> 低頻波動與房間模態
    -> 中高頻直接聲與反射路徑
    -> 因果的跨頻段合成
    -> RIR 與完整場景 metadata
    -> 與真實量測使用相同算法比較
```

最終用途是縮小 synthetic-to-real gap：用合成 RIR 訓練的語音模型，
應該能在未見過的真實房間中表現得更好。聲學指標相似是必要的診斷，
但不是最終成功條件；最終仍需要 room-disjoint 的下游任務驗證。

## 2. 為什麼採用混合解法

完整三維波動求解能描述繞射、干涉和模態，但全頻段、長時間、高解析度
的計算成本太高。幾何聲學能高效率地追蹤直接聲和反射，卻在波長接近
房間尺度的低頻區失效。因此目前採用 hybrid RIR：

| 頻段 | 方法 | 主要負責的現象 |
|------|------|----------------|
| 20 Hz 至 crossover | pytARD 模態波動求解；analytic 只作快速探針 | 低頻模態、干涉、位置耦合、模態衰減 |
| crossover 至 Nyquist | Pyroomacoustics 幾何聲學 | 直接聲、鏡像反射、射線尾場、空氣吸收 |
| crossover 附近 | 四階 Linkwitz–Riley 分頻；未校正 backend 才使用有界能量 bridge | 因果、平滑地合成兩個 backend |

目前預設 crossover 為 1000 Hz。這不是宣稱 1000 Hz 以下的所有現象
都必須由波動法處理，而是現有速度、網格解析度與高頻 backend 能力之間
的工程選擇。任何 crossover 變更都必須重新檢查因果性、頻率響應和能量。

## 3. 場景是聲學原因，不是結果

M1 引入 `RoomSceneV2`。一個 v1 場景會記錄：

- 房間長、寬、高；
- west、east、south、north、floor、ceiling 六個邊界；
- 各邊界材料的 octave-band 吸收、散射與穿透特性；
- 門窗或局部材料 patch；
- 家具幾何與材料；
- 溫度、濕度與聲速；
- 音源、麥克風的位置、增益和 transducer ID；
- 隨機種子與生成設定。

重要原則是 **material first**。在 v1 中，RT60 是生成 RIR 後量到的結果，
不是房間的主要輸入參數。設定的材料與幾何必須先決定聲音如何損失能量，
再由生成的 RIR 計算 realized RT60。

`calibrated` 輸出模式保留跨場景的相對增益；`peak_normalized` 會把每條
RIR 的峰值個別正規化，適合相容舊資料配方，但會破壞距離和材料造成的
絕對增益關係。物理驗證原則上使用 `calibrated`。

場景格式、材料先驗與序列化欄位詳見
[`rir_scene_v2.md`](rir_scene_v2.md)。

## 4. 低頻算法：從波動方程到房間模態

### 4.1 第一性原理

在均勻、靜止、線性的空氣中，小訊號聲壓 \(p(\mathbf{x}, t)\) 服從：

\[
\nabla^2 p -
\frac{1}{c^2}\frac{\partial^2 p}{\partial t^2}
= s(\mathbf{x}, t)
\]

其中 \(c\) 是聲速，\(s\) 是音源。邊界條件決定聲波到達牆面後有多少
能量被反射、吸收，以及反射相位如何改變。

對矩形房間先採用剛性牆近似，壓力特徵函數為：

\[
\phi_{\mathbf{n}}(x,y,z) =
\cos\left(\frac{n_x\pi x}{L_x}\right)
\cos\left(\frac{n_y\pi y}{L_y}\right)
\cos\left(\frac{n_z\pi z}{L_z}\right)
\]

對應的模態頻率為：

\[
f_{\mathbf{n}} =
\frac{c}{2}
\sqrt{
\left(\frac{n_x}{L_x}\right)^2 +
\left(\frac{n_y}{L_y}\right)^2 +
\left(\frac{n_z}{L_z}\right)^2
}
\]

\(\mathbf{n}=(n_x,n_y,n_z)\) 是非負整數三元組，DC 模態
\((0,0,0)\) 不當作可傳播的聲學共振。公式顯示房間尺寸直接決定低頻
共振位置；因此只調 RT60 無法修正錯誤的模態頻率。

### 4.2 音源與接收器的模態耦合

一個模態是否在 RIR 中清楚可見，不只取決於房間是否存在該模態，也取決
於音源和麥克風是否位於其節點。沿單一維度的 cosine norm 為：

\[
I_x =
\begin{cases}
L_x, & n_x=0 \\
L_x/2, & n_x>0
\end{cases}
\]

\(I_y\) 與 \(I_z\) 同理。模態在 RIR 中的耦合強度與下式成正比：

\[
G_{\mathbf{n}} \propto
\frac{
\phi_{\mathbf{n}}(\mathbf{x}_s)
\phi_{\mathbf{n}}(\mathbf{x}_r)
}{
I_x I_y I_z \, \omega_{\mathbf{n}}
}
\]

其中 \(\mathbf{x}_s\) 是音源位置、\(\mathbf{x}_r\) 是接收位置，
\(\omega_{\mathbf{n}}=2\pi f_{\mathbf{n}}\)。程式中等價地使用
\((2-\delta_{n_x0})(2-\delta_{n_y0})(2-\delta_{n_z0})\) 表示
inverse volume norm，並以 \(1/\omega_{\mathbf{n}}\) 表示 impulse response
的頻率權重。

這個算法具有三個必須維持的性質：

1. **節點抑制**：音源或麥克風在模態節點上時，該模態應消失或顯著減弱。
2. **互易性**：交換音源和麥克風後，無方向性系統的 RIR 應保持相同。
3. **因果性**：模態尾場從幾何直接到達時間之後才開始，不能在直接聲前出現。

早期 analytic probe 曾用位置/rank heuristic 設振幅與相位，並只保留
頻率最低的 64 個模態。這會漏掉大部分 200–300 Hz 共振，相關報告已作廢。
修正後枚舉每軸索引 0 至 5，最多保留 256 個模態；這足以容納 215 個
可能的非 DC 三元組。

### 4.3 每個模態的時間演化

將波動方程投影到特徵函數後，每個模態可視為獨立的受迫阻尼振盪器：

\[
\ddot q_{\mathbf{n}}
+ 2\gamma_{\mathbf{n}}\dot q_{\mathbf{n}}
+ \omega_{\mathbf{n}}^2 q_{\mathbf{n}}
= f_{\mathbf{n}}(t)
\]

\(\gamma_{\mathbf{n}}\) 是**振幅**衰減率。自由響應的振幅包絡為
\(e^{-\gamma t}\)，能量則以 \(e^{-2\gamma t}\) 衰減。因此：

\[
RT60_{\mathbf{n}} =
\frac{\ln(1000)}{\gamma_{\mathbf{n}}},
\qquad
Q_{\mathbf{n}} =
\frac{\omega_{\mathbf{n}}}{2\gamma_{\mathbf{n}}}
\]

這裡的 RT60 是振幅下降 60 dB，也就是降到 \(1/1000\) 所需的時間。
Q 越大，頻譜峰越窄、模態維持越久。

### 4.4 目前實作的材料邊界參與模型

M2 的第一個假設是：把六面材料在模態頻率上的 diffuse-field absorption
\(\alpha\) 內插出來，再依模態在各組牆面的參與量分配損失：

\[
P_{\mathbf{n}} =
\frac{\alpha_{west}+\alpha_{east}}{I_x}
+ \frac{\alpha_{south}+\alpha_{north}}{I_y}
+ \frac{\alpha_{floor}+\alpha_{ceiling}}{I_z}
\]

\[
\gamma_{\mathbf{n}} =
s_{loss}\frac{cP_{\mathbf{n}}}{8}
\]

直覺上，某一維的模態索引非零時，其 cosine norm 是 \(L/2\)，對該對牆面
的參與量會加倍。因此不同材料會對不同模態造成不同衰減，而不是所有頻率
共用一條 RT60 包絡。

\(cP/4\) 是一階 Sabine 能量損失率；振幅的指數衰減率是其一半，所以
使用 \(cP/8\)。`s_loss` 預設為 1.0，只是一個診斷旋鈕，不是物理常數。

這套映射已完成程式、metadata 與測試，但**目前實驗不支持它成為 production
模型**。主要原因是材料表中的 diffuse-field absorption 只提供反射能量
大小，沒有低頻複數阻抗的相位資訊，也沒有完整描述入射角、安裝方式、
板材共振與牆後空腔。第 9 節列出實驗證據。

### 4.5 精確取樣的阻尼 recurrence

對取樣間隔 \(\Delta t\)，令：

\[
r=e^{-\gamma\Delta t},\qquad
\omega_d=\sqrt{\omega^2-\gamma^2}
\]

離散極點與更新式為：

\[
a_1=2r\cos(\omega_d\Delta t),\qquad
a_2=-r^2
\]

\[
b=\frac{1-a_1-a_2}{\omega^2}
\]

\[
q[k+1]=a_1q[k]+a_2q[k-1]+bf[k]
\]

這個 recurrence 直接放置連續阻尼振盪器的取樣極點。當
\(\gamma=0\) 時，它會精確退化為原本 pytARD 的無阻尼 recurrence，
因此我們可以把「數值更新是否正確」和「材料如何映射成 \(\gamma\)」
分開驗證。

目前結論是：

- exact damped recurrence：保留，屬於已測試的數值基礎建設；
- boundary participation 計算：保留作研究基礎；
- diffuse absorption \(\rightarrow\gamma\) 的現行公式：未通過驗證；
- 全域 `s_loss=0.58` 校正：已否決，不可當 production 常數。

更完整的推導見 [`modal_damping.md`](modal_damping.md)。

## 5. 中高頻算法：直接聲、早期反射與尾場

中高頻 backend 使用 Pyroomacoustics，主要工作為：

1. 由音源到麥克風建立直接路徑；
2. 用 image-source method 產生可辨識的早期反射；
3. 用 ray tracing 補充高階、較擴散的反射尾場；
4. 對六個邊界使用頻率相依的材料吸收與散射；
5. 套用空氣吸收；
6. 將家具近似為高度與位置相關的遮蔽及散射。

目前材料 patch 會進入有效邊界材料計算，但幾何 backend 尚未把每個 patch
全部建成獨立的反射網格；家具也不是完整波動散射物。這些是已知的模型
邊界，不能把它們誤認為精確的 CAD 聲學模擬。

低頻與高頻響應分別經過互補的四階 Linkwitz–Riley filter。只有尚未具備
共同 source convention 的舊 backend，才在 crossover 附近使用有界的
RMS energy bridge；M2.10 impedance calibration 不再套用這個 gain。組合後
必須檢查：

- 直接聲之前沒有能量；
- crossover 附近沒有明顯凹洞或能量突增；
- 更換 backend 不會偷偷改變輸出正規化；
- calibrated 模式下，距離與材料的相對增益仍有意義。

## 6. 獨立 FDTD 參考解

若只用另一套 cosine recurrence 驗證 modal solver，兩者可能共享同一個
錯誤。`puresound.audio.fdtd_reference` 因此實作一個小型、validation-only
的三維 staggered-grid FDTD：

\[
\rho\frac{\partial\mathbf{v}}{\partial t}=-\nabla p
\]

\[
\frac{\partial p}{\partial t}=-\rho c^2\nabla\cdot\mathbf{v}
\]

聲壓位於 cell center，粒子速度位於 cell face；時間步長遵守三維 CFL
穩定條件。現行 reference 將 absorption 轉成純實數的 locally reacting
impedance：

\[
R=\sqrt{1-\alpha}
\]

\[
Z=\rho c\frac{1+R}{1-R},\qquad v_n=\frac{p}{Z}
\]

它只保留正的反射振幅，沒有反射相位，因此是目前材料近似的獨立數值檢查，
還不是最終的真實牆面模型。

固定參考案例為 3.0 × 2.5 × 2.0 m、網格約 0.15 m、各面
\(\alpha=0.08\)。結果如下：

| 模態 | 剛性矩形房頻率 | FDTD 估計 | 頻率誤差 |
|------|----------------|-----------|----------|
| (1, 0, 0) | 57.17 Hz | 57.33 Hz | 0.3% |
| (0, 1, 0) | 68.60 Hz | 68.35 Hz | 0.4% |

half-power Q 與牆面反射係數預期的衰減相差也在 20% 內。這證明基礎模態
頻率、阻尼 recurrence 和測量算法在簡單條件下彼此一致；它**不證明**
現有 absorption-to-loss 映射能代表真實材料。

詳見 [`modal_validation.md`](modal_validation.md)。

## 7. 我們如何量 RIR

合成與真實 RIR 必須通過同一套分析程式，否則比較沒有意義。

### 7.1 時域與殘響指標

直接到達點預設取 RIR 的絕對峰值。DRR 使用直接窗內與窗外的能量比：

\[
DRR=10\log_{10}\frac{E_{direct}}{E_{reverberant}}
\]

C50 與 C80 分別以直接到達後 50 ms、80 ms 為分界：

\[
C_T=10\log_{10}
\frac{\sum_{0\le t<T}h^2(t)}
{\sum_{t\ge T}h^2(t)}
\]

Schroeder energy decay curve 由能量反向積分：

\[
EDC[n]=\sum_{k=n}^{N-1}h^2[k]
\]

再對 dB 曲線做線性擬合：

| 指標 | 擬合範圍 | 外推 |
|------|----------|------|
| EDT | 0 至 -10 dB | 外推到 60 dB |
| T20 | -5 至 -25 dB | 外推到 60 dB |
| T30 | -5 至 -35 dB | 外推到 60 dB |

真實 RIR 尾端常含底噪。直接對整段反向積分會把噪音誤認為極長的殘響，
所以程式使用 Lundeby-style 修正：

1. 以 10 ms block 計算平均平方聲壓；
2. 從最後 20% 的 blocks 估計穩態噪音；
3. 對高於噪音至少 10 dB 的衰減區擬合；
4. 反覆估計 decay/noise intersection；
5. 在 intersection 截斷積分並扣除預期噪音能量。

若動態範圍小於 15 dB、尾噪不穩定或找不到可靠交點，程式保留 raw curve
並在 metadata 記錄原因，不會硬產生看似有效的 T30。

完整定義見 [`rir_metrics.md`](rir_metrics.md)。

### 7.2 低頻 peak 與 Q

低頻分析不是直接拿房間尺寸公式去標註「應該有幾個模態」，而是從實際
RIR 響應找可觀察到的頻譜峰：

1. 找直接到達；
2. 從直接到達後 20 ms 開始取固定 0.8 s gate；
3. 分析 35–300 Hz；
4. 找出符合 prominence 條件的峰；
5. 找峰值兩側的 -3 dB half-power crossing；
6. 計算 \(B=f_{high}-f_{low}\) 與 \(Q=f_{peak}/B\)。

若 crossing 因頻譜重疊或時間解析度不足而無法確定，峰仍保留，但 Q 設為
`null`。FFT zero padding 只用來改善 crossing 的內插，不宣稱增加真實
的頻率解析度。

量測房間可能不是矩形，鄰近模態會合併，位置節點會隱藏峰，噪音也會改變
prominence。因此 bank 驗證比較的是分布，不要求真實房間與矩形公式逐峰
一一對應。

### 7.3 分布距離

對每個 bank 計算 peak count、peak frequency、prominence、peak spacing、
half-power bandwidth 與 Q 的分布。主要距離使用一維 Wasserstein distance：

\[
W_1(P,Q)=\int_0^1
\left|F_P^{-1}(u)-F_Q^{-1}(u)\right|du
\]

為了讓不同單位可比較，再除以 measured distribution 的 IQR：

\[
d_{norm}=\frac{W_1(P_{synthetic},P_{measured})}
{\operatorname{IQR}(P_{measured})}
\]

這個值越小越接近量測分布，但不能單獨決定模型好壞。若 measured IQR
很小、樣本來自相同房間，或多個 channel 不獨立，數字會過度樂觀。

## 8. 目前生成流程的算法

概念上的單一場景生成流程如下：

```text
輸入：seed、room recipe、sample rate、duration、backend 設定

1. 由 seed 取樣 RoomSceneV2
   - 房間幾何
   - 六面材料頻譜與局部 patch
   - 家具、環境、音源、麥克風

2. 計算環境聲速與幾何直接到達時間

3. 生成低頻響應
   - 建立矩形房間的模態/ARD basis
   - 投影 source excitation 與 receiver observation
   - 若使用 experimental material backend：
       由 boundary participation 計算每模態 gamma
       用 exact damped recurrence 更新
   - 否則使用 M1 bridge 作對照

4. 生成中高頻響應
   - image sources 產生直接聲與早期反射
   - ray tracing 產生較高階尾場
   - 套用材料、散射與空氣吸收

5. 用 Linkwitz–Riley crossover 合成 low/high
   - crossover band 能量匹配
   - 保持因果與有限增益

6. 依 output mode 校準或 peak normalize

7. 輸出多聲道 WAV 與 JSON metadata

8. 使用共同分析器計算 realized metrics
```

其中 analytic backend 是快速、可解釋的結構探針，不是 pytARD production
wave backend 的替代品。它適合測試節點、互易性、因果性和模態覆蓋，
不適合用來宣稱完整的低頻聲場精度。

## 9. M2 實驗結果與目前結論

最新 controlled probe 對每個 variant 使用完全相同的 20 個房間 × 5 個
位置。量測資料 development 與 item-heldout 使用不同 seed，選中的 item
沒有重疊；但目前尚未證明它們來自不同實體房間。

item-heldout 的中位數如下：

| Variant | peaks/channel | Q | bandwidth | spacing | prominence |
|---------|---------------|---|-----------|---------|------------|
| 修正後 M1 bridge | 11 | 38.58 | 4.21 Hz | 14.77 Hz | 14.01 dB |
| M2 material loss，scale 1.0 | 8 | 22.24 | 6.06 Hz | 23.56 Hz | 12.89 dB |
| M2 material loss，scale 0.58 | 13 | 38.96 | 3.50 Hz | 15.50 Hz | 13.64 dB |
| measured | 6 | 42.58 | 3.48 Hz | 12.57 Hz | 14.07 dB |

相對 measured IQR 正規化後的 Wasserstein distance：

| Variant | Q | spacing | bandwidth | peak count |
|---------|---|---------|-----------|------------|
| 修正後 M1 bridge | 0.087 | 0.080 | 0.183 | 0.280 |
| M2 material loss，scale 1.0 | 0.505 | 0.614 | 0.364 | 0.325 |
| M2 material loss，scale 0.58 | 0.304 | 0.183 | 0.270 | 0.387 |

`scale=0.58` 是由 development aggregate 估出。它能把 Q 的中位數拉回來，
卻產生太多窄而可見的模態，joint distribution 仍不如修正後的 M1 bridge。
這表示單一全域 scalar 只能移動某個 summary statistic，無法修正錯誤的
loss physics。

所以目前的判斷是：

### 已接受

- 以矩形 eigenfunction 計算 source/receiver endpoint coupling；
- 模態節點、互易性與 causal onset；
- 完整的預設模態覆蓋，不再截斷為前 64 個；
- exact damped recurrence；
- boundary participation 與每模態 metadata 作為後續研究基礎；
- FDTD 與 response-level modal estimator 作為獨立驗證工具。

### 已否決或尚未接受

- 把 diffuse absorption 直接透過 \(cP/8\) 當成真實低頻 modal loss；
- 以全域 `material_modal_loss_scale=0.58` 作 production 校正；
- 用 item-disjoint 結果宣稱已通過 room-disjoint 驗收；
- 把 analytic probe 當 production wave solver。

當前最強的診斷 baseline 是**修正後的 M1 bridge**。M2 不是「已完成但需要
微調」；更精確地說，M2 已建立可驗證的數值基礎，並用實驗否決了第一個
材料損失假設。

完整 development/heldout 數據與決策記錄見
[`RIR_REALISM_PLAN.md` 第 10 節](../../RIR_EXP_LOG.md#10-m2-modal-validation-and-corrected-probe)。

## 10. 下一個算法：低頻複數阻抗

下一步不再尋找另一個全域 damping scalar，而是讓邊界在低頻具有複數、
頻率相依的阻抗：

\[
Z(f)=R(f)+jX(f)
\]

法向入射時，複數壓力反射係數為：

\[
\Gamma(f)=\frac{Z(f)-\rho c}{Z(f)+\rho c}
\]

其中：

- \(|\Gamma|\) 決定反射後保留多少能量；
- \(\angle\Gamma\) 決定反射相位；
- \(R(f)\) 描述耗散；
- \(X(f)\) 描述牆面質量、彈性、空腔等 reactive behavior。

### 10.1 已完成：複數阻抗的最小可驗證核心

目前已新增 `puresound.audio.acoustic_impedance`，以 Pa·s/m 為唯一阻抗
單位，實作並測試：

- 空氣特性阻抗 \(Z_0=\rho c\)；
- \(Z\rightarrow\Gamma\) 與 \(\Gamma\rightarrow Z\)；
- \(\alpha=1-|\Gamma|^2\)；
- matched、rigid 與 pressure-release 三個極限；
- 被動邊界的 \(\operatorname{Re}(Z)\ge0\) 與 \(|\Gamma|\le1\)；
- complex reflection 的 magnitude、phase 與 round-trip；
- 相同 absorption、不同 phase 必須得到不同的 impedance。

由 absorption 建立阻抗的 API 強制要求另外傳入 reflection phase：

\[
|\Gamma|=\sqrt{1-\alpha},\qquad
\Gamma=|\Gamma|e^{j\phi}
\]

這個設計刻意禁止程式把 absorption-only 資料誤當成唯一的低頻阻抗。
`SurfaceMaterial.impedance_real/imag` 現在明確使用 Pa·s/m，會檢查被動性、
頻率中心、對數頻率內插及 JSON round-trip。當 base wall 與所有 patch
都有阻抗時，有效邊界以等壓平行支路的 admittance 作面積混合：

\[
Y_{eff}=\sum_i\frac{a_i}{Z_i},\qquad Z_{eff}=\frac{1}{Y_{eff}}
\]

只要任一 component 缺少阻抗，有效阻抗就保持 unknown；不從 absorption
補猜 phase。現有 real-boundary FDTD 的 absorption helper 也已明確標記為
「zero-phase、high-impedance branch」的相容路徑。

### 10.2 已完成：一階被動、因果的 time-domain boundary

M2.2 新增一個研究用的一階 relaxation admittance：

\[
y(s)=Z_0Y(s)
=g_\infty+\frac{g_r}{1+s\tau},
\qquad
\tau=\frac{1}{2\pi f_r}
\]

其中 \(y\) 是無因次 normalized admittance、\(Z_0=\rho c\)。模型要求
高頻端 \(g_\infty\ge0\)，以及低頻端
\(g_0=g_\infty+g_r\ge0\)；\(g_r\) 本身可以是正或負。所有中間頻率的
real admittance 都位於兩個非負端點之間，因此模型維持 positive-real：
它不主動產生能量，且 pole 位於穩定的左半平面。低頻與高頻極限分別為：

\[
y(0)=g_\infty+g_r,\qquad y(\infty)=g_\infty
\]

阻抗和反射可由同一個 \(y\) 得到：

\[
Z(f)=\frac{Z_0}{y(f)},\qquad
\Gamma(f)=\frac{1-y(f)}{1+y(f)}
\]

time-domain 實作用 bilinear transform：

\[
s=\frac{2}{\Delta t}\frac{1-z^{-1}}{1+z^{-1}}
\]

若 \(u\) 是 relaxation low-pass state，更新式為：

\[
u[n]=b\,p[n]+b\,p[n-1]-a_1u[n-1]
\]

\[
b=\frac{1}{1+2\tau/\Delta t},\qquad
a_1=\frac{1-2\tau/\Delta t}{1+2\tau/\Delta t}
\]

邊界法向速度為：

\[
v_n[n]=
\frac{g_\infty p[n]+g_ru[n]}{\rho c}
\]

每一面牆的每一個 FDTD cell 都保有自己的 \(p[n-1]\) 和 \(u[n-1]\)
state。當 \(g_r=0\) 時，模型退化為 frequency-independent real
admittance；回歸測試確認新舊 FDTD RIR 逐 sample 相同。

單牆 reflection impulse test 會從離散 filter 量出 complex response，
並與連續模型的 magnitude/phase 比較。在 16 kHz 取樣率、0–300 Hz
驗證範圍內，complex response 絕對誤差小於 0.003；數位 pole 位於單位圓
內，掃頻的 \(|\Gamma|\le1\)。三維 FDTD 測試也確認 frequency-dependent
boundary 保持有限、因果並完整序列化。

這一步證明「一個被動、因果、帶相位的邊界可以進入 time-domain solver」，
但它仍是合成 reference model，不是特定建材的先驗。

### 10.3 已完成：第一批有 provenance 的 phase-aware reference prior

第一批 reference 使用 Miki 1990 的 rigid-backed porous-layer 模型。令
\(X=1000f/\sigma\)，其中 \(\sigma\) 是 airflow resistivity：

\[
Z_c=\rho c
\left[
1+5.50X^{-0.632}
-j\,8.43X^{-0.632}
\right]
\]

\[
k=\frac{\omega}{c}
\left[
1+7.81X^{-0.618}
-j\,11.41X^{-0.618}
\right]
\]

\[
Z_s=-j\frac{Z_c}{\tan(kd)}
\]

\(d\) 是多孔層厚度。程式保守限制在
\(0.01\le f/\sigma\le1\)，超出範圍會明確拒絕，不靜默外推。

目前包含兩個 100 mm glass-wool reference：

| Reference | 量測 flow resistivity | 有效頻率範圍 | 證據等級 |
|-----------|-----------------------|--------------|----------|
| 14 kg/m³ | 5.88 kPa·s/m² | 58.8–5880 Hz | 量測 flow resistivity + Miki model |
| 30 kg/m³ | 15.5 kPa·s/m² | 155–15500 Hz | 量測 flow resistivity + Miki model |

flow resistivity 來自 Tarnow 2002 的量測；complex impedance 是由 Miki
模型預測，不是假裝成直接 impedance measurement。參考來源與 DOI 記在
[`impedance_priors.md`](impedance_priors.md)。

將 reference 的 complex \(\Gamma(f)\) 擬合到一階 time-domain boundary：

| Reference | Fit band | RMS complex error | Max complex error | Max phase error |
|-----------|----------|-------------------|-------------------|-----------------|
| 14 kg/m³ | 60–300 Hz | 0.0189 | 0.0571 | 0.0710 rad |
| 30 kg/m³ | 155–300 Hz | 0.00782 | 0.0160 | 0.00741 rad |

兩者都通過 max complex error 0.08 的 diagnostic gate。擬合後
\(g_r<0\) 並不表示 active boundary；它代表 rigid-backed porous layer
的 compliance，而 \(g_0\) 和 \(g_\infty\) 仍然非負。

### 10.4 已完成：phase 對 modal frequency/Q 的 FDTD 診斷

在 2.0 × 1.2 × 1.0 m 的 controlled room 中，比較：

1. 14 kg/m³ phase-aware fitted boundary；
2. 在 80 Hz 具有相同 \(|\Gamma|\)、但 phase 為零的實數 boundary。

dominant response peak 的結果為：

| Boundary | Peak frequency | Q |
|----------|----------------|---|
| phase-aware prior | 64.0 Hz | 15.8 |
| magnitude-only | 85.7 Hz | 10.5 |

這不是「模型已符合真實房間」的證據，而是直接證明：即使固定某一頻率的
reflection magnitude，phase 仍會實質改變模態頻率和 Q。因此只用
absorption coefficient 無法完成低頻邊界建模。

這個 prior 首次接入 FDTD 時也暴露出 edge/corner 的額外 stability limit：
interior CFL 本身不足以限制高 admittance 的顯式牆面項。solver 現在要求
三個方向累積的 normalized boundary Courant number 小於 0.9，並同時記錄
interior 與 boundary time-step limit。

### 10.5 已完成：直接量測契約、被動多 pole 與 eigenproblem 最小連接點

M2.4 打通了一條可執行的垂直路徑：

```text
法向入射 complex impedance CSV + provenance JSON
    -> 嚴格的單位、phase、passivity 與來源檢查
    -> positive-real multi-pole admittance fitting
    -> 每牆面 cell、每 pole 的 FDTD 時域 state
    -> 使用同一個 Gamma(s) 的 1D 複數 modal eigenproblem
```

量測 CSV 強制包含：

```text
frequency_hz
impedance_real_pa_s_m
impedance_imag_pa_s_m
```

實部與虛部的 standard deviation 可以選填，但必須成對出現。JSON sidecar
則要求 schema version、measurement id、方法、`incidence="normal"`、
air density、sound speed、樣品描述、來源 URL 與 license。loader 會拒絕
非遞增頻率、負 resistance、\(|\Gamma|>1\)、diffuse incidence 與缺少
provenance 的資料。

這裡的關鍵是「量測一定保留相位」。ISO 10534-2 類型的 two-microphone
transfer-function 量測可以取得法向入射的 complex impedance；reverberation
room 的 diffuse absorption 不能直接冒充這份資料。

為了讓離散量測可以進入時域 solver，新增多極點 normalized admittance：

\[
y(s)=g_s
+\sum_k\frac{g_{L,k}}{1+s\tau_k}
+\sum_k g_{H,k}\frac{s\tau_k}{1+s\tau_k}
\]

\[
\tau_k=\frac{1}{2\pi f_{p,k}}
\]

所有 \(g_s\)、\(g_{L,k}\)、\(g_{H,k}\) 都限制為非負；每個 branch 都是
stable positive-real element，所以平行相加後仍然被動。換句話說，被動性
不是在擬合完後只掃幾個頻率「看起來沒爆掉」，而是由參數化本身保證。

目前 fitter 固定 pole frequency，以 bounded least-squares 同時擬合
complex pressure reflection 的實部與虛部。它會輸出 RMS/max complex
error、magnitude error、phase error 與 acceptance gate。這是一個刻意
保守的固定 pole 方法，不是假稱完整 vector fitting；若 gate 失敗，應增加
pole、調整 pole placement 或改善資料，不可關閉 passivity constraint。

在 FDTD 中，每個 pole 有一個 low-pass state：

\[
u_k[n]=b_kp[n]+b_kp[n-1]-a_{1,k}u_k[n-1]
\]

因為 high-pass branch 等於 \(p-u_k\)，牆面速度可以寫成：

\[
v_n[n]=\frac{
g_sp[n]
+\sum_k g_{L,k}u_k[n]
+\sum_k g_{H,k}(p[n]-u_k[n])
}{\rho c}
\]

因此 K 個 pole 只需要 K 組 relaxation state，而不需另外建立 K 組
high-pass state。實作在每一面牆、每一個 cell 都各自保存這些狀態。

同一份 rational \(\Gamma(s)\) 也已接到 1D cavity eigenproblem。兩端使用
相同 locally reacting boundary 時，複數 pole 滿足：

\[
1-\Gamma(s)^2e^{-2sL/c}=0
\]

\[
s_n=-\gamma_n+j\omega_n,\qquad
f_n=\frac{\omega_n}{2\pi},\qquad
Q_n=\frac{\omega_n}{2\gamma_n}
\]

靜態實數 reflection 的測試會對照解析解
\(f_n=nc/(2L)\) 與
\(\gamma_n=-(c/L)\ln|\Gamma|\)；phase-aware 多 pole 測試則確認：即使
在參考頻率匹配 \(|\Gamma|\)，複數相位仍會移動 cavity mode。

合成二 pole 量測在給定正確 pole 時，最大 complex reflection fitting
error 小於 \(10^{-7}\)。量測 round-trip、dense-band passivity、多 pole
FDTD 與 1D modal root 都已有自動測試。完整契約與推導見
[`impedance_measurements.md`](impedance_measurements.md)。

### 10.6 已完成：第一份直接 complex impedance 與被動 RLC resonance

M2.5 審核公開資料時，先明確拒絕只有 absorption coefficient 的資料。
例如 FOAM 01/02 的授權、樣品數量與標籤都很好，但公開檔沒有
reflection phase，因此不能進入這條 complex boundary 路徑。

第一份接受資料是
[Zenodo 15195587](https://zenodo.org/records/15195587) 的
`paper_data.hdf5`。它以 CC BY 4.0 發布，直接包含逐頻率的 normalized
resistance 與 reactance。論文同時報告：

- 3D-printed 穿孔 liner 的 chamber、孔徑、facesheet、POA 與背板；
- NASA GFIT 與 UFSC grazing-duct test rig；
- 無流、Mach 0.3/0.5、130/145 dB 與 eduction 方法；
- `exp(+i*omega*t)` phasor convention；
- 兩個 nominally identical samples 的實際 3D scan 結果。

Repository 納入 Figure 6 的 NASA/UFSC 無流、130 dB KT 結果，限定在論文
共同比較的 500–2500 Hz。原始 HDF5 的 SHA-256 為：

```text
ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72
```

來源提供的是

\[
z(f)=Z(f)/(\rho c)
\]

而 HDF5 沒有保存逐筆 normalization atmosphere。為了不任選標準大氣後
假稱為量測環境，新增
`puresound.normalized_complex_impedance_measurement.v1`，原樣保存
dimensionless real/imag，同時明列
`acoustic_field_geometry="grazing_duct"`。這個 contract 把「表面阻抗」
和「如何量到它」分開，不會把 grazing eduction 偷換成 normal-incidence
tube。

真實資料也暴露了 M2.4 模型的結構限制。只有實數 relaxation pole 的模型
無法重現 reactance 在約 1.6 kHz 穿越零的 Helmholtz resonance，
complex error 接近 0.9。因此不是繼續增加同類 pole，而是加入物理上對應
series RLC 的 passive resonant admittance：

\[
y_k(s)=
\frac{g_{\mathrm{peak},k}}{Q_k}
\frac{s/\omega_{0,k}}
{(s/\omega_{0,k})^2+s/(Q_k\omega_{0,k})+1}
\]

\(f_0>0,Q>0,g_{\mathrm{peak}}\ge0\) 使共軛 pole 永遠在左半平面，branch
為 positive-real。它在 FDTD 中以 resonance-prewarped bilinear biquad
實作，每一面牆、每一個 cell、每個 resonance 都有獨立 state；stability
gate 使用 `static + sum(g_peak)`，而不是只檢查 DC/高頻端點。

NASA 資料以 alternating frequency bins 訓練／held out，結果為：

| 指標 | 結果 |
|------|------|
| fitted resonance | 1646.66 Hz |
| fitted branch Q | 11.69 |
| training RMS / max complex error | 0.0437 / 0.0815 |
| held-out RMS / max complex error | 0.0385 / 0.0590 |
| 4096 點最大 Cayley magnitude | 0.9178 |
| NASA–UFSC cross-rig RMS difference | 0.1182 |

held-out error 小於兩個 nominally identical samples／rigs 的差異。
4096 點 dense sweep 全部滿足 positive-real 和 bounded reflection
coordinate。在以 resonance 設定第一軸模態的 1D diagnostic 中，RLC
boundary 的 Q 為 17.73；參考頻率匹配 magnitude 的實數 boundary 為
6.04，比值 2.94。FDTD resonant biquad 也通過 finite-output、per-wall-state
與 boundary time-step regression。

這項結果只證明「真實 complex data 已能完整走過 pipeline」。它是
高聲壓 aircraft liner 的 grazing-duct eduction，不代表油漆牆、地毯或
ceiling tile。兩份 sidecar 都設為
`automatic_scene_catalog_mapping=false`。完整接受／拒絕記錄見
[`complex_impedance_source_audit.md`](complex_impedance_source_audit.md)。

### 10.7 M2.6：把缺少的正常室內材料變成可執行量測

兩個 glass-wool reference 仍只是「量測 flow resistivity + Miki model」；
第一份直接 complex dataset 又是高聲壓穿孔 liner。下一階段需要：

1. 納入正常聲壓、normal-incidence、厚度／backing／air gap／環境完整的
   porous 或 room-finish direct complex measurement；
2. 加入 measurement uncertainty 加權與 material-disjoint split；
3. 只把量測配置相容的 prior 映射到 scene material；
4. 將 rational impedance 接入 3D modal eigenproblem；
5. 同時驗證模態頻率偏移、Q、reflection magnitude 與 phase；
6. 把真實 RIR 切成真正的 room-disjoint development/test；
7. 聲學 gate 通過後，執行固定配方的下游語音任務。

第二輪公開來源審核仍沒有找到同時保留 phase、完整 mounting／backing、
環境、正常室內 SPL、授權與 machine-readable values 的 room-finish
dataset。因此 M2.6 沒有 digitize 論文曲線或從 absorption 補猜 phase，
而是先完成真正取得資料所需的 two-microphone reduction：

```text
重複量測的 complex H12
    -> microphone-switch complex calibration
    -> 圓管 plane-wave cutoff + 麥克風間距 conditioning
    -> coherence gate
    -> complex reflection Gamma
    -> complex surface impedance Z
    -> 跨安裝 repeatability uncertainty
    -> uncertainty-weighted passive rational fit
```

座標、推導、CSV/JSON schema、實驗 checklist 與 CLI 詳見
[`impedance_tube_protocol_zh-TW.md`](impedance_tube_protocol_zh-TW.md)。
synthetic round-trip 已驗證：已知被動阻抗經 H12、複數通道失配和換麥校正
後，可以在數值精度內恢復。這代表 acquisition software gate 已通過；它
不代表第一批實體 room-finish specimen 已經量完。

尚未完成的 physical gate 是：至少三個獨立 specimen、每個多次重新安裝，
在匹配 backing／air gap 與正常聲壓下取得資料，然後做 material-disjoint
holdout。只有這個 gate 通過，`automatic_scene_catalog_mapping` 才能從
`false` 改為 `true`。

### 10.8 M2.7：從六面 impedance 同時解 3D frequency、Q 與 mode shape

舊的 `*-material` backend 先使用 rigid-wall cosine mode，再把 diffuse
absorption 經 surface participation 轉成一個 decay rate。它只能改 Q，
不能表達 reflection phase 造成的 modal frequency shift，也沒有改變
eigenfunction。

新的 `analytic-impedance` backend 不使用這條近似。對每個牆面指定被動
rational normalized admittance \(y(s)\)，由 momentum equation 得到：

\[
\partial_n p+\frac{s}{c}y(s)p=0,
\qquad s=-\gamma+j\omega.
\]

沿一個長度為 \(L\) 的軸，令 \(h_\pm=(s/c)y_\pm(s)\)，complex
wavenumber 必須滿足：

\[
(h_-h_+-k^2)\sin(kL)+k(h_-+h_+)\cos(kL)=0.
\]

三個軸不是各自挑一個 1D frequency；它們必須和同一個 temporal pole
一起滿足 3D dispersion：

\[
k_x^2+k_y^2+k_z^2+\left(\frac{s}{c}\right)^2=0.
\]

也就是同時求解 \(s,k_x,k_y,k_z\) 四個 complex unknowns。程式從弱
boundary 逐步 continuation 到完整 admittance，避免 root 跳到別的 mode
branch。求得的 complex separable eigenfunction 會在 source 與 receiver
位置取值，因此仍保留節點、互易性與位置耦合。

目前 gate：

| 檢查 | 結果 |
|------|------|
| 只有 x 軸 static impedance | 與 exact 1D frequency、decay、Q 一致 |
| 均勻立方體 | 三個 axial modes 保持 permutation degeneracy |
| 2.0 × 1.2 × 1.0 m phase-aware 3D root | 63.8846 Hz，Q 17.32 |
| 相同設定的 independent FDTD | 64.0049 Hz，Q 15.78 |

frequency 差 0.19%，Q 差 9.8%，通過 controlled-reference gate。完整
generator CLI smoke 也已成功，metadata 會保存六面 model、每個 complex
pole、frequency、decay、Q 與 residual。

M2.7 當時必須區分兩件事：

- nonlinear eigenvalue 與 complex eigenfunction 已經是 3D impedance
  boundary 的物理解；
- 第一版 RIR 中每個 pole 的 residue 仍使用 engineering amplitude scale ×
  separable source/receiver coupling，當時尚未用 FDTD transfer function
  或真實量測校正。

未提供 residue calibration 時，backend 仍保留這條舊路徑，metadata 寫入
`modal_residue_fdtd_validated=false`。下一節的 M2.8 已補上 numerical
reference calibration，但真實房間 transfer measurement 尚未通過，所以
`modal_residue_production_validated=false` 與
`production_material_mapping_enabled=false` 仍不會解除。

### 10.9 M2.8：固定 3D poles，以 FDTD 校正 transferable modal residue

#### 問題不是再調一次 Q

M2.8 不允許 fitting 改動 M2.7 求得的 \(s_m\)、\(k_x,k_y,k_z\) 或
eigenfunction。要估的只有：

1. 一個全域 complex scale \(C\)，修正正交相位與 normalization；
2. 一個平滑 frequency exponent \(\eta\)，取代手寫的 \(1/f\)。

這個限制很重要。若同時重 fit poles，即使 waveform 看起來更像，也無法
判斷改善來自正確 residue，還是拿錯 frequency/Q 補償錯 amplitude。

#### FDTD source 與 modal basis 必須使用相同物理約定

reference FDTD 每個 time step 在一個 pressure cell 注入 unit-peak Ricker
pulse \(u[n]\)：

\[
p[\text{source cell},n]\mathrel{+}=u[n].
\]

假設 complex eigenfunction 使用 bilinear volume normalization：

\[
\int_V \phi_m(\mathbf{x})^2\,dV=1,
\]

單一 FDTD cell 的 initial-pressure projection 會帶一個 cell volume
\(\Delta V\)。對 source \(j\) 和 mode \(m\)，先建立未受迫正相 pole：

\[
z_{jm}(t)=
\Delta V\,
\phi_m(\mathbf{x}_{s,j})
\phi_m(\mathbf{x}_{r,j})
e^{s_m t}.
\]

FDTD 並不是用理想 Dirac impulse 激勵，所以不能直接拿
\(\operatorname{Re}z\) 或 \(\operatorname{Im}z\) 比 waveform。兩個
quadrature basis 都必須和實際 Ricker source convolution：

\[
b^{(R)}_{jm}=u * \operatorname{Re}z_{jm},\qquad
b^{(I)}_{jm}=u * \operatorname{Im}z_{jm}.
\]

接著只在 source pulse 結束後的 free-decay window 比較，並把 FDTD
target 和 modal basis 都用同一個 60–240 Hz zero-phase bandpass。這避免：

- 把 source spectrum 誤認成 residue frequency law；
- 把有效頻帶外、根本沒有枚舉的 pole 算成 residue error；
- 因 FDTD cell snapping 而用錯 source／receiver 座標。

實作使用 FDTD 實際 cell center，不是使用者要求但尚未離散化的位置。

#### 可轉移模型與 fitting

頻率權重定義為：

\[
w_m(\eta)=
\left(\frac{f_\mathrm{ref}}{f_m}\right)^\eta,
\qquad f_\mathrm{ref}=100\ \text{Hz}.
\]

對每個位置 case：

\[
\hat p_j =
a_R\sum_m w_m b^{(R)}_{jm}
+a_I\sum_m w_m b^{(I)}_{jm}.
\]

對每一個候選 \(\eta\)，用四組 training positions 聯合解兩個 linear
least-squares coefficients \(a_R,a_I\)。每個 case 先除以自己的分析窗
RMS，避免最響的位置壟斷 objective。選出 training NRMSE 最小的
\(\eta\) 後，JSON 儲存的 complex convention 是：

\[
C=a_R-j a_I,\qquad
\hat h_m(t)=
\operatorname{Re}
\left[
C\,w_m\,
\phi_m(\mathbf{x}_s)\phi_m(\mathbf{x}_r)e^{s_m t}
\right].
\]

renderer 不再乘 FDTD 的 \(\Delta V\)，因為輸出已回到 continuous
point-source convention。為保持有限 mode sum 的 causality，使用 absolute
modal time 保留校正相位，但在 geometric direct arrival 前裁成零；不能把
time origin 重設到每個 source–receiver 的 direct arrival，否則會額外產生
位置相依的 phase rotation。

#### 資料切分與結果

校正使用兩個不同長寬比的 shoebox：

- room A：2.00 × 1.35 × 1.05 m，15 個 fixed poles；
- room B：1.85 × 1.55 × 1.15 m，19 個 fixed poles。

每房兩組位置進 training、一組位置完全 holdout，共四 train、兩 holdout。
房間尺寸刻意讓三軸 fundamental 都落在 boundary model 的 60 Hz 下限以上；
不能用有效帶外的 pole 幫忙通過 gate。

求得：

\[
C=0.444175+j\,0.129146,\qquad \eta=0.35.
\]

| split / model | mean NRMSE | mean correlation | mean energy ratio |
|---|---:|---:|---:|
| train / calibrated | 0.1299 | 0.9916 | 0.9831 |
| position holdout / calibrated | 0.2304 | 0.9734 | 1.0039 |
| position holdout / 舊 \(1/f\)+sine baseline | 0.9703 | 0.2467 | 0.0913 |

驗收門檻是 holdout correlation ≥ 0.90、NRMSE ≤ 0.35，而且必須優於舊
baseline；結果通過。這代表 fixed poles + complex eigenfunctions 對新的
source／receiver 位置有可轉移的 waveform residue，不只是逐 case
curve fitting。

版本化輸出：

- calibration：
  `egs/rir_generation/phases/m2_impedance/config/impedance_residue_calibration_glass_wool_14kgm3_100mm_m2_8.json`
- 完整 FDTD、fixed-pole、exponent search 與 per-case report：
  `egs/rir_generation/phases/m2_impedance/reports/impedance_residue_calibration_glass_wool_14kgm3_100mm_m2_8_report.json`

提供 calibration 時，backend metadata 會寫
`modal_residue_fdtd_validated=true` 和
`modal_residue_model=fdtd_calibrated_complex_scale_power_law`。未提供時仍
使用舊 engineering fallback，兩者不會在 metadata 中混淆。

#### 現在可以宣稱與不能宣稱的事

可以宣稱：

- 在這個 60–240 Hz、均勻六面、locally reacting rational boundary 的
  controlled shoebox domain，已能產生 pole、Q、位置 coupling、相位與
  free-decay energy 都和 independent FDTD 相符的低頻 modal RIR；
- calibration 對未見過的 source／receiver 位置通過 holdout；
- 完整 dataset generator CLI 已能讀取這份 calibration。

仍不能宣稱：

- 數值 FDTD 校正等於真實房間 transfer-function 校正；
- 單一 glass-wool model 可外推到一般牆面、不同 mounting 或 patchy room；
- 60–240 Hz 參數可外推到 300 Hz 以上；
- full hybrid RIR 的 direct-path sample amplitude 或 low/high crossover
  gain 已由 M2.8 校正；這次 gate 只涵蓋 source pulse 結束後的低頻 modal
  free decay；
- 已可開啟自動 scene material mapping 或取代 production bank。

### 10.10 M2.9：厚度變體、真正 room holdout 與 grid holdout

M2.8 還有兩個可能讓結果過度樂觀的地方：

1. holdout 位置仍在 training 已看過的兩個房間；
2. 只有 100 mm boundary，無法判斷 residue 是否只對單一 impedance 有效。

M2.9 不增加任意的 synthetic material。第二個 boundary 使用同一份 Tarnow
實測 flow resistivity \(\sigma=5.88\ \mathrm{kPa\,s/m^2}\)，只把
rigid-backed layer 厚度從 100 mm 改為 50 mm，再由 Miki model 產生 complex
impedance。50 mm 的一階 causal relaxation fit 在 60–300 Hz 的 RMS／最大
complex-reflection error 是 0.00313／0.00506。這是有 provenance 的
model-derived installation variant，但仍不是直接量測的 installed finish。

#### 三種互不混淆的 holdout

fitter 現在把 evaluation split 分開報告：

- `position_holdout`：room A/B 已見過，但 source／receiver 位置未見；
- `room_holdout`：room C 的 geometry 與全部位置都不參與 fitting；
- `grid_holdout`：room C 同一 requested geometry 改以 0.08 m nominal
  spacing 重跑；training 全部使用 0.12 m。

room A/B 只有四組 training positions。room C（1.72 × 1.42 × 1.12 m）
包含兩組 room holdout；第一組另有 fine-grid copy。每個 split 必須同時滿足：

\[
\overline{\mathrm{corr}}\ge 0.90,\qquad
\overline{\mathrm{NRMSE}}\le 0.35,
\]

而且 NRMSE 必須優於在相同 training cases 上 fit gain 的舊
\(1/f\)+sine baseline。pole、Q 與 eigenfunction 仍完全固定。

#### 每個 boundary 各自校正

| boundary | split | correlation | NRMSE | energy ratio |
|---|---|---:|---:|---:|
| 50 mm | position holdout | 0.9904 | 0.1366 | 0.9456 |
| 50 mm | unseen room | 0.9929 | 0.1187 | 0.9787 |
| 50 mm | fine grid | 0.9961 | 0.0919 | 0.9448 |
| 100 mm | position holdout | 0.9734 | 0.2304 | 1.0039 |
| 100 mm | unseen room | 0.9933 | 0.1177 | 1.0282 |
| 100 mm | fine grid | 0.9929 | 0.1193 | 0.9674 |

兩個 boundary 的三種 holdout 全部通過。各自的 residue 為：

\[
C_{50}=0.639407+j\,0.060622,\quad \eta_{50}=0.25,
\]

\[
C_{100}=0.444175+j\,0.129146,\quad \eta_{100}=0.35.
\]

scale magnitude ratio 是 1.388，phase span 0.188 rad，exponent span 0.10。
所以「各自通過」不等於參數完全相同。

#### 跨 boundary 共用參數的診斷

另外把兩個 boundary 的八組 training cases 合併，只 fit 一組 shared
parameters：

\[
C_\mathrm{shared}=0.532760+j\,0.103650,\qquad
\eta_\mathrm{shared}=0.10.
\]

shared model 的 position／room／grid holdout 分別得到：

| split | correlation | NRMSE | energy ratio |
|---|---:|---:|---:|
| position holdout | 0.9766 | 0.2561 | 0.9696 |
| unseen room | 0.9882 | 0.2161 | 1.0116 |
| fine grid | 0.9924 | 0.1838 | 0.9344 |

也全部通過同一門檻。因此在「同一實測 flow resistivity、50/100 mm
model-derived rigid-backed thickness variants」這個窄 scope 內，共用 residue
是可行的。不過 protocol 明確保存
`general_boundary_invariance_established=false`；不能把結果外推到另一種
材料、air gap、斜入射或實測牆面。

版本化輸出：

- `impedance_reference_glass_wool_14kgm3_50mm.json`；
- 50/100 mm 各自的 `*_m2_9.json` calibration 與完整 report；
- `impedance_residue_protocol_m2_9_report.json`，保存 shared test、三種
  holdout 與 acceptance decision。

50 mm calibration 已通過完整 generator smoke：60–240 Hz 共解出 123 個
modes，metadata 為 `modal_residue_fdtd_validated=true`，
`modal_residue_production_validated=false`，而 direct-path amplitude 仍明確
標成未由 residue fit 校正。

### 10.11 M2.10：統一 FDTD source 與數位 RIR 的物理 convention

M2.8/M2.9 解決了 pole residue 的 waveform fit，卻留下了一個不能靠提高
correlation 消除的尺度問題：reference FDTD 與 dataset renderer 的「輸入」
不是同一個物理量。

- FDTD 每個 time step 把 \(q[n]\) 直接加到一個 pressure cell；
- renderer 的 direct path 則是延遲後、振幅為 \(1/r\) 的離散 impulse；
- 若直接把 FDTD fit 得到的 residue 放進 renderer，modal tail 會缺少
  pole integration 與 \(1/f_s\)；改變 sample rate 時，direct/modal 比例也會
  改變。

#### 從一階聲學方程推導

令 cell 體積為 \(\Delta V\)、time step 為 \(\Delta t\)。FDTD 的 pressure
increment \(q(t)\) 對應到：

\[
\frac{\partial p}{\partial t}
+\rho c^2\nabla\cdot \mathbf v
=
\frac{\Delta V}{\Delta t}q(t)\,
\delta(\mathbf x-\mathbf x_s).
\]

再配合 particle-velocity 方程，可得：

\[
\left(\frac{\partial^2}{\partial t^2}-c^2\nabla^2\right)p
=
\frac{\Delta V}{\Delta t}\frac{dq(t)}{dt}\,
\delta(\mathbf x-\mathbf x_s).
\]

三維 free-field Green's function 因此給出 FDTD 的直達壓力：

\[
p_\mathrm{FDTD}(r,t)
=
\frac{\Delta V}{\Delta t\,4\pi c^2r}
\frac{dq(t-r/c)}{dt}.
\]

我們正式定義 dataset RIR 的 source convention 為：

\[
y[n]=\sum_k h[k]x[n-k],\qquad
h_\mathrm{direct}[n]=\frac{1}{r}\delta[n-D].
\]

要讓這個 RIR 與上述 FDTD 代表同一個 excitation，數位輸入必須是：

\[
x(t)=
\frac{\Delta V}{\Delta t\,4\pi c^2}\frac{dq(t)}{dt}.
\]

這裡的 \(\Delta V\) 不是 production RIR 的任意 gain。它只描述「一個
finite-volume cell 的 pressure increment」如何換算成連續 point source；
把 FDTD residue 轉回連續、volume-normalized eigenfunction coupling 時會
消掉。

#### Modal residue 的轉換

對 pole \(s_m\)，M2.9 fit 的 pressure-state residue 記為
\(R_m^{(p)}\)。production renderer 使用的 residue 為：

\[
R_m^{(\mathrm{RIR})}
=
R_m^{(p)}
\frac{4\pi c^2}{f_s s_m}.
\]

除以 \(s_m\) 是 time integration；乘上 \(1/f_s=\Delta t\) 是因為離散
convolution 使用 sample sum，而不是連續時間積分。這也說明一個看似奇怪但
必要的差別：

- smooth modal tail 的每個 sample 會隨 \(1/f_s\) 縮放；
- direct path 的 Kronecker impulse 振幅仍是 \(1/r\)；
- 兩者的離散頻率響應才會在不同 sample rate 下代表同一個房間。

三個 convention 都使用不可含糊的版本 id：

- `puresound.fdtd_pressure_cell_state_increment.v1`；
- `puresound.free_field_pressure_1_over_r_discrete_rir.v1`；
- `puresound.pressure_state_residue_to_free_field_1_over_r.v1`。

calibration schema 升為
`puresound.impedance_modal_residue_calibration.v2`。loader 仍可讀 M2.8/M2.9
的 v1 檔，並依其已知 FDTD calibration protocol 補上相同的明確轉換；新的
50/100 mm `*_m2_10.json` 則直接保存三個 id。

#### 驗證結果

第一層用 normalized admittance \(y=1\) 的 matched boundary，讓局部法向
反射係數為零，直接比較 FDTD 直達 pulse 與上面的 Green's-function 預測：

| case | correlation | fitted/theory amplitude | time offset |
|---|---:|---:|---:|
| axis, 50 mm grid | 0.9685 | 1.0023 | 0 sample |
| axis, 40 mm grid | 0.9753 | 0.9933 | 0 sample |
| diagonal, 30 mm grid | 0.9896 | 1.1080 | 0 sample |

斜向 case 的剩餘 10.8% 振幅誤差主要反映有限 Cartesian grid 的 dispersion
與 anisotropy；縮細網格時已由 40 mm 的 14.9% 降至 30 mm 的 10.8%。
gate 要求 correlation ≥ 0.95、振幅誤差 ≤ 15%、時間偏差 ≤ 1 sample。

第二層以 band-limited Ricker pulse 檢查轉換前後的 modal convolution
identity。4/8/32 kHz 的 NRMSE 分別是
0.00591、0.00148、0.0000925，correlation 都大於 0.9999999。

第三層把同一組 direct path 加兩個 synthetic poles，在 60–240 Hz 比較
4/8/16 kHz 與 32 kHz reference 的複數頻率響應。最大相對誤差分別為
4.07%、1.76%、0.591%，通過 5% gate。

完整 machine-readable 結果在
`egs/rir_generation/phases/m2_impedance/reports/rir_source_convention_m2_10_report.json`。
renderer metadata 現在分別記錄 fitted convention、rendered convention、
transform，以及 `direct_path_source_convention_matched=true`。

這個結果表示 modal residue 與 \(1/r\) direct tap 現在使用同一套數位輸入
定義；不表示 \(1/r\) 已被真實喇叭／麥克風的絕對 Pa、SPL、directionality
量測校正，也不表示 low/high crossover level 已通過 FDTD 或 measured-room
校正。後者仍是下一個 gate。

### 10.12 M2.11：保留 source convention 的複數 crossover gate

#### 稽核發現的兩個問題

舊 crossover matcher 會把 low/high 經過分頻後的輸出，再以 band-limited
RMS 算一個逐 channel 的 low-band scalar gain。這有兩個問題。

第一，預設分析頻帶固定為 700–1300 Hz。當 impedance experiment 把
crossover 改為 240 Hz 時，matcher 仍在 700–1300 Hz 估計 gain；這個區域的
low-pass branch 已幾乎沒有能量，gain 很容易直接撞到 2× 上限。

第二，即使 low/high 是完全相同的 common-source RIR，兩個
Linkwitz–Riley branch 在一段有限頻帶內的 RMS 也不必相等。240 Hz crossover
使用新的 168–312 Hz 分析頻帶時，舊公式仍會對相同輸入估出約
0.9503–0.9506 的 low gain，讓原本 flat 的 complex sum 產生約
0.44 dB 誤差。因此「branch RMS 相等」不是物理 calibration condition。

#### Linkwitz–Riley 真正需要滿足的條件

令二階 Butterworth low/high transfer 為 \(B_L(z)\)、\(B_H(z)\)。目前的
四階 Linkwitz–Riley branches 是：

\[
L(z)=B_L^2(z),\qquad H(z)=B_H^2(z).
\]

若兩個 backend 在 crossover 表示同一個 complex response \(X(z)\)，輸出為：

\[
Y(z)=X(z)\left[L(z)+H(z)\right].
\]

正確條件不是分別令 low/high RMS 相等，而是：

1. \(L\) 與 \(H\) 在 crossover region 具有相同 phase；
2. \(|L+H|=1\)，也就是 complex sum 為 flat-magnitude all-pass；
3. 兩個 backend 的 direct path 使用相同 source convention 與幾何延遲；
4. 已經有物理尺度的 branch 不可再被任意 scalar 改寫。

#### 新的執行政策

`generate_hybrid_rir` 現在依 low-backend metadata 決定：

- 若 `direct_path_source_convention_matched=true`，且
  `preserve_source_convention_at_crossover=true`，即使舊
  `match_crossover_energy` 開關仍為 true，也不套用 low gain；
- metadata 寫入 `policy=preserve_validated_source_convention`、
  `energy_matching_requested=true`、`energy_matching_applied=false`，
  並逐 channel 保存 gain；此路徑全部是 1.0；
- 尚未校正的 legacy backend 暫時保留有界 RMS bridge，但明確標成
  `policy=energy_rms_match`，不能視為物理 calibration；
- 自動分析頻帶改為
  \([0.7f_c,\,1.3f_c]\)；若使用者提供顯式頻帶，卻沒有包含
  \(f_c\)，程式直接拒絕執行；
- CLI 提供 `--no-preserve-crossover-source-convention`，只供重現舊式
  diagnostic，不是推薦 production 設定。

#### 複數響應驗證

第一層直接檢查 digital filter。在 8/16/48 kHz、\(f_c=240\) Hz 下：

- \(|L+H|\) 最大誤差為 \(3.21\times10^{-12}\) dB；
- crossover region 的 branch phase 最大差為
  \(1.42\times10^{-12}\) degree；
- 兩個 branch 在 crossover 各約為 -6.02 dB。

第二層使用相同的 \(1/r\) anechoic direct path：

- low branch 使用 impedance renderer 的整數 sample direct tap；
- high branch 使用 Pyroomacoustics `max_order=0`、關閉 ray tracing 與
  air absorption 的 direct path；
- reference 是理想 fractional geometric delay 乘上共同的
  \(L+H\) all-pass；
- 五個距離為 0.8、1.0、1.5、2.0、3.0 m，分析 60–1000 Hz。

| sample rate | 最大 magnitude error | 最大 phase error |
|---:|---:|---:|
| 8 kHz | 0.980 dB | 5.39° |
| 16 kHz | 1.049 dB | 2.78° |
| 48 kHz | 0.457 dB | 0.579° |

gate 是 1.1 dB／6°，全部通過。剩餘誤差主要來自 low backend 的 rounded
direct sample 與 Pyroomacoustics windowed-sinc fractional delay 不完全相同；
這要由 M3 的共用 fractional-delay `PathEvent` 解決，而不是再 fit 一個 gain。

完整結果保存在
`egs/rir_generation/phases/m2_impedance/reports/hybrid_crossover_m2_11_report.json`。完整
renderer smoke 在 8 kHz 解出 125 modes，輸出五個 finite channels，metadata
確認 energy matching 未套用、五個 gains 全為 1.0。

目前通過的是 digital filter 與 anechoic direct-path crossover。尚未通過：

- 同一房間中 modal wave 與 geometric reflection 的 full complex match；
- complex boundary reflection phase 穿越 crossover 的連續性；
- measured-room crossover transfer；
- production material mapping。

### 10.13 M2.12：full-room complex crossover failure baseline

M2.11 只證明「相同 direct path 經過兩個 Linkwitz–Riley branches」時，
filter 與 source convention 正確。M2.12 把問題提升到完整房間，不先調 gain：

- boundary：100 mm glass-wool Miki/relaxation reference；
- reference：相同 room、snapped FDTD cell center 與 source convention 的
  staggered-grid FDTD transfer；
- cases：一個 training position、一個 position holdout、一個 unseen-room
  holdout；
- low：M2.10 fixed-pole impedance modal renderer；
- high diagnostic：精確 shoebox image geometry，最多 12 階、2625 個 image
  sources；
- crossover：240 Hz，分析 168–300 Hz；
- source excitation 在 crossover band 的最小相對 magnitude 為 0.702，
  因此 failure 不是 source spectrum 除以接近零造成。

high branch 分別測試三種 reflection hypothesis：

1. `magnitude_only_normal`：每次反射只乘 normal-incidence
   \(|\Gamma|\)；
2. `complex_normal`：每次反射乘同一個 normal-incidence complex
   \(\Gamma\)；
3. `complex_angle`：依每條 image path 的入射角使用 locally reacting
   \[
   \Gamma(\theta,\omega)=
   \frac{\cos\theta-y(\omega)}{\cos\theta+y(\omega)}.
   \]

這些 high variants 是 boundary-phase diagnostic，並不宣稱已等同目前的
Pyroomacoustics production backend。

#### 先確認 low model 沒有整體失效

在原 calibration band 60–240 Hz，三個 case 的 modal-only aggregate 為：

| metric | result | gate |
|---|---:|---:|
| complex correlation | 0.934 | ≥ 0.90 |
| complex NRMSE | 0.363 | ≤ 0.50 |
| transfer-energy ratio | 0.919 | 0.5–2.0 |

所以 M2.8–M2.10 的低頻結果沒有被這次測試推翻。然而只看靠近 band 上緣的
168–300 Hz，raw modal transfer-energy ratio 升到 5.84；即使經 240 Hz
low-pass branch 仍是 3.67。這表示「整個 60–240 Hz aggregate 通過」不能保證
它適合直接在 240 Hz 接 geometric backend。

#### 240 Hz full-room crossover 明確失敗

目前 magnitude-only high proxy 的三-case aggregate：

| metric | result | gate |
|---|---:|---:|
| complex correlation | 0.420 | ≥ 0.90 |
| complex NRMSE | 4.285 | ≤ 0.50 |
| transfer-energy ratio | 3.952 | 0.5–2.0 |

train、position holdout、room holdout 三個 case 全部失敗。單看 high-pass
branch，magnitude-only reflection 的 energy ratio 是 3.20；改成
angle-aware complex reflection 後降到 0.336，但 phase/correlation 仍未通過。
這證明 absorption magnitude 不足以決定 crossover response，但只補一個
complex reflection formula 也還不能完成 full-room match。

#### 兩種「簡單修正」也沒有通過

第一個診斷把 crossover 掃過 120、150、180、210、240 Hz。最佳結果是
120 Hz 加 `complex_angle`：

- complex correlation 0.866；
- NRMSE 1.628；
- energy ratio 0.710。

雖然比 240 Hz 好很多，仍未通過 correlation／NRMSE gate。因此不能只把
crossover 往下移。

第二個診斷針對 modal response 在 geometric arrival 的硬 clipping，加入
0/2/4/8/12/20 ms raised-cosine onset。8 ms 是仍能保住原 low-band gate 的
最佳值，但 240 Hz `hybrid_complex_angle` NRMSE 只由 4.332 降到 3.967；
12/20 ms 雖繼續降低 crossover error，卻使原本的 60–240 Hz low validation
失敗。因此 smooth onset 不能當 production hotfix。

M2.12 的正式 decision 是：

- 保存 failure baseline；
- 不 fit scalar gain；
- 不重 fit poles 或 residues；
- 不因 scan 結果把 crossover 偷改成 120 Hz；
- 不把 onset smoothing 接進 renderer；
- 下一步建立共用 `PathEvent`、fractional delay 與 angle-aware complex
  early reflection，再重跑同一份 gate。

完整 machine-readable 結果位於
`egs/rir_generation/phases/m2_impedance/reports/full_room_crossover_m2_12_report.json`。

### 10.14 M3.1：把直接聲與早期反射變成可檢查的 PathEvent

M2.12 暴露的根本問題，不只是「high branch 少了一個 phase」。原本 low
與 high backend 各自決定到達 sample、振幅與 reflection phase；即使兩邊都
大致合理，也不保證它們在 crossover 表示同一條物理路徑。M3 因此先不急著
增加更高反射階數，而是建立一個兩個 backend 都能共同使用的中間表示。

`PathEvent` 的基本原則是：

> 先保存一條聲音路徑的物理原因，再於指定 sample rate 下把它渲染成樣本。

目前使用三個版本化 JSON contract：

| contract | 保存內容 |
|---|---|
| `puresound.path_gain_spectrum.v1` | 不含 propagation delay 的 complex pressure gain samples |
| `puresound.path_event.v1` | 一條 direct／reflection path 的幾何與聲學狀態 |
| `puresound.path_event_set.v1` | 一組相同 source-receiver channel 的 events |

每個 event 明確包含：

- source/receiver ID 與位置；
- path type、總距離 \(L_p\)、音速 \(c\) 與延遲
  \(\tau_p=L_p/c\)；
- source 出發與 receiver 到達的單位方向；
- 依序經過的 surface ID、interaction type、interaction point 與
  incidence cosine；
- 不含 delay phase 的 complex pressure-gain spectrum；
- source/receiver directivity ID 與 gain；
- visibility、diffraction model 與 scattering model。

這些欄位不是為了讓 metadata 變多，而是用來回答可驗證的物理問題：延遲是否
來自幾何？交換 source/receiver 後是否 reciprocal？反射點有沒有真的落在
牆上？gain 是否錯把 delay phase 乘了兩次？附近位置是否突然跳到另一條路徑？

#### 精確 shoebox direct 與一階鏡像路徑

令 source 與 receiver 位置為 \(\mathbf{x}_s,\mathbf{x}_r\)。直接聲為

\[
\mathbf{v}_0=\mathbf{x}_r-\mathbf{x}_s,\qquad
L_0=\|\mathbf{v}_0\|,\qquad
\tau_0=L_0/c .
\]

其 departure 與 arrival propagation direction 都是
\(\mathbf{v}_0/L_0\)，pressure gain 沿用整個專案已校正的 free-field
convention：

\[
G_0=\frac{1}{L_0}.
\]

對任一 shoebox 平面，先把 source 對該平面鏡射成
\(\mathbf{x}'_s\)。連接 \(\mathbf{x}'_s\) 與 receiver，與牆面的交點就是
specular reflection point \(\mathbf{q}\)。路徑長為

\[
L_p=
\|\mathbf{q}-\mathbf{x}_s\|
+\|\mathbf{x}_r-\mathbf{q}\|
=\|\mathbf{x}_r-\mathbf{x}'_s\|.
\]

departure 與 arrival direction 分別是

\[
\mathbf{d}_{out}=
\frac{\mathbf{q}-\mathbf{x}_s}
{\|\mathbf{q}-\mathbf{x}_s\|},
\qquad
\mathbf{d}_{in}=
\frac{\mathbf{x}_r-\mathbf{q}}
{\|\mathbf{x}_r-\mathbf{q}\|}.
\]

M3.1 會固定產生一條 direct path 與 west/east/south/north/floor/ceiling
六條一階 reflection。鏡像法對矩形平面給出解析解，因此它是未來 mesh tracer
與 production backend 的 geometry oracle。

#### complex gain 與 propagation delay 必須分開

對 locally reacting surface，入射角為 \(\theta\)、normalized admittance
為 \(y(f)\) 時，一階 pressure reflection 是

\[
\Gamma(\theta,f)=
\frac{\cos\theta-y(f)}
{\cos\theta+y(f)}.
\]

目前反射 event 保存的 gain 是

\[
G_p(f)=\frac{\Gamma(\theta,f)}{L_p}.
\]

其中刻意不含
\(\exp(-j2\pi f\tau_p)\)。完整頻率響應必須由 renderer 組合：

\[
H_p(f)=G_p(f)\exp(-j2\pi f\tau_p).
\]

這個分離很重要。如果 complex gain 已含 propagation phase，而 renderer
又依 event delay 移動一次，路徑會被延遲兩次；反過來，如果只保存 rounded
sample index，換到另一個 sample rate 就會改變物理路徑。

#### causal fractional delay 的離散定義

令指定 sample rate 為 \(f_s\)，連續延遲換成

\[
D=\tau_p f_s=N+\mu,\qquad
N=\lfloor D\rfloor,\quad 0\le\mu<1 .
\]

M3.1 使用三階 one-sided forward Lagrange，節點
\(k=0,1,2,3\)，係數為

\[
h_k(\mu)=
\prod_{\substack{m=0\\m\ne k}}^{3}
\frac{\mu-m}{k-m}.
\]

四個 taps 放在 \(N,N+1,N+2,N+3\)。因此：

- 所有 \(n<N\) 的樣本嚴格為 0；
- \(\sum_k h_k=1\)，所以 DC 與 \(1/L_p\) gain 不變；
- sample rate 改變時重新由同一個 \(\tau_p\) 求 \(D\)，而不是 resample
  一個已 rounded 的 impulse。

這裡的「到達 sample」採離散時間 bin 定義：連續 wavefront 落入
\([N/f_s,(N+1)/f_s)\) 時，第一個允許非零的 bin 是 \(N\)。報告中的
pre-arrival check 因而檢查所有 \(n<N\)，不是把 arrival 先 round 到最近
sample。

one-sided Lagrange 是為了同時保有共同 delay contract 與嚴格左側 support。
它不是全頻 exact delay；所以我們明確限制驗證頻帶，而不把低頻通過外推成
Nyquist 附近也通過。

#### 為什麼 complex spectrum 目前不直接渲染

任意一組 complex frequency samples 並不自動構成 causal、stable、passive
的 time-domain filter。直接補 conjugate symmetry 後 IFFT，可能在 event
到達以前產生 pre-ringing，也可能因截斷改變 reflection magnitude/phase。
所以 M3.1 renderer 只接受「實數且 frequency-independent」的 gain；
遇到 angle-aware complex spectrum 會明確拒絕，而不是偷偷丟掉 imaginary
part 或只用 magnitude。

下一個切片會把已知的 positive-real admittance model 直接離散化成 causal
reflection filter，再與 fractional propagation delay 串接。這樣 boundary
phase 的來源仍是物理模型，不是對 spectrum 做無約束 IFFT。

#### M3.1 驗證結果

驗證包含 ordinary、asymmetric 與 near-grazing 三個房間／位置 case，另在
8/16/48 kHz 掃描 fractional part 0.05–0.95、頻帶 60–1000 Hz。

| gate | 最壞結果 | 門檻 | 結論 |
|---|---:|---:|---|
| geometry distance error | \(8.88\times10^{-16}\) m | \(\le10^{-12}\) m | 通過 |
| reciprocity error | \(8.88\times10^{-16}\) | \(\le10^{-12}\) | 通過 |
| 1 mm 位移的 distance-change ratio | 0.9515 | \(\le1\) | 通過 |
| fractional-delay magnitude error | 0.1001 dB | \(\le0.11\) dB | 通過 |
| fractional-delay phase error | 0.5554° | \(\le0.60°\) | 通過 |
| arrival bin 前非零樣本數 | 0 | 0 | 通過 |

schema JSON round-trip 完全相同；passive admittance case 的最大
\(|\Gamma|\) 為 0.8062，沒有反射增益。正式 decision 是：

- 接受 M3.1 `PathEvent` 幾何、serialization、reciprocity、continuity 與
  scalar fractional-delay renderer；
- 尚未宣稱 complex boundary filter 已能在 time domain 正確渲染；
- 尚未宣稱 higher-order、mesh visibility、家具遮擋或 measured C50 已改善；
- 下一步完成 passive causal boundary-filter realization，接進 hybrid
  backend，然後原封不動重跑 M2.12 gate。

完整 machine-readable 結果位於
`egs/rir_generation/phases/m3_wave_path/reports/path_events_m3_1_report.json`。

### 10.15 M3.2：把 complex reflection 變成被動因果 filter

M3.1 能保存 angle-aware complex gain，但刻意拒絕把任意 complex samples
直接 IFFT。M3.2 解決的是這個缺口：不從 samples 猜 filter，而是從已知為
positive-real 的 rational admittance model 直接推導 digital reflection。

#### 從 positive-real admittance 到 bounded-real reflection

令某個 sample rate 下的 normalized digital admittance 為

\[
Y(z)=\frac{B(z)}{A(z)}.
\]

對 incidence cosine \(q=\cos\theta>0\)，reflection 是

\[
\Gamma_\theta(z)
=\frac{q-Y(z)}{q+Y(z)}
=\frac{qA(z)-B(z)}
{qA(z)+B(z)}.
\]

這不是 curve fitting，而是 boundary equation 的代數重寫。若
\(\operatorname{Re}Y(e^{j\omega})\ge0\)，則

\[
1-|\Gamma_\theta|^2
=
\frac{4q\,\operatorname{Re}Y}
{|q+Y|^2}
\ge0.
\]

所以 \(|\Gamma_\theta|\le1\)：被動邊界不會在一次反射後產生額外能量。
continuous positive-real rational model 經 bilinear transform 後仍保持
positive-real；Cayley transform 再把它轉成 bounded-real reflection。
只要 \(q>0\)，所得 denominator \(qA+B\) 的 poles 位於 unit circle 內。

目前支援三種已存在的模型：

- `FirstOrderRelaxationAdmittance`；
- `PassiveMultiPoleAdmittance` 的 low-pass／high-pass parallel branches；
- `PassiveResonantAdmittance` 的 prewarped RLC biquad branches。

各 branch 先形成 causal digital admittance，再以 polynomial common
denominator 精確相加。最後才套上 \(qA\pm B\)，不是先對各頻率算
\(\Gamma\) 再擬合 IIR。

filter 使用版本
`puresound.digital_boundary_reflection_filter.v1`，保存 numerator、
denominator、sample rate、incidence cosine、原 admittance metadata 與
discretization；反序列化時會重新檢查有限係數、normalized denominator 與
pole stability。

#### PathEvent renderer 如何使用 boundary filter

對一條 reflection event，renderer 現在依序做：

1. 由 event 的連續 \(\tau_p\) 建立共同 fractional-delay kernel；
2. 以 surface ID 找到明確提供的 rational admittance model；
3. 在 event 保存的每個頻率重新計算
   \(\Gamma(\theta,f)/L_p\)；
4. 若重新計算的 complex gain 與 serialized spectrum 不一致，直接拒絕；
5. 由相同 model 與 incidence cosine 建立
   \(\Gamma_\theta(z)\)；
6. 把 fractional-delay kernel 通過 causal reflection filter；
7. 乘上 \(1/L_p\) 與 source/receiver directivity gains，再疊加到 RIR。

boundary filter 的 direct feed-through 落在 path arrival bin，內部 state
只會向後產生 tail，因此不會出現 raw IFFT 的 pre-ringing。若 event 只有
complex samples、卻沒有可重建的 rational model，renderer 仍然拒絕；它不會
丟掉 phase、不會只取 magnitude，也不會假設 minimum phase。

#### 單牆 digital-versus-analog gate

M2.10 的 100 mm glass-wool relaxation reference 在 60–300 Hz、五個
incidence cosines \(0.05,0.1,0.25,0.5,1.0\) 下測試：

| sample rate | 最大 complex error | 最大 magnitude error | 最大 phase error |
|---:|---:|---:|---:|
| 8 kHz | 0.002788 | 0.02175 dB | 0.2397° |
| 16 kHz | 0.000696 | 0.00285 dB | 0.05981° |
| 48 kHz | 0.0000773 | 0.000316 dB | 0.00664° |

最壞 pole magnitude 分別為 0.9839、0.9919、0.9973，全部小於 1。較高
sample rate 時 pole 更靠近 1 是相同 analog relaxation time 的正常結果，
不是不穩定。

另外用 positive-real multi-pole 與 passive resonant RLC 做全 Nyquist
sweep；所有 case 都保持 \(|\Gamma|\le1\)、finite impulse 與 stable poles。
這些 general-model tests 是數值 realization gate，不代表 resonant liner
已可自動映射成一般室內牆面。

#### 回到 M2.12：先分辨 renderer error 與 model incompleteness

三個 frozen rooms 都使用相同 snapped source/receiver positions、boundary、
sample rate 與分析頻帶。對每個 case 建立七個 events：

- 一條 direct；
- west/east/south/north/floor/ceiling 六條一階 reflection。

先不跟 FDTD 比，而是跟完全相同七條路徑的 analytic frequency-domain
complex-angle 解比較：

| 指標 | 三個 case 最壞結果 | gate |
|---|---:|---:|
| complex NRMSE | 0.003846 | \(\le0.01\) |
| complex correlation | 0.999998 | \(\ge0.999\) |
| transfer-energy ratio | 0.99883–0.99953 | 0.99–1.01 |

這證明 fractional delay、\(1/r\)、angle filter 與 time-domain IIR 的組合
正確重建了它宣稱的 path set。

但是把這七條 paths 放入原 240 Hz full-room crossover 後：

| candidate | correlation | NRMSE | energy ratio |
|---|---:|---:|---:|
| first-order geometric raw | 0.596 | 7.565 | 7.137 |
| first-order geometric high-pass | 0.378 | 3.394 | 3.293 |
| first-order hybrid | 0.589 | 4.578 | 4.086 |

因此 full-room gate 仍然失敗。這次能更精確地下結論：

- 失敗不是 complex spectrum 被錯誤 IFFT；
- 不是 fractional delay renderer 與 analytic path 不一致；
- 也不是缺一個 scalar gain；
- 七條 direct／一階 paths 本身不足以重建該頻帶的完整多重反射與 wave
  interference；只加入更多正能量 taps 也不保證改善，因為 complex
  cancellation 與 path phase 同樣重要；
- 原 order-12 complex-angle diagnostic 的 full-room hybrid 也仍未通過，
  所以「增加反射階數」必須被測試，不能被先驗視為答案。

M3.2 的正式 decision 是接受 causal boundary-filter realization，但保持
full-room crossover gate 為 open。不把七條 paths 升格成 production high
backend。下一步要擴充 coherent event set 或接入 mesh engine，並將 direct、
early、late energy 分開審計，再重跑 full-room 與 measured C50。

完整報告：

- `egs/rir_generation/phases/m3_wave_path/reports/path_event_filters_m3_2_report.json`；
- `egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_2_report.json`。

### 10.16 M3.3：高階有序反射路徑

M3.2 已證明「一條已知路徑」可以被正確地轉成 causal RIR，但七條路徑仍
不足以重建 full-room response。M3.3 因此先回答一個更窄、但必要的問題：

> 能不能把任意反射階數的 shoebox image source，無損地轉成可檢查、
> 可序列化、可用被動因果 filter 渲染的 `PathEvent`？

答案是可以；但實驗同時否定了「把 image order 調高就會自然通過 FDTD」
這個假設。

#### 從 unfolded room 取得每一條有序路徑

令第 \(i\) 軸的房間長度為 \(L_i\)、source 座標為 \(x_{s,i}\)，並以整數
\(n_i\) 表示該軸的 image index。unfolded space 中的 image source 是

\[
x'_{s,i}
=
2\left\lfloor\frac{n_i+1}{2}\right\rfloor L_i
+(-1)^{n_i}x_{s,i}.
\]

三維 image order 定義為 Manhattan norm：

\[
N=|n_x|+|n_y|+|n_z|.
\]

因此 max order \(N_{\max}\) 不是在每一軸各取
\([-N_{\max},N_{\max}]\) 的立方體，而是枚舉
\(|n_x|+|n_y|+|n_z|\le N_{\max}\) 的八面體 lattice。未排除特殊幾何前，
累積路徑數為

\[
\#(N_{\max})
=
\frac{4N_{\max}^3+6N_{\max}^2+8N_{\max}+3}{3}.
\]

所以 order 1、2、4、8、12 分別有 7、25、129、833、2625 條 image
paths。

從 receiver 到 unfolded image source 畫一直線；每穿過一個
\(kL_i\) 平面，就代表在 physical room 的一個 wall interaction。將座標以
週期 \(2L_i\) fold 回 \([0,L_i]\)，便得到真正的 reflection point。依照
直線參數由小到大排序 crossing，可同時得到：

- 反射 surface 的時間順序；
- 每一個 reflection point；
- 每一面的 incidence cosine；
- 整條路徑距離與 delay；
- source departure 與 receiver arrival direction；
- 哪些 interactions 發生在同一個幾何點。

這比只保存「每面牆被撞了幾次」更重要，因為不同材料 filter 的串接順序、
mesh visibility、未來的 transmission 與 diffraction 都需要完整 path
topology。每個 event 也保存 `image_order_xyz`，而且其 Manhattan norm
必須與 interaction 數一致，serialization 時會重新檢查。

#### 多次反射如何渲染

若一路徑依序撞到 surfaces
\(s_1,\ldots,s_K\)，其 frequency-domain reference 為

\[
H_p(f)
=
\frac{D_sD_r}{L_p}
\exp(-j2\pi f\tau_p)
\prod_{k=1}^{K}
\Gamma_{s_k}\!\left(\theta_k,f\right).
\]

`PathEvent` 仍將 propagation delay 與 boundary gain 分開，避免重複計算
phase。time renderer 先產生共同的 causal fractional-delay impulse，再把它
依序通過

\[
\Gamma_{s_1}(z),\Gamma_{s_2}(z),\ldots,\Gamma_{s_K}(z),
\]

最後才乘上 \(D_sD_r/L_p\)。每一個 \(\Gamma_{s_k}(z)\) 都由 M3.2 的
positive-real admittance 經 Cayley transform 建立，因此 repeated
reflection 不需要對 complex samples 做 IFFT，也不會在幾何到達時間以前
產生 pre-ringing。

renderer 會先重新計算整個
\(\prod_k\Gamma_{s_k}(\theta_k,f)/L_p\)，並與 event 保存的 complex
spectrum 比較；任何 surface model、incidence angle 或 interaction
順序不一致都會被拒絕。

#### edge 與 corner 不是「同時撞很多平面」那麼簡單

若 unfolded ray 同時穿過兩個或三個 boundary planes，它落在 edge 或
corner。此時逐面乘上 specular face reflection 並不是一個已驗證的物理
模型；真實結果會涉及 edge diffraction、有限尺寸與局部幾何。

所以目前有兩個明確分開的 policy：

- `exclude`：production default；排除 simultaneous crossings，直到有物理
  diffraction model；
- `sequential_face_product_diagnostic`：只為重現舊 analytic image-source
  的 face-product。coincident surfaces 保存相同
  `interaction_group_id`，表示它們是同一點，而不是有零長度段的數個獨立
  bounce。

第二個 policy 不會被升格為 production physics。在三個 frozen cases 中，
兩個一般位置沒有 simultaneous paths；position holdout 因 source 與
receiver 的 snapped coordinate 對齊，在 2625 條 paths 中有 410 條特殊
path。若採 physical `exclude`，該 case 剩 2215 條。忽略這個差異會讓
transfer NRMSE 達 1.378，所以不能把 edge/corner 當成無關緊要的
implementation detail。

#### representation gate

對每個 order，先將 events 的 complex spectra 與現有、完全相同階數的
analytic complex-angle image-source 解比較。三個 frozen rooms 的最壞
spectral NRMSE 為 \(3.60\times10^{-14}\)，即 floating-point 誤差等級。
這驗證了 image 座標、surface 順序、incidence angle 與 repeated complex
gain product。

再將 order-12 events 經 causal time renderer 轉成 RIR，與同一組 analytic
paths 比較：

| case | paths | NRMSE | correlation | energy ratio |
|---|---:|---:|---:|---:|
| train position | 2625 | 0.00853 | 0.999964 | 0.99900 |
| position holdout | 2625 | 0.01751 | 0.999910 | 0.99245 |
| room holdout | 2625 | 0.01707 | 0.999921 | 0.99553 |

因此 M3.3 的 higher-order representation 與 repeated causal filter
renderer 通過；這個 gate 只證明「程式忠實渲染它宣稱的 paths」，不證明
這些 paths 已經是完整房間物理。

#### order convergence 與 full-room 結論

以 order 12 當作這次有限 image enumeration 的比較基準，各階 analytic
transfer 的三-case 平均差異為：

| order | paths | 對 order 12 的 mean complex NRMSE | correlation | energy ratio |
|---:|---:|---:|---:|---:|
| 1 | 7 | 9.551 | 0.394 | 9.364 |
| 2 | 25 | 8.093 | 0.358 | 8.132 |
| 4 | 129 | 3.300 | 0.298 | 3.309 |
| 8 | 833 | 1.718 | 0.470 | 1.865 |
| 12 | 2625 | 0 | 1 | 1 |

這裡的 order 12 不是 ground truth，只是固定 reference。數字也不必隨
order 單調改善：RIR 是 coherent pressure sum，新增 paths 會同時改變
constructive 與 destructive interference，不能把每條反射當成只會增加的
正能量。

最重要的是，order-12 causal PathEvent 放回相同 240 Hz full-room protocol
後，aggregate 結果是：

| candidate | correlation | NRMSE | energy ratio |
|---|---:|---:|---:|
| geometric raw | 0.504 | 1.002 | 0.770 |
| geometric high-pass | 0.455 | 1.204 | 0.335 |
| hybrid | 0.760 | 4.332 | 3.787 |

full-room gate 仍為 false，而且與舊 order-12 analytic complex-angle
diagnostic 幾乎相同。這反而是有用的定位結果：causal PathEvent
implementation 已忠實重現 analytic failure，所以剩餘問題不是 renderer
bug，也不是 image order 太低；shoebox specular model、visibility、
edge/corner physics、wave/geometric crossover 或 low/high branch energy
分工仍不完整。

M3.3 的正式 decision 是：

- 接受任意階 shoebox `PathEvent`、有序交互與 repeated causal filters；
- 拒絕「只增加 image order」作為 full-room 修正；
- 不替換 production high backend；
- 下一步以 FDTD 為 reference，分別比較 direct、50 ms 內 early 與 later
  complex response，再評估具有 explicit visibility 的 mesh engine。

完整報告：

- `egs/rir_generation/phases/m3_wave_path/reports/higher_order_path_events_m3_3_report.json`；
- `egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json`。

### 10.17 M3.4：direct／early／later 誤差歸因

M3.3 證明 order-12 PathEvent 忠實重現了 analytic image-source model，但
它沒有回答 full-room error 主要來自哪一段 response。M3.4 的目的不是再
發明一個較好的 scalar metric，而是建立一個可以精確重建原 response 的
分解，分辨應該先修改 early geometry／phase，還是補更多 late paths。

#### 為什麼不能直接切出 direct sound

三個 frozen rooms 的 direct 與第一條 reflection 只相差約
0.7–1.1 ms；FDTD 使用的 180 Hz Ricker source pulse 遠比這個間距寬。
因此在量測到的 pressure output 上切一個「direct window」，一定同時包含
first reflections。把這種 window 稱為純 direct 會製造錯誤的物理解釋。

本協議改用 M2.10 已驗證 source convention、M3.1–M3.3 已驗證的
\(1/r\) causal direct `PathEvent` 作為明確 anchor \(y_D(t)\)。它先與
完全相同的 FDTD-equivalent band-limited input convolution；M3.4 不重新
宣稱從 FDTD 單獨抽出了 direct。

#### exactly reconstructive 分解

令完整 output 為 \(y(t)\)，direct 以外的 residual 為

\[
r(t)=y(t)-y_D(t).
\]

以 direct arrival 後 50 ms 為中心，建立寬度 8 ms 的 raised-cosine early
mask \(w_E(t)\)，並令

\[
w_L(t)=1-w_E(t).
\]

則

\[
\begin{aligned}
y_E(t)&=w_E(t)r(t),\\
y_L(t)&=w_L(t)r(t),\\
y(t)&=y_D(t)+y_E(t)+y_L(t).
\end{aligned}
\]

因為兩個 masks 是逐 sample 精確互補，這不是三個彼此獨立的估計；
direct、early、later 必須重建原 output。三個 cases 的 FDTD 與 PathEvent
reconstruction NRMSE 都低於 \(10^{-16}\)，PathEvent 依事件分桶再相加的
time-domain NRMSE 也低於 \(5\times10^{-15}\)。

PathEvent 另外保留第二種分法：

- direct event；
- arrival \(\le t_D+50\) ms 的 reflection events；
- arrival \(>t_D+50\) ms 的 events。

這個 geometry partition 讓 boundary-filter tail 繼續屬於產生它的 path；
共同時間窗版本則讓 FDTD 與 PathEvent 可以在相同 output mask 下公平比較。
兩種含義不能混為一談。

#### complex error 也必須保留 interference

將 output components 除以相同 input spectrum 後，在 168–300 Hz 比較
complex transfer。若

\[
e_E=H_{E,\mathrm{path}}-H_{E,\mathrm{FDTD}},\qquad
e_L=H_{L,\mathrm{path}}-H_{L,\mathrm{FDTD}},
\]

則因 direct anchor 相同，

\[
e_{\mathrm{full}}=e_E+e_L
\]

而

\[
\|e_{\mathrm{full}}\|^2
=
\|e_E\|^2+\|e_L\|^2
+2\operatorname{Re}\langle e_E,e_L\rangle.
\]

最後一個 cross term 不能省略。early 與 later 是 coherent pressure，不是
兩袋可以直接相加的正能量；報告中的 component norm 也明確標為
non-additive。

#### 三個 frozen rooms 的結果

共同 output-window 分解的三-case aggregate：

| component | NRMSE | correlation | path/FDTD component norm | error norm/FDTD full norm |
|---|---:|---:|---:|---:|
| early reflections | 0.331 | 0.948 | 0.999 | 1.045 |
| later reflections | 2.537 | 0.719 | 2.473 | 0.224 |
| direct + early | 1.044 | 0.509 | 0.819 | 1.045 |
| full geometric | 1.002 | 0.504 | 0.770 | 1.002 |

later component 的相對 NRMSE 很大，但 FDTD later norm 平均只有 full
transfer norm 的 0.086；PathEvent later 是 0.221。相反地，direct 與
early-reflection component norms 分別是 full norm 的 3.912 與 3.576。
這些看似大於 1 的值不是能量爆炸，而是 direct 與 reflections 在完整
transfer 中發生強烈 destructive interference。

因此最重要的現象是：

- early reflection 單獨看仍有 0.948 correlation，而且 transfer norm
  幾乎相同；
- 但它剩下的 complex phase／shape error 已達 full FDTD norm 的 1.045；
- 把它與 direct 相加後，應有的 cancellation 沒有對準，所以 early
  cumulative correlation 降到 0.509；
- later error 確實存在，卻不是目前 full-transfer error 的主導項。

逐 case 的 squared self terms、early/later cross term 與總 error 完全閉合；
position／room holdout 中 cross term 為負，表示 later error 還部分抵銷了
early error。這也是為什麼不能用「哪一段 NRMSE 最大」直接決定工作優先序。

M3.4 的正式 decision 是：

- 接受 direct-anchor／early／later attribution protocol；
- 將主要問題定位在 early-reflection complex phase 與 direct/early
  interference，而不是 50 ms 後 path 數量不足；
- 不以更多 image order、更多 late rays 或 scalar energy matching 作為
  下一個修正；
- 下一步先建立 oblique single-wall FDTD gate，直接比較相同 locally
  reacting boundary 的 \(\Gamma(\theta,f)\) magnitude/phase，再審核 modal
  low-pass 與 geometric high-pass 在 crossover 的分工；
- 在這兩個 gate 之前，mesh engine evaluation 不會被誤當成低頻相位修正。

完整報告位於
`egs/rir_generation/phases/m3_wave_path/reports/direct_early_later_m3_4_report.json`。

### 10.18 M3.5：FDTD 斜入射邊界其實在算什麼

M3.4 將主要差異定位到 early-reflection complex phase，但這仍不能直接
判定 PathEvent 錯了。作為 reference 的 FDTD 也是離散模型；它的 pressure
與 velocity 不在同一時間、也不在同一空間位置。M3.5 因此從 solver update
本身推導其真正的 plane-wave reflection，再與 continuous physics 比較。

#### 3D discrete dispersion

令時間步為 \(\Delta t\)，三軸網格間距為
\(\Delta_x,\Delta_y,\Delta_z\)，
\(\Omega=\omega\Delta t\)。staggered-grid plane wave 滿足

\[
\frac{\sin^2(\Omega/2)}{(c\Delta t)^2}
=
\sum_{i\in\{x,y,z\}}
\frac{\sin^2(k_i\Delta_i/2)}{\Delta_i^2}.
\]

指定 frequency 與 tangential direction 後，程式先由上式解出 normal
wavenumber \(k_n\)。該 discrete wave 的 normalized normal characteristic
admittance 不是幾何 \(\cos\theta\)，而是

\[
q_d
=
\frac{c\Delta t}{\Delta_n}
\frac{\sin(k_n\Delta_n/2)}{\sin(\Omega/2)}.
\]

這一項記錄 interior numerical dispersion；它在網格趨近零時才收斂至
\(q=\cos\theta\)。

#### half-time 與 half-cell phase

目前 FDTD update 的 pressure \(p^n\) 位於第一個 cell center
\(x_n=\Delta_n/2\)，boundary velocity 則位於 wall face \(x_n=0\) 和下一個
half time step。若 rational admittance 經 solver 相同的 bilinear filter
得到 \(Y_d(e^{j\Omega})\)，wall update 實際看到的 admittance 帶有
half-time factor

\[
\widetilde Y_d=Y_d e^{-j\Omega/2}.
\]

將 incident 與 reflected pressure 從 cell center 外推至 wall face，可得
目前程式真正實現的 reflection：

\[
\Gamma_{\mathrm{FDTD}}
=
\frac{
q_d-\widetilde Y_d e^{+jk_n\Delta_n/2}
}{
q_d+\widetilde Y_d e^{-jk_n\Delta_n/2}
}.
\]

對照之下，continuous locally reacting plane wave 是

\[
\Gamma_{\mathrm{continuous}}
=
\frac{\cos\theta-Y(j\omega)}
{\cos\theta+Y(j\omega)},
\]

而 PathEvent digital filter 是用幾何 \(\cos\theta\) 對
\(Y_d(z)\) 做 bounded-real Cayley transform。三者的差異因此可以分成：

- analog-to-digital admittance error；
- interior dispersion 造成的 \(q_d-\cos\theta\)；
- half-time staggering；
- wall face 到 pressure cell center 的 half-cell phase。

#### canonical 與實際 early-path 掃描

掃描範圍為 168–300 Hz。canonical probe 包含 incidence cosine
0.1、0.25、0.5、0.75、1.0，三個 tangential azimuth、三個 wall axes，
並使用三個 frozen rooms 真正的 anisotropic grid spacing 與 FDTD
sample rate。另一組 probe 從 order-12、50 ms 內 early paths 的每軸
incidence distribution 取 0/10/25/50/75/90/100% representative
directions。

實際 early paths 比 canonical scan 更 grazing；最低 incidence cosine
約 0.056。因此 actual-path worst case 會比 cosine 下限 0.1 的 canonical
scan 更嚴格。

| comparison | complex error | magnitude error | phase error |
|---|---:|---:|---:|
| PathEvent digital vs continuous，canonical | 0.00287 | 0.0218 dB | 0.240° |
| PathEvent digital vs continuous，actual early paths | 0.00288 | 0.0218 dB | 0.243° |
| FDTD discrete vs continuous，canonical | 0.1238 | 0.475 dB | 7.68° |
| FDTD discrete vs continuous，actual early paths | 0.1665 | 0.656 dB | 9.75° |

PathEvent filter 的誤差仍是 M3.2 已知的 bilinear discretization 等級。
相反地，目前約 6 cm FDTD grid 明顯超過預先設定的
0.02 complex／0.1 dB／1° continuous-parity gate。actual early-path
representatives 的 mean phase error 為 4.26°，已足以影響 M3.4 中很強的
direct／early cancellation。

#### 被動不等於 phase 已準確

所有 canonical cases 仍保持
\(|\Gamma_{\mathrm{FDTD}}|\le1\)，所以這不是 unstable 或 active boundary。
它是被動但有 phase bias 的低階離散邊界。將最壞 canonical direction 的
空間與時間網格一起做 linear refinement：

| linear grid scale | complex error | magnitude error | phase error |
|---:|---:|---:|---:|
| 1 | 0.1238 | 0.475 dB | 7.68° |
| 1/2 | 0.03997 | 0.148 dB | 2.44° |
| 1/4 | 0.01332 | 0.0463 dB | 0.815° |
| 1/8 | 0.00492 | 0.0158 dB | 0.303° |

誤差單調收斂，而且 1/4 grid 才通過目前的 continuous parity thresholds。
這證明 coarse-grid 差異是可解釋的 discretization error，而不是應該加入
material prior 的新 reflection phase。

M3.5 的正式 decision 是：

- 接受 discrete reflection 推導、被動性與 grid-convergence gate；
- 拒絕把目前約 6 cm FDTD boundary phase 當成 continuous ground truth；
- 不修改 PathEvent 的物理 reflection phase 去擬合 coarse-grid artifact；
- 下一步實作／評估 face-pressure 或 boundary-phase-compensated update，
  並用 time-domain plane-wave probe 驗證 harmonic 推導；
- 只有 boundary reference 通過後，才重新審核 modal low-pass 與 geometric
  high-pass crossover；mesh 或更多 late paths 仍不是目前的優先修正。

完整報告位於
`egs/rir_generation/phases/m3_wave_path/reports/oblique_fdtd_boundary_m3_5_report.json`。

### 10.19 M3.6：wall-face／half-time 補償是否足夠

M3.5 指出目前 boundary update 同時有 half-cell、half-time 與 interior
dispersion。M3.6 先測試兩個局部、因果且計算成本低的修正，但不直接改掉
production default。

#### 兩個 opt-in pressure schemes

令 \(p_0\) 是第一個 pressure cell center、\(p_1\) 是向室內的第二個 cell。
`face_extrapolated` 用線性外推估計 wall-face pressure：

\[
p_{\mathrm{face}}^n
=
\frac{3}{2}p_0^n-\frac{1}{2}p_1^n.
\]

對一個 harmonic incident／reflected wave，其 spatial factors 是

\[
\begin{aligned}
S_i&=\frac32e^{+j\phi}-\frac12e^{+j3\phi},\\
S_r&=\frac32e^{-j\phi}-\frac12e^{-j3\phi},\\
\phi&=k_n\Delta_n/2.
\end{aligned}
\]

一階 phase term 會互相抵銷，所以它比直接使用
\(e^{\pm j\phi}\) 的 cell-center pressure 更接近 wall face。

`face_time_extrapolated` 再用 causal linear predictor：

\[
p_{\mathrm{pred}}^n
=
\frac32p_{\mathrm{face}}^n
-\frac12p_{\mathrm{face}}^{n-1},
\]

其 transfer 為

\[
T(z)=\frac32-\frac12z^{-1}.
\]

它乘上原本的 \(e^{-j\Omega/2}\) 後，會抵銷 half-time phase 的一階項。
兩者都只是 experimental `FDTDReferenceConfig.boundary_pressure_scheme`；
既有 `cell_center` 仍是 default。

#### time-domain 如何獨立驗證 harmonic equation

只看推導本身不夠，所以建立 20 m 的 1D normal-incidence FDTD。分別以
target material、rigid wall 與已知 constant-admittance reference wall
執行相同 source。對第一批只撞一次 test wall 的 response，計算

\[
C_{\mathrm{measured}}
=
\frac{P_{\mathrm{target}}-P_{\mathrm{reference}}}
{P_{\mathrm{rigid}}-P_{\mathrm{reference}}}.
\]

所有不經 test wall 的 direct／top-wall paths 在分子、分母中代數消失。
harmonic equation 預測

\[
C_{\mathrm{predicted}}
=
\frac{\Gamma_{\mathrm{target}}-\Gamma_{\mathrm{reference}}}
{1-\Gamma_{\mathrm{reference}}}.
\]

第一次使用較短 domain 時，在約 190 Hz 出現 cross-ratio denominator
comb null；該點只剩 peak 的 4%，造成不適定除法。正式 protocol 將 domain
延長至 20 m，讓一次與多次 test-wall interactions 分離，而沒有放寬
acceptance threshold。

168–300 Hz 的正式 time-domain 結果：

| scheme | complex error | magnitude error | phase error |
|---|---:|---:|---:|
| cell center | \(1.08\times10^{-4}\) | 0.00067 dB | 0.00480° |
| face extrapolated | \(1.05\times10^{-4}\) | 0.00075 dB | 0.00483° |
| face + time extrapolated | \(1.10\times10^{-4}\) | 0.00078 dB | 0.00506° |

所以 M3.5/M3.6 的 harmonic equations 確實描述了 time-domain update；
後面的 continuous mismatch 不是公式寫錯。

#### 修正改善平均值，但沒有通過 worst-case gate

對三個 frozen grids 與實際 order-12 early-path representative directions：

| scheme | worst complex | worst magnitude | worst phase | mean complex | mean phase |
|---|---:|---:|---:|---:|---:|
| cell center | 0.1665 | 0.656 dB | 9.75° | 0.0574 | 4.26° |
| face extrapolated | 0.1390 | 0.662 dB | 7.77° | 0.0296 | 1.32° |
| face + time extrapolated | 0.1390 | 0.582 dB | 7.99° | 0.0224 | 1.54° |

face-only 的平均 phase 較低，但 magnitude bias 反而較大；face+time 是較
平衡的結果，mean magnitude error 降至 0.061 dB。兩者都保持
\(|\Gamma|\le1\)，短 3D stability probes 也沒有 non-finite output。

然而 continuous-reference gate 看的是 worst actual geometry，而不是只看
平均。actual early paths 包含 incidence cosine 約 0.056 的 grazing
interactions；這時 interior dispersion 造成的
\(q_d-\cos\theta\) 仍然存在，局部 pressure extrapolation 無法移除。

若把比較 target 改成「接受相同 \(q_d\) 的 dispersion-matched reflection」，
face+time 的 actual-path worst error 會降至 0.0237 complex、0.0338 dB、
2.10°。這證明 wall centering／time staggering 大部分已修正，但仍沒有通過
0.02 complex／1° gate。

M3.6 的正式 decision 是：

- 接受三種 discrete equations 的 time-domain cross-ratio 驗證；
- 接受兩個 experimental schemes 在此 material／頻帶的被動性與短時
  stability；
- 拒絕把任一簡單 extrapolation 升格為 continuous FDTD reference；
- 保持 production/default `cell_center` 不變；
- 不重跑 full-room crossover，因為 reference candidate gate 沒有通過；
- 下一步評估 higher-order characteristic、dispersion-aware boundary，
  或可負擔的更高精度 wave reference。這類方法必須處理 tangential
  wavenumber，而不只是修補 wall pressure 的取樣位置。

完整報告位於
`egs/rir_generation/phases/m3_wave_path/reports/corrected_fdtd_boundary_m3_6_report.json`。

### 10.20 M3.7：fourth-order dispersion-aware harmonic candidate

M3.6 已經把 wall-face 與 half-time 的低階偏差分離出來；剩餘最壞誤差會
隨入射角變得 grazing 而快速增加。這是 interior discrete derivative
改變了 plane wave 的 characteristic admittance，不是再調一個 local
material phase 就能修正。

#### 從 staggered derivative 重新推導 dispersion

二階 staggered derivative 的 Fourier symbol 含有
\(\sin(k\Delta/2)/\Delta\)。M3.7 測試的四階 stencil 為

\[
\left.\frac{\partial p}{\partial x}\right|_0
\approx
\frac{9}{8}
\frac{p_{+1/2}-p_{-1/2}}{\Delta}
-
\frac{1}{24}
\frac{p_{+3/2}-p_{-3/2}}{\Delta}.
\]

代入 \(p(x)=e^{-jkx}\) 後，它的半 derivative symbol 可整理成

\[
K_4(k,\Delta)
=
\frac{\sin(k\Delta/2)}{\Delta}
\left(
1+\frac{\sin^2(k\Delta/2)}{6}
\right).
\]

因此四階 staggered-grid dispersion relation 是

\[
\left[
\frac{\sin(\Omega/2)}{c\Delta t}
\right]^2
=
\sum_{a\in\{x,y,z\}}
K_4(k_a,\Delta_a)^2.
\]

對指定的 physical tangential wavenumbers，先由上式數值求出可傳播的
normal wavenumber \(k_n\)，再得到 discrete normal characteristic
admittance

\[
q_{d,4}
=
\frac{c\Delta t}{\sin(\Omega/2)}
K_4(k_n,\Delta_n).
\]

這一步直接修正 grazing angle 最敏感的
\(q_d-\cos\theta\)，而不是把 interior dispersion 誤當成 wall material
的 reflection phase。

#### 為什麼還需要 quadratic wall-face／half-time predictor

只換 interior stencil，M3.6 的 linear predictor 仍留下二階 interpolation
error。三個 pressure cell centers 位於距牆
\(\Delta/2,3\Delta/2,5\Delta/2\)，把二次多項式外推到 wall face 得

\[
p_{\mathrm{face}}^n
=
\frac{15}{8}p_0^n
-
\frac54p_1^n
+
\frac38p_2^n.
\]

相同係數從 \(n,n-1,n-2\) 預測 boundary velocity 所需的 half step：

\[
p_{\mathrm{pred}}^{n+1/2}
=
\frac{15}{8}p_{\mathrm{face}}^n
-
\frac54p_{\mathrm{face}}^{n-1}
+
\frac38p_{\mathrm{face}}^{n-2},
\]

其 transfer function 為

\[
T_2(z)
=
\frac{15}{8}
-
\frac54z^{-1}
+
\frac38z^{-2}.
\]

對 incident wave 的 spatial factor 則是

\[
S_{i,2}
=
\frac{15}{8}e^{j\phi}
-
\frac54e^{j3\phi}
+
\frac38e^{j5\phi},
\qquad
\phi=k_n\Delta_n/2,
\]

reflected factor \(S_{r,2}\) 將指數符號反轉。最後仍使用與 M3.5 相同的
discrete boundary equation：

\[
\Gamma_d
=
\frac{
q_{d,4}-Y_d e^{-j\Omega/2}T_2S_{i,2}
}{
q_{d,4}+Y_d e^{-j\Omega/2}T_2S_{r,2}
}.
\]

這不是任意 curve fitting；空間與時間係數都由對 wall location／half step
的二次 Lagrange interpolation 唯一決定。

#### frozen grid harmonic gate

在 168–300 Hz、三個 frozen full-room grids、canonical directions 與 63
個 actual order-12 early-path representative directions 上比較四組方程：

| equation | actual worst complex | magnitude | phase |
|---|---:|---:|---:|
| second-order + cell center | 0.1665 | 0.656 dB | 9.75° |
| second-order + linear face/time | 0.1390 | 0.582 dB | 7.99° |
| fourth-order + linear face/time | 0.02372 | 0.0349 dB | 2.10° |
| fourth-order + quadratic face/time | **0.00750** | **0.0458 dB** | **0.595°** |

最後一組的 actual-direction mean complex／magnitude／phase error 是
0.00229、0.0104 dB、0.165°；canonical worst 是
0.00750、0.0459 dB、0.595°。因此它同時通過既定
0.02 complex、0.10 dB、1° gate。只用 fourth-order interior 而保留
linear predictor 仍不通過，表示 interior 與 boundary centering 必須一起
處理。

所有掃描點仍滿足 \(|\Gamma_d|\le1\)。frozen grids 的四階 CFL number
最大只有 0.3927，低於 harmonic stability bound 1。將空間與時間一起縮小：

| linear grid scale | worst complex | worst phase | max \(|q_{d,4}-\cos\theta|\) |
|---:|---:|---:|---:|
| 1 | 0.00750 | 0.595° | 0.00154 |
| 1/2 | 0.00106 | 0.0640° | 0.000405 |
| 1/4 | 0.000267 | 0.0158° | 0.000103 |
| 1/8 | 0.0000669 | 0.00396° | 0.0000257 |

所以候選方程不是只在單一 coarse grid 偶然抵銷誤差；reflection error 與
characteristic-admittance error 都隨 refinement 單調下降。

M3.7 的正式 decision 是：

- 接受 `fourth_order_quadratic_face_time` 為唯一通過的 **harmonic
  candidate**；
- 不把它稱為已驗證的 time-domain FDTD reference，因為 near-wall
  fourth-order closure 與 cross-ratio probe 尚未實作；
- 不改 `FDTDReferenceConfig` 的 production/default `cell_center`；
- 不重跑 full-room crossover；
- 下一步先建立 1D fourth-order time-domain prototype，以 cross-ratio
  比較實測 reflection 與本節 harmonic equation；通過後才移植到 3D。

完整報告位於
`egs/rir_generation/phases/m3_wave_path/reports/higher_order_fdtd_candidate_m3_7_report.json`。

### 10.21 M3.8：把 harmonic candidate 落成 1D time-domain closure

M3.7 只證明「如果 time-domain update 具有指定的 Fourier symbol 與
boundary transfer」，reflection 會通過 gate。真正實作時，四階 centered
stencil 在靠牆處會要求 domain 外的 pressure／velocity samples；若任意退回
二階差分，near-wall defect 可能重新改變 reflection。因此 M3.8 先在 1D
建立明確、可檢查的 closure，而不直接修改 3D production solver。

#### interior 與第一層 near-wall stencil

pressure \(p_j\) 位於 \((j+1/2)\Delta x\)，velocity \(u_i\) 位於
\(i\Delta x\)。距邊界足夠遠時，對兩種場都使用 M3.7 的 centered
fourth-order staggered derivative：

\[
\Delta x\,D_c f_i
=
\frac98(f_i-f_{i-1})
-
\frac1{24}(f_{i+1}-f_{i-2}).
\]

第一個 interior velocity face 與第一個 pressure cell 無法使用這個
stencil。以相對 evaluation point 的四個 locations
\(-1/2,1/2,3/2,5/2\) 建立 cubic interpolation，再對 evaluation point
微分，可得

\[
\Delta x\,D_L f
=
-
\frac{23}{24}f_0
+
\frac78f_1
+
\frac18f_2
-
\frac1{24}f_3.
\]

這組 one-sided weights 對三次以下多項式的 derivative 是精確的；右端
closure 使用鏡射後的 weights。它同時用於：

- 由 \(p_0,p_1,p_2,p_3\) 更新第一個 interior velocity face；
- 由 \(u_0,u_1,u_2,u_3\) 更新第一個 pressure cell；
- far rigid wall 的對稱位置。

test wall 的 boundary velocity \(u_0\) 不由 derivative stencil 更新，而是
由 M3.7 的 admittance filter 與 quadratic face/time pressure 決定：

\[
\begin{aligned}
p_f^n
&=
\frac{15}{8}p_0^n-\frac54p_1^n+\frac38p_2^n,\\
\hat p_f^{n+1/2}
&=
\frac{15}{8}p_f^n-\frac54p_f^{n-1}+\frac38p_f^{n-2},\\
u_0^{n+1/2}
&=
-\frac{Y_d(z)\hat p_f^{n+1/2}}{\rho c}.
\end{aligned}
\]

每個 sample 的順序是：更新 interior velocity、計算 boundary pressure
predictor、寫入 boundary velocity、更新全部 pressure、加入 source。兩份
face-pressure history 都是顯式狀態，因此 update 是 causal。

#### 20 m cross-ratio 與 joint refinement

沿用 M3.6 的 20 m target／rigid／constant-reference 三次模擬。direct path
與不經 test wall 的 contributions 在 cross-ratio 中消失；比較的是實際
time-domain closure 與 M3.7 harmonic reflection equation，而不是只看一條
波形是否相似。

168–300 Hz 結果：

| linear scale | worst complex | magnitude | phase | mean complex |
|---:|---:|---:|---:|---:|
| 1 | 0.000276 | 0.00184 dB | 0.00471° | 0.000104 |
| 1/2 | 0.0000156 | 0.000108 dB | 0.000345° | 0.00000616 |
| 1/4 | 0.0000184 | 0.0000912 dB | 0.000716° | 0.000000963 |

三組都遠低於 0.02 complex／1° gate。1/4 grid 的 single-bin worst complex
比 1/2 稍高，發生在 183 Hz，量級仍只有 \(1.84\times10^{-5}\)；mean
complex error 持續下降，所以不把 analysis-window leakage 誤報成 closure
不收斂。

另執行 1 秒 passive stability smoke。base grid 的一維 fourth-order CFL
number 是 0.2267，所有 samples finite，最大絕對 pressure 2.575，
0.9–1.0 s tail RMS 與 0.1–0.2 s early RMS 比值為 0.2367。這是實證的長時
smoke，不等於一般化的 discrete energy proof。

M3.8 的正式 decision 是：

- 接受 `third_order_one_sided` near-wall closure 與
  `face_quadratic_time_quadratic` boundary state 的 1D time-domain
  prototype；
- 接受 normal-incidence time-domain equation 與 M3.7 harmonic equation
  一致；
- 尚未接受 3D reference，因為 tangential derivatives、edge/corner
  closures 與 oblique time-domain reflection 都未驗證；
- production/default `cell_center` 保持不變，也不重跑 full-room
  crossover；
- 下一步將同一組 operator 以 opt-in 模式移植到 3D reference，先做
  plane-wave／oblique validation，再決定是否用它裁決 crossover。

完整報告位於
`egs/rir_generation/phases/m3_wave_path/reports/higher_order_fdtd_time_domain_m3_8_report.json`。

### 10.22 M3.9：opt-in fourth-order 3D FDTD reference

M3.8 只更新一個 normal 軸。M3.9 將同一組 operator 真正移植到
`simulate_fdtd_reference` 的三個 velocity grids 與 pressure grid，但仍
保留原本二階 solver 作為 default。

#### 三軸 update 與六面 closure

新的 opt-in config 必須同時指定：

```python
FDTDReferenceConfig(
    spatial_derivative_order=4,
    near_wall_closure="third_order_one_sided",
    boundary_pressure_scheme="face_quadratic_time_quadratic",
)
```

對 \(u_x,u_y,u_z\) 的 interior pressure gradients，分別沿自己的 staggered
軸使用 fourth-order centered stencil；每一軸的第一／最後 interior
velocity face 使用 M3.8 的 mirrored one-sided stencil。更新 pressure 時，
先對三個 velocity grids 做相同的 fourth-order divergence，再相加：

\[
p^{n+1}
=
p^n-\rho c^2\Delta t
\left(
D_xu_x+D_yu_y+D_zu_z
\right).
\]

因此 face 上只有一個 one-sided contribution，edge 有兩個，corner 有三個；
它們是三個可分離方向的物理 divergence 累加，不是另外乘上的 edge/corner
reflection gain。

六個 wall faces 各自保存
\(p_f^{n-1},p_f^{n-2}\)，先沿 face normal 以
\(15/8,-5/4,3/8\) 外推 wall pressure，再用相同三係數預測 half-time。
admittance 的 relaxation／resonant states 仍逐 face cell 更新，所以
second-order production path 與既有 material models 不需要改格式。

#### bulk CFL 不足以保證 closure stability

第一次直接使用 M3.7 frozen-grid 的 effective CFL 約 0.39。harmonic
reflection 仍被動，但長 3D run 會由浮點 roundoff 激發 transverse
near-wall mode，最後 overflow。把 CFL 降至 0.30 的 normal run 與 0.25 的
0.6 s oblique run 則保持 finite。

這說明兩件事：

- bulk Fourier symbol 的 CFL bound 只描述無限／週期 interior；
- one-sided closure、quadratic time predictor 與 edge/corner coupling
  仍可能給出更嚴格的 timestep limit。

目前 opt-in fourth-order path 因此強制

\[
\mathrm{CFL}_{\mathrm{effective}}
\le 0.25,
\]

並在 result metadata 序列化 `effective_interior_cfl`。這是由目前 closure
實證得到的保守 cap，不宣稱已完成一般化 energy-stability proof。二階
default 不套用這個 cap。

#### normal 3D plane-mode reduction

solver 新增 validation-only 的 spatial source weights、receiver projection、
controlled source signal 與 DC-removal bypass；預設 point source／point
receiver／Ricker／DC removal 完全不變。

先在 tangential directions 使用 uniform pressure sheet。理論上所有
tangential derivatives 都是零，所以 3D update 必須退化成 M3.8 1D
equation。sample-by-sample 測試在 \(10^{-12}\) tolerance 內一致。

再以 20 m target／rigid／constant-reference cross-ratio 比較
168–300 Hz：

| gate | worst complex | magnitude | phase |
|---|---:|---:|---:|
| 3D normal plane mode vs discrete equation | 0.000294 | 0.00195 dB | 0.00520° |

第一次測試曾得到約 0.0127 complex；原因不是 wave update，而是一般 RIR
輸出會對每個 wall model 各自減去 tail mean，之後再做 window subtraction
不再代數等價。plane-mode transfer gate 關閉這個 validation-irrelevant
後處理後，即恢復與 1D 同量級的結果。

#### oblique plane mode 與雙探針分解

固定 tangential rigid walls 時，連續 cosine 不一定是 one-sided discrete
operator 的精確 eigenvector。因此先建立 pressure-to-velocity gradient
\(G_t\) 與 velocity-to-pressure divergence \(H_t\)，解

\[
H_tG_t\phi_m=\lambda_m\phi_m.
\]

用 \(\phi_m\) 同時作 source sheet 與 receiver projection，可避免把
不同 tangential modes 混在一起。其等效 fourth-order tangential symbol 是

\[
K_{t,m}
=
\frac{\sqrt{-\lambda_m}}{2\Delta_t},
\]

再反解成 harmonic equation 所用的 \(k_t\)。

near-cutoff broadband pulse 的 group velocity 對頻率變化很大，單一時間窗
無法同時完整保留所有 first reflections 並排除下一次 wall interaction；
早期實驗因此出現 0.45 complex 的假 mismatch。正式 gate 改用
270/285/300 Hz steady harmonic drive。在 west wall 與 source plane
之間量測兩個 pressure probes：

\[
\begin{bmatrix}P(x_1)\\P(x_2)\end{bmatrix}
=
\begin{bmatrix}
e^{jk_nx_1}&e^{-jk_nx_1}\\
e^{jk_nx_2}&e^{-jk_nx_2}
\end{bmatrix}
\begin{bmatrix}A_i\\A_r\end{bmatrix},
\qquad
\Gamma_{\mathrm{measured}}=\frac{A_r}{A_i}.
\]

這個分解不依賴 arbitrary reflection window，也不受 source 另一側 cavity
paths 影響。270 Hz、incidence cosine 約 0.25 的 mode 收斂最慢；1 s 時
尚有 0.0206 complex／1.39° transient bias，延長到足夠的 round trips 後
降至 0.000608／0.0253°。

三個正式 frequencies 的 cosine 範圍是 0.250–0.490，總結為：

| gate | worst complex | magnitude | phase |
|---|---:|---:|---:|
| 3D oblique measured vs discrete equation | 0.000608 | 0.00592 dB | 0.0253° |

M3.9 的正式 decision 是：

- 接受 fourth-order／quadratic solver 為 **opt-in 3D reference**，但只限
  已驗證的 plane-mode scope；
- 接受 `effective_interior_cfl <= 0.25` 為目前 closure 的必要保守限制；
- 不改 production/default 二階 `cell_center`；
- 尚不重跑 full-room crossover，因為目前只驗證 normal axis \(x\)、
  一個 \(y\)-tangential rigid mode 與三個 oblique frequencies；
- 下一步補 normal axis \(y/z\)、兩個 tangential axes／azimuths、不同
  mode indices，以及 edge/corner holdouts，之後才重新打開 full-room
  early/crossover attribution。

完整報告位於
`egs/rir_generation/phases/m3_wave_path/reports/fourth_order_fdtd_3d_m3_9_report.json`。

預計驗證順序為：

```text
單一邊界、法向入射的 reflection magnitude/phase
    -> 矩形房間單模態的頻率與 Q
    -> 多頻帶、多材料的小型 FDTD
       （目前完成同材料兩種厚度、跨房間／位置／網格）
    -> 合成 bank 的 modal distribution
    -> room-disjoint measured bank
    -> 固定訓練配方的 synthetic-to-real 結果
```

只有前一層通過，才進入下一層。這可避免在大型 bank 或模型訓練中，用昂貴
實驗掩蓋一個在單牆反射就能發現的物理錯誤。

### 10.23 M3.10：擴展 3D reference，而不是只相信單一平面波

M3.9 的一組 x-normal plane mode 只能證明「那一個方向能工作」。三維
operator 可能在軸交換、兩個切向波數同時非零、或靠近 edge/corner 的點源
幾何中出現不同誤差。因此 M3.10 固定同一套 fourth-order operator 與
quadratic face/time boundary，新增四類未參與先前選型的 holdout：

1. x-normal、y-tangential；
2. y-normal、z-tangential；
3. z-normal、x-tangential；
4. x-normal，且 y/z 兩個不相等的 tangential mode 同時存在。

在 285 Hz，time-domain 雙探針分解的最差 complex reflection error 是
\(4.42\times10^{-4}\)，最差 phase error 是 \(0.0268^\circ\)。這些值都
明顯低於預先固定的 \(0.02\) 與 \(1^\circ\) gate。三個 normal 軸與
非對稱 azimuth 因此都通過，而不是靠 x/y/z 對稱的假設直接宣稱通過。

#### 點源互易性為什麼需要另外處理

連續、線性、時不變且具有 reciprocal boundary 的聲場 Green function
應滿足

\[
G(\mathbf{x}_r,\mathbf{x}_s)
=G(\mathbf{x}_s,\mathbf{x}_r).
\]

但目前 staggered fourth-order closure 的「在 pressure cell 加一個點源」
與「讀取另一個 pressure cell」並不是離散 operator 下嚴格互為 adjoint
的 source/readout pair。靠近 face、edge、corner 時，raw point-source
reciprocity NRMSE 最差可到 0.637。這不是可以隱藏的數值細節，因此 reference
API 同時保存 raw 誤差，並明確提供雙向 Green 平均：

\[
G_{\mathrm{recip}}(s,r)
=\frac{G_{\mathrm{raw}}(s,r)+G_{\mathrm{raw}}(r,s)}{2}.
\]

`simulate_reciprocal_fdtd_reference()` 實際執行 forward 與 reverse 兩次
simulation，回傳上式結果，並在 metadata 寫入
`raw_reciprocity_nrmse` 與 `reciprocity_averaged=true`。三個
face/edge/corner holdout 的 reciprocalized NRMSE 都是 0。這表示它可作為
reciprocal reference transfer 使用；它不等於證明 raw update 本身已成為
energy-conserving／self-adjoint discretization。

開發中也試過 `face_quadratic_time_centered` 加 mirrored mimetic divergence，
希望直接修復 raw reciprocity。它在穩定性上可運作，但實際 reflection
phase 約有 \(12^\circ\) 級誤差，沒有通過既有 plane-mode 物理 gate，因此
保留為實驗路徑，沒有取代 M3.9 operator，也沒有改 production default。

### 10.24 M3.11：用通過擴展 gate 的 reference 重做 full-room 歸因

M3.11 沒有把 M2.12 的 second-order failure report 改名當成新結果。它以
`fourth_order_reciprocal` profile 重新執行三個 frozen room cases，每個
source/receiver pair 都使用 M3.10 的雙向平均 reference，並保留原本不 fitting
scalar gain 的規則。

240 Hz production proxy 仍未通過 full-room complex gate：

| 分支 | mean correlation | mean NRMSE | energy ratio |
|------|-----------------:|-----------:|-------------:|
| modal low，60–240 Hz gate | 0.935 | 0.367 | 0.902 |
| hybrid magnitude-only normal | 0.399 | 4.567 | 4.217 |
| hybrid complex-angle | 0.759 | 4.594 | 4.049 |
| hybrid causal PathEvent order 12 | 0.759 | 4.594 | 4.049 |

120 Hz、complex-angle 是 crossover scan 中最好的診斷點，但
correlation／NRMSE／energy ratio 仍只有
0.869／1.622／0.702，不能宣稱 full-room gate 通過。8 ms modal onset 也只
改善部分數值，沒有被包裝成 production 修正。

重新做 direct-anchor／early／later 歸因後：

- direct anchor correlation 近 1、NRMSE 為 0；
- early-reflection component correlation 0.953、NRMSE 0.315、
  energy ratio 1.004；
- later component correlation 0.730、NRMSE 2.477、energy ratio 2.447；
- early component 的 error norm 相對 full FDTD norm 是 1.069；
- later component只有 0.240。

因此結論仍是：主要 full-transfer 誤差來自 early reflection phase／coherent
interference，並且 modal crossover raw branch 在上緣過度增益；不能把問題
簡化為「缺少更多 late rays」。M3.11 完成的是 reference-qualified audit 與
可重現歸因，不是宣稱 240 Hz full-room crossover 已被修好。

### 10.25 M3.12：先評估既有 mesh engine，再決定自己實作多少

我們以 M3 的必要契約檢查本機 Pyroomacoustics 0.10.1：

- general 3D polygon walls 與 non-convex outer-room visibility 可用；
- 一個 non-convex 3D smoke case 能產生有限、非零 RIR；
- frequency-dependent energy absorption 與 controlled energy scattering
  可用；
- public API 不提供 ordered complex causal `PathEvent` filters；
- 也沒有 interior solid furniture transmission、edge diffraction 與
  lossless PathEvent serialization 的共同契約。

因此決策不是「Pyroomacoustics 不好」，而是分工：

- Pyroomacoustics 繼續是 production default 與獨立 visibility cross-check；
- PureSound 的 `PathEvent` 保持可檢查、可序列化的 authoritative event
  contract；
- M3 只新增目前需要的 vertical-prism furniture geometry，不重寫一個泛用
  triangle-mesh ray tracer。

`trimesh` 在此環境未安裝，而且即使安裝，它本質上也只解 geometry
intersection，不會自動提供 acoustic filter、diffraction 或 energy
partition，所以不能單獨滿足 M3。

### 10.26 M3.13／M3.14：家具真正進入路徑幾何

#### SceneObject 是閉合的垂直稜柱

每件家具由平面多邊形 \(P\)、高度區間
\([z_{\min},z_{\max}]\) 與 material 定義：

\[
\mathcal O
=\{(x,y,z)\mid (x,y)\in P,\ z_{\min}\le z\le z_{\max}\}.
\]

schema 驗證多邊形至少三點、面積非零、係數有限且在 \([0,1]\)、物件完全
位於房間內，並拒絕 source 或 receiver 落在 solid object 內。`transmission`
是能量係數；舊 JSON 沒有此欄時以 0 讀取，因此仍可向後讀取。

對每個 `PathEvent`，先重建折線頂點：

\[
\mathbf{s}\rightarrow \mathbf{q}_1\rightarrow\cdots
\rightarrow\mathbf{q}_K\rightarrow\mathbf{r},
\]

其中同一 `interaction_group_id` 的 edge/corner simultaneous hits 只算一個
頂點。每一段都做：

1. z 軸 slab interval；
2. xy 線段與閉合多邊形的 intersection interval；
3. 兩個 interval 有重疊即代表穿過該 vertical prism。

只要任何一段被任何物件截住，specular event 就標記
`visible=false`，metadata 同時保存 event 到 occluder IDs 的對應。這個
判定使用完整 3D 高度，所以一條越過矮桌上方的 ray 不會被 2D footprint
誤殺。source／receiver 對調會遍歷相同線段集合，因此 visibility
保持 reciprocal。

#### 穿透：能量係數必須先轉成壓力增益

對被擋住的 direct segment，依路徑順序求每個物件的 entry／exit 點。若
穿過物件集合 \(\mathcal B\)，總能量 transmission 是

\[
\tau_{\mathrm{path}}=\prod_{o\in\mathcal B}\tau_o.
\]

RIR 儲存的是聲壓振幅，因此 event gain 乘上

\[
g_T=\sqrt{\tau_{\mathrm{path}}},
\]

不是直接乘 \(\tau_{\mathrm{path}}\)。entry／exit surface、distance、
delay 與 `energy_partition_fraction` 都會序列化。

#### 繞射：目前是有界、reference-frequency 的 edge model

對每個 blocker 的垂直 edge，找 source-edge-receiver 的可見 detour，
依總路徑長度排序並保留最短兩條。設相對直接路徑的 excess distance 為
\(\Delta d\)，reference wavelength 為 \(\lambda=c/f_{\mathrm{ref}}\)，則

\[
v=\sqrt{\frac{2\max(0,\Delta d)}{\lambda}},
\qquad
g_D=
\frac{0.5\sqrt{\max(0,1-\alpha-\tau)}}{\sqrt{1+v^2}}.
\]

實際壓力 gain 為 \(g_D/d_D\)。這個
`puresound.bounded_fresnel_edge.v1` 保證有限且不憑空放大可用反射能量，
但它不是完整 UTD，也只在目前的 1 kHz reference frequency 建立 scalar
event。頻率相依 diffraction 留給後續模型。

#### 受控散射：守恆的 deterministic energy partition

目前只對 visible first-order wall reflection 建立散射分支。若材料在
reference frequency 的 scattering coefficient 是 \(s\)，則：

\[
E_{\mathrm{specular}}=(1-s)E_{\mathrm{parent}},
\qquad
E_{\mathrm{scatter},j}=\frac{s}{N}E_{\mathrm{parent}}.
\]

所以 pressure gain 分別乘
\(\sqrt{1-s}\) 與 \(\sqrt{s/N}\)。四個 deterministic nearby surface
samples 保證每次生成相同，且每個 parent 的 partition sum 精確等於 1。
散射候選仍要重新通過家具 visibility，不能因為名稱是 diffuse 就穿牆。

正式 controlled scene 產生 1 個 transmission event、2 個 diffraction
events 與 12 個 visible scattering events；forward/reverse 各 22 events，
最大 path-distance reciprocity error 為 0。visibility、transmission、
diffraction、energy partition、serialization 與 time rendering gates
全部通過。

#### Directivity 與 interaction metadata

`speech_cardioid` 依 source yaw/pitch 形成 forward unit vector
\(\hat{\mathbf f}\)，每條 event 的 pressure gain 為

\[
g_{\mathrm{cardioid}}
=\operatorname{clip}\left(
\frac{1+\hat{\mathbf f}\cdot\hat{\mathbf d}_{\mathrm{departure}}}{2},
0,1\right).
\]

因此 direct 與不同 reflection departure directions 不再共用一個事後
scalar。receiver 目前仍只接受 omni。每條 event 還保存 diffraction model、
scattering model、energy partition 與 interaction types；未知 directivity
不會默默當 omni。

### 10.27 M3 production opt-in、量測出口與最終結論

`PathEventHighFrequencyBackend` 把 M3 路徑接入既有 hybrid generator。它：

- 只接受 material-first `RoomSceneV2`；
- 由 1 kHz 的 \(\alpha,\tau\) 計算剩餘反射能量
  \(1-\alpha-\tau\) 與 pressure reflection
  \(\sqrt{1-\alpha-\tau}\)；
- 生成最高指定 image order 的 exact paths；
- 套用家具 visibility、transmission、diffraction、first-order scattering
  與 source cardioid；
- 以 shared causal fractional delay 渲染；
- 保持既有 Pyroomacoustics backend 為預設。

CLI 必須明確選擇：

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m3/path_events \
  --scene-version v1 \
  --high-backend path-events-m3 \
  --pra-max-order 8 \
  --low-backend analytic \
  --n-rooms 1 --rir-per-room 1 \
  --sample-rate 16000 --duration 1.0
```

`--pra-max-order` 在這個 opt-in backend 中沿用為 PathEvent max order，
限制為 0–20。這是為了不破壞既有 recipe 的參數介面；metadata 仍會寫出
實際 backend class。未指定 `--high-backend` 時行為完全不變。

#### 量測 exit 不是拿 M3 high-only 去比 M1 hybrid

公平比較必須讓兩邊都是完整 RIR。正式 protocol 使用：

- frozen M1 paired bank 的 20 items／100 channels；
- 完全相同的 scene geometry、source/receiver、config；
- 相同 analytic modal low backend；
- M1 使用 frozen Pyroomacoustics high output；
- M3 只把 high branch 換成 opt-in PathEvent backend；
- held-out measured RIR view 抽 100 channels，seed 20260731；
- 依距離分桶比較中位數，再平均共同且樣本數足夠的 bucket gap。

早期反射 timing 不使用「2.5 ms 後第一個 local peak」：該方法幾乎總是抓到
direct-filter 殘留。也不以單一最高 peak 作 exit metric，因為兩條 coherent
reflection 稍微相消就會交換最高峰身份，造成不連續跳動。正式 timing 定義
是 direct-excluded 2.5–50 ms early-energy centroid：

\[
t_E=
\frac{\sum_{n=n_D+n_{2.5}}^{n_D+n_{50}}
((n-n_D)/f_s)\,h[n]^2}
{\sum_{n=n_D+n_{2.5}}^{n_D+n_{50}}h[n]^2}.
\]

它仍是 waveform distribution proxy，不是 measured path annotation，但能
連續反映整組早期能量的到達位置，也與 C50 共用 50 ms 邊界。

結果：

| 指標 | M1 mean absolute median gap | M3 gap | 相對改善 |
|------|----------------------------:|-------:|---------:|
| C50 | 5.289 dB | 3.239 dB | 38.8% |
| early-energy centroid | 3.046 ms | 2.243 ms | 26.4% |

診斷用 dominant-peak gap 則由 1.104 ms 退步到 2.740 ms。這項負結果保留
在正式 report；它提醒我們 M4 需要更好的 spatial multiband late／diffuse
field 與更連續的 shadow-boundary diffraction，而不是宣稱所有 waveform
peak 都已改善。

M3 exit 的四類要求全部通過：

| Exit requirement | 證據 |
|------------------|------|
| path delay 符合 geometry | 最大 distance error \(8.88\times10^{-16}\) m；最大 delay error \(3.47\times10^{-18}\) s |
| 鄰近位置路徑連續 | 1 mm 擾動保持 event IDs，最大 distance-change/displacement 0.951 |
| 預期範圍內 reciprocity | PathEvent 最大 \(8.88\times10^{-16}\) m；scene interaction 0；reciprocalized FDTD NRMSE 0 |
| measured timing 與 C50 改善 | early centroid 26.4%，C50 38.8% |

因此 M3 已完成，下一個里程碑是 M4 late field、echo density、multiband
decay 與 spatial output。這不會抹去 M3.11 的 full-room 240 Hz complex
crossover failure；該 failure 與 M4 late-field 工作都保留為明確限制。

### 10.28 M4.1：先定義「何時成為後期擴散場」

M3 能逐條說明 direct 與 early paths 從哪裡來，但若一直枚舉 coherent
高階路徑，成本會快速增加，而且規則的路徑間距容易形成 comb／metallic
coloration。M4 因此把問題拆成兩段：

1. direct 與 early response 繼續由可檢查的 PathEvents 負責；
2. 到達 mixing region 後，改由高 echo density、frequency-dependent、
   可產生多輸出的 late-field renderer 負責。

這裡的 **mixing time** 不是 RT60。RT60 說明能量衰減多快；mixing time
說明原本可分辨、稀疏的反射何時密集到可視為統計性尾場。M4.1 採用
[Abel 與 Huang 的 normalized echo density](https://research.aalto.fi/en/publications/a-simple-robust-measure-of-reverberation-echo-density/)
作第一個可重現定義；公式與 FDN mixing-time 討論亦可見
[Schlecht 的 FDN 論文集／博士論文](https://theses.eurasip.org/media/theses/documents/sebastian-jiro-schlecht-feedback-delay-networks-in-artificial-reverberation-and-reverberation-enhancement.pdf)。

#### Normalized echo density

在以 \(n\) 為中心的局部 window \(W_n\) 中，先計算標準差
\(\sigma_n\)，再計算超過一個標準差的樣本比例：

\[
\eta(n)=
\frac{
  \frac{1}{|W_n|}
  \sum_{\tau\in W_n}
  \mathbf{1}\{|h[\tau]|>\sigma_n\}
}{
  \operatorname{erfc}(1/\sqrt{2})
}.
\]

其中

\[
\operatorname{erfc}(1/\sqrt{2})\approx 0.3173105
\]

是零均值高斯變數落在正負一個標準差以外的理論比例。因此：

- \(\eta\approx 1\)：局部振幅的 exceedance rate 接近高斯擴散尾場；
- \(\eta\ll 1\)：脈衝仍稀疏；
- \(\eta>1\) 是允許的有限樣本結果，實作不會強制 clip；
- 對整條 RIR 乘固定 gain 不會改變 \(\eta\)。

PureSound 的 `abel_normalized_echo_density_profile()` 使用 20 ms full
window 與 1 ms hop。第一個 window 從 direct sample 開始，時間軸以 direct
為零；不把 pre-arrival silence 混入統計。若直接使用原始定義，
mixing time 是

\[
t_\mathrm{mix}
=
\min\{t:\eta(t)\ge 0.9\}.
\]

有限 window 的 profile 會明顯波動，所以正式 bank baseline 另記錄一個
明確的工程延伸：必須連續 10 ms 高於 0.9，回傳該段的起點。原始
first crossing 仍可用 `minimum_sustain_ms=0` 重現，兩種定義不會混成
同一個未版本化數值。

相關 API：

- `abel_normalized_echo_density_profile()`：回傳 direct-relative 時間與
  完整 \(\eta(t)\)；
- `estimate_abel_mixing_time()`：原始或 sustained threshold crossing；
- `analyze_echo_density()`：JSON-safe mixing time、20/50/100/200 ms
  probes、末段 median；
- `analyze_rir(..., echo_density=True)`：opt-in 加入摘要，既有呼叫預設
  不增加成本或欄位。

#### measured／M1／M3 baseline

固定 seed `20260731`，從每個 item 最多抽一個 source channel，使用 100
個 measured、100 個 M1，以及 10 個 opt-in M3 room：

| bank | mixing coverage | median mixing time | NED 20 ms | NED 50 ms | NED 100 ms | NED 200 ms | late median NED |
|------|----------------:|-------------------:|----------:|----------:|-----------:|-----------:|----------------:|
| measured | 100% | 16.0 ms | 0.952 | 0.987 | 1.026 | 1.021 | 1.500 |
| M1 | 100% | 21.0 ms | 0.908 | 1.016 | 0.987 | 1.021 | 1.188 |
| M3 | 100% | 20.5 ms | 0.977 | 0.992 | 1.036 | 1.232 | 1.154 |

M1 與 M3 的六個 broadband median checks 全部落在 measured 的
10th–90th percentile envelope 內。這是重要的**負結果**：它不能解讀成
M1/M3 late field 已經和真實房間相同，只能說 broadband、monophonic
NED 對目前差異沒有足夠辨識力。Abel criterion 本來就不量測 octave
decay、metallic coloration、方向性、array coherence 或 IACC；因此 M4
不能以這 6/6 當 exit。

完整結果在
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_late_field_baseline.json`，validator 是
`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_late_field_baseline.py`。

#### M4 renderer 決策：PathEvent early + multiband FDN late

M4 選擇 FDN 作 dense tail，而不是把無限多高階 paths 直接渲染成
waveform。FDN 的典型 transfer 可寫成

\[
H(z)
=
\mathbf{c}^{T}
\left[
\mathbf{D}(z^{-1})-\mathbf{A}(z)
\right]^{-1}
\mathbf{b}
+d,
\]

其中 \(\mathbf{D}\) 是多條不同長度的 delay lines，\(\mathbf A\) 是
feedback mixing 與 frequency-dependent attenuation，
\(\mathbf b,\mathbf c\) 分別是 injection/output mapping。lossless core
會使用正交／酉 feedback mixing；相關穩定性結構見
[Schlecht 與 Habets 的 lossless FDN formulation](https://arxiv.org/abs/1606.07729)。

對 delay line \(i\)，若 traversal 時間為
\(\tau_i=d_i/f_s\)，目標 octave-band RT60 為 \(T_{60}(f)\)，每次 loop
所需的 pressure gain 是

\[
g_i(f)=10^{-3\tau_i/T_{60}(f)}.
\]

因為 pressure 在 \(T_{60}\) 後必須下降 60 dB，也就是振幅乘
\(10^{-3}\)。所有有限正 RT60 都給出 \(0<g_i(f)<1\)，再由被動
frequency-dependent loop filter 逼近各 band gains；frequency-dependent
FDN 衰減控制可參考
[Schlecht 與 Habets 的濾波器設計](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_46.pdf)。

後續順序已固定：

1. M4.2 先增加 octave-band NED／decay 與 noise-aware target；
2. M4.3 實作 deterministic、passive、multiband FDN core；
3. M4.4 才在 mixing region 接上 PathEvent early response；
4. M4.5 加 Ambisonic／array output，凍結 coherence 與 IACC gate；
5. M4.6 最後才加完整 source/receiver directivity 與 optional HRTF。

也就是說，我們不會先聽一個「殘響很濃」的 FDN 就宣稱完成；先用
multiband 與 spatial measurement 說明它改善了哪個可觀察量，再做
room-disjoint listening/downstream validation。

### 10.29 M4.2：multiband target 與 spatial contract

M4.1 的 broadband 結果讓 M1/M3 都得到 6/6，表示把所有頻率混在一起會
掩蓋問題。M4.2 因此同時處理兩件事：把 late-field 指標拆成 octave
bands，並定義未來多 receiver output 的空間驗收方法。

#### 為何低頻不能固定使用 20 ms window

20 ms 在 4 kHz 有 80 個週期，在 125 Hz 卻只有 2.5 個週期。若每個
octave 都硬套相同 window，低頻 NED 主要反映短窗相位，而不是 reflection
density。對 center frequency (f_c)，M4.2 使用

\[
T_W(f_c)=
\max\left(
20\ \mathrm{ms},
\frac{4}{f_c/\sqrt{2}}
\right),
\]

也就是在 octave lower edge 至少觀察四個週期。125 Hz 的 window 約
45.3 ms，250 Hz 約 22.6 ms，500 Hz 以上維持 20 ms。63 Hz 仍以低頻
modal/wave validation 為主，不先假設它已進入 diffuse late field，所以
本階段 target 固定為 125、250、500、1000、2000、4000 Hz。

每一 band 都保留 broadband direct sample 作共同物理時間原點，不讓
band-pass 後較大的 late peak 被誤認為新的 direct arrival。

#### Noise-aware echo density 與 decay

真實量測的末端通常含底噪。如果直接在最後 25% 算 NED，stationary
Gaussian noise 很容易得到 η≈1，看起來像完美 diffuse tail。M4.2 先對
每個 octave 執行既有 Lundeby-style analysis：只有在 dynamic range、
stationarity 與 intersection 都可靠時，才在 decay/noise intersection
截斷 echo-density profile；否則保留完整 response 並記錄失敗原因。

同一條 corrected Schroeder curve 產生 EDT/T20/T30，所有 fit 都保存
R²。bank distribution 只接受 R²≥0.9 的 decay；coverage 另外報告，避免
「只剩少數可 fit channels」被一個漂亮 median 隱藏。

#### M4.2 measured／M1／M3 結果

固定 seed `20260731`，抽 50 measured、50 M1 與全部 10 個 M3 probe：

| band | measured mixing | M1 mixing | M3 mixing | measured late NED | M1 late NED | M3 late NED |
|------|----------------:|----------:|----------:|------------------:|------------:|------------:|
| 125 Hz | 22.6 ms | 38.6 ms | 22.6 ms | 1.124 | 1.080 | 1.057 |
| 250 Hz | 19.3 ms | 28.3 ms | 11.3 ms | 1.124 | 1.270 | 1.107 |
| 500 Hz | 16.0 ms | 26.0 ms | 18.0 ms | 1.070 | 1.210 | 1.235 |
| 1 kHz | 18.0 ms | 22.0 ms | 41.0 ms | 1.038 | 1.212 | 1.178 |
| 2 kHz | 24.0 ms | 32.0 ms | 152.0 ms | 1.021 | 1.212 | 1.168 |
| 4 kHz | 24.0 ms | 46.0 ms | 149.0 ms | 1.001 | 1.198 | 1.163 |

48 個可比較的 band/metric median checks 中，M1 通過 31、M3 通過 32。
這比 M4.1 的 broadband 6/6 有辨識力。最重要的結果是：

- M3 在 2/4 kHz 的 mixing time 為 152/149 ms；measured median 是
  24/24 ms，p90 只有 70.2/88.2 ms，表示 M3 coherent high paths 太久才
  進入密集尾場；
- M1 雖然 late-quarter median 偏高，1/2/4 kHz 在 400 ms 的 NED 只有
  0.515/0.471/0.407，低於 measured p10 0.929/0.918/0.884；這表示其高頻
  尾場密度維持方式不對，而不是只需增加一個 broadband gain；
- measured 各 band 有 16–38% 使用可靠 noise intersection，這些 channel
  若不截斷會把底噪當成真實 tail；目前 deterministic M1/M3 是 0%。

正式 report 是
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json`。

#### Binaural IACC 契約

若左右耳 RIR 為 (h_L,h_R)，時間窗 ([a,b]) 的 IACC 定義為

\[
\mathrm{IACC}_{a,b}
=
\max_{|\tau|\le 1\ \mathrm{ms}}
\left|
\frac{
\int_a^b h_L(t)h_R(t+\tau)\,dt
}{
\sqrt{\int_a^b h_L^2(t)\,dt\int_a^b h_R^2(t)\,dt}
}
\right|.
\]

M4.2 固定 early 為 direct 後 0–80 ms、late 為 80 ms 到 response 結束，
並分別輸出 broadband IACC_E/IACC_L。IACC_E4 與 IACC_L4 是 500、1000、
2000、4000 Hz 四個有效 octave 值的平均；80 ms early/late 分法與
[Hidaka、Beranek、Okano 的 IACC 研究](https://doi.org/10.1121/1.404472)
一致。實作另外保存 signed peak correlation 與 lag，不只保留絕對值。

#### 一般 omni array 的 diffuse coherence 契約

兩支 omni microphones 相距 (d)，理想 isotropic 3D diffuse field 的
complex coherence target 是

\[
\Gamma_\mathrm{diffuse}(f,d)
=
\frac{\sin(2\pi f d/c)}{2\pi f d/c}
=
\operatorname{sinc}(2fd/c).
\]

這個 signed sinc 關係源自隨機 reverberant field 的 point-to-point
correlation；Cook 等人的經典量測給出同一關係，後續 microphone-array
文獻也以它作 diffuse coherence model。M4.2 使用 late window 的 overlap
Welch auto/cross spectra 估計 complex coherence，輸出各 octave 與整體
complex RMSE，同時保留 imaginary RMS；不把 sign/phase 丟掉只看
magnitude-squared coherence。[Cook et al. 1955](https://doi.org/10.1121/1.1908122)

FDN 的多輸出不能只複製同一 tail。過高 inter-channel correlation 會
破壞 source width、造成 uncontrolled coloration；多輸出 FDN 的 correlation
主要受 feedforward/output paths 影響，因此 M4.3/M4.5 必須驗證 output
matrix，而不能只增加 feedback lines。[Schlecht、Fagerström、Välimäki 2023](https://aaltodoc.aalto.fi/items/ab902c17-c91b-4274-b3c6-7d976a7f1b25)

#### 為何目前沒有 measured spatial 分數

現有 training bank 的五個 WAV channels 表示「五個不同音源到同一支
receiver」，不是「同一音源同時到五支 receivers」。拿它們互算 IACC
會量到不同 source positions 的差異，不是 spatial coherence。M4.2
validator 因此明確輸出
`spatial_contract.evaluable_from_these_banks=false`。

真正的 spatial exit 需要：同一 source event、至少兩個同步 receiver
RIR、receiver positions／orientation、channel semantics 與 sound speed。
在取得這類 measured fixture 前，API 與解析 invariant 已完成，但不會
捏造一個 measured IACC pass。M4.3 先依已確認的 500 Hz–4 kHz density／
mixing gap 驗證 mono FDN core；M4.5 再以合法 multi-receiver fixture 關閉
spatial exit。

### 10.30 M4.3：deterministic internally contractive multiband FDN

M4.3 的目標不是立刻替換完整 RIR，而是先把 dense late-field renderer
隔離出來，使它能獨立回答三個問題：相同 seed 是否逐 sample 可重現、
feedback 是否不會自激發散，以及每個有效 octave 的衰減與 echo density
是否符合 M4.2 measured envelope。實作位於
[`puresound/audio/multiband_fdn.py`](../../puresound/audio/multiband_fdn.py)，
production hybrid、Pyroomacoustics default 和 `path-events-m3` opt-in 選擇
都沒有改變。

#### 架構與狀態更新

目前採用「每個 octave 一個平行 FDN branch，再由因果 octave filter
限制輸出頻帶」的 reference 架構。所有 branch 共用 \(N=16\) 條 delay
長度與正交 mixing matrix，但各自有 band-specific attenuation、output taps
與 state。若第 \(b\) 個 band 在時間 \(n\) 讀出的 delay-line 向量是
\(\mathbf y_b[n]\)，寫回向量為

\[
\mathbf w_b[n]
=
\mathbf U\,\mathrm{diag}(\mathbf g_b)\,\mathbf y_b[n]
+q_b\,x[n]\,\mathbf b,
\]

而 raw output 為

\[
r_b[n]=\mathbf c_b^T\mathbf y_b[n].
\]

每個 \(r_b\) 通過 nominal octave filter 後相加得到 mono late tail。輸入向量
\(\mathbf b\)、每個 band 的輸出向量 \(\mathbf c_b\) 與 signs/permutations
全部由 seed 決定；band injection weights \(q_b\) 滿足
\(\sum_b q_b^2=1\)。因此相同 target、sample rate 與 seed 會產生完全相同
的係數與 waveform。

#### Delay 與 feedback matrix

delay range 由 measured high-band target mixing time \(T_m=24\) ms 導出：
預設取 \(0.125T_m\) 到 \(0.75T_m\)，再在附近挑選互不相同的質數 sample
length。16 kHz、seed `20260731` 的結果是

```text
53, 59, 61, 67, 79, 89, 97, 113,
127, 139, 157, 179, 211, 223, 251, 283 samples
```

也就是 3.31–17.69 ms。不同質數彼此互質，可減少多條 delay 很快同時回到
共同週期所造成的規則脈衝群；這不是「質數本身保證無 coloration」，所以
仍需頻譜診斷與 listening test。

feedback matrix \(\mathbf U\) 是 normalized Hadamard matrix，再做
seeded row/column permutation 與 sign flips。這些操作不破壞

\[
\mathbf U^T\mathbf U=\mathbf I,
\]

因此 mixing 本身不增加 state energy；16-line 版本也能用固定、精確、
容易測試的係數避免一般隨機矩陣的數值不確定性。

#### Band-dependent decay 與「被動」的精確含義

對 delay line \(i\) 的長度 \(d_i\)、sample rate \(f_s\) 與第 \(b\) 個
octave target \(T_{60,b}\)，每次 traversal 的聲壓 gain 為

\[
g_{b,i}=10^{-3d_i/(f_sT_{60,b})}.
\]

經過 \(f_sT_{60,b}/d_i\) 次 traversal 後，累積 gain 正好是
\(10^{-3}\)，即聲壓下降 60 dB。內部 feedback operator 是

\[
\mathbf A_b=\mathbf U\,\mathrm{diag}(\mathbf g_b),
\qquad
\lVert\mathbf A_b\rVert_2=\max_i g_{b,i}<1.
\]

因此這裡的「passive」明確指 **zero-input internal feedback 是嚴格
contractive、不會產生 runaway energy**。output taps 只是觀察 state，並未
從 delay buffers 扣除輸出能量，所以此契約不主張整個 input/output transfer
在所有 normalization 下都具有小於 1 的總能量 gain。

目前採平行 octave branches，而非在單一 FDN loop 中逼近一個高階
frequency-dependent attenuation filter。這讓 M4.3 的衰減與穩定性可直接
驗證；若 M4.4 coupling 顯示 crossover ripple 或運算量不可接受，再把相同
target 收斂到共用 state／loop-filter realization。

#### M4.3 正式結果

validator 從 M4.2 report 讀取 measured median T20，不用手寫另一套 RT60
target。正式 16 kHz、1.5 s impulse 結果如下：

| band | gate | target T20 | rendered T20 | 相對誤差 | mixing time | late NED |
|------|------|-----------:|-------------:|---------:|------------:|---------:|
| 125 Hz | diagnostic | 0.396 s | 0.482 s | 21.72% | 22.6 ms | 0.982 |
| 250 Hz | diagnostic | 0.376 s | 0.435 s | 15.57% | 15.3 ms | 1.155 |
| 500 Hz | qualified | 0.548 s | 0.555 s | 1.15% | 12.0 ms | 1.060 |
| 1 kHz | qualified | 0.463 s | 0.485 s | 4.85% | 26.0 ms | 1.050 |
| 2 kHz | qualified | 0.472 s | 0.458 s | 3.03% | 10.0 ms | 1.031 |
| 4 kHz | qualified | 0.474 s | 0.474 s | 0.02% | 24.0 ms | 0.982 |

8/8 structural checks 全數通過：distinct prime delays、orthogonal feedback、
strict contraction、unit band-weight energy、same-seed determinism、finite
output、causal delayed onset，以及 render 結束前已衰減。500 Hz–4 kHz 的
12 個 T20／mixing-time／late-NED qualified checks 也全數通過 frozen M4.2
條件。

125/250 Hz 仍是 diagnostic，不是被結果隱藏的通過項。125 Hz 的 T20 與
late NED、250 Hz 的 T20 未過；完整 hybrid 在這裡還有 modal branch，平行
octave filter crossover 也會影響 fitted decay。它們必須在 M4.4 完整
early/modal/late coupling 後重新量測。現行 M4.4 明確只把不低於 500 Hz／
半個 crossover 的 bands 交給 FDN，所以 125/250 Hz 仍沒有被宣告通過；
不能為了讓 isolated FDN 報表全綠就調整低頻 gain。

coloration report 另輸出 octave 內 spectral flatness 與 p95/median ripple，
但它只用於找異常共振，不能取代聽感。正式 JSON 在
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_fdn_report.json`，可播放 impulse 在
`egs/rir_generation/exp/rir_realism/m4/rir_m4_fdn_core/rir_m4_fdn_core.wav`。

M4.4 已依下節的 coupling 契約完成這個步驟；`path-events-m3` 與
Pyroomacoustics default 均未改變。

### 10.31 M4.4：PathEvent early／FDN late 的因果 coupling

M4.3 只證明 FDN core 本身穩定且能達到 decay/density target；完整 RIR
仍缺少一個關鍵決策：何時停止把高階反射視為可分辨的 coherent paths，
何時改由統計 late field 接手。M4.4 將此決策落在獨立
[rir_late_coupling.py](../../puresound/audio/rir_late_coupling.py)
中，並新增明確 opt-in 的 PathEventFDNHighFrequencyBackend／CLI
**--high-backend path-events-m4**。

#### 時間軸與 complementary crossfade

每個 source channel 使用自己的物理 direct sample \(n_d\)。transition
center 是

\[
n_m=n_d+\operatorname{round}(f_sT_m),
\]

目前 frozen \(T_m=24\) ms；transition width 為 16 ms。若 transition
start/end 是 \(n_0,n_1\)，區間內令

\[
\theta[n]
=
\frac{\pi}{2}\,
\frac{n-n_0}{n_1-n_0},
\qquad n_0\le n\le n_1,
\]

\[
w_E[n]=\cos\theta[n],
\qquad
w_L[n]=\sin\theta[n],
\qquad
w_E^2[n]+w_L^2[n]=1.
\]

區間前 \(w_E=1,w_L=0\)；區間後 \(w_E=0,w_L=1\)。因此不是把兩段 RIR
在某個 sample 硬切開，也不會在 transition 內因兩個 unity gains 直接造成
3–6 dB 疊加。

這個 timing 是 direct-relative，不是 absolute sample。不同 source distance
會有不同 \(n_d\)，所以 near/far channel 的物理到達時間不會被拉到同一個
index。正式 fixture 的 transition 是 direct 後 16–32 ms，中心 24 ms。

#### PathEvent 如何注入 FDN

原始 coherent PathEvent response 記為 \(p[n]\)。FDN excitation 不是另一個
任意 impulse，而是

\[
x_{\mathrm{FDN}}[n]=w_E[n]p[n].
\]

也就是 direct 與 early reflections 的實際 amplitude／sign／timing 會驅動
FDN state；transition 結束後的 sparse high-order paths 不再持續注入。
FDN raw output \(f[n]\) 再乘 \(w_L[n]\)。coupled response 是

\[
h[n]=w_E[n]p[n]+a\,w_L[n]f[n].
\]

因為 FDN、octave filters 與 crossfade 全部 causal，且 \(w_L=0\) 直到
transition start，M4.4 不會建立 direct 前能量。更強的實作 invariant 是：
從 sample 0 到 transition start（包含 endpoint），\(h[n]\) 與原始
PathEvent response 逐 sample 完全相同。

每個 channel 的 FDN seed 由 base seed、serialized scene_id 與 source
index 經 stable BLAKE2 digest 導出；不使用 Python process-randomized
hash。相同場景可完全重現，不同 source channels 也不會共享完全相同的
delay signs/output taps。

#### 為何不能只做 RMS match

在 transition 中 coherent 與 FDN component 可能有非零內積。若分別匹配
RMS 再相加，cross term 會改變總 early/late energy。M4.4 固定有限 RIR 中
transition start 後的能量等於原始 PathEvent：

\[
E_T=\lVert p\rVert^2,
\qquad
E_E=\lVert w_Ep\rVert^2,
\qquad
E_F=\lVert w_Lf\rVert^2,
\qquad
C=\langle w_Ep,w_Lf\rangle.
\]

gain \(a\ge0\) 由

\[
a^2E_F+2aC+E_E=E_T
\]

的正根得到：

\[
a
=
\frac{-C+\sqrt{C^2+E_F(E_T-E_E)}}{E_F}.
\]

由於 \(0\le w_E\le1\)，理想精度下 \(E_T-E_E\ge0\)，所以根存在。這個
契約保存的是 transition start 到 finite render end 的總能量，並不聲稱
每個瞬間或每個 octave 的能量都與 sparse PathEvent 相同；octave shape
改由 scene material RT60 與 FDN branch 決定。

#### Per-room octave decay

M4.3 validator 使用 measured median T20 作固定 reference target。M4.4
生成資料時不能讓所有房間共享同一組 RT60，因此改由
RoomSceneV2.predicted_octave_rt60_s() 取得 serialized surface absorption
與面積導出的 octave decay。只保留 full nominal octave 位於 Nyquist 下、
且不低於 500 Hz／半個 crossover frequency 的 bands；較低頻仍交給 modal
branch 與 low/high crossover。

這表示 M4.4 已從「固定 measured target 的核心驗證」前進到「每個
material-first room 的 decay cause」，但 Sabine-style material prediction
仍不是 measured-room inverse calibration。

#### 正式 M4.4 結果

正式 validator 使用一個 deterministic office fixture、5 個 source
channels、16 kHz、1.2 s、order-4 PathEvents：

| band | material target T20 | M4 median T20 error | M3 mixing | M4 mixing | M3 late NED | M4 late NED |
|------|--------------------:|--------------------:|----------:|----------:|------------:|------------:|
| 500 Hz | 0.303 s | 2.83% | 28 ms | 22 ms | 0.560 | 1.041 |
| 1 kHz | 0.264 s | 5.32% | 40 ms | 26 ms | 0.000 | 1.109 |
| 2 kHz | 0.256 s | 0.69% | 無 crossing | 30 ms | 0.000 | 0.992 |
| 4 kHz | 0.220 s | 3.47% | 無 crossing | 34 ms | 0.000 | 1.001 |

四個 M4 median mixing times 都落在 frozen M4.2 measured p10–p90 envelope，
每個 band 到 measured late-NED median 的距離也都比 M3 小。

8/8 coupling structural gates 通過：shape、same-seed exact determinism、
finite output、no prearrival、transition 前 exact preservation、
post-transition energy、distinct channel seeds，以及 production default
未變。完整 modal/high hybrid 另通過 5/5 相容性 gate：

- transition 前逐 sample 完全相同；
- median/max absolute C50 change 不超過 1/3 dB；
- median/max early-energy-centroid change 不超過 1/2.5 ms。

正式報告是
egs/rir_generation/phases/m4_spatial_late_field/reports/m4_path_event_fdn_coupling_report.json；M3/M4
high-only 與 full-hybrid 對照 WAV 位於 egs/rir_generation/exp/rir_realism/m4/rir_m4_coupling/。

M4.4 本身的 WAV channel 仍是不同 source 到同一 receiver，不能拿來算
IACC/coherence。M4.5 因此另外建立同一 source 到同步 multi-receiver／
Ambisonic 的 renderer，而不是重新解釋 M4.4 的 channel 語意。

### 10.32 M4.5：同一個 late field 的 receiver array 與 Ambisonics

#### 為什麼不能各自生成兩條 mono RIR

如果左、右接收器各自用不同亂數生成 late tail，即使兩條 RIR 的 RT60、
頻譜與 echo density 都正確，兩者之間也沒有共同波場。這種作法無法保證
麥克風間距改變時的 coherence、相位差與 IACC 會遵循聲學。M4.5 的核心
約束因此是：**所有輸出 channel 都必須是同一組平面波的不同投影**。

先用 M4.3 的 passive multiband FDN 生成各 octave band 的衰減包絡。對 band
\(b\)，以長度 10 ms 的 causal RMS 表示包絡 \(e_b(t)\)，再為第 \(k\) 個
平面波產生獨立、同頻帶的 Gaussian carrier \(n_{k,b}(t)\)：

\[
s_k(t)=\sum_b e_b(t)n_{k,b}(t).
\]

方向 \(\mathbf q_k\) 不是亂抽後留下 clustering，而是先用 Fibonacci
sphere 作近似等面積取樣，再施加 seeded 3D rotation。正式 gate 使用
\(K=256\) 個方向。令 \(\mathbf q_k\) 為傳播方向、\(\mathbf r_m\) 為第
\(m\) 個接收器位置，接收器相對延遲為

\[
\tau_{mk}=\frac{R+(\mathbf r_m-\bar{\mathbf r})\cdot\mathbf q_k}{c},
\qquad
R=\max_j\|\mathbf r_j-\bar{\mathbf r}\|.
\]

固定的 aperture margin \(R/c\) 讓所有方向與 receivers 的 delay 都非負；
它不依賴方向，因此 array centroid 與 FOA 共用明確、固定的時空 reference，
同時不改變任兩個接收器的相對 delay。接收器輸出為

\[
p_m(t)=\frac{1}{\sqrt K}\sum_{k=1}^{K}
D_m(-\mathbf q_k)\,s_k(t-\tau_{mk}),
\]

其中 \(-\mathbf q_k\) 是接收器看到的 direction of arrival，\(D_m\) 是
接收器 pressure directivity。fractional delay 使用與 PathEvent 相同的
causal forward Lagrange kernel。

同一批 \(s_k\) 也直接編碼成 first-order Ambisonics。PureSound 固定使用
**ACN channel order、SN3D normalization、W/Y/Z/X channel labels**：

\[
\begin{bmatrix}W\\Y\\Z\\X\end{bmatrix}
=\frac{1}{\sqrt K}\sum_k
\begin{bmatrix}1\\d_{k,y}\\d_{k,z}\\d_{k,x}\end{bmatrix}s_k(t),
\qquad \mathbf d_k=-\mathbf q_k.
\]

這裡 SN3D 的一階 basis 是 direction cosine，不能額外乘上 \(\sqrt 3\)；
否則會把 N3D 與 SN3D 混用。

direct／early 部分仍由每個 receiver 的 PathEvents 產生；FOA early 也依每個
PathEvent 的 arrival direction 使用同一個 \([1,y,z,x]\) basis。receiver
array 與四個 FOA channel 都沿用 M4.4 的 direct-relative equal-power
transition：transition start 以前逐 sample 完全保留，之後以 positive
quadratic root 各 channel 保留有限 RIR 的 post-transition energy。

#### coherence 與 IACC gate

相距 \(d\) 的兩個 omni point receivers，在 3D isotropic diffuse field 的
理論 coherence 是

\[
\Gamma(f,d)=\operatorname{sinc}\!\left(\frac{2fd}{c}\right)
=\frac{\sin(2\pi fd/c)}{2\pi fd/c}.
\]

正式 fixture 使用 17 cm 間距、16 kHz、1.2 s、late window 從 direct 後
80 ms 開始。RIR 是 exponential decay 的非平穩訊號，因此 Welch estimator
使用 48-sample segment；若使用很長 segment，包絡在同一 segment 內明顯
改變，估計 variance 反而升高。這是 estimator contract，不是把 RIR 當作
無限長 stationary noise。

M4.5 正式結果：

- 12/12 structural gates 通過，包括 array/FOA shape、exact determinism、
  causality、early exact preservation、post-transition energy 與 ACN/SN3D；
- 5/5 spatial gates 通過；500 Hz–4 kHz qualified octave 的 bin-weighted
  complex coherence RMSE 為 `0.187197`，低於 `0.20` gate，imaginary
  coherence RMS 也低於 `0.15`；全可見頻譜（包含未建模的低頻與 octave
  外 leakage）RMSE `0.216982` 保留為 diagnostic，不拿來取代 M4.2 gate；
- 500 Hz、1/2/4 kHz 每個 reported octave 的 complex RMSE 都低於 `0.23`；
- late IACC_L4 為 `0.323144`，低於 model-derived `0.50` gate；
- Pyroomacoustics production default 完全沒有變更。

報告位於
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_spatial_rir_report.json`；可播放的 receiver
array 與 FOA RIR 位於 `egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/`。

### 10.33 M4.6：接收器指向性與 optional binaural BRIR

#### 一階 real pressure directivity

source 與 receiver 共用可檢查的一階 pressure pattern：

\[
D(\theta)=\alpha+(1-\alpha)\cos\theta.
\]

`cardioid` 使用 \(\alpha=0.5\)，`hypercardioid` 使用 \(0.25\)，
`figure_eight` 使用 \(0\)，`omnidirectional` 固定為 1。方向由 transducer
yaw/pitch 的 forward axis 與 path direction 內積得到。hypercardioid 與
figure-eight 背面可以得到負值；這是 pressure phase reversal，不應錯誤
clamp 到零。source gain 使用 departure direction；receiver gain 使用
arrival direction 的負向量，也就是從 receiver 指回聲音來向。

#### HRTF decoder 是可注入契約，不是假資料

完整 FOA RIR \(a_c(t)\) 透過兩耳、四 channel 的 causal FIR
\(g_{e,c}(t)\) 解碼：

\[
h_e(t)=\sum_{c=0}^{3} a_c(t)*g_{e,c}(t),
\qquad g\text{ shape}=[2,4,L].
\]

`AmbisonicBinauralDecoder` 強制保存 sample rate、ACN/SN3D input contract、
dataset/reference ID、decoder kind 與 provenance。專案目前沒有把來源、
受試者與授權不明的 measured HRTF 塞進 repository。正式 artifact 使用的
one-tap lateral decoder 明確標示為 `analytic_demonstration_not_hrtf`；另以
multi-tap fixture 驗證外部 HRTF-derived FIR 能被注入、保留 filter tail
與 provenance，但 fixture 也不宣稱是實測 HRTF。

M4.6 的 11/11 decoder／causality／determinism checks 全數通過。analytic
BRIR 的 IACC_E4/L4 為 `0.442999 / 0.384164`，只作 pipeline diagnostic，
不是 perceptual HRTF gate。報告位於
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_binaural_brir_report.json`，示範 BRIR 是
`egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_analytic_brir_2ch.wav`。

#### 「M4 完成」的精確意思

`validate_m4_exit.py` 把兩種完成狀態分開：

- **implementation exit：PASS。** M4.1–M4.6、receiver array、FOA、
  directivity、decoder contract、正式 reports 與 audition artifacts 都存在；
- **empirical／production exit：OPEN。** 尚缺同步 measured multi-receiver
  RIR、授權且已校正的 measured HRTF，以及針對 metallic coloration 與
  spatial plausibility 的受控聽測。

因此現在可以合成可用的 mono、receiver-array、FOA 與 demonstration BRIR，
也可以注入自己的 HRTF FIR；但不能把 model-derived pass 描述成已通過真實
裝置／真人聽覺的 production validation。

### 10.34 M5.1：先讓反演問題可識別、可稽核

M5 的方向是從少量真實量測估計 renderer 參數，再將剩餘且可重現的誤差交給
learned residual。這不是把一條 measured RIR 當 waveform target 任意 overfit。
對實體房間 \(r\)，我們只用 train positions \(\mathcal T_r\) 求解

\[
\theta_r^*=\arg\min_{\theta_r}
\sum_{p\in\mathcal T_r}
\mathcal L\left(h^{\mathrm{meas}}_{r,p},F(\theta_r;p)\right),
\]

並在未參與 fitting 的 positions 評估。最後還要把整個 room 留出，判斷模型
是否學到跨房間規律，而不是只記住 room identity。

這個反問題會把 room、source、receiver、latency 與 environment 混在一起，
所以 M5.1 先新增 `puresound.rir_measurement_campaign.v1`。每筆受控 capture
必須保存：

- 可追溯房間 geometry／mesh 與座標系；
- source／receiver position、orientation 與量測不確定度；
- source／receiver calibration response 與實體序號；
- temperature、humidity 與 pressure；
- 至少兩次 raw exponential-sine-sweep recording；
- inverse sweep、background noise、deconvolution config 與 latency correction；
- 同步 multi-receiver channel semantics；
- 每個 asset 的 campaign-relative path 與 SHA-256；
- 以實體 `room_id` 唯一決定的 train／validation／test split。

Repeated ESS 反卷積可把線性 RIR 與 harmonic distortion 在時間上分開，方法
基礎見 [Farina 的 AES ESS 論文](https://angelofarina.it/Public/Papers/134-AES00.PDF)。
重複錄音同時提供 repeatability 與 noise-floor 證據；因此 schema 不允許只
保留最後一條處理後 RIR 就宣稱是 controlled campaign。

#### M5 calibration loss

`puresound.rir_calibration_loss.v1` 將目標拆開記錄：

\[
\mathcal L =
w_sL_{\mathrm{MRSTFT}}+w_eL_{\mathrm{EDC}}+w_aL_{\mathrm{arrival}}+
w_oL_{\mathrm{octave}}+w_cL_{\mathrm{coherence}}+
w_pR_{\mathrm{causality}}+w_dR_{\mathrm{decay}}.
\]

- multiresolution STFT 同時比較多種 time／frequency resolution 的 spectral
  convergence 與 log magnitude；
- EDC 從 measured／synthetic 各自 direct arrival 對齊後比較 Schroeder
  decay curve，不把 timing error 混入 decay error；
- arrival loss 另外保留絕對飛行時間；
- octave term 比較 band energy 與 qualified T20；
- spatial term 比較同步 receiver pair 的 late complex coherence；
- causality 與 decay regularization 禁止 pre-arrival energy 和持續增長尾場。

Mono input 的 spatial term 會明確回報 `evaluable=false`，而不是用 0 假裝
通過。現行 NumPy／SciPy 實作是可稽核 metric oracle，不是 autograd
optimizer；M5.2 的 differentiable objective 必須先對這個 oracle 做數值一致性
測試。

#### 正式 M5.1 結果與證據邊界

Deterministic probe 已驗證 identity loss、8-sample arrival shift、pre-arrival
energy、spatial perturbation 與 mono unevaluable semantics，針對 schema／hash／
loss 的 10 個測試全數通過。因此 M5.1 **implementation exit：PASS**。

同一 validator 抽查既有 measured train／held-out bank 各 64 筆 metadata。
兩者可用於 DRR、C50、decay、頻譜與 echo-density 分布比較，但九類 controlled
acquisition evidence 都是 0/64，所以 **controlled measurement readiness：
OPEN**。缺失的 raw sweep、calibration 與 pose provenance 不能從最終 WAV
推回來；M5.3 不會假裝已完成。

接下來的 M5.2 先做 synthetic recovery：由已知 scene 生成 target，隱藏一小組
material／late-field parameters，以多起點反演後在未見位置驗證。若連無量測
噪聲的 synthetic case 都找不回某參數，該參數就不應在真實房間被自由 fitting。
完整欄位、公式與量測 SOP 見
[`rir_measurement_campaign_zh-TW.md`](rir_measurement_campaign_zh-TW.md)。

### 10.35 M5.2：先做可精確反駁的 synthetic inverse baseline

在真實量測加入 noise、pose uncertainty 與 model mismatch 以前，M5.2 先問一個
更基本的問題：optimizer 是否能從多個位置找回我們自己藏起來的參數？實作
建立一個 causal approximate renderer，固定 geometry/direct component，只
開放 mixing time、early-reflection gain、四個 octave RT60 與四個 octave late
gain，共十個有 box bounds 的 shared-room parameters。

三個 train positions 具有不同距離、early paths 與 deterministic late basis，
但共用同一組 hidden truth。三組相距很遠的初值都收斂到同一解：

- 最大 multi-start parameter spread：`2.35e-12`；
- scaled-Jacobian condition number：`17.22`；
- local Jacobian：full column rank；
- 兩個 unseen positions 的 M5.1 oracle mean loss：
  `2.26933 -> 1.69e-14`；
- direct arrival 前樣本：全部嚴格為零。

因此 M5.2 noise-free synthetic-recovery gate 是 **PASS**。但這是同一 model
family 生成與反演的 inverse-crime baseline；它只驗證參數順序、bounds、
multi-position residual、multi-start、local sensitivity、held-out-position 與
M5.1 loss oracle 的接線。它不證明 noise/model-mismatch robustness、global
identifiability、完整 M4 inversion 或真實房間校準。

下一節的 M5.2b 已加入 controlled noise、model mismatch 與 smooth
M4-consistent proxy residual。M5.3 measured fit 仍由 controlled campaign
readiness 阻擋，不能拿舊 measured WAV 補做不存在的 acquisition provenance。
M5.2 noise-free 正式 report 為
`egs/rir_generation/phases/m5_calibration/reports/m5_synthetic_recovery_report.json`。

### 10.36 M5.2b：不要把 noise floor 擬合成房間

M5.2b 對每個位置加入 32–38 dB SNR、已知 gain／latency error、未建模 early
paths，以及具有 1.25 倍 RT60 的獨立 late component。Fitter 只依 acquisition
metadata 移除已知 gain 與 latency；noise 和 model mismatch 都保留。

新的 smooth M4-proxy residual 同時比較 waveform、direct-relative coherent
early、broadband log-energy decay 與 octave log-energy decay。Decay window
只有在 target energy 高於估計 noise energy 20 dB 時才參與 optimization；否則
noise plateau 會被錯解為慢衰減。實作使用 finite-difference least squares，
並不宣稱是 autograd 或完整 M4 differentiable renderer。

15/15 gates 通過：mixing time／early gain error 分別為 `0.0235 ms`／
`0.0214 dB`，最大 band RT60 error `1.43%`，最大 late-gain error
`0.106 dB`。兩個初值收斂 spread `1.03e-6`，condition number `8.12`；兩個
held-out positions 的獨立 M5.1 total 相對初始值下降 `61.4%`。相較
waveform-only ablation，octave error 由 `0.03341` 降至 `0.00846`，最大
RT60／late-gain error 也同時下降。

五個案例中有兩個的 global peak 不是 direct arrival。因此後續 M5 measured
pipeline 必須由 geometry (d/c) 限制 onset search window，不能把全域
`argmax(abs(rir))` 當成可靠 direct detector。正式 report 為
`egs/rir_generation/phases/m5_calibration/reports/m5_robust_recovery_report.json`。

這仍是 synthetic perturbation，不是 M5.3。下一節把 robust objective 的第一組
參數接到 actual M4 PathEvent／FDN renderer；真正 measured fit 繼續等待
controlled campaign。

### 10.37 M5.2c：真正跨過 M4 的離散 topology

M4 的 mixing time 不只是 smooth scalar。它會決定 multiband FDN 的 prime
delay lengths；mixing time 稍微改變時，delay topology 可能整組跳到下一個
質數。因此不能假設整個 renderer 對 mixing time 可微。M5.2c 採兩層 profile：

\[
m^*=\arg\min_{m\in\{20,24,28\}\,\mathrm{ms}}
\min_{g,\boldsymbol\tau}
\mathcal L_{\mathrm{robust}}
\left(h^{\mathrm{target}},F_{\mathrm{M4}}(m,g,\boldsymbol\tau)\right).
\]

外層固定一個 mixing-time candidate，讓 M4 建立對應的 prime delays；內層以
bounded finite-difference least squares 估計 aggregate coherent-reflection
gain \(g\) 與 500／1000／2000 Hz RT60 \(\boldsymbol\tau\)。每個 candidate 都
保存 cost、delay lengths、Jacobian rank 與 scaled condition number，因此被
淘汰的 topology 也能稽核。

這裡直接呼叫 `couple_path_event_rir_with_fdn`：前段是 material-first scene 的
order-4 PathEvent RIR，中間是 direct-relative equal-power transition，後段是
deterministic multiband FDN。為避免 coherent gain 去補償 FDN 誤差，early
waveform residual 只看 direct 後 12 ms；8 ms broadband／octave log-energy
windows 則只納入高於 noise floor 15 dB 的區段。

驗證 target 不是 fitter 的完全同模型輸出：late field 混入 8% 不同 seed 的
FDN topology，再加 42 dB SNR broadband noise。兩個位置 fitting、兩個未見位置
holdout 的正式結果為：

- hidden mixing time `24 ms` 在 `20/24/28 ms` 中正確被選中；
- best／second profile cost ratio：`0.0820`；
- 所有 profile inner solves 收斂且 local full rank；最佳 condition number
  `3.14`；
- coherent gain absolute error：`0.661 dB`；
- 500／1000／2000 Hz RT60 relative error：`0.392%`、`1.86%`、`4.23%`；
- held-out independent M5.1 total：`3.59367 -> 1.01550`，下降 `71.7%`；
- physical-arrival causality、M4 pre-transition exact preservation 與 production
  default 不變全部通過。

14/14 gates 因此為 **PASS**。正式報告是
`egs/rir_generation/phases/m5_calibration/reports/m5_m4_parameter_mapping_report.json`，三個
holdout target／initial／recovered WAV 位於
`egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping/`。

這個 PASS 的範圍必須說清楚。`coherent_reflection_gain_db` 仍是多條反射路徑
共用的 aggregate proxy，不等於已找回哪一面牆的 absorption 或 scattering；
外層也只證明給定三個 mixing candidates 中的選擇，不是 global
identifiability。正式 gate 使用 order-4 PathEvent，不能外推到資訊較少的低階
path set。下一個 M5.2d 會把 coherent proxy 拆成可解釋的 material／path groups，
逐組做 sensitivity、correlation、multi-start 與 holdout ablation；M5.3 仍需
等待符合 M5.1 契約的真實 campaign。

### 10.38 M5.2d：先問 Jacobian，再決定能不能 fit material

對 PathEvent (p)，令 (n_{p,g}) 是它撞到 boundary group (g) 的次數，
effective pressure adjustment 為 (delta_g) dB。M5.2d 在進入 M4 coupling
前把 path gain 改成

\[
a_p(\boldsymbol\delta)=a_{p,0}
10^{\frac{1}{20}\sum_g n_{p,g}\delta_g}.
\]

這保留「一次反射與多次反射必須一致受同一牆面影響」的物理結構，而不是
對任意 time sample 加 gain。三個 order-4 train positions 在 48 dB SNR 下，
west／east／south／north／floor／ceiling 六欄 Jacobian 的結果為：

- numerical rank `6/6`；
- normalized condition number `2.74`；
- maximum absolute column correlation `0.540`；
- 最大 recovery error `0.00123 dB`；
- multi-start spread `4.21e-10 dB`；
- held-out M5.1 total 下降 `64.8%`。

接著做刻意的反例：每面牆同時放入 absorption-loss 與
specular-scattering-loss，但兩者在 mono coherent amplitude 中產生同一條
sensitivity column。結果 rank 是 `6/12`；priority-preserving selector 留下六個
`effective_reflection`，拒絕六個 `scattering_specular_loss`。因此「optimizer
有輸出 12 個數」不等於 12 個物理量可辨識；scattering 必須移到 M5.4。

### 10.39 M5.3：measured runner 必須 fail closed

M5.3 runner 先執行 campaign audit，只有全部 asset 存在、SHA-256 相符、每房
至少規定數量的 unique configurations、repeated raw ESS／inverse／noise／
deconvolved RIR 都保留，且 train／validation／test rooms 齊全時才可 fitting。

Train room 內的位置以

\[
H(\text{campaign id},\text{room id},\text{measurement id})
\]

的 SHA-256 固定分出 `position_fit` 和 `position_holdout`，所以重跑不會改 split。
Validation/test physical rooms 從不參與 room fitting，只接收 train-room
population median parameters。報告分開保存 train-fit、position-holdout、
validation-room 與 test-room loss，避免 item-disjoint 冒充 room-disjoint。

完整 synthetic campaign fixture（3 rooms、同步 receivers、retained assets／
hash）已讓所有 runner 階段 8/8 PASS，且明確標為 non-evidence。當同一 CLI
讀取目前 template 時，readiness FAIL、optimizer 完全不啟動並回傳 exit code
2。這表示 M5.3 runner implementation 完成；沒有表示真實 measured fit 完成。

### 10.40 M5.4：spatial parameter 只能用同步觀測

M5.4 對每個 actual M4 candidate (k) 計算含 spatial coherence 的 loss：

\[
k^*=\arg\min_k\left(
0.25L_{\mathrm{STFT}}+0.5L_{\mathrm{EDC}}+
0.5L_{\mathrm{octave}}+2L_{\mathrm{coherence}}+
R_{\mathrm{causal}}+R_{\mathrm{decay}}
\right).
\]

輸入少於兩個同步 receiver 時直接拒絕。Actual M4 四候選 fixture 正確選回
truth scattering＋opposed-cardioid，best／second ratio `0.812`，11/11 gates
PASS。

現行 scattering 只改 first-order coherent PathEvents；80 ms 後 FDN field
尚未依 scattering 改變。所以 scattering 是由同步 early／spectral／octave
total 分辨，而 late coherence 驗證 shared field／directivity。文件與 report
明確保存這個邊界，沒有把相同 late coherence 說成 scattering evidence。

### 10.41 M5.5：residual 的自由度也必須受物理限制

Train observation (i) 的 raw residual 是

\[
r_i(t)=h_i^{\mathrm{target}}(t)-h_i^{\mathrm{physical}}(t).
\]

每條 residual 以 direct arrival 為 (t=0)，除以 physical tail norm，再跨多個
train rooms 取 median template。套用到新位置時只恢復該位置的 physical norm。
模型不是任意 waveform memorizer，還必須滿足：

- (r(t)=0\) for (t<0)；
- 50 ms 後每 10 ms block 的 pressure RMS 不得比 `RT60=0.8 s` envelope
  衰減得更慢；
- \(\|r\|_2^2/\|h_{\mathrm{physical}}\|_2^2\le0.15\)。

正式 fixture 用兩個 train rooms 學 template，在完全未見的第三房測試。
physical-only total `0.8941`、residual-only `4.0490`、combined `0.2873`；
combined 相對 physical 降 `67.9%`。12/12 causality、decay、energy、ablation
與 interpolation gates PASS。這是 constrained residual reference，尚不是在
真實 campaign 上完成 neural architecture／downstream 選型。

### 10.42 M5.6：兩個 exit，不混成一個 PASS

`puresound.m5_exit.v1` 彙整九個 implementation gates、七個必要 artifacts，
以及四個禁止 false claim 的 invariants。結果是：

- `implementation_exit.passed=true`：M5.1–M5.6 code、完整／blocked M5.3
  路徑、synthetic gates、spatial profile、residual ablations 都已完成；
- `empirical_exit.passed=false`：尚缺真實 controlled campaign、measured
  position/room holdouts、measured synchronized calibration、measured residual、
  controlled listening 與 room-disjoint downstream；
- `production_enablement.ready=false`：不更換 production default。

這是 M5 目前可誠實達成的 terminal implementation state。剩餘項目需要新的
外部量測／受試者／下游訓練結果，不能靠再寫一個 synthetic validator 變成
empirical PASS。

### 10.43 M6.1：先凍結 bank 身分，再大量生成

舊 `PreGeneratedRoomBank` 的同 stem WAV／JSON layout 保持相容，但目錄結構本身
無法證明 room-disjoint、renderer 版本或檔案未被修改。M6.1 因此新增
`puresound.rir_bank.v2`。每個 item 保存 `item_id`、`room_id` 與
`acoustic_space_id`；split 是對最後一個 identity 做 deterministic SHA-256
assignment，而不是逐 item random split。

Manifest 以 strict canonical JSON 計算 SHA-256；每個 WAV、metadata 與 canonical
scene 另有獨立 hash，並用實際 audio header 核對 sample rate、channels、frames。
Generator 保存 recipe/version/code/config hash/seed；renderer profile 保存 low/high
backend、scene schema、config hash、calibration/residual/approval evidence 與
`development -> empirical_candidate -> production_approved` tier。

`release_status=production` 只有在所有 profile 都 production-approved、所有 items
的 M6.3 QC 都 PASS、approval/QC hashes 完整且 code revision 已固定時才成立。
M6.1 的三 split fixture 明確是 development/non-evidence；12/12 schema、split、
tamper、path、false-production 與 legacy-reader gates PASS，所以 **M6.1
implementation PASS**，但 `ready_for_production=false`。

詳細欄位與重現方式見
[`rir_bank_v2_zh-TW.md`](rir_bank_v2_zh-TW.md)。

### 10.44 M6.2：worker scheduling 不得改變 bank

M6.2 在 rendering 前先固定完整 task plan。Item seed 由 base seed 與
`sample_id` 的 SHA-256 導出；每個 worker 在進入 renderer 前重設 Python、NumPy
與 Torch RNG，Pyroomacoustics 還會同時固定 NumPy 與 libroom 的兩個 RNG。
因此相同 recipe 不論使用一個或兩個 workers，或刪除輸出後重新 fresh run，都
產生相同 WAV、metadata、task-plan 與 manifest hashes。

M6 resume 只有在 task/config/profile/split、code revision、scene hash、WAV hash
與實際 audio header 全部相符時才 skip。Manifest 同時保存關鍵 runtime package
versions。正式負向控制修改一個 WAV 後，resume 只重建 `1/6` 並恢復原 manifest；
config 或 code revision 改變時則拒絕把新舊 renderer 結果拼成混血 bank。

生成器會寫三份 content-addressed split JSONL、`rir_bank_manifest.json` 與
`rir_bank_generation_audit.json`。17/17 M6.2 gates PASS，而且 reproducibility
fixture 使用實際預設的 Pyroomacoustics high backend。功能仍由
`--emit-m6-manifest` 明確 opt-in，default high backend 保持 Pyroomacoustics，
release 仍是 `draft/development/QC-pending`。Reader 遇到 M6 root 時也必須明確
指定 split，避免舊式 directory scan 混入 validation/test。

### 10.45 M6.3：不能把「算不出來」偽裝成「零」

M6.3 對每個 item 執行同一份 content-addressed
`puresound.rir_bank_qc.physical.v1` policy。先驗證 asset／scene／audio shape 與
finite/non-silent/peak invariants，再以距離 (d)、metadata 聲速 (c) 和 sample
rate (f_s) 計算幾何 direct sample (n_d=df_s/c)。`floor(n_d)` 之前的顯著能量
是因果 failure；DRR、clarity 和 decay 則錨定在幾何 arrival 附近的 local direct
peak，不能把全域最大晚期 reflection 當成 direct。

每個 channel 計算 tail-energy fraction、DRR、C50/C80、spectral tilt、octave
metrics、noise-aware EDT/T20/T30 與 fit (R^2)，以及 Abel normalized echo
density／mixing time。結果區分：

- `fail`：違反 hard physical/structural invariant，必須 quarantine；
- `not_evaluable`：例如短 RIR 或 noise intersection 使 decay fit 不可靠；
- `not_applicable`：metric 的物理 channel semantics 不成立。

現有 5-channel bank 是五個不同 source 到同一 receiver，不是同步 receiver
array，所以 spatial IACC/coherence 必須是 `not_applicable`。硬算 channel pair
correlation 雖然能得到數字，但那不是空間聲學證據。

QC 為每個 item 寫獨立 report path/hash，只把 PASS items 放入
train/validation/test candidate indexes；FAIL items 保留原始資產並進 quarantine
index。Reader 預設排除 failure。任一 candidate split 變空時 release 保持
`draft`；三 split 完整才可成為 `candidate`，而 renderer approval、M6.4
distribution、M6.5 listening/downstream 仍阻止它成為 production。

正式 synthetic validator 對 actual M6.2 output 得到 `6/6` PASS，另注入 silent、
pre-arrival、direct-only sparse late field 與晚 3 ms direct arrival 四種負控制，
`4/4` 正確隔離。Direct-arrival 搜尋窗大於允許誤差，因此 timing gate 不再是
不可達死碼；octave T20/R2 coverage 也實際參與 admission。QC report tamper 會使
audit FAIL。13/13 gates PASS，因此 **M6.3 implementation
PASS**，但沒有把 synthetic fixture 宣稱為 measured/production evidence。

### 10.46 M6.4：signal variant 只能改它聲明要改的量

M6.4 將 QC-passed bank 發布成 content-addressed variant。Calibrated variant
保留 renderer 的物理 level；peak-normalized variant 對同一 item 的所有 channels
施加唯一共同增益

\[
g_i=\frac{0.98}{\max_{c,n}|h_{i,c}[n]|},\qquad
\tilde h_{i,c}[n]=g_i h_{i,c}[n].
\]

因為 DRR 與 clarity 都是能量比，T20 是衰減曲線斜率，所以共同 scalar gain 應
滿足

\[
\Delta\mathrm{DRR}=\Delta C_{50}=\Delta C_{80}=\Delta T_{20}=0.
\]

Validator 不只相信公式，而是由輸出的兩個 variants 重算 metrics，要求最大差異
小於 `1e-4`，並要求 acoustic-space／room／scene／split lineage 完全相同。若改成
逐 channel normalization，上述跨 channel level semantics 將被破壞，因此契約
明確禁止把它描述成相同的 level policy。

每個 distribution 保存 room volume、RT60、distance、peak、tail fraction、DRR、
C50/C80、T20、spectral tilt、mixing time 與 late density 的完整 rows 和分位數。
Recipe 再把 variants、origin weights、三 split indexes 與 hashes 凍結。
`PreGeneratedReleaseBank` 依 origin weight 取樣，而不是讓檔案數量偷偷決定
synthetic/real 比率，並把 release SHA、recipe、variant 帶入 sample metadata。

Float WAV 另有一個與聲學無關的 reproducibility 問題：libsndfile 的 `PEAK`
chunk 會寫 wall-clock timestamp。M6.4 只把此 timestamp 歸零，保留 waveform 與
peak value，使同一 task 在不同時間仍 byte-identical。

目前 synthetic calibrated／peak-normalized recipes ready；沒有 QC-passed
measured variant，因此 real／mixed recipes 必須 blocked。正式 12/12 gates PASS，
所以 **M6.4 implementation complete**；這不等於 measured/mixed evidence complete。

### 10.47 M6.5：implementation evidence 與 empirical evidence 是兩條 gate

M6.5 的 bank-level acoustic comparison 先驗證 calibrated/normalized invariance；
若 release 含 measured variant，再對每項指標計算 normalized quantile-Wasserstein
distance：

\[
d_m=\frac{1}{101}\sum_{q\in\{0,.01,\ldots,1\}}
\frac{|Q_{\mathrm{syn},m}(q)-Q_{\mathrm{real},m}(q)|}
{Q_{\mathrm{real},m}(.95)-Q_{\mathrm{real},m}(.05)+\epsilon}.
\]

缺少 measured reference 時回報 `not_evaluable`，不能把另一個 synthetic variant
當作 real reference。

Throughput report 會重算 `generated / elapsed`，並要求 generated、skipped、failed
之和等於 task count。Controlled listening 需要 randomized double blind、共同
loudness master gain、hidden reference、degraded anchor、room-disjoint stimuli、
完整 assignment／response／analysis records 與 hashes；empirical tier 至少 20
人，而且 participant-level endpoint 與信賴區間會由 records 重算。Downstream
report 由 recipe indexes 重算 train/test acoustic-space hashes，至少三個 unique
seeds，training／model recipe 必須凍結；paired improvement 的 95% t confidence
interval 由 per-seed values 重算，其 lower bound 必須大於零。

Formal validator 使用真正 generator 的 wall time，但 listening 與 downstream
只使用明確標為 `contract_fixture` 的結構測試。Unblinded、split hash tamper、
single-seed、偽造正 CI 與把 non-human response 改標 empirical 的五種負控制皆被
拒絕，14/14 implementation gates PASS；因沒有真人、
trained-model 與 measured-distribution evidence，empirical exit 和 production
enablement 都保持 false。這就是目前 **M6.5 implementation PASS、empirical
OPEN** 的精確含義。

### 10.48 M6.6：不可變 promotion certificate

若直接改寫 M6.4 的 `release_status`，release SHA、variant manifest hashes、
distribution references 與 recipes 的證據鏈會斷掉。M6.6 因此使用 append-only
promotion：candidate bytes 完全不動，production certificate 參照

\[
H_R=\mathrm{SHA256}(\text{release}),\qquad
H_E=\mathrm{SHA256}(\text{M6.5 evaluation}),
\]

並把所有 promotion gates 的布林結果、blocker list、evidence-bundle hash 一起做
canonical hash。只有

\[
\mathrm{production\_ready}=\bigwedge_k g_k
\]

為真時，decision 才能是 `approved`；其餘情況一律 `blocked`。

Gates 同時要求四個 recipes ready、real/mixed origin semantics 正確、全 item QC
PASS、generator revision pinned、全部 renderer profiles 為
`production_approved`、M6.5 implementation/empirical/enablement 三者 PASS。外部
evidence bundle 必須保存實際 listening/downstream/renderer-approval files，路徑
不得 absolute 或包含 `..`，檔案 hash 需和 M6.5 report 的 artifact hashes 相同。
最後還要求 acoustics、ML、release owner 三個 role-specific approve records。

`PreGeneratedReleaseBank(require_production=True)` 不只檢查 certificate 的內部
hash／boolean consistency，還會重跑所綁定 release 與 evidence audit、驗證固定的
canonical check set，並重算 decision。研究用 candidate reader 不受影響。Formal
validator 的 unsafe path、evaluation hash tamper、forged approved flag、全 true
且重算外層 hash 的稱職偽造、direct status edit 與 production-reader 負控制全部
通過，共 15/15 gates。因此 **M6.6 implementation PASS**，但現有 certificate
忠實回報 **production promotion BLOCKED**。

Hash 只能證明 reviewed bytes 未改，不能自行建立 reviewer 身份信任。Sign-off
records 的作者權限屬部署治理邊界；若需抵抗可任意改 workspace 的攻擊者，應由
受控 CI 或外部 signing service 對 certificate 再做組織級數位簽章。

### 10.49 Training integration 與 matched pilot

`AudioEffectAugmentor.init_room_bank()` 現在以 `bank_type: release` 建立
`PreGeneratedReleaseBank`；若省略 `bank_type`，存在 `recipe_id` 也會自動選 release
mode。Release training config 必須同時指定 recipe、split 與相同的
`usage_role`，dynamic dataset 也會和自己的 train／validation／test role 對帳；
room mode 則拒絕 release-only options。這讓既有 dynamic dataset 不需改變
`apply_rir()` 路徑，就能保留 release、variant、origin、acoustic-space 與 split
provenance。

下一個實驗不是盲目放大，而是 matched backend A/B：各生成 1,000 rooms × 4 RIR，
兩組固定 scene v1、calibrated level、low backend、seed 與 room/item count，只比較
Pyroomacoustics 與 opt-in PathEvents-M4 high backend。兩組都經 M6 manifest、QC 與
variant release，再用同一份 training recipe 做 room-disjoint downstream 比較。
生成／QC／release pipeline 已凍結在公開入口 `generate_m6_bank.py`（底層仍保留
`generate_m6_training_pilot.sh` 相容腳本）；尚未在目標
generation host 執行完整 4,000-item × 2 pilot，因此 throughput 與 QC yield 不能
由六項 validator fixture 外推。

### 10.50 2026-08-02 renderer 與資料鏈 hardening

這一輪修正的是實際演算法與 fail-closed 邊界，不是只 tune Pyroomacoustics：

- pytARD low band 改用單一 sample 的 causal delta excitation，移除舊 bipolar
  excitation 的固定 `1-z^{-L}` comb；M6 CPU/GPU 建議預設分別改為
  `pytard-material`／`pytard-cupy-material`，讓低頻 decay 使用材料頻率相關阻尼；
- M4 FDN 使用 endpoint-complete Butterworth power partitions，從 DC 覆蓋至
  Nyquist；晚場 target energy 由材料 RT60 外插，不再被 finite-order coherent
  path tail 截短；render recurrence 以不超過最短 delay 的 block 等價向量化；
- PathEvent early reflections 會把 surface absorption endpoints 映成 passive
  one-pole admittance phase prior。這是可追溯的模型先驗，不冒充量測 impedance；
- source directivity 在 Pyroomacoustics 與 PathEvent 兩側都實際渲染；障礙物只
  作用於被遮蔽的 direct／early paths，late diffuse tail 會逐步恢復，不再把整條
  RIR 乘同一增益；高階 scattering 的 specular energy removal 交由 FDN diffuse
  field，而不是默認成零散射；
- coherent path 與 FDN RT60 都加入 ISO 9613-1 大氣衰減；有限 duration 尾端加
  cosine-squared fade，最後 sample 精確歸零；
- receiver array／FOA 的 late field 使用單一 aggregate coupling gain，保留同一
  plane-wave field 的 channel energy ratio，不再逐 channel normalize 而破壞
  diffuse isotropy；
- M6 reader／QC 加入安全 item ID、manifestless M6 rejection、split/usage-role
  對帳、pass-only candidate semantics、active-tail echo density、calibrated level
  policy、octave coverage、sample-exact variant lineage 與 batch provenance；
- augmentor RIR cache 改為 32-entry LRU，避免 worker 在長 epoch 無界成長。

因此 2026-08-02 以前生成的 RIR bank 即使舊 manifest 自洽，也不能聲稱包含這些
修正。新的 matched A/B 必須換 output root 重新生成；先前中止的 pilot 只保留為
diagnostic，不可直接拿來做修正後的 winner 判定。

### 10.51 強化後 matched preflight 的實測結論（2026-08-02）

為了先驗證資料鏈與 renderer，再決定是否放大到訓練 bank，我們在同一批 30 個
rooms 上各生成 2 個 items，固定 `v1/mixed`、seed `1337`、calibrated
`16 kHz / 1.6 s`，並以 `pytard-cupy-material` 作共同低頻 backend。Pyroomacoustics
與 PathEvents-M4 各產生 60 items／300 channels；60/60 matched identity、60/60
QC PASS、0 quarantine、release audit PASS，且兩個 release 的 train reader 都能
載入 56 個 PASS items。可重算的完整報告是
[`preflight_validation_summary.json`](../../egs/rir_generation/exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/preflight_validation_summary.json)。

| 聲學量中位數 | Pyroomacoustics | PathEvents-M4 | M4 相對變化 |
|---|---:|---:|---:|
| DRR | -4.83 dB | -3.24 dB | +1.59 dB |
| C50 | 6.10 dB | 10.28 dB | +4.18 dB |
| C80 | 8.03 dB | 15.01 dB | +6.97 dB |
| T20 | 0.990 s | 0.472 s | -0.518 s |
| `|T20 − scene RT60|` | 0.327 s | 0.080 s | M4 較小 |

M4 在這個 matched synthetic sample 呈現較乾、較早期能量主導的聲場，而其
broadband T20 更接近由材料推導的 scene RT60；這是 M4 FDN late-field 與
PathEvent early-field 真正介入的結果，不是單純改 Pyroomacoustics 參數。兩邊的
causal arrival error 中位數為 0.033 ms，最後 sample 全部精確為零；固定 390 Hz
comb 與 M4 的 6–7.8 kHz late-tail coverage 也通過。這些結果只證明實作機制與
M6 admission/release 流程可運作，不能單憑它們判定「哪一個更接近真實」：沒有
measured reference、human listening 或 downstream model，而且 validation/test
各只有 2 items。

觀察到的 aggregate throughput 約為 Pyroomacoustics（2 workers）6.5 秒／item，
以及 M4（8 workers、resume 後）8.2 秒／item；M4 的瓶頸目前在 CPU
PathEvent／material-boundary，GPU 主要只加速共同低頻 solve。

另有兩個 provenance 邊界必須保留：此次生成記錄的是 dirty code revision，因而
只能作 candidate preflight；Pyroomacoustics 雖在 renderer 內使用設定的 air
absorption，但目前 high-band metadata 沒有完整保存該 policy 與 coefficients，
可追溯性不如 M4。下一步仍是以乾淨、pinned revision 生成完整 4,000-item matched
pilot，再進行 measured/listening/downstream evidence。

## 11. 已知限制

閱讀實驗結果時必須保留以下邊界：

- 低頻 eigenfunction 目前以矩形房間為基礎；
- production material-modal damping 尚未使用新的低頻複數阻抗，也沒有
  angle dependence；
- FDTD reference 支援被動多極點 locally reacting admittance，但尚未
  代表已映射的真實 installed material；
- impedance eigenproblem 已完成 separable shoebox 3D；modal residue 已
  通過兩個 model-derived thickness variants 的 FDTD position／room／grid
  holdout，FDTD pressure-cell residue 也已轉為與 \(1/r\) direct tap 一致的
  數位 RIR convention，anechoic direct crossover 亦已通過 complex-response
  gate，但尚未通過不同實測材料、full-room modal/geometric crossover 或
  measured-room production validation；
- analytic probe 只枚舉有限模態，不是完整波場；
- M3.4 已將 order-12 與 FDTD response 做可精確重建的
  direct-anchor／early／later 歸因，主要 full-transfer error 位於 early
  reflection phase/interference；
- M3.10 已完成 multi-axis、non-equal-azimuth 與 face/edge/corner holdout；
  opt-in fourth-order plane-mode reference 通過，但 raw point-source
  discretization 不嚴格 reciprocal，正式 reference 需執行雙向 Green 平均；
  production/default 二階 solver 沒有改變；
- M3.11 使用擴展 reference 重跑後，240 Hz full-room complex crossover
  仍未通過；主要誤差仍在 early reflection phase/interference 與 modal
  crossover 上緣增益；
- M3.13／M3.14 的家具是 closed vertical prisms，不是一般 triangle mesh；
  diffraction 是 1 kHz bounded reference model；first-order scattering 有明確
  path branch，高階互動則以 specular energy removal 餵入 FDN diffuse field，
  尚未列舉每條高階 diffuse path；
- PathEvent high backend 已可生成完整 hybrid RIR，但仍是明確 opt-in；
  Pyroomacoustics 保持 production default；
- source／receiver 已支援 omni、cardioid、hypercardioid 與 figure-eight
  一階 real pressure pattern；任意量測 directivity 仍需轉成可追溯模型；
- M3 measured exit 的 C50 與 continuous early-energy centroid 有改善，
  但 diagnostic dominant-peak timing 退步；目前 measured heldout 也是
  item-disjoint，不是已證明的 room-disjoint path annotation；
- M4.6 實作 exit 已完成，且 model-derived 17 cm array fixture 通過
  coherence／IACC gate；它不是 measured multi-receiver room bank，內附的
  BRIR decoder 也明確不是 measured HRTF，因此 empirical／production exit
  仍未執行；125/250 Hz 仍由 modal/crossover 處理；
- M5.2d 在 synthetic shoebox 可識別六個 effective boundary reflection
  groups，但 mono 下 absorption／scattering 的 12-column split rank 只有 6；
  M5.4 必須使用同步 receiver 證據，且目前仍不是 measured-room fit；
- M5.3 runner、M5.4 spatial profile、M5.5 constrained residual 與 M5.6
  aggregate exit 的 implementation 已完成；沒有合格 controlled campaign、
  listening 與 room-disjoint downstream 證據，所以 empirical／production
  exit 仍為 OPEN；
- M6.1–M6.6 已完成 bank contract、正式 parallel/resume generator、per-item
  QC/quarantine、distribution/variant recipe release 與 fail-closed evaluation
  contract、immutable production-decision certificate 與 production reader；
  real/mixed recipes、measured distribution、真人 listening、trained room-disjoint
  downstream、renderer approval 與三方 sign-off 仍缺，因此 promotion BLOCKED；
- M6 release recipe 已接進 training augmentation；修正前的 A/B pilot 已中止，
  2026-08-02 已完成 30-room × 2-item 的 hardened matched preflight（兩個 backend
  各 60 items，candidate PASS），但完整 4,000-item Pyroomacoustics／PathEvents-M4
  pilots 尚未在目標 host 完成，不能先宣稱大規模 throughput、QC yield 或
  downstream winner；
- 家具已進入 PathEvent visibility，但材料 surface patches 尚未在所有
  backend 中成為逐 polygon 的精確幾何邊界；
- modal peaks 是 response-level 特徵，不是可靠的一對一 room-mode 標籤；
- 聲學分布接近不等於下游模型一定改善。

這些限制會記錄在輸出 metadata 和 benchmark 報告中，而不是留作隱含假設。

## 12. 如何重現

生成並驗收可播放的 M1 audition bank：

```bash
.venv/bin/python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1_bank \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --record-realized-metrics \
  --n-rooms 10 --rir-per-room 5 \
  --sample-rate 16000 --duration 1.0 \
  --low-backend analytic --pra-max-order 8 --pra-n-rays 2000 \
  --num-workers 8 --seed 1337

.venv/bin/python egs/rir_generation/tools/audition/build_audition_bank.py \
  --bank egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1_bank \
  --output-dir egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1 \
  --dry-wav test/test_case/1272-141231-0008.flac \
  --dry-wav test/test_case/61-70970-0040.flac \
  --min-items 50 --num-previews 6
```

這個 gate 實際讀取全部 50 items、250 個 RIR channel，檢查 WAV/metadata、
有限值、clipping、尾場能量，以及每個 channel 在物理
\(t=d/c\) 以前必須完全沒有聲能。第一次掃描因此發現 Pyroomacoustics
的 fractional-delay/ray-tracing 低幅能量在 direct alignment 後被移到
`t=0`；修正為逐 channel 裁掉物理到達前樣本並重生後，pre-arrival peak
全部為 0。

最終 audition gate 通過：near/far median DRR gap 為 10.63 dB，
\(R^2\ge0.9\) 的有效 T20 比例為 79.6%，沒有格式、非有限值、clipping、
尾場或因果失敗。完整結果在 `egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1/report.json`，播放
索引在 `manifest.json`。這代表 M1 已通過本地結構與 audition usability
gate，不代表已通過 measured-room 或 M2 阻抗驗收。

生成 experimental M2 bank：

```bash
python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m2/hybrid_rir_material_modal_m2 \
  --scene-version v1 --room-type mixed \
  --low-backend pytard-material \
  --output-mode calibrated --record-realized-metrics \
  --n-rooms 20 --rir-per-room 5 \
  --sample-rate 16000 --duration 1.6
```

用相同 estimator 比較 synthetic 與 measured bank：

```bash
python egs/rir_generation/compare_modal_acoustics.py \
  m1=/path/to/m1 \
  m2=/path/to/m2 \
  measured=/path/to/measured \
  --per-bank 100 \
  --analysis-duration-s 0.8 \
  --reference-tag measured \
  --json-output egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics.json
```

執行核心 RIR 測試：

```bash
pytest -q test/test_rir_metrics.py \
  test/test_rir_scene_v2.py \
  test/test_rir_path_events.py \
  test/test_fdtd_reference.py \
  test/test_hybrid_rir.py \
  test/test_rir_benchmark_cli.py
```

重現 M3.1 PathEvent gate：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/validate_path_events.py \
  --output-report \
  egs/rir_generation/phases/m3_wave_path/reports/path_events_m3_1_report.json
```

重現 M3.2 causal filter 與 full-room diagnostic：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m2_impedance/scripts/validate_full_room_crossover.py \
  --include-causal-path-event-first-order \
  --output-report \
  egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_2_report.json

PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/validate_path_event_filters.py \
  --full-room-report \
  egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_2_report.json \
  --output-report \
  egs/rir_generation/phases/m3_wave_path/reports/path_event_filters_m3_2_report.json
```

重現 M3.3 higher-order ordered paths：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m2_impedance/scripts/validate_full_room_crossover.py \
  --causal-path-event-max-order 12 \
  --output-report \
  egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json

PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_path_events.py \
  --full-room-report \
  egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json \
  --output-report \
  egs/rir_generation/phases/m3_wave_path/reports/higher_order_path_events_m3_3_report.json
```

重現 M3.4 direct／early／later attribution：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_direct_early_later_attribution.py \
  --full-room-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/direct_early_later_m3_4_report.json
```

重現 M3.5 oblique FDTD boundary audit：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_oblique_fdtd_boundary.py \
  --full-room-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/oblique_fdtd_boundary_m3_5_report.json
```

重現 M3.6 corrected boundary experiment：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_corrected_fdtd_boundary.py \
  --full-room-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/corrected_fdtd_boundary_m3_6_report.json
```

重現 M3.7 higher-order harmonic candidate gate：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_fdtd_candidate.py \
  --full-room-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/higher_order_fdtd_candidate_m3_7_report.json
```

重現 M3.8 fourth-order 1D time-domain closure gate：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_fdtd_time_domain.py \
  --full-room-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_3_report.json \
  --harmonic-candidate-report \
    egs/rir_generation/phases/m3_wave_path/reports/higher_order_fdtd_candidate_m3_7_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/higher_order_fdtd_time_domain_m3_8_report.json
```

重現 M3.9 opt-in fourth-order 3D FDTD gate：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_fourth_order_fdtd_3d.py \
  --time-domain-report \
    egs/rir_generation/phases/m3_wave_path/reports/higher_order_fdtd_time_domain_m3_8_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/fourth_order_fdtd_3d_m3_9_report.json
```

重現 M3.10 expanded holdouts：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_fourth_order_fdtd_holdouts.py \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/fourth_order_fdtd_holdouts_m3_10_report.json
```

重現 M3.11 fourth-order reciprocal full-room audit：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m2_impedance/scripts/validate_full_room_crossover.py \
  --fdtd-reference-profile fourth_order_reciprocal \
  --causal-path-event-max-order 12 \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_11_report.json

PYTHONPATH=. python \
  egs/rir_generation/phases/m3_wave_path/scripts/validate_direct_early_later_attribution.py \
  --full-room-report \
    egs/rir_generation/phases/m3_wave_path/reports/full_room_crossover_m3_11_report.json \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/direct_early_later_m3_11_report.json
```

重現 M3.12 engine evaluation 與 M3.13／M3.14 interaction gate：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/evaluate_mesh_engine.py \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/mesh_engine_m3_12_report.json

PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/validate_scene_interactions.py \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/scene_interactions_m3_13_m3_14_report.json
```

重現 measured exit 與總 M3 exit。第一個命令需要本機 frozen M1 paired bank
與 held-out measured view：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/validate_m3_measured_exit.py \
  --m1-bank egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  --measured-bank \
    /work/any_exp_link/puresound_exp/real_rir_16k_heldout_view/items \
  --synthetic-items 20 --measured-channels 100 \
  --path-event-max-order 8 \
  --output-report \
    egs/rir_generation/phases/m3_wave_path/reports/m3_measured_exit_report.json

PYTHONPATH=. python egs/rir_generation/phases/m3_wave_path/scripts/validate_m3_exit.py \
  --output-report egs/rir_generation/phases/m3_wave_path/reports/m3_exit_report.json
```

重現 M4.1 measured／M1／M3 late-field baseline：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_late_field_baseline.py \
  measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items \
  m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  m3=egs/rir_generation/exp/rir_realism/m3/rir_m3_late_baseline \
  --reference-tag measured --per-bank 100 --seed 20260731 \
  --json-output \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_late_field_baseline.json
```

重現 M4.2 noise-aware multiband target 與 spatial-input audit：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_late_field.py \
  measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items \
  m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  m3=egs/rir_generation/exp/rir_realism/m3/rir_m3_late_baseline \
  --reference-tag measured --per-bank 50 --seed 20260731 \
  --json-output \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json
```

重現 M4.3 isolated deterministic multiband FDN core：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_fdn.py \
  --target-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json \
  --output-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_fdn_report.json \
  --output-wav egs/rir_generation/exp/rir_realism/m4/rir_m4_fdn_core/rir_m4_fdn_core.wav
```

此命令只生成與驗證 late-field impulse，不會改寫 production RIR bank。

重現 M4.4 coupling gate 與 M3/M4 對照 WAV：

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_path_event_fdn_coupling.py \
  --target-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json \
  --output-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_path_event_fdn_coupling_report.json \
  --output-dir egs/rir_generation/exp/rir_realism/m4/rir_m4_coupling
```

生成 opt-in M4 bank：

```bash
PYTHONPATH=. python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m4/path_event_fdn \
  --scene-version v1 --room-type office \
  --high-backend path-events-m4 \
  --fdn-mixing-time-ms 24 --fdn-transition-ms 16 \
  --fdn-delay-lines 16 --fdn-seed 20260731 \
  --low-backend analytic-material \
  --sample-rate 16000 --duration 1.2 \
  --n-rooms 1 --rir-per-room 1
```

上面的 bank CLI 保留既有「5 個 sources、1 個 receiver」資料集格式。要生成
同一 source 的同步 receiver array、完整 FOA 與示範 BRIR，使用新的空間
CLI：

```bash
PYTHONPATH=. python egs/rir_generation/render_spatial_rir.py \
  --sample-rate 16000 --duration 1.2 \
  --max-order 4 --delay-lines 16 --plane-waves 128 \
  --binaural-spacing-m 0.17 --binaural-decoder analytic \
  --output-dir egs/rir_generation/exp/rir_realism/m4/spatial_demo
```

輸出包含 `scene.json`、`metadata.json`、`receiver_rir.wav`、
`ambisonic_acn_sn3d_rir.wav` 與 `binaural_brir.wav`。`analytic` decoder 只驗證
pipeline；production HRTF 應以 `AmbisonicBinauralDecoder` 注入具有 dataset
ID／license／processing provenance 的 `[2,4,L]` FIR。

重現 M4.5、M4.6 與 aggregate implementation exit：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_spatial_rir.py
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_binaural_brir.py
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_m4_exit.py
```

最後一個命令應顯示 `M4 implementation exit: PASS` 與
`M4 empirical/production exit: OPEN`；兩者不是互相矛盾，而是證據層級
不同。

## 13. 名詞對照

| 名詞 | 中文說明 |
|------|----------|
| RIR | 房間脈衝響應 |
| RT60 | 聲壓級衰減 60 dB 所需時間 |
| DRR | 直接聲能量對殘響能量比 |
| C50 / C80 | 50/80 ms 前後的 early-to-late energy ratio |
| Room mode | 房間邊界與尺寸允許的聲學特徵振盪 |
| Modal node | 該模態聲壓為零的位置 |
| Q factor | 峰值頻率除以 half-power bandwidth |
| Absorption coefficient | 被邊界吸收的能量比例 |
| Acoustic impedance | 聲壓與粒子速度的複數比例 |
| Reactive component | 儲存並釋放能量、造成相位變化的阻抗成分 |
| FDTD | 有限差分時域波動求解 |
| Image-source method | 用鏡像音源建立鏡面反射路徑 |
| PathEvent | 渲染成樣本前，保存單一路徑距離、方向、surface、delay 與 complex gain 的中間表示 |
| Fractional delay | 不把連續到達時間只 round 成整數 sample 的次取樣延遲 |
| Echo density | 局部時間窗內可視為密集反射／擴散尾場的統計程度 |
| Mixing time | 稀疏、可分辨反射轉為密集統計尾場的時間，不等於 RT60 |
| FDN | Feedback Delay Network；以多條 delay line、feedback mixing 與衰減濾波器生成密集殘響尾場 |
| IACC | 左右耳 RIR 在有限 lag 內的最大絕對 normalized cross-correlation |
| Spatial coherence | 兩個 receiver 在各頻率的 normalized complex cross spectrum |
| Positive-real | 在右半平面具有非負 real part 的被動 impedance/admittance transfer property |
| Bounded-real | unit circle 上 transfer magnitude 不超過 1；此處對應被動 reflection |
| Cayley transform | 將 positive-real admittance 映射為 bounded-real reflection 的分式轉換 |
| Wasserstein distance | 比較兩個一維分布搬移成本的距離 |
| Room-disjoint | 開發與測試資料沒有共享任何實體房間 |

## 14. 程式與文件入口

| 入口 | 用途 |
|------|------|
| [`puresound/audio/acoustic_impedance.py`](../../puresound/audio/acoustic_impedance.py) | 複數阻抗、反射係數、吸收率與 phase-aware 轉換 |
| [`puresound/audio/impedance_priors.py`](../../puresound/audio/impedance_priors.py) | Miki/glass-wool reference priors、證據等級與 relaxation fitting |
| [`puresound/audio/impedance_measurements.py`](../../puresound/audio/impedance_measurements.py) | 法向入射 complex impedance CSV/JSON 契約 |
| [`puresound/audio/impedance_fitting.py`](../../puresound/audio/impedance_fitting.py) | 被動多極點 complex reflection fitting |
| [`puresound/audio/impedance_modes.py`](../../puresound/audio/impedance_modes.py) | 1D 複數 impedance cavity eigenproblem |
| [`puresound/audio/rir_source_convention.py`](../../puresound/audio/rir_source_convention.py) | FDTD pressure-cell source、\(1/r\) 數位 RIR 與 modal residue 的版本化轉換 |
| [`puresound/audio/rir_path_events.py`](../../puresound/audio/rir_path_events.py) | M3 PathEvent schema、任意階 shoebox paths、家具 visibility、穿透、繞射、受控散射與因果 renderer |
| [`puresound/audio/rir_attribution.py`](../../puresound/audio/rir_attribution.py) | M3.4 exact direct-anchor／early／later complementary decomposition |
| [`puresound/audio/hybrid_rir.py`](../../puresound/audio/hybrid_rir.py) | hybrid generator、modal recurrence、crossover，以及 opt-in M3 coherent／M4 FDN PathEvent backends |
| [`puresound/audio/rir_scene.py`](../../puresound/audio/rir_scene.py) | v2 material-first 場景、SceneObject 幾何與序列化 |
| [`puresound/audio/rir_materials.py`](../../puresound/audio/rir_materials.py) | 材料頻譜與 priors |
| [`puresound/audio/fdtd_reference.py`](../../puresound/audio/fdtd_reference.py) | 獨立 FDTD 參考解與 staggered-grid discrete oblique reflection |
| [`puresound/audio/low_frequency_modes.py`](../../puresound/audio/low_frequency_modes.py) | 低頻 peak、bandwidth 與 Q estimator |
| [`puresound/audio/rir_metrics.py`](../../puresound/audio/rir_metrics.py) | DRR、C50/C80、EDT/T20/T30、noise-aware multiband NED、mixing time、IACC 與 diffuse coherence |
| [`puresound/audio/multiband_fdn.py`](../../puresound/audio/multiband_fdn.py) | M4.3 deterministic prime delays、orthogonal feedback、band-dependent contraction 與 isolated impulse renderer |
| [`puresound/audio/rir_air_absorption.py`](../../puresound/audio/rir_air_absorption.py) | ISO 9613-1 atmospheric attenuation、causal minimum-phase FIR 與 air-adjusted RT60 |
| [`puresound/audio/rir_late_coupling.py`](../../puresound/audio/rir_late_coupling.py) | M4.4 direct-relative equal-power crossfade、PathEvent injection 與 exact post-transition energy match |
| [`puresound/audio/spatial_late_field.py`](../../puresound/audio/spatial_late_field.py) | M4.5 isotropic plane-wave quadrature、receiver delays、ACN/SN3D FOA 與同步 array coupling |
| [`puresound/audio/spatial_rir.py`](../../puresound/audio/spatial_rir.py) | RoomSceneV2 到完整 receiver-array／FOA／optional BRIR 的高階 opt-in API |
| [`puresound/audio/binaural_renderer.py`](../../puresound/audio/binaural_renderer.py) | M4.6 可追溯 `[2,4,L]` Ambisonic binaural FIR decoder 與 analytic non-HRTF demo |
| [`puresound/audio/rir_measurement_campaign.py`](../../puresound/audio/rir_measurement_campaign.py) | M5.1 受控房間、transducer、pose、repeated ESS、room split、asset hash 與 readiness audit schema |
| [`puresound/audio/rir_calibration.py`](../../puresound/audio/rir_calibration.py) | M5.1 MR-STFT、EDC、arrival、octave、spatial coherence、causality 與 decay reference loss |
| [`puresound/audio/rir_inverse_calibration.py`](../../puresound/audio/rir_inverse_calibration.py) | M5.2/M5.2b causal approximate renderer、noise/nuisance/model-mismatch fixture、noise-aware M4-proxy fitting、multi-start 與 local Jacobian evidence |
| [`puresound/audio/rir_m4_inverse_calibration.py`](../../puresound/audio/rir_m4_inverse_calibration.py) | M5.2c actual M4 coupling、離散 mixing topology profile、bounded coherent-gain／octave-RT60 inner solve 與 local-identifiability evidence |
| [`puresound/audio/rir_m5_pipeline.py`](../../puresound/audio/rir_m5_pipeline.py) | M5.2d grouped PathEvent inverse、local identifiability 與 M5.4 synchronized spatial candidate profile |
| [`puresound/audio/rir_measured_calibration.py`](../../puresound/audio/rir_measured_calibration.py) | M5.3 readiness-gated campaign fitting、deterministic position split、train-room fit 與 room-disjoint evaluation |
| [`puresound/audio/rir_constrained_residual.py`](../../puresound/audio/rir_constrained_residual.py) | M5.5 direct-relative causal residual、decay／energy constraints 與三路 ablation |
| [`puresound/audio/rir_bank_manifest.py`](../../puresound/audio/rir_bank_manifest.py) | M6.1/M6.2 deterministic acoustic-space split、content-addressed assets/scenes/split indexes/task plan/manifest、renderer provenance 與 audit contract |
| [`puresound/audio/rir_bank_qc.py`](../../puresound/audio/rir_bank_qc.py) | M6.3 content-addressed physical QC、pass/not-evaluable/not-applicable/fail semantics、candidate indexes 與 quarantine |
| [`puresound/audio/rir_bank_release.py`](../../puresound/audio/rir_bank_release.py) | M6.4 distribution snapshots、calibrated/peak-normalized lineage、release recipes 與 audit |
| [`puresound/audio/rir_bank_evaluation.py`](../../puresound/audio/rir_bank_evaluation.py) | M6.5 acoustic/throughput/listening/downstream evidence contract 與 implementation/empirical 雙 exit |
| [`puresound/audio/rir_bank_production.py`](../../puresound/audio/rir_bank_production.py) | M6.6 immutable promotion certificate、evidence bundle/sign-off audit 與 production-reader gate |
| [`egs/rir_generation/tools/audition/build_audition_bank.py`](../../egs/rir_generation/tools/audition/build_audition_bank.py) | M1 全量品質 gate 與乾／濕 near/far audition previews |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_path_events.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_path_events.py) | M3.1 geometry、reciprocity、continuity 與跨 sample-rate fractional-delay gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_path_event_filters.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_path_event_filters.py) | M3.2 positive-real／bounded-real filter、PathEvent time renderer 與 full-room diagnostic gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_path_events.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_path_events.py) | M3.3 order 1/2/4/8/12 路徑、解析一致性、edge/corner policy 與 full-room convergence gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_direct_early_later_attribution.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_direct_early_later_attribution.py) | M3.4 FDTD／PathEvent early-later complex error 與 coherent cross-term 歸因 |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_oblique_fdtd_boundary.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_oblique_fdtd_boundary.py) | M3.5 continuous／PathEvent／FDTD 斜入射 reflection 與 grid-convergence gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_corrected_fdtd_boundary.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_corrected_fdtd_boundary.py) | M3.6 wall-face／half-time 補償、被動性與 time-domain cross-ratio gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_fdtd_candidate.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_fdtd_candidate.py) | M3.7 fourth-order dispersion、quadratic face/time、被動性、CFL 與 refinement harmonic gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_fdtd_time_domain.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_higher_order_fdtd_time_domain.py) | M3.8 fourth-order 1D one-sided closure、cross-ratio refinement 與長時 stability gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_fourth_order_fdtd_3d.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_fourth_order_fdtd_3d.py) | M3.9 opt-in 3D fourth-order 六面 closure、normal cross-ratio 與 oblique two-probe gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_fourth_order_fdtd_holdouts.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_fourth_order_fdtd_holdouts.py) | M3.10 multi-axis／azimuth plane modes 與 face/edge/corner reciprocal reference gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/evaluate_mesh_engine.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/evaluate_mesh_engine.py) | M3.12 既有 mesh acoustic engine 能力與整合決策 |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_scene_interactions.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_scene_interactions.py) | M3.13／M3.14 家具 visibility、穿透、繞射、散射、互易性與 rendering gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_m3_measured_exit.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_m3_measured_exit.py) | M1/M3 paired complete-hybrid 對 held-out measured 的 C50 與 early-energy centroid gate |
| [`egs/rir_generation/phases/m3_wave_path/scripts/validate_m3_exit.py`](../../egs/rir_generation/phases/m3_wave_path/scripts/validate_m3_exit.py) | 彙整 M3 implementation steps、四類 exit 條件與已知限制 |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_late_field_baseline.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_late_field_baseline.py) | M4.1 measured/M1/M3 echo-density／mixing-time 分布基線 |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_late_field.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_late_field.py) | M4.2 noise-aware octave target、measured envelope 與 spatial channel-semantics audit |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_fdn.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_fdn.py) | M4.3 FDN 結構、因果／穩定性、T20、mixing time 與 late-NED gate |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_path_event_fdn_coupling.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_path_event_fdn_coupling.py) | M4.4 coupling 結構、完整 hybrid 相容性與 M3/M4 multiband 比較 |
| [`egs/rir_generation/render_spatial_rir.py`](../../egs/rir_generation/render_spatial_rir.py) | 生成同步 receiver array、ACN/SN3D FOA 與 optional analytic BRIR artifacts |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_spatial_rir.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_spatial_rir.py) | M4.5 causality、determinism、early/energy、diffuse coherence 與 IACC gate |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_binaural_brir.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_binaural_brir.py) | M4.6 decoder injection、provenance、causality 與 BRIR contract gate |
| [`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_m4_exit.py`](../../egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_m4_exit.py) | 分開彙整 M4 implementation PASS 與 empirical／production OPEN |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py) | M5.1 schema／loss deterministic probes、non-evidence template 與 existing-bank readiness audit |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_synthetic_recovery.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_synthetic_recovery.py) | M5.2 hidden-parameter recovery、multi-start、unseen-position oracle gate 與三段對照 WAV |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_robust_recovery.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_robust_recovery.py) | M5.2b controlled noise／gain／latency／model mismatch、waveform ablation、held-out gate 與六段 RIR artifacts |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_m4_parameter_mapping.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_m4_parameter_mapping.py) | M5.2c actual PathEvent＋multiband-FDN parameter profile、held-out M5.1 oracle、14 gates 與三段 RIR artifacts |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_group_identifiability.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_group_identifiability.py) | M5.2d 六面 effective reflection recovery、Jacobian／multi-start／holdout 與 absorption-scattering rank ablation |
| [`egs/rir_generation/phases/m5_calibration/scripts/fit_m5_measured_campaign.py`](../../egs/rir_generation/phases/m5_calibration/scripts/fit_m5_measured_campaign.py) | M5.3 campaign audit、fail-closed CLI、train-position fit 與 position／room holdout report |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measured_runner.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measured_runner.py) | 以完整 non-evidence fixture 驗證 M5.3 runner 能執行且不冒充 measured evidence |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_spatial_calibration.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_spatial_calibration.py) | M5.4 actual-M4 synchronized scattering／directivity profile 與 mono hard rejection |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_constrained_residual.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_constrained_residual.py) | M5.5 causal／decay／energy-bounded residual 與 held-out three-way ablation |
| [`egs/rir_generation/phases/m5_calibration/scripts/validate_m5_exit.py`](../../egs/rir_generation/phases/m5_calibration/scripts/validate_m5_exit.py) | M5.6 implementation PASS、empirical OPEN 與 production-disabled aggregate decision |
| [`egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_contract.py`](../../egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_contract.py) | M6.1 三 split fixture、12 個 schema／hash／leakage／false-production gates 與正式 report |
| [`egs/rir_generation/phases/m6_bank/scripts/validate_m6_reproducible_generation.py`](../../egs/rir_generation/phases/m6_bank/scripts/validate_m6_reproducible_generation.py) | M6.2 actual Pyroom generator serial／parallel／fresh exact hashes、tamper resume、revision/config isolation 與 17-gate report |
| [`egs/rir_generation/phases/m6_bank/scripts/validate_m6_item_qc.py`](../../egs/rir_generation/phases/m6_bank/scripts/validate_m6_item_qc.py) | M6.3 actual generator QC、四種 quarantine 負控制與 report-tamper gate |
| [`egs/rir_generation/phases/m6_bank/scripts/validate_m6_variant_release.py`](../../egs/rir_generation/phases/m6_bank/scripts/validate_m6_variant_release.py) | M6.4 deterministic variants/distributions/recipes、release reader 與 index-tamper gate |
| [`egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_evaluation.py`](../../egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_evaluation.py) | M6.5 distribution/throughput、listening/downstream contract 與 fail-closed empirical/production gates |
| [`egs/rir_generation/phases/m6_bank/scripts/validate_m6_production_decision.py`](../../egs/rir_generation/phases/m6_bank/scripts/validate_m6_production_decision.py) | M6.6 15-gate decision/certificate、unsafe/tamper/competent-forgery 負控制與 BLOCKED 正式結果 |
| [`egs/rir_generation/compare_modal_acoustics.py`](../../egs/rir_generation/compare_modal_acoustics.py) | bank-level modal distribution 比較 |
| [`egs/rir_generation/README.md`](../../egs/rir_generation/README.md) | 生成、分析與資料 bank 操作 |
