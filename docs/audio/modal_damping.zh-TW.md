# 材質推導之低頻模態阻尼（Material-derived low-frequency modal damping）

English version: `modal_damping.md`

完整背景、算法與目前實驗判斷見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md)。

M2 在 `puresound.audio.rir.render.low_frequency` 中引入一條明確的實驗性低頻
路徑（low-frequency path）。可用以下任一種方式選用：

```text
analytic-material
pytard-material
pytard-cupy-material
```

這些變體都需要 `--scene-version v1`。對應的 legacy backend 名稱維持共用的
RT60 行為，讓 M1 與 M2 可以繼續獨立做 benchmark。

## Boundary participation（邊界能量參與）

對一個矩形房間，rigid-wall 壓力特徵函數（eigenfunction）為：

```text
phi_n(x,y,z) =
  cos(nx pi x / Lx) cos(ny pi y / Ly) cos(nz pi z / Lz)
```

沿某一軸的體積 norm，在該軸 mode index 為零時是 `L`，否則是 `L/2`。針對每個
模態頻率，PureSound 會內插六個序列化的表面吸音頻譜，並計算：

```text
P_n =
  (alpha_west + alpha_east) / Ix
  + (alpha_south + alpha_north) / Iy
  + (alpha_floor + alpha_ceiling) / Iz

gamma_n = s_loss c P_n / 8
zeta_n = gamma_n / omega_n
RT60_n = ln(1000) / gamma_n
Q_n = omega_n / (2 gamma_n)
```

`c P_n / 4` 是一階 Sabine 能量損耗率；模態振幅（amplitude）的衰減率是它的
一半。這個 weak-loss 模型刻意採用吸音係數，而不是對整個房間擬合單一 RT60。
若某個模態在某一維度的 index 非零，該維度對向牆面對它的能量參與度，會是
index 為零時的兩倍，所以改動一片表面，對不同模態的影響並不相同。

`s_loss` 預設為 1.0。CLI 提供 `--material-modal-loss-scale`，僅供受控診斷
使用，而且每次結果都會把這個值序列化下來。開發階段推得的 0.58 這個值能
還原整體 median Q，卻讓 joint heldout peak-count／bandwidth 分布變差，
因此並未被接受為物理常數。

## Source/receiver modal coupling（聲源／接收端模態耦合）

Analytic probe backend 會用每個矩形模態在實際端點（聲源與接收端）上的
特徵函數值，去激發並觀測它：

```text
A_n is proportional to
  (2 - delta_nx0) (2 - delta_ny0) (2 - delta_nz0)
  phi_n(source) phi_n(receiver) / omega_n
```

括號中的因子是 cosine-mode 體積 norm 的倒數。因此，若聲源或接收端剛好落在
某個模態節點（modal node）上，該模態就會被壓抑；交換聲源與接收端的位置，
回應完全不變（reciprocity）。模態尾段從幾何 direct-arrival 的時間點、以零
相位開始，所以這個有限項的解析展開不會產生 pre-arrival tail。

Probe 會列舉每個維度 index 從 0 到 5、且落在低頻頻段內的所有 index
triplet。它明確的安全上限是 256 個模態，高於該 index 上限下 215 種可能的
非 DC triplet 數。先前固定只取前 64 個的截斷方式，會漏掉絕大多數
200–300 Hz 的共振，新版 probe 已經不再使用它。Metadata 會記錄耦合模型、
index 上限與模態數上限。

## Exact damped recurrence（精確阻尼遞迴）

pytARD adapter 早已將房間對角化成彼此獨立的模態。M2 把原本
無阻尼的 homogeneous 遞迴式，換成以下方程式的精確取樣極點（pole）解：

```text
q_n'' + 2 gamma_n q_n' + omega_n^2 q_n = f_n
```

給定取樣間隔 `dt`、阻尼頻率 `wd` 與 pole 半徑 `r`：

```text
r   = exp(-gamma_n dt)
wd  = sqrt(omega_n^2 - gamma_n^2)
a1  = 2 r cos(wd dt)
a2  = -r^2
b   = (1 - a1 - a2) / omega_n^2

q[k+1] = a1 q[k] + a2 q[k-1] + b f[k]
```

在零阻尼時，這個式子會精確化簡回先前的 pytARD 遞迴式。Material path 會
停用 `apply_rt60_decay_envelope`；metadata 會記錄
`global_rt60_envelope_applied: false`。

## Metadata and validation（Metadata 與驗證）

每個套用 material damping 的項目都會記錄：

- 模態 indices 與頻率；
- 振幅衰減率；
- 預測的模態 RT60 與 Q；
- 取樣模態集合中 RT60 與 Q 的最小／中位數／最大值；
- 邊界模型，以及沒有 global envelope 這件事。

目前測試驗證了：

- 與零阻尼相容的遞迴式維持穩定；
- 改動西牆時，對 x 軸向模態的影響，是 x index 為零之模態的兩倍；
- 同一個 scene 中，不同模態的衰減率彼此不同；
- 強吸音邊界能讓精確 pytARD 晚場模態能量，比反射邊界低超過 40 dB；
- 一個獨立的 staggered-grid 3D FDTD 參考解，能在頻率誤差 1% 內、Q 誤差
  20% 內還原前兩個房間模態；
- 一個 damped-sinusoid 測試 fixture，能在 2% 誤差內還原 spectral Q；
- analytic probe 遵守模態節點、causality 與 source/receiver reciprocity；
- 既有的 causal crossover 與 bounded-gain 回歸測試仍然通過。

FDTD 的公式推導、數值參考值、response-level 的 peak/Q 估計器，以及 bank
比較指令，記錄在 [`modal_validation.zh-TW.md`](modal_validation.zh-TW.md)。

## Current M2 boundary（目前 M2 的邊界）

這是第一版 M2 實作，還不是它的最終驗收結果：

- 邊界損耗目前是一階吸音／參與度模型；
- 複數阻抗相位已經在 validation FDTD 中被驗證過，同一個 rational boundary
  現在也驅動一個已驗證的 1D 複數 cavity 特徵值問題，但兩者都還沒有套用到
  這個以 production 為導向的 3D 模態遞迴上；角度相依性也尚未納入；
- 非 shoebox 幾何造成的模態耦合，目前是缺席的；
- 修正後的密集成對比較顯示，目前的阻尼過量且形狀不對；單純調整一個
  scalar loss 校正值，並不能修正整個 joint distribution；
- 在把 material backend 變成 production 預設之前，還需要直接的
  installed-material 阻抗證據、3D 模態特徵值問題的整合，以及下游測試。

複數量測契約、被動 multi-pole fitting、FDTD 狀態，以及 1D 特徵值參考解，
記錄在 [`impedance_measurements.zh-TW.md`](impedance_measurements.zh-TW.md)。
