# 低頻 modal validation

English: [modal_validation.md](modal_validation.md)

本頁說明低頻 impedance 與 modal implementations 的驗證方式。內容是 validation
method，不代表模型已適用於 production rooms。

## 驗證分層

實作使用彼此獨立的檢查：

1. impedance/reflection 代數 round trip；
2. passive causal digital boundary response；
3. one-dimensional complex modes；
4. separable three-dimensional modes；
5. 獨立 three-dimensional FDTD reference；
6. held-out source/receiver positions。

不同分層彼此一致才有意義，因為 validation 不會直接拿 scene metadata 當預期
聲學答案。

## Impedance primitives

令 (Z_0=ho c)：

```text
Gamma = (Z - Z0) / (Z + Z0)
alpha = 1 - |Gamma|^2
Z = Z0 (1 + Gamma) / (1 - Gamma)
```

Tests 涵蓋 complex round trip、phase、magnitude、matched/rigid limits，以及
passive surface 的 real impedance 不得為負。

Absorption 本身無法提供 reflection phase。因此除非有 phase-aware evidence 或
明確版本化的 prior，scene material 不會設定 impedance。

## Passive time-domain boundary

Normalized admittance model 為 positive-real，並使用 bilinear transform
離散化。結果的 poles 必須位於 digital unit circle 內。

FDTD solver 在每個 wall cell 保存獨立 boundary state。驗證內容：

- output 與 state 皆為 finite；
- strict causality；
- 宣告頻帶內的 passivity；
- zero-relaxation limit 與 real-boundary implementation 一致；
- digital reflection 符合 analog target。

Multi-pole 與 resonant boundary 為每個 branch 保存 state。Fitting equation 與
measurement contract 詳見[複數阻抗](impedance_measurements.zh-TW.md)。

## One-dimensional modes

長度 (L) 的 cavity，兩端使用相同 locally reacting boundary 時：

[
1-Gamma(s)^2 e^{-2sL/c}=0,
qquad
s=-	ext{decay rate}+j,	ext{angular frequency}.
]

[
Q=rac{	ext{angular frequency}}{2,	ext{decay rate}}.
]

Frequency-independent real boundary 會和 closed-form mode frequency 與 decay
比較。Phase-aware boundary 則必須相對於 magnitude-matched real boundary
產生合理位移。

## Three-dimensional modes

Wall 的 normalized admittance (y(s)) 滿足：

```text
normal_derivative(p) + (s/c) y(s) p = 0
```

對長度 (L) 的單一 axis，令 (h_pm=(s/c)y_pm(s))，spatial wavenumber
滿足：

[
(h_-h_+-k^2)sin(kL)+k(h_-+h_+)cos(kL)=0.
]

三個 axes 共用同一 temporal pole：

[
k_x^2+k_y^2+k_z^2+(s/c)^2=0.
]

Solver 從 rigid-room mode 做 continuation，避免跳到無關的 root。驗證包含：

- 還原 exact 1D case；
- uniform cube 中預期的 degeneracy；
- frequency 與 Q 對照獨立 FDTD output；
- source/receiver eigenfunction evaluation。

## Residue calibration

`ImpedanceModalLowFrequencyBackend` 同時需要 complex poles 與 residues。
可選的 residue calibration 使用多個 source/receiver positions 的 fixed-pole
FDTD responses；evaluation positions 不得參與 fitting。

Metadata 會區分：

- 在已驗證 geometry 與 frequency range 內使用的 calibrated residue；
- 未經 FDTD validation 的 engineering fallback；
- 需要 measured-room transfer functions 的 production validation。

Synthetic FDTD fit 良好，不等於取得 production approval。

## Evidence tiers

Evidence source 必須明確：

| Tier | 意義 |
|---|---|
| Direct complex measurement | Real 與 imaginary impedance 都是實測 |
| Measured property plus model | 實測 property 輸入具名 physical model |
| Engineering prior | 為 simulation 或 diagnosis 選定的參數 |
| Synthetic reference | 一個實作和獨立 solver 比較 |

例如，由 measured flow resistivity 建立的 porous model，其 impedance phase 仍是
model-derived。公開 grazing-duct liner data 應保留原始 geometry，不能改標成
normal-incidence wall data。

## 重現檢查

相關實作：

```text
puresound/audio/rir/physics/impedance/
puresound/audio/rir/physics/wave/
puresound/audio/rir/render/low_frequency/
```

Validators 與 calibration tools：

```text
egs/rir_generation/phases/m2_impedance/scripts/
```

各 script 的輸入請使用 `--help` 查詢。產生的 reports 應放在本機 experiment
output；只有 release 流程明確要求時才納入版控。

Production boundary model 還需要可追溯的 real-room evidence、held-out rooms
與 positions、宣告 applicability limits，以及經 review 的 material catalog
mapping。
