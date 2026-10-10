# 材質推導之低頻模態阻尼

English version: [modal_damping.md](modal_damping.md)

`puresound.audio.rir.render.low_frequency.modal_damping` 讓矩形房間的每個模態
都有自己的衰減率，由六面牆的頻率相依吸音率計算，而不是用一個全房間 RT60。
啟用材質阻尼時，pytARD 與 analytic 低頻 backend 會使用它。這條損耗律沒有對
實測房間驗證過，因此它是實驗用的基礎設施，不是 production 預設。

## 選用方式

`egs/rir_generation/generate_hybrid_rir.py --low-backend` 接受
`analytic-material`、`pytard-material` 與 `pytard-cupy-material`。它們會在
`AnalyticModalLowFrequencyBackend`、`GpuARDPytARDBackend` 或
`GpuARDPytARDCuPyBackend` 上設定 `material_modal_damping=True`，並需要
`--scene-version v1`（帶表面材料的 `RoomSceneV2`）。一般的 `analytic`、
`pytard` 與 `pytard-cupy` backend 維持共用的 RT60 衰減包絡，兩條路徑可以各自
獨立比較。

## 邊界能量參與

$L_x \times L_y \times L_z$ 房間的剛性牆壓力特徵函數為

$$\phi_n(x,y,z) = \cos\frac{n_x\pi x}{L_x}\cos\frac{n_y\pi y}{L_y}\cos\frac{n_z\pi z}{L_z}$$

沿某一軸的體積 norm，index 為零時 $I = L$，否則 $I = L/2$。
`material_modal_decay_rates(scene, nx, ny, nz, omega_rad_s, sound_speed,
loss_scale=1.0)` 在每個模態頻率內插六個有效邊界吸音頻譜（log 頻率內插，超出
端點頻帶時保持常數），並計算

$$P_n = \frac{\alpha_\text{west} + \alpha_\text{east}}{I_x} + \frac{\alpha_\text{south} + \alpha_\text{north}}{I_y} + \frac{\alpha_\text{floor} + \alpha_\text{ceiling}}{I_z}$$

$$\gamma_n = \frac{s_\text{loss}\,c\,P_n}{8},\qquad \zeta_n = \frac{\gamma_n}{\omega_n},\qquad \mathrm{RT60}_n = \frac{\ln 1000}{\gamma_n},\qquad Q_n = \frac{\omega_n}{2\gamma_n}$$

$cP_n/4$ 是一階 Sabine 能量損耗率；模態振幅以其一半、即 $\gamma_n$（1/s）衰減。
某軸 index 非零的模態，對該軸牆面對的參與度是兩倍，因此改變單一表面對不同
模態的影響不同，這是單一 RT60 無法表達的。$\gamma_n$ 上限為 $0.95\,\omega_n$，
確保每個模態都是欠阻尼；接近 DC 的模態不加阻尼。

$s_\text{loss}$（`--material-modal-loss-scale`，預設 1.0）是未校正的物理 prior。
它只供受控診斷使用，會隨每個結果序列化；未選 `*-material` backend 時，CLI
會拒絕非預設值。它不是公認的物理常數。

## Analytic probe：聲源／接收端耦合

`AnalyticModalLowFrequencyBackend`（預設 `num_modes_per_axis=5`、
`max_modes=256`、`physical_mode_coupling=True`）以實際端點上的特徵函數值激發並
觀測每個模態：

$$A_n \propto (2-\delta_{n_x0})(2-\delta_{n_y0})(2-\delta_{n_z0})\;\phi_n(\text{source})\,\phi_n(\text{receiver})\;\frac{f_1}{f_n}$$

括號內的係數是 cosine 模態體積 norm 的倒數。位於模態節點上的聲源或接收端會
抑制該模態；聲源與接收端互換時響應不變。每個模態都在幾何直達聲抵達時刻以零
相位開始，因此有限展開不會產生抵達前的尾巴；同一時刻另加一個
$1/\max(d, 0.1)$ 的直達脈衝。

Probe 會列舉 0 到 `num_modes_per_axis` 之間、剛性頻率落在
`[low_fmin_hz, low_fmax_hz]` 內的所有 index 三元組。256 個模態的上限是安全
上限，高於 0–5 index 範圍內 215 個非 DC 三元組，因此頻帶內不會有模態被上限
截掉。

## pytARD 中的精確阻尼遞迴

pytARD 求解會把房間對角化成彼此獨立的模態。啟用材質阻尼時，`solve_modal_ard`
以精確取樣極點積分每個模態

$$\ddot q_n + 2\gamma_n\dot q_n + \omega_n^2 q_n = f_n$$

時間步長為 $\Delta t$ 時：

$$r = e^{-\gamma_n\Delta t},\quad \omega_d = \sqrt{\omega_n^2 - \gamma_n^2},\quad a_1 = 2r\cos(\omega_d\Delta t),\quad a_2 = -r^2,\quad b = \frac{1 - a_1 - a_2}{\omega_n^2}$$

$$q[k+1] = a_1 q[k] + a_2 q[k-1] + b f[k]$$

$\gamma_n = 0$ 時它就是原本無阻尼的 pytARD 遞迴。材質路徑不套用
`apply_rt60_decay_envelope`，metadata 記錄 `global_rt60_envelope_applied: false`。

## Metadata

`material_modal_damping_metadata(scene, config, max_mode_index=16,
max_modes=128, loss_scale=1.0)` 以 model `surface_participation_sabine` 記錄：
每個模態的 index、頻率、振幅衰減率、預測 RT60 與 Q；各模態 RT60 與 Q 的
min/median/max；loss scale；以及沒有套用全域包絡。

## 測試

- `test/rir/test_rir_scene_v2.py`：阻尼依表面與模態而異（改變 west 牆對
  x 軸向模態的影響，是 x index 為零之模態的兩倍）；loss scale 使衰減率與 Q
  呈倒數變化；材質 backend 拿掉全域 RT60 包絡；精確 pytARD 遞迴在吸音邊界下
  衰減較快。
- `test/rir/test_hybrid_rir.py`：analytic probe 遵守模態節點、因果性與
  聲源／接收端互易性。
- `test/rir/test_fdtd_reference.py`：獨立 FDTD reference 還原前幾個軸向
  模態的頻率與邊界 Q，峰值／Q 估計器能還原阻尼正弦
  （見 [modal validation](modal_validation.zh-TW.md)）。

## 限制

- 邊界損耗是一階吸音／參與度模型：沒有反射相位、沒有角度相依，也沒有非
  shoebox 幾何造成的模態耦合。
- 複數阻抗相位由另外的 impedance modal backend 與 FDTD reference 處理
  （見[複數聲學阻抗](impedance_measurements.zh-TW.md)），不是這個遞迴。
- 單一 loss scale 無法修正阻尼分佈的形狀；要讓此 backend 成為預設，需要已安裝
  材料的阻抗證據與下游測試。
