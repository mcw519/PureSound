# puresound.nnet.loss.inherit

English version: [inherit.md](inherit.md)

錨點繼承（anchor inheritance）hinge。它防的是：模型把「最後說話的人」當成前景，
把替前一位講者決定的增益直接交給下一位近場講者。在使用者起音窗內，模型對使用者達成的
增益必須比它對「使用者出現之前那段聲音」施加的增益高出至少 `margin_db`。

## Class: `AnchorInheritanceLoss`

逐幀計算，網格為 `EnergyVADLabeler` 的網格（`frame_length` 400、`hop_length` 160，
即 16 kHz 下 100 fps——也就是 `vad_target` 所在的網格）。`E` 是增強後的幀、`R` 是
target（使用者）、`N = batch[reference_key]`（`consistency_noise = noisy − target`）：

```
a_t = <E_t, R_t> / max(<R_t, R_t>, eps_ratio)     # 對使用者達成的增益
q_t = <E_t, N_t> / max(<N_t, N_t>, eps_ratio)     # 對先前那段聲音達成的增益
pos(z) = softplus(β z) / β                         # β = softplus_beta
P_t = clamp(10 log10(pos(a_t)^2 + eps_db), floor_db, 0)
Q_t = 同上，作用在 q_t
```

每個 row 有兩個幀集合：

- `O` —— 從使用者第一個活躍幀起，`vad_target` 的前 `onset_frames` 個活躍幀。
- `B` —— 起音前、`vad_target == 0`，且 `consistency_noise` 能量落在該前綴自身峰值幀
  `pre_energy_window_db` 以內的幀。

```
Pbar_on  = Σ_O w_t P_t / Σ_O w_t                      w_t  = <R_t, R_t>
Qbar_pre = detach( Σ_B w'_t Q_t / Σ_B w'_t )          w'_t = <N_t, N_t>
L        = mean over eligible rows of relu(margin_db − (Pbar_on − Qbar_pre))
```

row eligible 的條件：`n_interferers ≥ 1`、使用者有活躍幀、起音位置至少在 row 的第
`min_onset_frame` 幀之後、起音後至少有 `onset_frames` 個活躍幀、`B` 非空，且
`background_vad_target` 顯示起音之前至少有 `min_pre_interferer_frames` 個干擾者幀。
沒有任何 eligible row 的 batch 回傳帶計算圖的零 `enhanced.sum() * 0.0`。

### Constructor

```python
AnchorInheritanceLoss(
    margin_db: float = 10.0,                   # 起音與前綴之間要求的增益差
    frame_length: int = 400,                   # EnergyVADLabeler 網格；只在測試裡改
    hop_length: int = 160,
    onset_frames: int = 50,                    # |O|；0.5 s 的語音；> 0
    min_onset_frame: int = 100,                # 起音至少在 row 開始 1 s 之後
    min_pre_interferer_frames: int = 100,      # 起音前 1 s 的干擾者語音
    fallback_pre_interferer_frames: int = 50,  # 較寬鬆的規則，只由 row_scores 回報；
                                               #   不可超過 min_pre_interferer_frames
    pre_energy_window_db: float = 25.0,        # B 的納入窗
    floor_db: float = -30.0,                   # P 與 Q 的下限；必須 < 0
    softplus_beta: float = 16.0,               # pos() 的銳利度
    eps_ratio: float = 1e-10,
    eps_db: float = 1e-12,
    reference_key: str = "consistency_noise",  # 存放 noisy - target 的 batch key
)
```

### 輸入

`required_inputs = ("enhanced", "target", "batch", "vad_target")`。
`enhanced` 與 `target` 為 `[B, T]`（`[B, C, T]` 輸入會對 channel 取平均，`[T]` 輸入
會補上 batch 軸）。`vad_target` 需要 recipe 裡有 `vad_label`；energy backend 就夠。
Batch key：`reference_key`（必要）、`n_interferers`（必要，`[B]`）與
`background_vad_target`（可選；缺席表示沒有任何 row 帶背景語音，因此沒有 row
eligible）。缺少必要輸入時丟出 `ValueError` / `KeyError`。計分以 float32 進行。

### 方法

- `row_scores(enhanced, target, batch, vad_target) -> Dict[str, Tensor]` —— 同樣的
  算式但完全不做 reduce，每個值都是 `[B]`：`hinge`、`pbar_on`、`qbar_pre`、
  `contrast`、`eligible`、`eligible_fallback`、`onset_frame`、
  `pre_interferer_frames`、`n_onset_frames`、`n_pre_frames`、
  `n_active_after_onset`、`n_interferers`、`n_frames`。只有 `eligible`（或
  `eligible_fallback`）為真的位置才有意義；彙總前先 mask。preflight 檢查與驗證期監看
  讀的就是它，所以被監看的數字與被訓練的數字出自同一份實作。
- `floored_gain_db(ratio)` —— 有下限的包絡
  `clamp(10 log10(pos(ratio)^2 + eps_db), floor_db, 0)`，公開出來讓測試能釘住它的值。

### Config 用法

沒有任何已出貨的 recipe 啟用這個 loss。

```yaml
vad_label:
  used: True
  backend: energy
  args: {frame_length: 400, hop_length: 160}
loss_func:
  - type: AnchorInheritanceLoss
    weighted: 0.25
    args: {margin_db: 10.0}
```

## 設計說明

- **只出現差值。** 沒有任何絕對 dB 門檻：絕對校準無法跨收音鏈或跨 checkpoint 成立，
  相對的形式可以。
- **前綴項 detach。** hinge 不能靠少壓一點前面的干擾者來滿足——那等於拿遠場壓制去換
  起音保留。
- **起音窗，而非整 row 平均。** 1 s 的事件幾乎動不了整 row 的平均，整 row 平均的形式
  幾乎沒有梯度。
- **下限限制了統計量。** 沒有下限時，pooling 後的 hinge 會被少數輸出與參考反相關的起音
  主導。`softplus` 本身並不會消除負比值的死區；是 clamp 讓 `a ≤ −0.026`（預設值下）
  以下的梯度歸零。逐 row 的值用中位數彙總。
- **`B` 需要能量。** 只有噪音或靜音的前綴會讓 hinge 輕易滿足，因此有
  `pre_energy_window_db` 的納入規則。
- **用哪個參考。** `consistency_noise` 由變速擾動之後的最終波形建立，而起音前的幀中
  使用者依構造是靜音的，所以它就是干擾者加噪音。變速前的 `background_vad_target` 只用
  在 row 層級的 1 s 計數，幾個百分比的時間誤差翻不動它。
- **沒有參數。** 這個 loss 既不改變 checkpoint 佈局，也不影響 streaming export。
