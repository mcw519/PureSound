# puresound.nnet.loss.stft_loss

English version: [stft_loss.md](stft_loss.md)

STFT 域的 loss。匯出給 recipe 使用的有三個 class：
`MultiResolutionSTFTLoss`（在多個 STFT 解析度上的頻譜保真）、
`OverSuppressionLoss`（輸出幅度低於 target 時的單邊懲罰）與
`SpectralLoss`（在預先算好的頻譜上計算幅度與複數誤差）。
`STFTLoss`、`SpectralConvergengeLoss`、`LogSTFTMagnitudeLoss` 以及 helper
`stft`、`as_complex`、`angle` 是內部元件，只能以
`puresound.nnet.loss.stft_loss.<name>` 取用。

## Helper：`stft`

```python
stft(x, fft_size, hop_size, win_length, window, power_floor=1e-8, relative_floor=False)
# x: [B, T] -> magnitude [B, frames, fft_size // 2 + 1]
```

執行 `torch.stft(..., return_complex=True)`，回傳
`sqrt(clamp(|X|^2, min=floor))`。`window` 是 window tensor，每次呼叫時搬到
`x` 所在的 device。

- `power_floor` 是對 `|X|^2` 的 clamp。預設 `1e-8` 即幅度 `1e-4`，約為相對
  full-scale bin 的 −80 dB。它是梯度的 floor：低於它的 bin 對兩個 STFT 項都
  不可見。
- `relative_floor=True` 讓 `power_floor` 變成相對於每個 row 自己最大 bin 功率
  的比例（每個 row `floor = power_floor * max |X|^2`），floor 因此跟著 row
  的音量走，而不是固定在一個絕對值。

## Class：`STFTLoss`

單一解析度：在 `stft` 算出的幅度上計算 spectral convergence 加上 log-magnitude
L1。

$$\mathcal{L}_{sc} = \frac{\||S| - |\hat{S}|\|_F}{\||S|\|_F}, \qquad \mathcal{L}_{mag} = \operatorname{mean}\left|\ln|S| - \ln|\hat{S}|\right|$$

$\hat S$ 是預測，$S$ 是 reference。Frobenius norm 是對整個 batch tensor
計算，不是逐 row。

```python
STFTLoss(fft_size=1024, shift_size=120, win_length=600, window="hann_window",
         power_floor=1e-8, relative_floor=False)
# window: torch.*_window 的 factory 名稱，以 getattr(torch, window) 解析
```

`forward(x, y) -> (sc_loss, mag_loss)` 回傳一對值，由呼叫端加權相加。
`SpectralConvergengeLoss` 與 `LogSTFTMagnitudeLoss` 是這兩項各自獨立的
module，皆為 `forward(x_mag, y_mag) -> Tensor`。

## Class：`MultiResolutionSTFTLoss`

每個 `(fft_size, hop_size, win_length)` 三元組建一個 `STFTLoss`，兩項各自
對解析度取平均：

$$\mathcal{L} = f_{sc} \cdot \frac{1}{R}\sum_r \mathcal{L}_{sc}^{(r)} + f_{mag} \cdot \frac{1}{R}\sum_r \mathcal{L}_{mag}^{(r)}$$

### Constructor

```python
MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 512],
    hop_sizes=[120, 240, 50],
    win_lengths=[600, 1200, 240],   # 三個 list 長度必須相同
    window="hann_window",
    factor_sc=0.1,                  # f_sc
    factor_mag=0.1,                 # f_mag
    power_floor=1e-8,               # 傳給每個解析度，見 `stft`
    relative_floor=False,
)
```

### 輸入

`forward(x, y, inactive_labels=None) -> Tensor`，waveform `[B, T]`。
`required_inputs = ("enhanced", "target", "inactive_labels")`。

被標為 inactive（reference 為靜音）的 row 在做 STFT 之前就被丟掉。沒有
`inactive_labels` 時 loss 自己推導 mask（`y.abs().amax(-1) > 0`），所以直接
呼叫也能用。若所有 row 都是 inactive，回傳 `x.new_zeros(())`，這個零與 `x`
之間沒有 graph edge。

### Config usage

```yaml
loss_func:
  - type: MultiResolutionSTFTLoss
    weighted: 0.5
    args:
      fft_sizes: [512, 1024, 256]
      hop_sizes: [128, 256, 64]
      win_lengths: [512, 1024, 256]
      window: hann_window
      factor_sc: 0.5
      factor_mag: 0.5
```

### 設計說明

- 用多個解析度，是因為單一 STFT 無法同時看清時間與頻率細節：長 window
  解析諧波，短 window 解析起音與暫態。
- 兩項對誤差的加權不同：spectral convergence 作用在線性幅度上，由大聲的
  bin 主導；log-magnitude L1 則在整個動態範圍內對相對誤差一視同仁。
- reference 為靜音的 row 是被丟掉而不是被評分：spectral convergence 要除以
  `||ref||`，而對著落在 floor 的 reference 算 log-magnitude，loss 會遠高於
  active row 的水準，其梯度會淹沒整個 batch。這些 row 由
  [`SDRLoss`](sdr.zh-TW.md) 的 target-absent 項與 VAD loss 監督。
- 這一項相對於 SDR 項的量級由 `factor_sc`、`factor_mag` 與該項的 `weighted`
  決定；factor 太小時它在 SDR 旁邊幾乎是常數，貢獻的梯度很少。

## Class：`OverSuppressionLoss`

單邊的幅度懲罰：只有輸出低於 reference 的 bin 才計入。

$$\mathcal{L} = \operatorname{mean}\left(\max\left(0,\ |S|^{p} - |\hat{S}|^{p}\right)^2\right)$$

### Constructor

```python
OverSuppressionLoss(
    p: float = 0.5,          # power-law 壓縮指數
    fft_size: int = 512,
    hop_size: int = 128,
    win_length: int = 512,   # 此長度的 Hann window；window 種類固定
)
```

`forward(enh, ref) -> Tensor`，waveform `[B, T]`。它沒有宣告
`required_inputs`，因此拿到的是 `("enhanced", "target")`。使用預設絕對 floor
的 `stft`。

### Config usage

```yaml
loss_func:
  - type: OverSuppressionLoss
    weighted: 1.
    args: {p: 0.5, fft_size: 512, hop_size: 128, win_length: 512}
```

### 設計說明

對稱的 loss（SDR、MR-STFT）對高估與低估一視同仁。對下游 ASR 而言，刪掉
target 語音是代價較高的錯誤，所以這一項只往一個方向施壓：保住 target 的能量。
高估仍由對稱項負責。`p = 0.5` 壓縮幅度，讓最大聲的 bin 不會主導懲罰。
該項的 `weighted` 決定刪字與殘留噪音之間的取捨。

## Class：`SpectralLoss`

在呼叫端已算好的頻譜上計算幅度與複數頻譜的 MSE。不接受任何 STFT 設定。

### Constructor

```python
SpectralLoss(
    gamma: float = 1,             # 幅度的 power-law 壓縮
    factor_magnitude: float = 1,  # 幅度項權重
    factor_complex: float = 1,    # 複數項權重；<= 0 時關閉
    factor_under: float = 1,      # |enh| < |ref| 的 bin 額外加權
)
```

### `forward(enh, ref) -> Tensor`

`enh`、`ref`：complex tensor `[N, *, C, T]`，或最後一軸為實部/虛部的 real
tensor `[N, *, C, T, 2]`（由 `as_complex` 轉換）。

```
A_e, A_r = |enh|, |ref|                        # ** gamma if gamma != 1 (clamped at 1e-12)
mag      = mean((A_e - A_r)^2 * w) * factor_magnitude,   w = factor_under where A_e < A_r, else 1
cplx     = MSE(view_as_real(A_e * e^{j angle(enh)}), view_as_real(A_r * e^{j angle(ref)})) * factor_complex
loss     = mag + cplx                          # cplx only when factor_complex > 0
```

`gamma == 1` 時複數項直接比較 `enh` 與 `ref`。相位用 `angle.apply` 取得，它是
一個 `torch.autograd.Function`，backward 時把 `1 / |x|^2` clamp 在 `1e-10`，
避免幅度接近零的 bin 產生爆炸的相位梯度。

訓練 module 交給 loss 的是 waveform，所以 `SpectralLoss` 是給自己計算頻譜的
程式碼用的；若設成 `loss_func` 項目，它會收到 waveform 並在 `as_complex` 失敗。

### 設計說明

power-law 壓縮（`gamma < 1`）平衡大聲與小聲的 bin；複數項除了幅度也約束相位；
`factor_under` 把對低估的單邊強調併進同一個 loss。
