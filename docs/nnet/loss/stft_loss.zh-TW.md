# puresound.nnet.loss.stft_loss

English version: [stft_loss.md](stft_loss.md)

語音增強裡,用來衡量頻譜保真度的 STFT-domain loss。只有
`MultiResolutionSTFTLoss`、`SpectralLoss`、`OverSuppressionLoss` 有被 import
進 `puresound/nnet/loss/__init__.py`,能透過 recipe 的 `loss_func[].type`
取用。`STFTLoss`、`SpectralConvergengeLoss`、`LogSTFTMagnitudeLoss`,以及
module-level 的 `stft` / `as_complex` / `angle` 這幾個 helper,都只是內部的
building block——只能透過 `puresound.nnet.loss.stft_loss.X` 這樣取用,不會
出現在 package 的頂層。

## Helper:`stft(x, fft_size, hop_size, win_length, window) -> Tensor`

`STFTLoss` 與 `OverSuppressionLoss` 用的內部 helper:執行
`torch.stft(..., return_complex=True)`,回傳**幅度（magnitude)**頻譜,轉置成
`[B, #frames, fft_size // 2 + 1]`。`window` 是一個已經算好的 window tensor
（例如 `torch.hann_window(win_length)`),每次呼叫都會被搬到 `x` 所在的
device 上。

## Helper:`as_complex(x: Tensor) -> Tensor`

吃的是**單一**個 tensor（不是 `(re, im)` 這種一對兩個的形式):如果 `x`
本身已經是 complex,原樣回傳;否則要求 `x.shape[-1] == 2`（實部虛部疊在
最後一軸——這個 codebase 裡「用實數 tensor 表示 complex」的慣例),再用
`torch.view_as_complex` 轉換。`SpectralLoss` 用它來同時接受這兩種表示法。

## Helper:`angle`（`torch.autograd.Function`)

帶有數值穩定梯度的相位角,呼叫方式是 **`angle.apply(x)`**——不是
`angle(x)`（`angle` 是一個 `torch.autograd.Function` 子類別;它的 static
`forward`/`backward` 一定要透過 `.apply` 呼叫)。Forward 就是單純的
`atan2(x.imag, x.real)`;backward 會把連鎖律裡的 `1 / |x|^2` 這一項 clamp
到 `1e-10`,避免在幅度接近零的 bin 上梯度爆炸——這正是直接對
`torch.angle` 微分會遇到的問題。`SpectralLoss`（見下)在需要用壓縮過的
幅度加上原始相位重建一個 gamma-compressed complex tensor 時,會用到它。

## Class: `SpectralConvergengeLoss`

兩個幅度頻譜之間、以 Frobenius norm 衡量的頻譜散度。

$$\mathcal{L}_{sc} = \frac{\||\hat{S}| - |S|\|_F}{\||S|\|_F}$$

`forward(x_mag, y_mag) -> Tensor`——`x_mag` = 預測值,`y_mag` = 參考值。

## Class: `LogSTFTMagnitudeLoss`

對 log 壓縮過的幅度頻譜做的 L1 loss。

$$\mathcal{L}_{log} = \| \log|\hat{S}| - \log|S| \|_1$$

`forward(x_mag, y_mag) -> Tensor`。

## Class: `STFTLoss`

單一解析度的 building block,在一組 `(fft_size, hop, win_length)` 設定下,
把上面兩個 loss 合在一起。

```python
STFTLoss(
    fft_size=1024,
    shift_size=120,
    win_length=600,
    window="hann_window",   # 任何 torch.*_window 工廠函式的名稱，透過 getattr(torch, window) 解析
)
```

`forward(x, y) -> Tuple[Tensor, Tensor]`——**回傳一組 2-tuple**
`(sc_loss, mag_loss)`,不是單一個 scalar;要怎麼加權、加總這兩個值,是
呼叫端（也就是下面的 `MultiResolutionSTFTLoss`)的責任。它沒有被 import
進 package 的 `__init__.py`,所以 `type: STFTLoss`不是一個合法的
`loss_func[].type`——它只是 `MultiResolutionSTFTLoss` 每個解析度會各建一個
的元件而已。

## Class: `MultiResolutionSTFTLoss`

把 `STFTLoss` 在好幾組解析度上的結果加總起來（粗解析度、寬頻的 bin 抓的是
寬頻失真;細解析度、窄 bin 抓的是諧波/音高結構)。

### Constructor

```python
MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 512],
    hop_sizes=[120, 240, 50],
    win_lengths=[600, 1200, 240],
    window="hann_window",
    factor_sc=0.1,
    factor_mag=0.1,
)
```

依 `zip(fft_sizes, hop_sizes, win_lengths)` 每一組三元組,建一個
`STFTLoss(fs, ss, wl, window)`（這三個 list 長度必須一致)。`factor_sc` /
`factor_mag` 是最終加總時,套用在（跨解析度取平均後的)spectral
convergence 與 log-magnitude 這兩項上的權重。

`uses_inactive_labels = True`——設定的理由跟 `SDRLoss` 一樣（見
[loss/sdr](sdr.md)),但處理方式不同:頻譜類 loss 在 reference 全零時是
無定義的（spectral convergence 除以 `||ref||`,log-magnitude 會撞到 eps
clamp),所以單一一個靜音 target 的 row,產生的 loss 可能比 active row
高上好幾個數量級,梯度會把整個 batch 淹沒。`MultiResolutionSTFTLoss` 不像
`SDRLoss` 那樣用替代公式給靜音 row 計分,而是直接把它們**丟掉**。

### `forward(x, y, inactive_labels=None) -> Tensor`

```python
if inactive_labels is None:
    active = y.abs().amax(dim=-1) > 0        # 自給自足的備援機制
else:
    active = ~inactive_labels.to(torch.bool).reshape(-1)
if not bool(active.any()):
    return x.new_zeros(())                   # 見下方說明
x, y = x[active], y[active]
```

如果沒有給 `inactive_labels`,這個 loss 會自己從 `y` 推出 active-row 的
mask,所以就算不靠訓練系統把 label 路由進來,也能單獨使用。如果整個
batch 裡每一個 row 都是 inactive,它會回傳一個單純的 `x.new_zeros(())`——
跟 `DistHeadRegressionLoss` 的 `dist_preds.sum() * 0.0`（見
[loss/dist](dist.md))不同,這個零值**沒有**接回 `x` 在 autograd 計算圖
上的邊。實務上,這只有在 `MultiResolutionSTFTLoss` 是某個 recipe*唯一*的
loss 項時才有影響;同一個 step 裡,對同一份 `enhanced` tensor 算的其他
loss 項,仍然會讓 backbone 保持在計算圖裡。

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml（現役 recipe)
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

## Class: `OverSuppressionLoss`

[loss/index](index.md) 裡提到的那個單邊 anti-deletion 壓力:懲罰 enhanced
的幅度掉到 reference **以下**,但當它等於或高於 reference 時完全**不**
懲罰（over-estimation/洩漏另有其他 loss 項負責——見下方 Config usage)。
原始碼裡,這個 class 正上方留著一段註解掉的自由函式雛型
（`over_suppression_loss`),是同一個構想較早的版本,當作歷史紀錄保留,
不是會被執行的程式碼。

### Constructor

```python
OverSuppressionLoss(
    p: float = 0.5,          # 冪次壓縮的指數
    fft_size: int = 512,
    hop_size: int = 128,
    win_length: int = 512,   # 固定用 Hann window；跟 STFTLoss 不同，這裡不可配置
)
```

### `forward(enh, ref) -> Tensor`

```python
enh_mag = stft(enh, fft_size, hop_size, win_length, hann_window)
ref_mag = stft(ref, fft_size, hop_size, win_length, hann_window)
loss = ref_mag.pow(p) - enh_mag.pow(p)
loss = loss.clamp_min(0) ** 2      # (先取 mask > 0，再平方) -- enh_mag >= ref_mag 的地方全部歸零
return loss.mean()
```

只要（經過冪次 `p` 壓縮的)reference 幅度超過 enhanced 幅度——也就是
deletion——這個差距的平方就會被懲罰;只要 enhanced 幅度相等或更高,這一項
就會被蓋成零。預設值（也是現役 recipe 用的值)`p=0.5` 是一種平方根壓縮,
會相對地把最大聲的 bin 壓低、比較不明顯,凸顯較安靜的 bin。

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml
# ANTI-SUPPRESSION: pure-magnitude one-sided loss = mean(ReLU(|T|^p - |E|^p)^2).
# Penalises enhanced magnitude falling BELOW target (deletion); over-estimation is
# already covered by SD-SDR + STFT above, so this adds net asymmetric pressure toward
# preservation. `weighted` trades deletions against substitutions.
- type: OverSuppressionLoss
  weighted: 3.0
  args: {p: 0.5, fft_size: 512, hop_size: 128, win_length: 512}
```

## Class: `SpectralLoss`

跟這一頁其他 class 都不一樣,`SpectralLoss`**不**接受任何 STFT 設定
（`n_fft` / `hop_length` / `win_length`)——它處理的是呼叫端**已經算好**的
頻譜 tensor,而不是 waveform。

### Constructor

```python
SpectralLoss(
    gamma: float = 1,             # 幅度的冪次壓縮指數
    factor_magnitude: float = 1,  # magnitude 項的權重
    factor_complex: float = 1,    # 實部+虛部（對相位敏感)項的權重
    factor_under: float = 1,      # enhanced 低於 reference 時的額外權重
)
```

### `forward(enh, ref) -> Tensor`

```
enh, ref: Tensor | ComplexTensor, 形狀 [N, *, C, T, 2] 或 [N, *, C, T]
```

1. 透過 `as_complex` 把兩者都轉成 complex（接受原生 complex tensor,或是
   最後一軸為 2 的實部/虛部 tensor)。
2. 取幅度;若 `gamma != 1`,兩者都做
   `.clamp_min(1e-12).pow(gamma)` 壓縮。
3. Magnitude 項:`(enh_abs - ref_abs)` 的均方,乘上 `factor_magnitude`。
   若 `factor_under != 1`,enhanced 幅度低於 reference 的 bin
   （`enh_abs < ref_abs`——跟 `OverSuppressionLoss` 針對的是同一種
   deletion 狀況)會再乘上一個額外的 `factor_under`——把單邊加重的想法,
   折進一個合併的 loss 裡,而不是另外拉出一個獨立的純懲罰項。
4. 若 `factor_complex > 0`:用 `angle.apply`（不是 `torch.angle`,是為了
   梯度穩定)重建 gamma-compressed 的 complex tensor,再加上
   `factor_complex * MSE(view_as_real(enh), view_as_real(ref))`——同時比對
   實部與虛部,隱含地也限制了相位,不只是幅度。當 `factor_complex <= 0`
   時,這一步會整段跳過。

目前這個 repo 裡沒有任何 recipe 在用它。
