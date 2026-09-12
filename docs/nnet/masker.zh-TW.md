# puresound.nnet.masker

English version: [masker.md](masker.md)

## Class: `Masker`

一組無狀態的 `@staticmethod`，負責把 backbone 預測出來的 mask（或 filter
係數）套用到 time-frequency representation 上。`Masker` 是個單純的 class，
不是 `nn.Module` —— 它包起來用的那些 multi-frame filter（`DeepFilter`、
`MultiFrameWienerFilter`、`MultiFrameMvdrFilter`，見
[lobe/multiframe](lobe/multiframe.md)）本身沒有可學習參數，所以每次呼叫都是
現場重新建一個，而不是讓 Masker 持有 submodule 狀態。

**Shape convention:** 這裡每個 method 都是用明確的 real/imag 軸，
`[N, 2, C, T]`（batch、real-or-imag、frequency、time）——**不是**疊在
channel 軸上的 `[N, 2*F, T]` 排法。`C` 就是 encoder 實際輸出的頻率 bin 數
（例如 512-pt FFT 做完 `drop_stft_first_bin` 後是 256）。

**Wiring:** `Masker` 被 `system.siso.EncDecMaskBase.forward` 與
`system.miso.EncDecCondMaskBase.forward`（增強模型的訓練/推論迴圈）使用，
也被 streaming 的 `streaming.dpcrn.*` / `streaming.dparn.*` 使用。這四處都是
依 `mask_type` 這個字串做分派：

| `mask_type` | Masker 呼叫 | Backbone 輸出 |
|---|---|---|
| `complex` | `apply_complex_mask_on_reim` | 一個 `[N, 2, C, T]` 的 complex mask |
| `deepfilter` | `apply_df_on_reim` | 一個 `[N, 2*order, C, T]` 的 deep-filter mask |
| `wiener` | 先 `apply_complex_mask_on_reim`，**再** `apply_wiener`，兩者拼接 | `(mask, ifc, cov)` 三元組 |
| `mvdr` | 先 `apply_complex_mask_on_reim`，**再** `apply_mvdr`，兩者拼接 | `(mask, ifc, cov)` 三元組 |
| `mapping` | 不呼叫任何 method —— backbone 輸出本身就是增強後頻譜 | `[N, 2, C, T]` |

`apply_real_on_real` / `apply_mag_mask_on_reim` 則是例外：目前 `EncDecMaskBase`/
`EncDecCondMaskBase` 裡沒有任何一條 `mask_type` 分支會呼叫這兩個，streaming 的
port 也不會，所以它們現在純粹是 library 層級的元件，為的是一套「real
mask（magnitude/mapping）」的 pipeline —— 但目前沒有任何東西把它組裝起來用。

### `envelope_postfiltering_on_cpx_mask(tf_rep, est_masks, tau=0.02) -> Tensor`

```python
envelope_postfiltering_on_cpx_mask(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2, C, T]
    tau: float = 0.02,
) -> Tensor             # [N, 2, C, T] -- 是一個「mask」，不是增強後的頻譜
```

在套用之前，先把 complex mask 的 magnitude response 銳化一下（envelope
post-filtering，也就是 McAulay-Malpass 風格的 musical-noise 抑制器）。作法是算出
`mask = |est_masks| / |tf_rep|`（mask 本身隱含的增益），用 `sin(pi * mask / 2)`
做 warp，再用這個比例去重新縮放 `est_masks` —— 讓接近 1 的增益更靠近 1、
小增益更靠近 0，把「保留/抑制」這個決策磨得更銳利。這個函式是在
`apply_complex_mask_on_reim` 內部、`postfilter=True` 時才會被呼叫;目前所有呼叫點
都把這個 flag 留在預設值 `False`，所以這條路徑目前沒有任何 recipe 會真正走到。

### `apply_real_on_real(tf_rep, est_masks) -> Tensor`

```python
apply_real_on_real(tf_rep: Tensor, est_masks: Tensor) -> Tensor  # [N, C, T]
```

`tf_rep * est_masks` —— 單純的 elementwise 乘法，給實數 mask 套在實數
（已經是 magnitude/mapping domain）representation 上用。完全不涉及 complex 軸。

### `apply_mag_mask_on_reim(tf_rep, est_masks) -> Tensor`

```python
apply_mag_mask_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, C, T]  -- 實數的 magnitude mask
) -> Tensor             # [N, 2, C, T]
```

把單一個實數 mask 同時廣播到 real 與 imaginary 兩個平面
（`torch.stack([est_masks, est_masks], dim=1)` 再相乘）—— 也就是把 magnitude
mask 套在 complex representation 上，同時完全不動 phase。

### `apply_complex_mask_on_reim(tf_rep, est_masks, postfilter=False) -> Tensor`

```python
apply_complex_mask_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2, C, T]
    postfilter: bool = False,
) -> Tensor             # [N, 2, C, T]
```

Complex 乘法，拆成四個實數項來算：
`y_real = real1*real2 - imag1*imag2`、`y_imag = real1*imag2 + imag1*real2`。
這就是 DPCRN/DPARN 在 `mask_type: complex` 下套 mask 的方式（已發佈的
voice-isolate 路徑 —— 見 [algorithms/dpcrn](algorithms/dpcrn.md)），streaming
的 port 也是用這個。

### `apply_df_on_reim(tf_rep, est_masks, num_feats, order) -> Tensor`

```python
apply_df_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2*order, num_feats, T] -- deep-filter 的 taps
    num_feats: int,     # filter 涵蓋的頻率 bin 數(<= C)
    order: int,         # tap 數（context 涵蓋幾個 frame）
) -> Tensor             # [N, 2, C, T]
```

每次呼叫都現場建一個新的 `DeepFilter(num_freqs=num_feats, frame_size=order,
lookahead=0)`（見 [lobe/multiframe](lobe/multiframe.md)），把 `tf_rep`/
`est_masks` reshape 成它要的 `[N, 1, T, C, 2]` / `[N, order, T, C, 2]`
排列方式，算完再 reshape 回來。`est_masks` 的 `2*order` channel 軸打包的是
`order` 組 complex tap（real/imag 交錯）—— 也就是對每個頻率、每個 frame 做一個
跨 `order` 個相鄰時間 frame 的線性 filter，而不是單一 frame 的乘法式 mask
（DeepFilterNet 風格的 deep filtering）。呼叫端是直接從 mask tensor 自己的 shape
反推出 `num_feats`/`order`（`system/siso.py`：`n_filter = mask.shape[2]`、
`n_order = int(mask.shape[1] / 2)`），所以 backbone 輸出的 channel 數本身
*就是*合約 —— 沒有另外的 config 開關。

### `apply_wiener(tf_rep, est_ifc, est_cov, order) -> Tuple[Tensor, int]`

```python
apply_wiener(
    tf_rep: Tensor,   # [N, 2, C, T]
    est_ifc: Tensor,  # [N, nbins, T, 2*order]   -- inter-frame correlation vector
    est_cov: Tensor,  # [N, nbins, T, 2*order^2] -- (inverse) covariance matrix
    order: int,       # tap 數
) -> Tuple[Tensor, int]
# enh:   [N, 2, C, T] -- 完整頻譜，但只有前面 `nbins` 個頻率 bin 是真的經過
#        Wiener filter;其餘 bin 只是底層 MultiFrameWienerFilter 把 tf_rep
#        原封不動傳出來的未濾波拷貝。
# nbins: est_cov.shape[1] -- 涵蓋了幾個低頻 bin
```

現場建一個 `MultiFrameWienerFilter(num_freqs=nbins, frame_size=order,
lookahead=0)`。`nbins` 是直接讀 covariance tensor 自己的 shape 得來的，也就是
backbone 自己的 `ifc`/`cov` 預測 head 實際產出的頻寬 —— 通常比模型完整的
頻率數還少，因為以 covariance 為基礎的 multi-frame filtering 只有在受限的
低頻頻段才可靠。**[algorithms/](algorithms/) 裡記錄的 backbone，目前沒有一個
會回傳所需要的 `(mask, ifc, cov)` 三元組**（`DPCRN`、`DPARN`、`DPRNN`、
`SkiM`、`TFGridNet`、`ConvTasNet`、`Unet` 系列全部都只回傳單一個
`Tensor`），所以這條路徑——就跟上面的 `apply_real_on_real` /
`apply_mag_mask_on_reim` 一樣——雖然在 `EncDecMaskBase`/`EncDecCondMaskBase`
裡是可以走到的（`mask_type in ["wiener", "mvdr"]` 時的
`mask, ifc, cov = mask`），但目前這個 library 裡沒有任何 backbone 會真正
用到它;這是為某種目前這裡還沒有人實作出來的模型架構預留的管線。呼叫端
（`system/siso.py`、`system/miso.py`）一定是先算出 `complex` mask 的增強
結果，再把精修過的低頻頻段拼進去：

```python
enh = Masker.apply_complex_mask_on_reim(tf_rep=features_for_enhanced, est_masks=mask)
enh_filter, n_bins = Masker.apply_wiener(
    tf_rep=features_for_enhanced, est_ifc=ifc, est_cov=cov, order=n_order
)
enh[:, :, :n_bins, :] = enh_filter[:, :, :n_bins, :]  # 低頻頻段用 Wiener，高頻頻段用 ratio mask
```

### `apply_mvdr(tf_rep, est_ifc, est_cov, order) -> Tuple[Tensor, int]`

```python
apply_mvdr(
    tf_rep: Tensor,   # [N, 2, C, T]
    est_ifc: Tensor,  # [N, nbins, T, 2*order]
    est_cov: Tensor,  # [N, nbins, T, 2*order^2]
    order: int,
) -> Tuple[Tensor, int]  # 合約跟 apply_wiener 一樣
```

跟 `apply_wiener` 有一樣的 shape 合約與低頻拼接方式，但建的是
`MultiFrameMvdrFilter(num_freqs=nbins, frame_size=order, lookahead=0,
enforce_constraints=True, cholesky_decomp=True)` —— 是一個
distortionless-response（空間性）filter，而不是 minimum-MSE filter。
