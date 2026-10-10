# puresound.nnet.masker

English version: [masker.md](masker.md)

`Masker` 把 backbone 的輸出（mask、filter taps 或 filter 統計量）套用到混音的
時頻表示上。它是一組 `@staticmethod` 的一般 class，不是 `nn.Module`：它包裝的
multi-frame filter（`DeepFilter`、`MultiFrameWienerFilter`、
`MultiFrameMvdrFilter`，見 [lobe/multiframe](lobe/multiframe.zh-TW.md)）沒有可學
參數，所以每次呼叫都重新建一個。

**Shape 慣例。** complex tensor 帶一個明確的實部/虛部軸，`[N, 2, C, T]`
（batch、re/im、頻率、時間）。`C` 是 encoder 的 bin 數，例如 512 點 FFT 經
`drop_stft_first_bin` 後是 256。

## 依 `mask_type` 分派

`EncDecMaskBase.forward`（`puresound/system/siso.py`）依 recipe 的 `mask_type`
選擇呼叫：

| `mask_type` | Masker 呼叫 | backbone 輸出 |
| --- | --- | --- |
| `complex` | `apply_complex_mask_with_df` | `[N, 2, C, T]` complex mask（backbone 有 `df_head` 時另有 `last_df_coefs`） |
| `deepfilter` | `apply_df_on_reim` | `[N, 2*order, F, T]` filter taps |
| `wiener` | `apply_complex_mask_on_reim`，再對低頻段 `apply_wiener` | `(mask, ifc, cov)` |
| `mvdr` | `apply_complex_mask_on_reim`，再對低頻段 `apply_mvdr` | `(mask, ifc, cov)` |
| `mapping` | 無；輸出本身就是增強後的頻譜 | `[N, 2, C, T]` |

`deepfilter` 的 `F = mask.shape[2]`、`order = mask.shape[1] / 2`：輸出 shape 就是
契約，沒有另外的 config key。streaming port（`puresound/streaming/dpcrn.py`、
`dparn.py`）逐 frame 呼叫 `apply_complex_mask_on_reim`；DPCRN port 另外從過去
frame 的 cache 自行加上 `df_head` residual。本 library 沒有任何 backbone 回傳
`(mask, ifc, cov)` 三元組，所以 `wiener` 與 `mvdr` 是為預測 filter 統計量的
backbone 預留的接線。

## Methods

### `apply_complex_mask_on_reim(tf_rep, est_masks, postfilter=False)`

```python
apply_complex_mask_on_reim(
    tf_rep: Tensor,           # X, [N, 2, C, T]
    est_masks: Tensor,        # M, [N, 2, C, T]
    postfilter: bool = False, # 先對 M 做 envelope post-filter
) -> Tensor                   # [N, 2, C, T]
```

以實數運算做 complex 乘法 `Y = M X`：
`Y_re = X_re M_re - X_im M_im`、`Y_im = X_re M_im + X_im M_re`。

### `apply_complex_mask_with_df(tf_rep, est_masks, df_coefs=None)`

```python
apply_complex_mask_with_df(
    tf_rep: Tensor,            # X, [N, 2, C, T]
    est_masks: Tensor,         # M, [N, 2, C, T]
    df_coefs: Tensor = None,   # w, [N, K, 2, F_df, T]，來自 DeepFilterResidualHead
) -> Tensor                    # [N, 2, C, T]
```

全頻段是 `M X`，最低的 `F_df` 個 bin 另加因果 deep-filter residual：

```
Y[f, t] = M[f, t] X[f, t] + sum_{k=0}^{K-1} w_k[f, t] X[f, t - k]    for f < F_df
```

residual 讀的是 noisy 頻譜（往過去補 `K - 1` 個零 frame），經由
`deep_filter_residual` 計算，它同時也是逐 frame streaming 用的運算。
`df_coefs=None` 時結果與 `apply_complex_mask_on_reim` 完全相同。見
[DPCRN](dpcrn.zh-TW.md) 的 `df_head` 選項。

### `apply_df_on_reim(tf_rep, est_masks, num_feats, order)`

```python
apply_df_on_reim(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2*order, num_feats, T]；channel 2k + {0, 1} = tap k 的 re/im
    num_feats: int,     # 從最低頻起算要 filter 的 bin 數（<= C）
    order: int,         # tap 數 K
) -> Tensor             # [N, 2, C, T]
```

以 `DeepFilter(num_freqs=num_feats, frame_size=order, lookahead=0)` 做 deep
filtering（Mack and Habets, "Deep Filtering: Signal Extraction and
Reconstruction Using Complex Time-Frequency Filters", IEEE SPL 2020）：最低的
`num_feats` 個 bin 各自換成當前與前 `order - 1` 個 frame 的 complex 線性組合。
`num_feats` 以上的 bin 原樣通過。

### `apply_wiener(tf_rep, est_ifc, est_cov, order)` / `apply_mvdr(...)`

```python
apply_wiener(
    tf_rep: Tensor,   # [N, 2, C, T]
    est_ifc: Tensor,  # [N, nbins, T, 2*order]    inter-frame correlation 向量
    est_cov: Tensor,  # [N, nbins, T, 2*order^2]  （inverse）correlation 矩陣
    order: int,       # tap 數
) -> Tuple[Tensor, int]   # (增強結果 [N, 2, C, T], nbins)
```

`nbins = est_cov.shape[1]`。`apply_wiener` 建立
`MultiFrameWienerFilter(num_freqs=nbins, frame_size=order, lookahead=0)`；
`apply_mvdr` 參數相同，建立 `enforce_constraints=True, cholesky_decomp=True` 的
`MultiFrameMvdrFilter`。兩者都只在最低的 `nbins` 個 bin 上跨 `order` 個 frame
做 filter，其餘原樣通過。呼叫端把 filter 後的低頻段接回 complex-mask 的估計：

```python
enh = Masker.apply_complex_mask_on_reim(tf_rep=X, est_masks=mask)
enh_filter, n_bins = Masker.apply_wiener(tf_rep=X, est_ifc=ifc, est_cov=cov, order=n_order)
enh[:, :, :n_bins, :] = enh_filter[:, :, :n_bins, :]
```

其中 `n_order = ifc.shape[-1] / 2`。

### `envelope_postfiltering_on_cpx_mask(tf_rep, est_masks, tau=0.02)`

```python
envelope_postfiltering_on_cpx_mask(
    tf_rep: Tensor,     # [N, 2, C, T]
    est_masks: Tensor,  # [N, 2, C, T]
    tau: float = 0.02,
) -> Tensor             # [N, 2, C, T]，post-filter 後的 mask
```

```
g  = clamp(|est_masks| / (|tf_rep| + eps), eps, 1)
pf = (1 + tau) / (1 + tau / sin^2(pi * g / 2))
return est_masks * pf
```

`g = 1` 時 `pf` 為 1，`g` 越小 `pf` 越趨近 0，所以小的 gain 會被壓得更低。它只
經由 `apply_complex_mask_on_reim(..., postfilter=True)` 執行；沒有任何 system
傳這個旗標。

### `apply_real_on_real(tf_rep, est_masks)` / `apply_mag_mask_on_reim(tf_rep, est_masks)`

```python
apply_real_on_real(tf_rep: Tensor, est_masks: Tensor) -> Tensor      # [N, C, T] * [N, C, T]
apply_mag_mask_on_reim(tf_rep: Tensor, est_masks: Tensor) -> Tensor  # [N, 2, C, T] * [N, C, T] -> [N, 2, C, T]
```

real mask 的逐元素乘積：作用在實數表示上，或廣播到 complex 表示的 re/im 兩個
平面（相位不變）。`EncDecMaskBase` 沒有任何 `mask_type` 會呼叫它們。

## 設計說明

- complex mask 在單一 frame 上同時改變振幅與相位。deep filtering 與 multi-frame
  filter 對每個 bin 結合多個 frame，能解析單一 frame 會模糊掉的諧波細節；它們只
  用在低頻段，也就是語音諧波成分所在之處。
- deep-filter residual 是加在 mask 估計上而不是取代它，所以 `df_head` 初始化為零
  的模型算出的正是只有 mask 的輸出。
