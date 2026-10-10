# puresound.nnet.lobe.multiframe

English version: [multiframe.md](multiframe.md)

Multi-frame complex filter。單一 frame 的 mask 每個 bin、每個 frame 只給一個增益；
每個 bin 一組 K-tap filter 則會組合這個 bin 最近的幾個 frame，能解析 10 ms hop
會糊掉的諧波結構（Schröter et al., "DeepFilterNet: A Low Complexity Speech
Enhancement Framework for Full-Band Audio based on Deep Filtering", ICASSP 2022）。

這個 module 分成兩組：

- **Deep-filter residual**——`deep_filter_residual` 與
  `DeepFilterResidualHead`，給 `DPCRN(df_head=...)` 用。實數運算、causal、可串流。
- **DeepFilterNet 風格的運算子**——`DeepFilter`、`MultiFrameWienerFilter`、
  `MultiFrameMvdrFilter` 與 `DeepFilterDecoder`，用複數運算，經由
  `Masker.apply_df_on_reim` / `apply_wiener` / `apply_mvdr` 呼叫
  （`EncDecMaskBase` 的 `mask_type: deepfilter | wiener | mvdr`）。

**兩組的 shape 慣例不同。** residual 路徑用 `[N, 2, F, T]`（實/虛部、頻率、時間），
與 [`Masker`](../masker.zh-TW.md) 的 `[N, 2, C, T]` 同順序。DeepFilterNet 風格的運算子用
`[B, C, T, F, 2]`（時間在頻率之前、實/虛部在最後）；`Masker` 在每個呼叫點都會
permute 成這個順序。

## Function: `deep_filter_residual`

每個 bin 的 causal multi-frame complex filter：

```
out[t] = sum_{k=0}^{K-1} coefs[k, t] * x[t - k]        (complex multiply)
```

```python
deep_filter_residual(history: Tensor, coefs: Tensor) -> Tensor
```

- `history`——`[N, 2, F, T + K - 1]` 的實/虛部頻譜，最舊的 frame 在前。離線路徑在
  左邊補 `K - 1` 個零 frame；串流路徑傳入它快取的 `K - 1` 個 frame，後面接目前這個。
- `coefs`——`[N, K, 2, F, T]`；第 `k` 個 tap 乘上 frame `t - k`。
- 回傳 `[N, 2, F, T]`。`history` 不是 `T + K - 1` 個 frame 時丟 `ValueError`。

它用實/虛部成對的實數寫成，而不是 complex tensor，所以同一個函式既是離線運算也是
逐 frame 的串流運算，並且原封不動匯出成 ONNX。

## Class: `DeepFilterResidualHead`

從 decoder 的 feature map 預測最低 `bins` 個細 bin 的 K-tap 係數。filter 的輸出是
**加到** complex mask 過的頻譜上：低頻帶 `Y = M * X + DF(X)`，其中 `DF` 濾的是
noisy 頻譜 `X`。

```python
DeepFilterResidualHead(
    in_channels: int,     # 接出來的 decoder feature map 的 channel 數
    upsample: int,        # 每個粗 bin 對應的細 bin 數
    bins: int = 128,      # 從底部起濾的細 bin 數；必須是 upsample 的倍數
    order: int = 5,       # tap 數 K：目前 frame 加 K - 1 個過去 frame
    hidden: int = 32,     # pointwise projection 的寬度
)
```

結構：在前 `bins // upsample` 個粗 bin 上做
`Conv2d 1x1 -> PReLU -> Conv2d 1x1（零初始化）-> tanh`。最後一層 conv 輸出
`upsample * 2 * order` 個 channel，reshape 成細 bin `f = coarse * upsample + u`。

`forward(x [N, C, F_coarse, T]) -> [N, K, 2, bins, T]`，由 `tanh` 限制在 `(-1, 1)`。
`bins % upsample != 0`、`order < 1`、或 feature map 少於 `bins // upsample` 個 bin 時
丟 `ValueError`。

### 在 DPCRN 裡的用法

```yaml
backbone_args:
  df_head: {bins: 128, order: 5, hidden: 32}
```

`DPCRN` 以 `in_channels = channels[1]`、`upsample = stride_f[0]` 建這個 head，在
decoder 倒數第二層的 feature map 上執行，結果存成 `backbone.last_df_coefs`。
`mask_type: complex` 的 `EncDecMaskBase` 會把它交給
`Masker.apply_complex_mask_with_df`；沒有這個 head 的 backbone 讓它維持 `None`，
結果就是單純的 complex mask。串流 runner（`puresound/streaming/dpcrn.py`）另外保存
低頻帶最近 `K - 1` 個 noisy frame 當作狀態。

### 設計說明

- **是殘差，不是取代。** 最後一層 projection 從零開始，所以初始化時模型算的與只有
  mask 的網路完全相同，那個網路的 checkpoint 可以直接 warm-start 這個模型，不必重新
  初始化任何東西。
- **Head 沒有狀態。** 只有 pointwise convolution、沒有時間 kernel，所以唯一新增的
  串流狀態是 filter 要讀的 `K - 1` 個 noisy 頻譜 frame。
- **讀的是 noisy 頻譜**，與 DeepFilterNet 的 filter 一樣，而不是 mask 過的頻譜。

測試：`test/nnet/test_multiframe.py` 固定住 filter 本身、因果性、從零開始，以及
只有 mask 的 checkpoint 可直接 warm-start。

---

## Class: `MultiFrameModule`

DeepFilterNet 風格運算子的基礎 class：在時間軸上 padding 並展開成 `frame_size` 個
frame 的 window。

```python
MultiFrameModule(num_freqs: int, frame_size: int, lookahead: int, real: bool = False)
```

- `num_freqs`——filter 套用的 bin 數，從底部算起；更高的 bin 原樣通過。
- `frame_size`——window 長度 N（frame 數）。
- `lookahead`——N 個 frame 中有幾個是未來 frame（`frame_size - 1 - lookahead` 個是
  過去 frame）；`0` 為 causal。沒有預設值。
- `real`——改用 `spec_unfold_real`（實數輸入、多一個尾軸），而不是 complex 的
  `spec_unfold`。下面的 filter 都走 complex 路徑。

方法：

- `spec_unfold(spec)`——complex `[B, C, T, F]` → `[B, C, T, F, N]`，前面補
  `frame_size - 1 - lookahead` 個 frame、後面補 `lookahead` 個。
- `solve(Rxx, rss, diag_eps=1e-8, eps=1e-7)`（static）——帶 Tikhonov 正規化的
  `Rxx⁻¹ rss`。
- `apply_coefs(spec, coefs)`（static）——`einsum("...n,...n->...")`：每個 window
  套一個係數向量。

Module 層級的 helper：

- `psd(x, n)`——`n` 個 frame window 上的相關矩陣 `X Xᴴ`：
  `[B, C, T, F]` → `[B, C, T, F, n, n]`。
- `df(spec, coefs)`——deep-filter 的乘加 `einsum("...tfn,...ntf->...tf")`，
  `DeepFilter` 使用。
- `_tik_reg(mat, reg=1e-7, eps=1e-8)`——`mat + (trace(mat).real * reg + eps) * I`。

## Class: `DeepFilter`

直接套用學出來的複數係數（DeepFilterNet 的運算子）：
`Y[t, f] = sum_n c_n[t, f] * X[t - N + 1 + lookahead + n, f]`。

```python
DeepFilter(num_freqs: int, frame_size: int, lookahead: int,
           real: bool = False,
           conj: bool = False)   # 套用前先取係數共軛
```

`forward(spec [B, C, T, F, 2], coefs [B, C*N, T, num_freqs, 2]) -> [B, C, T, F, 2]`。
只取代前 `num_freqs` 個 bin。

## Class: `MultiFrameWienerFilter`

由 noisy 相關矩陣 `Rxx` 與語音 inter-frame correlation（IFC）向量 `rss` 算出的
multi-frame Wiener filter（Huang and Benesty, "A Multi-Frame Approach to the
Frequency-Domain Single-Channel Noise Reduction Problem", IEEE TASLP 2012）：

```
w = Rxx⁻¹ rss,        Y[t, f] = sum_n w_n[t, f] * X_window[t, f, n]
```

```python
MultiFrameWienerFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    cholesky_decomp: bool = False,   # covariance 輸入是 Cholesky 因子 L
    inverse: bool = True,            # covariance 輸入已經是反矩陣
    enforce_constraints: bool = True,
    eps: float = 1e-8,
    dload: float = 1e-7,             # inverse=False 時的對角加載
)
```

`forward(spec [B, 1, T, F, 2], ifc [B, T, num_freqs, N*2], iRxx [B, T, num_freqs, N*N*2]) -> [B, 1, T, F, 2]`。

## Class: `MultiFrameMvdrFilter`

Multi-frame MVDR filter，對目前 frame 的語音成分無失真，由噪音相關矩陣 `Rnn` 算出：

```
w = Rnn⁻¹ r · conj(r_cur) / (rᴴ Rnn⁻¹ r)
```

它等於 `Rnn⁻¹ γ / (γᴴ Rnn⁻¹ γ)`，其中正規化 IFC `γ = r / r_cur`，`r_cur` 是 window
最後一個元素（`lookahead=0` 時就是目前 frame）。constructor 與
`MultiFrameWienerFilter` 相同；`forward(spec, ifc, iRnn)` 的 shape 也相同。

### Covariance 輸入形式（兩個 filter 共用）

| `cholesky_decomp` | `inverse` | 網路預測的是 | filter 的動作 |
|---|---|---|---|
| False | True | `R⁻¹` | 直接相乘 |
| True | True | `L`，`R⁻¹ = L Lᴴ` | 重建後相乘 |
| False | False | `R` | 對角加載、`linalg.solve` |
| True | False | `L`，`R = L Lᴴ` | 重建、對角加載、求解 |

`enforce_constraints` 會把 Cholesky 輸入的嚴格上三角歸零，或讓一般的 `R` 成為
Hermitian（對角為實數、上三角 = 下三角的共軛）。`get_r_factor()` 依輸入形式回傳一個
常數 `f`，使預測矩陣除以 `f` 大致落在 `[-1, 1]`，用來縮放網路輸出。
`Masker.apply_wiener` 用預設形式（`R⁻¹`）；`Masker.apply_mvdr` 用 `R⁻¹` 的
Cholesky 因子。

## Class: `DeepFilterDecoder`

從 bottleneck embedding 與 complex spectrogram 預測上面兩個 filter 需要的 IFC 向量
與 covariance：embedding 經 [`SqueezedGRU`](group_op.zh-TW.md) 接兩個
[`GroupedLinear`](group_op.zh-TW.md) 輸出，再各自加上 spectrogram 上的 separable-conv
路徑。

```python
DeepFilterDecoder(
    df_bins: int,                       # 要預測的 bin 數
    cpx_in_channels: int,               # spectrogram 輸入的 channel 數
    emb_in_dim: int,                    # embedding 寬度
    emb_hid_dim: int = 256,             # SqueezedGRU hidden size
    df_order: int = 3,                  # tap 數 N
    df_n_layer: int = 3,                # SqueezedGRU 層數
    df_pathway_kernel_size_t: int = 1,  # conv 路徑的時間 kernel
    df_lin_groups: int = 1,             # 輸出 GroupedLinear 的 group 數
)
```

`forward(emb [N, emb_in_dim, T], c0 [N, CH, df_bins, T]) -> (ifc [N, df_bins, T, df_order*2], cov [N, df_bins, T, df_order**2*2])`。
輸出是頻率在時間之前；`Masker.apply_wiener` / `apply_mvdr` 會 permute 成 filter 的
`[B, T, F, ...]` 順序。`mask_type: wiener | mvdr` 需要回傳 `(mask, ifc, cov)` 的
backbone；`puresound.nnet` 裡沒有 backbone 這樣回傳。

## 經由 `Masker` 串接

```python
# Masker.apply_wiener（節錄）
tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1)   # [N, 2, C, T] -> [N, 1, T, C, 2]
est_ifc = est_ifc.permute(0, 2, 1, 3)              # [N, C, T, *] -> [N, T, C, *]
est_cov = est_cov.permute(0, 2, 1, 3)
wiener = MultiFrameWienerFilter(num_freqs=nbins, frame_size=order, lookahead=0)
enh = wiener(tf_rep, est_ifc, est_cov)
```

`Masker.apply_df_on_reim` 對 `DeepFilter` 做同樣的事，
`Masker.apply_complex_mask_with_df` 則套用 residual 路徑。

## 範例

```python
from puresound.nnet.lobe.multiframe import DeepFilter

deepfilter = DeepFilter(num_freqs=257, frame_size=3, lookahead=0)
spec = torch.rand(2, 1, 100, 257, 2)     # [B, C, T, F, 2]
coefs = torch.rand(2, 3, 100, 257, 2)    # [B, N, T, F, 2]
enh = deepfilter(spec, coefs)            # [B, C, T, F, 2]
```
