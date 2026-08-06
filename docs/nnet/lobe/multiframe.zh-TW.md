# puresound.nnet.lobe.multiframe

English version: `multiframe.md`

DeepFilterNet 風格的 multi-frame 語音強化 module:把 spectrogram 依
(frequency, time) 為單位,展開成小範圍的時間 context window,再套用
學習出來的、或閉式解(Wiener/MVDR)的 filter 係數,取代單一 frame 的
complex mask。

**這個 module 的 shape 慣例是 `[B, C, T, F, ...]`**——batch、channel、
time、frequency,選擇性地在最後加上代表堆疊實部/虛部的 `2`。這與
[`masker.Masker`](../masker.md) 的 `[N, 2, C, T]`(實部/虛部緊接在
batch 之後、再來是 frequency、最後是 time)是**兩種不同、而且都是真實
存在的慣例**——順序並不相同,而 `nnet/masker.py` 在每個呼叫點都會明確
`.permute(...)` 在兩者之間轉換。不要假設這兩個 module 的 tensor 可以
不經轉置直接互通。

## Class: `MultiFrameModule`

基礎 class:把 `[B, C, T, F]` spectrogram 轉成每個時間步彼此重疊的
`frame_size` 個 frame context window。

```python
MultiFrameModule(num_freqs: int, frame_size: int, lookahead: int, real: bool = False)
```

**Parameters:**
- `num_freqs` – filter 實際套用的頻率 bin 數(可能小於 spectrogram 的完整寬度——見下方子類別裡的 `narrow` 呼叫)
- `frame_size` – context window 的 frame 數
- `lookahead` – **必填,沒有預設值**——`frame_size` 個 frame 裡有多少是*未來*的 frame(`frame_size - 1 - lookahead` 個是過去的 frame);`lookahead=0` 就是完全 causal
- `real` – 決定要用 `spec_unfold`(complex 輸入)還是 `spec_unfold_real`(帶有明確額外尾軸的 real-valued 輸入)的 padding/unfold 邏輯。**這個檔案內外都沒有任何呼叫端設定 `real=True`**——下面每個具體子類別都只會走 complex(`real=False`)那條路

### `spec_unfold(spec: Tensor) -> Tensor`

**Parameters:** `spec` – complex `[B, C, T, F]`

在時間軸上 padding(前面 `frame_size - 1 - lookahead` 個、後面
`lookahead` 個)後展開。**Returns:** `[B, C, T, F, N]`,`N = frame_size`。

### `spec_unfold_real(spec: Tensor) -> Tensor`

Real-valued 版本(只有 `real=True` 時才會用到,但目前整個 codebase
都沒有任何地方觸發這個路徑)。

### `solve(Rxx, rss, diag_eps=1e-8, eps=1e-7)`(staticmethod)

透過 `torch.inverse` 加上 Tikhonov 正規化([`_tik_reg`](#module-level-helpers))
求解通用的 `Rxx⁻¹ @ rss`。定義出來是為了重複使用,但**下面三個具體的
filter 都沒有呼叫它**,repository 裡其他地方也沒有。

### `apply_coefs(spec, coefs)`(staticmethod)

`einsum("...n,...n->...", spec, coefs)`——把每個 window 的係數向量套用到
展開後的 spectrogram 上。`MultiFrameWienerFilter` 與 `MultiFrameMvdrFilter`
會用它(`DeepFilter` 不會,它改用 module-level 的 `df()` 函式)。

---

## Module-level helpers

- **`psd(x: Tensor, n: int) -> Tensor`** – 在 `n` 個步長展開的時間 window 上算 `X · conj(X)ᵀ` 外積 PSD/相關矩陣。`x`:`[B, C, T, F]` → 回傳 `[B, C, T, F, n, n]`。有定義但 repository 裡沒有任何地方呼叫它——給還沒有現成 covariance 估計、想從原始 spectrogram 算出 `Rxx` 的呼叫端用的 library 工具函式。
- **`df(spec, coefs) -> Tensor`** – `einsum("...tfn,...ntf->...tf", spec, coefs)`,真正的 deep-filtering 加總乘積。`DeepFilter.forward` 會用到。
- **`_compute_mat_trace(input, dim1=-2, dim2=-1)`** – 沿兩個軸算 trace;`_tik_reg` 會用到。
- **`_tik_reg(mat, reg=1e-7, eps=1e-8) -> Tensor`** – 對相關矩陣做 Tikhonov 正規化:`mat + (trace(mat).real * reg + eps) * I`。當輸入*不是*已經是 inverse 時(`inverse=False`),`MultiFrameWienerFilter`/`MultiFrameMvdrFilter` 會用到。

---

## Class: `DeepFilter`

直接套用**學習出來**的 filter 係數(沒有 Wiener/MVDR 的結構)——
DeepFilterNet 的核心算子。

```python
DeepFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    real: bool = False,
    conj: bool = False,
)
```

**Parameters:**
- `num_freqs`、`frame_size`、`lookahead`、`real` – 與 `MultiFrameModule` 相同
- `conj` – 若為 `True`,套用係數前先取共軛

**Reference:** Schröter et al., "DeepFilterNet: A Low Complexity Speech
Enhancement Framework for Full-Band Audio based on Deep Filtering," ICASSP 2022。

### `forward(spec: Tensor, coefs: Tensor) -> Tensor`

**Parameters(原始碼 docstring 自己的慣例——見上方的提醒):**
- `spec` – `[B, C, T, F, 2]`
- `coefs` – `[B, N, T, F, 2]`(`N = frame_size`)

只有前 `num_freqs` 個 bin 會被處理——`spec[..., :num_freqs, :]` 會被換成
過濾後的結果,更高的 bin 則原樣通過。

**Returns:** `[B, C, T, F, 2]`,與 `spec` 形狀相同。

---

## Class: `MultiFrameWienerFilter`

從 inter-frame correlation(IFC)向量與 noisy covariance matrix 算出的
multi-frame Wiener filter。

```python
MultiFrameWienerFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    cholesky_decomp: bool = False,
    inverse: bool = True,
    enforce_constraints: bool = True,
    eps: float = 1e-8,
    dload: float = 1e-7,
)
```

**Parameters:**
- `num_freqs`、`frame_size`、`lookahead` – 與 `MultiFrameModule` 相同
- `cholesky_decomp` – 若為 `True`,covariance 輸入是 `Rxx = L·Lᴴ` 裡的 `L`,而不是 `Rxx` 本身
- `inverse` – 若為 `True`(預設),covariance 輸入已經是 `Rxx⁻¹`,所以不需要 `torch.linalg.solve`,只做線性組合;若為 `False`,輸入是一般的 `Rxx`,會先用 `_tik_reg` 正規化再求解
- `enforce_constraints` – 重新套用 Hermitian 對稱性(一般矩陣輸入時),或把不合法的上三角區域清零(Cholesky 輸入時)
- `eps`、`dload` – `inverse=False` 時傳給 `_tik_reg` 的正規化常數

### `forward(spec, ifc, iRxx) -> Tensor`

**Parameters:**
- `spec` – `[B, 1, T, F, 2]`
- `ifc` – `[B, T, F, N*2]`,inter-frame 語音相關向量
- `iRxx` – `[B, T, F, (N**2)*2]`,(inverse)noisy covariance matrix 或其 Cholesky 因子

**Returns:** `[B, C, T, F, 2]`。

---

## Class: `MultiFrameMvdrFilter`

Multi-frame MVDR(distortionless)beamformer——建構子與
`MultiFrameWienerFilter` 完全一樣,只是權重公式不同(正規化到讓 steering
方向維持 distortionless)。

```python
MultiFrameMvdrFilter(
    num_freqs: int,
    frame_size: int,
    lookahead: int,
    cholesky_decomp: bool = False,
    inverse: bool = True,
    enforce_constraints: bool = True,
    eps: float = 1e-8,
    dload: float = 1e-7,
)
```

### `forward(spec, ifc, iRnn) -> Tensor`

與 `MultiFrameWienerFilter.forward` 的 shape 完全相同,只是 `iRnn`
扮演 `iRxx` 的角色(是 noise、而不是 noisy 的 covariance)。

---

## Class: `DeepFilterDecoder`

從 bottleneck embedding 加上原始 complex spectrogram,預測
`MultiFrameWienerFilter` / `MultiFrameMvdrFilter` 需要的 `ifc`/covariance
tensor——結合了一個便宜的 [`group_op.SqueezedGRU`](group_op.zh-TW.md) 分支
(處理 embedding)與一個小型的 separable-conv 分支(直接處理 spectrogram)。

```python
DeepFilterDecoder(
    df_bins: int,
    cpx_in_channels: int,
    emb_in_dim: int,
    emb_hid_dim: int = 256,
    df_order: int = 3,
    df_n_layer: int = 3,
    df_pathway_kernel_size_t: int = 1,
    df_lin_groups: int = 1,
)
```

**Parameters:**
- `df_bins` – 要預測係數的頻率 bin 數
- `cpx_in_channels` – complex-spectrogram 輸入分支的 channel 數
- `emb_in_dim` / `emb_hid_dim` / `df_n_layer` – 傳給內部的 `SqueezedGRU(emb_in_dim, emb_hid_dim, num_layers=df_n_layer, linear_groups=8)`
- `df_order` – filter tap 數(即上面幾個 filter 的 `N` / `frame_size`)
- `df_pathway_kernel_size_t` – 處理原始 spectrogram 的兩條 separable-conv 路徑的 kernel 大小
- `df_lin_groups` – 兩個輸出 `GroupedLinear` 投影使用的 group 數(與 `SqueezedGRU` 自己寫死的 `linear_groups=8` 無關)

### `forward(emb, c0) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `emb` – `[N, C, T]`,bottleneck embedding
- `c0` – `[N, CH, C, T]`,complex spectrogram(`CH = cpx_in_channels`)

**Returns:** `(ifc, cov)`:
- `ifc` – `[N, C, T, df_order * 2]`
- `cov` – `[N, C, T, df_order**2 * 2]`

(這裡的 `C` 是頻率軸,寬度為 `df_bins`——**frequency 在 time 之前**,
與 `MultiFrameWienerFilter`/`MultiFrameMvdrFilter` 自己的 `[B, T, F, ...]`
輸入慣例剛好相反)。目前 repository 裡沒有任何地方把這個直接接到那兩個
filter 上(也沒有任何測試涵蓋它),但 `nnet/masker.py` 的
`Masker.apply_wiener` / `Masker.apply_mvdr` 示範了在兩種慣例之間橋接
所需要的確切轉置方式——見下方 Wiring。

## Wiring

`nnet/masker.py` 的 `Masker` 靜態方法是這個 module 目前唯一的呼叫端,
每次呼叫都重新建構 filter,並先轉置成它預期的軸順序:

```python
# Masker.apply_wiener / Masker.apply_mvdr,節錄
tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1)          # [N,2,C,T] -> [N,1,T,C,2]
est_ifc = est_ifc.permute(0, 2, 1, 3)                     # [N,C,T,*] -> [N,T,C,*]
est_cov = est_cov.permute(0, 2, 1, 3)                     # [N,C,T,*] -> [N,T,C,*]
wiener = MultiFrameWienerFilter(num_freqs=nbins, frame_size=order, lookahead=0)
enh = wiener(tf_rep, est_ifc, est_cov)
```

這正是把 `DeepFilterDecoder` 的 `(ifc, cov)` 輸出接到這些 filter 上所需要
的轉置方式,即使目前沒有任何 model 真的把兩者端到端接在一起。
`Masker.apply_df_on_reim` 則是 `DeepFilter` 對應的膠水程式碼。

## Example

```python
from puresound.nnet.lobe.multiframe import DeepFilter

deepfilter = DeepFilter(num_freqs=257, frame_size=3, lookahead=0)
spec = torch.rand(2, 1, 100, 257, 2)     # [B, C, T, F, 2]
coefs = torch.rand(2, 3, 100, 257, 2)    # [B, N, T, F, 2]
enh = deepfilter(spec, coefs)            # [B, C, T, F, 2]
```
