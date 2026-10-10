# puresound.nnet.lobe.ssm

English version: [ssm.md](ssm.md)

給串流增強用的選擇性狀態空間（Mamba / S6）區塊（Gu and Dao, "Mamba: Linear-Time
Sequence Modeling with Selective State Spaces", 2023）。

`MambaInter` 是 `DPRNNblock2D` inter（時間）路徑 `SingleRNN` 的替換件：輸入
`[N, D, T]`、輸出 `[N, D, T]`、因果。inter 路徑是 dual-path 模型裡唯一跨時間攜帶
上下文的元件，而 LSTM 在那裡學到的上下文受限於訓練列的長度，以及 backpropagation
through time 能回傳多遠。SSM 保有逐幀的遞迴形式——沒有 look-ahead、每步 CPU 成本與
LSTM 一步同一量級——而它隨輸入變化的衰減是為更長的上下文設計的。

## 計算

```
[x', z]      = in_proj(x)                        # D -> 2 * d_inner
u            = SiLU(causal depthwise Conv1d(x')) # kernel d_conv
[dt_r, B, C] = x_proj(u)                         # dt_rank + 2 * d_state
Δ            = softplus(dt_proj(dt_r))           # 每個 channel、每個 frame
A            = -exp(A_log)                       # [d_inner, d_state]，負實數
h_t          = exp(Δ_t A) ⊙ h_{t-1} + (Δ_t u_t) B_t
y_t          = <h_t, C_t> + D ⊙ u_t
out          = out_proj(Dropout(y ⊙ SiLU(z)))    # d_inner -> D
```

## Class: `MambaInter`

```python
MambaInter(
    d_model: int,                  # D，inter 路徑寬度
    d_state: int = 16,             # 每個 channel 的 SSM 狀態大小
    d_conv: int = 4,               # 因果 depthwise conv 的 kernel
    expand: int = 2,               # d_inner = expand * d_model
    dt_rank: Optional[int] = None, # 預設 ceil(d_model / 16)
    dropout: float = 0.0,          # 在 out_proj 之前
    dt_min: float = 0.001,         # 初始步長 Δ 的範圍
    dt_max: float = 0.1,
    dt_init_floor: float = 1e-4,
    zero_init_out: bool = False,   # out_proj 從零開始
)
```

- `forward(x [N, D, T]) -> [N, D, T]`。
- `initial_stream_state(batch, device=None, dtype=None) -> (conv_cache [N, d_inner, d_conv-1], h [N, d_inner, d_state])`；
  conv cache 用模型的 dtype，`h` 一律 fp32。
- `step(x_t [N, D], state) -> (y_t [N, D], state)` 前進一幀，結果與串行掃描完全一致。

初始化：`A` 用 S4D-real（每個 channel `A_log = log(1..d_state)`）；`dt_proj` 的 bias
設成讓 `softplus(bias)` 在 `[dt_min, dt_max]` 間呈 log-uniform（下限
`dt_init_floor`），這決定了初始的記憶長度。

**參數預算。** `d_model` 128 時預設值約 116k 參數，對上
`SingleRNN("LSTM", 128, 96)` 連同 projection 約 99k，所以兩者互換是等量的容量交換。

### 在 DPCRN 裡的用法

```yaml
backbone_args:
  inter_type: lstm+mamba      # lstm | mamba | mamba_context | lstm+mamba
  mamba_args: {d_state: 16, d_conv: 4, expand: 2}
```

`mamba` 取代 inter LSTM；`lstm+mamba` 保留 LSTM，另加一條
`MambaInter(zero_init_out=True)` 並聯分支；`mamba_context` 在感知頻帶上執行這個區塊。
見 [DPCRN](../dpcrn.zh-TW.md)。

### 設計說明

- **`zero_init_out`。** 輸出投影為零時區塊輸出恰好是 0，所以並聯到已訓練 LSTM 上的
  分支在第 0 步不會改變它的輸出：warm start 不必重新初始化任何東西，而 SSM 內部參數
  要等 `out_proj` 動了之後才開始收到梯度。
- **fp32 狀態。** `h` 每一幀都乘上 `exp(Δ A)`；在 bf16 下每步的微小增量會 underflow，
  所以不論 autocast 設定為何，`h` 與遞迴都以 fp32 計算。

## 執行路徑

`forward` 在執行期決定走哪條掃描：

```python
use_kernel = selective_scan_fn is not None and u.is_cuda and not torch.jit.is_tracing()
```

| 路徑 | 時機 | 說明 |
|---|---|---|
| `mamba_ssm` 的融合 `selective_scan_fn` | CUDA、kernel 可載入、非 tracing | 訓練路徑 |
| `_scan_fallback` | CPU、tracing、或沒有 kernel | 以 `SCAN_CHUNK = 64` 分塊的逐幀 Python 迴圈；結果精確 |
| `step()` | 串流 | 一幀、明確的 `(conv_cache, h)` 狀態；串流匯出執行的就是它 |

三條路徑共用同一組參數。`_load_selective_scan` 只載入
`mamba_ssm.ops.selective_scan_interface`（不 import 整個套件，套件的 generation
工具會要求特定 `transformers` 版本），任何失敗都回傳 `None`，因為 fallback 是受支援的
設定。

**少了 kernel 是無聲而且昂貴的。** fallback 正確但慢很多，差距幾乎全落在 backward。
串流匯出與它的 real-time-factor 檢查逐幀執行 `step()`，看不到這件事；訓練成本必須
另外檢查。訓練慢得不合理時，在訓練實際執行的環境裡查：

```python
from puresound.nnet.lobe.ssm import selective_scan_fn
print(selective_scan_fn is not None)
```

不同的工作目錄可能讓 `mamba_ssm` 解析到原始碼樹而不是已安裝的套件，給出不同的答案。

## 記憶體與分塊

訓練時整個區塊包在一個 checkpoint 裡（否則 projection 的中間結果會保留整段序列），
fallback 路徑上每個 `SCAN_CHUNK` 幀的分塊再各自 checkpoint，所以 backward 只保留分塊
邊界。`u`、`Δ`、`B`、`C`、`z` 的 fp32 副本只存在於 checkpoint 的分塊內部，從不以全長
存在。調大 `SCAN_CHUNK` 不會讓 fallback 變快——成本在串行迴圈——反而會保留更大的
autograd 圖。

## 對新的 PyTorch 重編 kernel

預編的 `selective_scan_cuda` 擴充連結的是 PyTorch 內部 C++ ABI，而那個 ABI 跨版本
不穩定。升級 torch 後 import 可能失敗：

```
ImportError: selective_scan_cuda...so: undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

解碼之後，擴充要的是
`c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool)`，
而安裝的 torch 匯出的是
`(int, char const*, char const*, unsigned int, bool, c10::cuda::CUDAErrorLogCapture*)`。
是簽名變了；symlink 或 `LD_PRELOAD` 都沒用。修法是對著已安裝的 torch 重新編譯。

直接 `pip install mamba-ssm` 會撞到三件事：

**PyPI 的 sdist 沒有 CUDA 原始碼。** `csrc/` 不存在，編譯會失敗在
`selective_scan.cpp: No such file or directory`。改從所需版本在 GitHub 上的 tag 編
（下例為 `v2.2.4`）：

```bash
git clone --depth 1 --branch v2.2.4 https://github.com/state-spaces/mamba.git
```

**近期的 torch header 要求 C++20**，而 mamba 的 `setup.py` 寫死 `-std=c++17`：

```
ATen/ATen.h:5: error: C++20 or later compatible compiler is required to use ATen.
```

把 `setup.py` 裡四處 `-std=c++17` 改成 `-std=c++20`。

**編譯器必須真的是 C++20。** GCC 10 接受 `-std=c++20`，但回報
`__cplusplus = 201709L`（草案），而 torch 檢查 `>= 202002L`，所以一樣過不了。需要
GCC 11+。發行版沒有更新的版本時，用一組隔離的工具鏈，不動系統也不動專案 venv：

```bash
conda create -y -p <toolchain-prefix> -c conda-forge 'gxx_linux-64=12' 'gcc_linux-64=12'
```

選的 GCC 版本也必須是你的 CUDA toolkit 的 `nvcc` 接受的 host 編譯器（例如 CUDA 12.3
最高接受 GCC 12）；選定前先查該 toolkit 的 release notes。

然後**就地編譯**，驗證通過之前什麼都不安裝：

```bash
cd mamba
export MAMBA_FORCE_BUILD=TRUE                  # 絕不抓預編 wheel
export TORCH_CUDA_ARCH_LIST=<major.minor>      # 只編你這張 GPU 的 compute capability
export CUDA_HOME=<cuda-toolkit-dir>            # 與 torch.version.cuda 相符的 toolkit
export MAX_JOBS=8                              # 每個 nvcc job 需要數 GB 記憶體
export CC=<toolchain-prefix>/bin/x86_64-conda-linux-gnu-gcc
export CXX=<toolchain-prefix>/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
python setup.py build_ext --inplace
```

`TORCH_CUDA_ARCH_LIST` 設成要執行的那張 GPU 的 compute capability
（`python -c "import torch; print(torch.cuda.get_device_capability())"` 會印成一組
數字，`(8, 6)` 就寫 `8.6`）；只列這一個架構可以縮短編譯時間。`CUDA_HOME` 指向版本與
`torch.version.cuda` 相符的 CUDA toolkit，`<toolchain-prefix>` 指向上一步的 GCC
安裝位置。

`NVCC_PREPEND_FLAGS` 很容易漏掉：`CC`/`CXX` 只管 `.cpp` 檔，`nvcc` 編 `.cu` 時用的是
它自己預設的 host 編譯器（`/usr/bin/c++`），於是 C++20 檢查會在編譯途中再次失敗。

### 安裝前先驗證

一個 import 得了的擴充仍可能算錯。拿它旁邊附的參考實作比對，梯度與輸出都要比：

```python
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
y_k = selective_scan_fn(u, dt, A, B, C, D, z=z, delta_bias=db, delta_softplus=True)
y_r = selective_scan_ref(u, dt, A, B, C, D, z=z, delta_bias=db, delta_softplus=True)
# forward 與對 u, dt, B, C 的梯度在 fp32 下應一致到 ~1e-7
```

先備份舊的 `.so`，比對通過後才把新的複製進 `site-packages`，再重跑
`test/streaming/test_dpcrn_streaming.py`。串流路徑走的是 `MambaInter.step()` 而不是
掃描，所以它確認的是匯出的計算圖沒有改變。
