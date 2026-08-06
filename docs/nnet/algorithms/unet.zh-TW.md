# puresound.nnet.unet

English version: [unet.md](unet.md)

Status: *library*（見 [nnet index](../index.zh-TW.md)）——這是就直接使用而言的
狀態，但這個 module 同時也是 [DPCRN](dpcrn.zh-TW.md) 跟
[DPARN](dparn.zh-TW.md) 繼承的**底盤（chassis）**——這兩個都是 active 狀態——
而 `UnetTcn` 本身也是 config-reachable 的（`getattr(nnet, "UnetTcn")`），
有自己的 forward smoke test（`test/test_backbone.py::test_unet_tcn_backbone`）。
這個檔案裡有三個 class；只有 `Unet` 跟 `UnetTcn` 有被 export。

## Class: `Unet`

一個在 `[frequency, time]` feature map 上運作的 2D CNN encoder/decoder，
搭配 down/up 對應階段之間的 skip connection。每個結構性開關都是一個
per-layer 的 tuple（每個 down-conv 階段一個值），而且頻率/時間軸是**各自獨立**
設定的——並沒有單一一個 `encoder_kernel: List[Tuple[int,int]]`。

### Constructor

```python
Unet(
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),   # 長度為 n_cnn+1；每個階段 channels[i] -> channels[i+1]
    transpose_t_size: int = 2,                 # up-path 的 ConvTranspose2d 時間軸 kernel（每個階段共用）
    skip_conv: bool = False,
    kernel_t: Tuple = (5, 1, 9, 1, 1),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),            # 每個 down-layer 的 look-ahead frame 數
    multi_output: int = 1,                     # 最後一層 up-layer 的 channel *倍率*，不是 bool
)
```

**Parameters:**
- `input_dim` – 輸入的頻率 bin 數
- `channels` – CNN 的 channel 遞增序列;`len(channels) == n_cnn + 1`，其中
  `n_cnn = len(kernel_t)`。`channels[0]` 是餵給 `input_norm` 的輸入 channel 數
- `kernel_t` / `stride_t` / `dilation_t` 以及 `kernel_f` / `stride_f` /
  `dilation_f` – 每個 down-conv 階段各一個值，這六個 tuple 長度必須一致（會
  assert）。公開的 API 裡沒有合併成單一 `(freq, time)` pair;內部才會用
  `self.kernel = list(zip(kernel_f, kernel_t))` 把它們 zip 起來
- `delay` – 每一層的 look-ahead frame 數。`0` = 那一層完全 causal（padding
  全部放在過去那一側）；down-conv 的時間軸 padding 是
  `((kernel_t[i]-1)*dilation_t[i] - delay[i], delay[i])` —— `delay[i]`
  越大，padding 就越往未來（look-ahead）那一側移動，所以 `delay[i]`
  **就是**那一層增加的延遲 frame 數。頻率方向的 padding 永遠是對稱的
  （兩側都是 `kernel_f[i]//2 * dilation_f[i]`）—— 只有時間軸才有 causality
  的問題。
- `transpose_t_size` – 每個 up-conv 階段的 `ConvTranspose2d` 都用這個值當
  時間軸的 kernel size（不是 `kernel_t`，那個只用在往下的路徑）；
  `ConvTranspose2d` 在時間軸 `stride=1` 時會把序列拉長
  `transpose_t_size - 1` 個 frame，`forward` 會把這段裁掉
- `skip_conv` – `False`（預設）：把 skip connection concat 到 up-path 的
  輸入上（up-conv 的輸入 channel 數會加倍，`channels[i+1] * 2`）。
  `True`：把 skip connection 投影過一個 `1x1 Conv2d + activation`
  （`self.skip_cnn`）之後改成用**加**的（up-conv 輸入 channel 數維持
  `channels[i+1]`）
- `multi_output` – 只作用在最後（`i == 0`）那層 up-conv 輸出 channel 上的
  **倍率**（`channels[0] * multi_output`）。不是 bool，也不是「回傳每個
  階段輸出的 list」——`forward` 永遠回傳單一個 `Tensor`；`multi_output > 1`
  只是把這個 tensor 最後的 channel 數變寬（例如把多個 mask head 打包進
  同一個 conv）。**這個 module 底下四個 subclass**（`DPCRN`、`DPARN`、
  `UnetTcn`、`UnetFsmn`）**沒有一個**會在自己的 `super().__init__()`
  呼叫裡把這個參數傳下去，所以實際上永遠固定是 `1`——要用到多輸出的最後一層，
  只能直接建構 `Unet` 本身才行。

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, C, T] 或 [N, C, T]（會 unsqueeze 成 [N, 1, C, T]）
# returns: [N, channels[0] * multi_output, C, T]
```

`input_norm`（對攤平後的 `channels[0] * input_dim` 做的一個 `iLN`，見
[lobe/norm](../lobe/norm.md)）→ CNN-down（每個階段的輸出都會 append 進
`skip` 這個 list）→ CNN-up 反過來跑，每個階段把對應的 skip connection
concat 或加上去，裁掉 transpose-conv 多出來的 `transpose_t_size - 1` 個
尾端 frame。在這個 base class 這一層，沒有 `transpose_delay` 這個選項——
裁切永遠是裁掉*尾端*（causal-safe 的方向）;`transpose_delay` 是下面這些
subclass 自己重新實作 `forward` 才加上的功能。

### `shape_info()` 與 `get_args`

`shape_info() -> (down_shape, up_shape)` 會依設定的 stride 走一遍，回報每個
階段的頻率 bin 數（用來檢查某組 `channels`/`stride_f` 設定是否合理的輔助
工具，`forward` 本身不會用到）。`get_args` 回傳一個完整的 `Dict`，涵蓋全部
15 個 constructor 參數，一個都不少——跟 [`DPARN.get_args`](dparn.zh-TW.md)
不一樣。

## Class: `UnetTcn`

`Unet` 把 bottleneck 換成一疊 [`TCN`/`GatedTCN`](conv_tasnet.zh-TW.md)
block，作用在攤平後的 `(channel * frequency)` 軸上，而不是用遞迴或
attention 當 bottleneck。

### Constructor

```python
UnetTcn(
    embed_dim: int = 0,
    embed_norm: bool = False,
    input_type: str = "RI",           # 有這個參數，但完全沒有任何地方會讀它 -- 是個死參數
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t: Tuple = (5, 1, 9, 1, 1),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),
    tcn_layer: str = "normal",         # "normal" -> TCN，"gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,
    repeat_tcn: int = 4,
    tcn_with_embed: List = [1, 0, 0, 0, 0],
    tcn_use_film: bool = False,
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",           # tcn_layer == "gated" 時會被忽略（會印出警告）
    causal: bool = False,
)
```

`input_dim` 到 `delay` 這幾個會原封不動傳給 `Unet.__init__`，跟 base class
一樣（`multi_output` 沒有被暴露出來——永遠是 `1`，見上方說明）；`tcn_layer`
到 `causal` 這幾個則是照 [`ConvTasNet`](conv_tasnet.zh-TW.md) 的方式建
TCN stack（一樣會 assert `len(tcn_with_embed) == per_tcn_stack`，dilation
的排程也一樣），作用在
`temporal_input_dim = (input_dim 經過所有 stride_f downsample 之後) *
channels[-1]` 個 channel 上。有兩個參數值得特別提醒：
- `input_type` 雖然接受這個參數，但這個 class **完全沒有任何地方存它或讀
  它**——傳什麼值進去都完全沒有效果。
- **`UnetTcn.forward` 從來不會呼叫 `self.input_norm`**——`Unet.__init__`
  建的那個 `iLN` layer 確實存在也持有參數，但這個 subclass 自己覆寫的
  `forward` 是直接從 input unsqueeze 跳到 CNN-down 迴圈，中間完全跳過它。
  `DPCRN` 跟 `DPARN`（RNN-bottleneck 的 subclass）都會呼叫
  `input_norm`；`UnetTcn` 跟 `UnetFsmn`（TCN/FSMN-bottleneck 的 subclass）
  則兩個都不會。這是不是故意的，原始碼裡完全沒有任何說明——先當作
  `UnetTcn`（還有 `UnetFsmn`）的輸入實際上是沒有做過 normalize 的。

`transpose_delay`（跟 base `Unet` 不一樣）在這裡是真的有作用的選項：
`transpose_delay=True` 時，`forward` 會裁掉*前段*的 `transpose_t_size - 1`
個 frame，`False`（預設）時裁掉尾端——跟
[`DPCRN`](dpcrn.zh-TW.md) 用的是同一套慣例。

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, CH, C, T] 或 [N, C, T]
# dvec: [N, embed_dim]，只有 tcn_with_embed[i] == 1 時才需要
# returns: [N, CH, C, T]
```

CNN-down → 把 `(CH, C)` 攤平成單一 channel 軸 → `repeat_tcn` 組、每組
`per_tcn_stack` 個 TCN/GatedTCN block（依 `tcn_with_embed` 決定是否用
`dvec` 條件化）→ 展開回 `(CH, C)` → CNN-up（skip-concat 或 `skip_conv`，
裁切時會考慮 `transpose_delay`）。

### `get_args` property

除了 `input_type`（沒有存起來，想拿也拿不回來）跟 `multi_output`（這個
subclass 根本沒有暴露，見上面 `Unet` 的說明）之外，其餘都完整。

### Example（對照 `test/test_backbone.py::test_unet_tcn_backbone`）

```python
from puresound.nnet import UnetTcn

model = UnetTcn(
    embed_dim=192, embed_norm=True, input_dim=256,
    activation_type="PReLU", norm_type="gLN",
    channels=(2, 32, 64, 128, 128, 128, 128),
    transpose_t_size=2, transpose_delay=True, skip_conv=False,
    kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
    stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2),
    dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1),
    delay=(0, 0, 0, 0, 0, 0),
    tcn_layer="gated", tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2,
    per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm=None, causal=False,
)
y = model(torch.rand(1, 2, 256, 100), torch.rand(1, 192))  # [1, 2, 256, 100]
```

## Class: `UnetFsmn`

這個檔案裡還有第三個 subclass——把 `Unet` 的 bottleneck 換成一疊
`FSMN`/`ConditionFSMN` layer（見 [lobe/rnn](../lobe/rnn.md)），而不是 TCN
block。它**目前沒有被 export**——`puresound/nnet/__init__.py` 從這個
module 只 import 了 `Unet` 跟 `UnetTcn`，所以 `UnetFsmn` 不是
config-reachable 的（`getattr(nnet, "UnetFsmn")` 會失敗），也完全沒有
test 覆蓋。另外有一個獨立的清理任務在追蹤到底要把它 export 出來還是直接
刪掉；在那個決定確定之前，這裡先不寫完整的文件。
