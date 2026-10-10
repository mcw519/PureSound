# puresound.nnet.lobe.banding

English version: [banding.md](banding.md)

把 bottleneck 的均勻頻率網格池化到感知尺度（ERB 或 mel）的頻帶上，再展開回去。
循環區塊因此只跑在較少的頻率位置上——低頻密、高頻疏——而它前後的所有東西仍維持
完整網格。

## 尺度與頻帶邊界

```python
erb_rate(hz)             # 21.4 * log10(1 + 0.00437 * hz)   (Glasberg & Moore ERB number)
erb_rate_to_hz(rate)
mel_rate(hz)             # 2595 * log10(1 + hz / 700)
mel_rate_to_hz(rate)

band_edges_hz(n_bands: int, f_min: float, f_max: float, scale: str = "erb") -> Tensor
# n_bands + 1 個邊界，在 `scale` 上等距，以 Hz 回傳。
# 除非 scale 屬於 {"erb", "mel"}、n_bands >= 1 且 0 <= f_min < f_max，否則 ValueError。
```

## Function: `triangular_band_matrix`

```python
triangular_band_matrix(
    n_bands: int, n_units: int, *, f_min: float, f_max: float, scale: str = "erb"
) -> Tensor   # [n_bands, n_units]
```

第 `u` 個位置視為位於 `f_min + (u + 0.5) * (f_max - f_min) / n_units`。
第 `b` 帶涵蓋邊界 `[e_b, e_{b+1}]`、中心 `c_b`；權重從 `e_{b-1}` 線性上升到 `c_b`，
再下降到 `e_{b+2}`，所以每一帶都與相鄰帶重疊（最外側的兩帶以鏡射方式延伸）。
每一列正規化為總和 1。比一個位置還窄的頻帶會把權重 1 放在最近的位置上，因此沒有空列。

## Class: `BandBottleneck`

```python
BandBottleneck(
    n_units: int,               # 輸入網格的頻率位置數
    n_bands: int,               # 必須 <= n_units，否則 ValueError
    *,
    sample_rate: int = 16000,
    f_min: float = 50.0,
    f_max: float | None = None, # None -> sample_rate / 2
    scale: str = "erb",         # "erb" | "mel"
    learnable: bool = False,    # 兩個映射都可訓練，以固定映射初始化
)
```

| method | shape |
| --- | --- |
| `to_bands(x)` | `[N, CH, n_units, T] -> [N, CH, n_bands, T]`，使用 `pool` `[n_bands, n_units]` |
| `to_units(x)` | `[N, CH, n_bands, T] -> [N, CH, n_units, T]`，使用 `expand` `[n_units, n_bands]` |

`expand` 是 `pool` 的轉置，再重新正規化成每個位置收到的總權重為 1。
`learnable=False` 時兩個映射都是 buffer。

## 在 recipe 中的用法

`DPCRN` 在兩個地方建它，兩個選項互斥。

- `band_bottleneck`：兩個 `DPRNNblock2D` 的輸入都先分帶，結果在進 head 與 decoder
  之前展開回來，所以只有 dual-path 區塊看到頻帶。
- `mamba_context`（搭配 `inter_type: mamba_context`）：intra-frequency 路徑留在
  完整網格，只把 inter-time 的 Mamba 路徑分帶；其輸出展開後以 residual 加回。

`n_units` 是 bottleneck 的頻率數，由 `DPCRN` 自動填入；其餘參數由 config 給。

```yaml
backbone:
  type: DPCRN
  backbone_args:
    stride_f: [2, 2, 1]          # 頻帶是從這個網格池化來的
    band_bottleneck:
      n_bands: 32
      scale: erb                 # 或 mel
      sample_rate: 16000
      f_min: 50.0
      learnable: false
```

`stride_f` 決定最低幾帶最細能到多細。16 kHz、256 bin、`stride_f: [2, 2, 1]` 時，
池化來源網格是 125 Hz 寬，所以不管尺度說最低幾帶該多窄，實際上都是 125 Hz。
`stride_f: [1, 1, 1]` 直接從完整解析度分帶，代價是 convolution 變貴。

## 設計說明

- 均勻的頻率 stride 在 200 Hz（人聲諧波所在）丟解析度的速度，跟在 7 kHz（幾乎沒
  東西可丟）一樣。感知尺度的頻帶把同樣數量的位置密集地花在低頻、稀疏地花在高頻。
  RNNoise、PercepNet、DeepFilterNet 也是這樣分組。
- 分帶不是 STFT 的替代品。分析、合成與 mask 都留在完整的複數網格上，因為逐帶增益
  無法重建帶內的結構。
- 每列總和為 1，池化是平均，數值不會隨帶寬變大。展開方向逐位置正規化，避免帶邊界的
  位置回來比帶中心安靜——那會形成梳狀濾波。
- 三角形、互相重疊的權重，避免音高移動時諧波跨帶處出現硬邊界。
- `learnable` 預設關閉，頻帶配置因此是明確、固定的設計選擇，其效果可以歸因。
- 兩個映射都是沿頻率的固定線性映射，時間上無狀態。串流版 DPCRN
  （`puresound/streaming/dpcrn.py`）重寫了 backbone 的 forward 並套用同樣的分帶，
  循環狀態的大小跟著帶數走；`test/streaming/test_dpcrn_streaming.py` 把它與離線
  模型對照。
