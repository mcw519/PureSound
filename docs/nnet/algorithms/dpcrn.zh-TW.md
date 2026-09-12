# puresound.nnet.dpcrn

English version: [dpcrn.md](dpcrn.md)

DPCRN —— 建立在 `Unet` chassis 上的 dual-path convolutional recurrent
network：CNN 沿頻率軸做 down/up stack，bottleneck 是兩個 `DPRNNblock2D` block
（intra-frequency 用雙向 LSTM、inter-time 用單向 LSTM，可選 FiLM
conditioning）。是目前已發佈的 voice-isolate backbone。

## Class: `DPCRN`

### Constructor

```python
DPCRN(
    input_dim: int = 512,
    dvec_dim: Optional[int] = None,        # speaker-embedding FiLM conditioning
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),   # CNN stack；channels[-1] = bottleneck C
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t / stride_t / dilation_t: Tuple,      # 每個 down-layer，時間軸
    kernel_f / stride_f / dilation_f: Tuple,      # 每個 down-layer，頻率軸
    delay: Tuple = (0, 0, ...),            # 每層的 look-ahead;streaming 時
                                           # 的 algorithmic latency 為 sum(delay) 個 frame
    rnn_hidden: int = 128,
    spectral_compress: bool = False,
    vad_head:  Optional[Dict] = None,      # {enabled, hidden, kernel_t} -> lobe.heads.VADHead
    dist_head: Optional[Dict] = None,      # {enabled, hidden}           -> lobe.heads.DistHead
)
```

`forward(x [N, CH, C, T], dvec=None) -> [N, CH, C, T]` 會在 feature domain
裡預測出一個 mask（由 `EncDecMaskBase` 負責套用）。啟用這些 head 之後，
backbone 每次 forward 完會多暴露一些側向輸出：`last_vad_logits [N, T]` 與
`last_dist_preds [N, 3]`（見 [nnet.lobe.heads](../lobe/heads.md)）；兩者預設都是
關閉的，不啟用就完全不增加任何成本。

**Streaming**：`delay=[0,...]` 可以做到 bit-exact、零延遲的 streaming；
`delay > 0`（已發佈的 recipe 用的是 `[1,1,1]`，在 hop 160 下等於 30 ms）匯出時
會把 future-buffering 直接烤進 ONNX graph 裡 —— 見
[streaming/dpcrn_onnx](../../streaming/dpcrn_onnx.md)。

## Class: `DPRNNblock2D`

Bottleneck block 本體（intra-frequency BiLSTM + inter-time LSTM + 可選
FiLM）。

```python
DPRNNblock2D(input_size: int, hidden_size: int, dropout: float = 0.0,
             embedding_size: Optional[int] = None, fused_type: Optional[str] = None)
```
