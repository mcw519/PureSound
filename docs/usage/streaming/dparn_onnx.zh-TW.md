# DPARN 串流 ONNX

English: [dparn_onnx.md](dparn_onnx.md)

> **狀態：legacy**——保持可用並凍結：不加新功能、不重寫。沒有任何已發布的 checkpoint
> 用它；部署路徑是 [DPCRN](dpcrn_onnx.zh-TW.md)。

DPARN 串流推論使用特徵幀 ONNX 模型，分工與 DPCRN 相同：Python 負責音訊緩衝、固定的
Hann STFT、overlap-add iSTFT 與狀態張量；ONNX Runtime 一次跑一個 DPARN 特徵幀。兩種
backbone 共用逐幀模型基底、exporter、manifest 格式與 runtime
（`puresound/streaming/base.py`）。

## 支援的設定

`validate_streaming_dparn_config` 在匯出前檢查 recipe。以下是硬性要求——違反就丟
`ValueError`：

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` encoder，Hann 窗、`sr: 16000`、`fmax: 8000`、`trainable: False`、
  `win_length <= fft_length`、`hop_length > 0`
- `features.feats_type: complex`、`drop_stft_first_bin: True`、`trainable: False`，
  不用 `include_specaug`
- `DPARN` backbone，`input_dim: 256`，`norm_type` 為 `cLN` / `iLN` / `bN2d` 之一，
  `skip_conv: False`，每個 down layer 都是 `stride_t: 1` 與 `dilation_t: 1`

有兩項檢查只會警告，匯出仍會成功：

- `transpose_delay: True`——匯出的模型是逐幀的，但不是因果的
- 非零的 `delay`——down layer 會看到未來的幀

和 DPCRN 路徑不同，**DPARN 的 exporter 沒有 look-ahead 補償**：它不緩衝未來的幀，也不在
暖機期間閘住遞迴狀態。非因果的 `delay` / `transpose_delay` 設定照樣能匯出、也照樣通過匯出時
的幀檢查，但結果不是真正的串流——那個檢查只拿一個幀對照 PyTorch 逐幀模型，不是在只有過去
的即時緩衝下的行為。

**repo 裡的 DPARN 範例不是串流 recipe。** `egs/noise_suppression/config/dparn.yaml` 設了
`encoder_args.trainable: True` 並加上 `freq_eq` 區塊：它是離線 recipe，而學習出來的前端
無法匯出成這個 runtime 所依賴的固定 STFT。匯出它會在 `encoder.trainable` 失敗，那是檢查
正常運作，不是缺陷。要匯出可串流的 DPARN checkpoint，就用 `encoder_args.trainable: False`
與固定的 Hann 前端訓練。

## 匯出

DPARN 沒有命令列包裝；直接呼叫函式庫：

```python
from puresound.streaming import export_streaming_dparn_onnx
from puresound.system.postprocess import Postprocessor

manifest = export_streaming_dparn_onnx(
    "path/to/dparn_recipe.yaml",
    "path/to/model.ckpt",
    "path/to/model.onnx",                      # manifest：旁邊的 model.json
    postprocess=Postprocessor(dry_blend=1.0),  # 記錄下來，由 runtime 套用
)
```

exporter 寫出與 DPCRN 相同的 manifest：音訊設定、輸入輸出與狀態的名稱與形狀、偏好的
provider、`streaming_delay_frames`，以及 graph 後處理的 `recommended_inference` 區段。它
不接受 onset guard；但帶有 `onset_guard` 區段的 DPARN manifest 仍會被 runtime 套用。graph
後處理階段只在一處說明：[dpcrn_onnx.zh-TW.md](dpcrn_onnx.zh-TW.md#runtime-套用的-graph-後處理)。

匯出時會拿 ONNX 的幀輸出對照 PyTorch 逐幀模型；兩者在容許誤差內不一致，匯出就失敗。

## 推論

runtime 是共用的 `StreamingOrt`，以 `StreamingDparnOrt` 的名字重新 export：

```python
from puresound.streaming import StreamingDparnOrt

runtime = StreamingDparnOrt("/path/to/model.onnx", provider="auto")
runtime.reset()

enhanced_0 = runtime.process_samples(chunk_0)
enhanced_1 = runtime.process_samples(chunk_1)
tail = runtime.flush()
```

`process_samples()` 接受任意 chunk 大小，輸出已完成的 overlap-add 部分。`flush()` 補零並
排空剩下的音訊。provider 可為 `auto`（CUDA、macOS 上再 CoreML，否則 CPU）、`cuda`、
`coreml`、`mps`（CoreML 的別名）與 `cpu`。DPCRN 命令列工具的 `infer` 與 `benchmark` 子指令
只讀 ONNX 檔與它的 manifest，所以也能跑 DPARN 的匯出：

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer model.onnx in.wav out.wav
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark model.onnx --seconds 10
```

做較底層的測試或自訂匯出流程時：

```python
from puresound.streaming import load_streaming_dparn_model

model = load_streaming_dparn_model("path/to/dparn_recipe.yaml", "path/to/model.ckpt")
state = model.initial_state(batch_size=1)
enhanced_frame, next_state = model.forward_frame(noisy_frame, state)
```

`noisy_frame` 與 `enhanced_frame` 的形狀是 `[batch, 257, 2]`，最後一軸是實部與虛部。

## 可攜式 SDK

[`sdk/python`](../../../sdk/python/README.zh-TW.md) 的獨立 SDK 以與 DPCRN 相同的方式載入
DPARN 匯出（`processor: stft_frame_ort`）：

```python
from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")
enhanced = runtime.process_samples(audio_float32)
tail = runtime.flush()
out_pcm = runtime.process_int16(in_pcm)     # 給即時框架用的 int16 輔助函式
tail_pcm = runtime.flush_int16()
```

## 附註

- ONNX 模型只處理特徵幀；音訊的 STFT 與 iSTFT 刻意放在 graph 之外。
- 學習出來的 STFT kernel 不在串流契約之內；請用上面固定 Hann 前端的設定訓練或微調。
- runtime 的波形串流只支援 batch size 1。
