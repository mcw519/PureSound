# DPARN Streaming ONNX Runtime

English version: [dparn_onnx.md](dparn_onnx.md)

> **Status: legacy** — 維持可運作但已凍結：不新增功能、不重寫。

DPARN 的 streaming 推論用的是一個逐 feature frame 的 ONNX 模型。Python 端
負責音訊緩衝、固定的 Hann STFT、overlap-add iSTFT，以及 ONNX Runtime 狀態
管理。ONNX Runtime 每次只跑一個 DPARN feature frame。

這條路徑是設計給設定成 16 kHz 音訊的 voice-isolate DPARN recipe 用的。

## Supported Configuration

`validate_streaming_dparn_config` 會在 export 前檢查 recipe。大多數檢查都是
**硬性規定**——只要違反就會立刻丟出 `ValueError`：

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` 前端，用 Hann window、`encoder_args.sr: 16000`、
  `fmax: 8000`、`win_length <= fft_length`、`hop_length > 0`
- `encoder_args.trainable: False`
- `features.feats_type: complex`、`drop_stft_first_bin: True`、
  `features.trainable: False`、不能開 `include_specaug`
- `DPARN` backbone，`input_dim: 256`
- `norm_type` 只能是 `cLN` / `iLN` / `bN2d` 其中之一（`BatchNorm2d` 在
  `eval()` 模式下套用的是固定的 running stats，因此與 frame 無關）
- `skip_conv: False`
- 每個 down layer 的 `stride_t: 1` 且 `dilation_t: 1`

還有兩項檢查是**軟性的**——只會發出 `warnings.warn`，export 依然會成功：

- `transpose_delay` 不是 `True`——會警告匯出的 ONNX 雖然是逐 frame，但並非
  真正 causal（實際音訊 streaming 時會洩漏未來的 frame）
- `delay` 有任何非零值——會警告 down layer 會偷看未來的 frame

跟 [DPCRN streaming 路徑](dpcrn_onnx.zh-TW.md)不同，**DPARN 的 exporter
完全沒有 look-ahead 補償機制**：它不會緩衝未來的 frame，也不會在暖機期間
gate RNN 狀態。一個非 causal 的 `delay`/`transpose_delay` 設定依然可以
export 成功（附帶警告），也依然能通過 export 當下做的 offline-vs-ONNX
比對——但產出的模型並不是真正的 streaming，因為那項檢查只驗證了整段
utterance（offline）層級的比對，並沒有驗證在只看得到過去的實際 streaming
緩衝下的行為。

**repo 裡的 DPARN recipe 不是 streaming recipe。** 唯一一份真實的 DPARN
config，`egs/noise_suppression/config/dparn.yaml`，設的是
`encoder_args.trainable: True` 並加了一個 `freq_eq` 區塊——它按設計就是一份
offline 的 enhancement recipe，而一個學習出來的前端沒辦法被 export 成這個
runtime 依賴的固定 STFT（Python 端負責 STFT/iSTFT；ORT 只跑逐 frame 的
backbone）。export 它會卡在 `encoder.trainable` 這項檢查上，那是檢查照設計
正常運作，不是缺陷。要把一個 DPARN checkpoint export 成 streaming，請用
`encoder_args.trainable: False` 加固定的 Hann 前端去訓練；它的
`norm_type: bN2d` 維持原樣就可以。

## Export

把訓練好的 checkpoint export 成 streaming 的 feature-frame ONNX 模型：

```bash
uv run python <your_recipe>/streaming_onnx.py export \
  <your_recipe>/config/dparn.yaml \
  /path/to/model.ckpt \
  /path/to/model.onnx
```

Exporter 也會寫出 `/path/to/model.json`。這份 manifest 包含：

- 音訊設定：sample rate、FFT length、window length、hop length
- ONNX 的 input/output 名稱
- 明確的 streaming 狀態 tensor 名稱與 shape
- 偏好的 ONNX Runtime providers
- DPARN 的 streaming delay（幾個 frame）
- 由 RUNTIME 套用的 graph 後階段（因為 graph 只含模型本身、之後什麼都沒
  有）：`recommended_inference`（`dry_blend`），以及——如果 export 時有給
  ——`onset_guard`。這兩個 key 都由共用的 `StreamingOrt` 與 portable SDK
  讀取，所以帶著它們的 DPARN manifest 行為一致；沒有那個 key 就等於該階段
  關閉。這支 legacy exporter 本身不會寫出 `onset_guard`（寫的是
  `export_streaming_dpcrn_onnx`）——機制、成本與那條「差一個 hop 的分析延
  遲」規則只記錄一次，在
  [dpcrn_onnx.zh-TW.md](dpcrn_onnx.zh-TW.md#graph-之後的第二個階段onset-guardonset_guard)。

Export 過程中，PureSound 會拿 ONNX 的 frame 輸出跟 PyTorch 的 frame
wrapper 輸出做比對。誤差超過容忍範圍就會讓 export 失敗。

## Inference

對一個音訊檔跑 streaming 推論：

```bash
uv run python <your_recipe>/streaming_onnx.py infer \
  /path/to/model.onnx \
  input.wav \
  output.wav \
  --provider auto
```

Provider 選項：

- `auto`：優先使用 CUDA；在 macOS 再嘗試 CoreML，否則使用 CPU
- `cuda`：要求用 CUDA，不行就退回 CPU
- `coreml`：要求 Apple 的 CoreML execution provider（macOS）
- `mps`：`coreml` 的 alias；ONNX Runtime 沒有原生 MPS provider
- `cpu`：強制用 CPU

Benchmark 即時率（realtime factor）：

```bash
uv run python <your_recipe>/streaming_onnx.py benchmark \
  /path/to/model.onnx \
  --provider cuda \
  --seconds 10
```

## Library API

```python
from puresound.streaming import StreamingDparnOrt

runtime = StreamingDparnOrt("/path/to/model.onnx", provider="auto")
runtime.reset()

enhanced_0 = runtime.process_samples(chunk_0)
enhanced_1 = runtime.process_samples(chunk_1)
tail = runtime.flush()
```

`process_samples()` 可以接受任意大小的區塊，並吐出目前所有已經完成的
overlap-add 輸出。`flush()` 會補零並清空剩下的音訊。

若要做更底層的測試或自訂 export 流程：

```python
from puresound.streaming import load_streaming_dparn_model

model = load_streaming_dparn_model(
    "<your_recipe>/config/dparn.yaml",
    "/path/to/model.ckpt",
)
state = model.initial_state(batch_size=1)
enhanced_frame, next_state = model.forward_frame(noisy_frame, state)
```

`noisy_frame` 與 `enhanced_frame` 的 shape 都是 `[batch, 257, 2]`，最後一軸
是 real 與 imaginary 部分。

## Portable SDK

只需要做推論的專案不必安裝完整的 PureSound 訓練套件。可以用
[sdk/python](../../sdk/python) 裡獨立的 SDK：

```bash
python -m pip install sdk/python
```

接著載入 export 出來的產物：

```python
from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")
enhanced = runtime.process_samples(audio_float32)
tail = runtime.flush()
```

對於 LiveKit agent 這類即時系統，可以用 int16 的輔助函式：

```python
out_pcm = runtime.process_int16(in_pcm)
tail_pcm = runtime.flush_int16()
```

這個 SDK 只依賴 NumPy 與 ONNX Runtime，不會 import `puresound`、PyTorch、
torchaudio、Lightning，或整套訓練用的 recipe stack。DPARN 是第一個使用
SDK `stft_frame_ort` processor profile 的 export。之後 PureSound 新增的
streaming export 應該優先新增或重用 manifest-driven 的 processor
profile，而不是另外造一個 model-specific 的 runtime class。

## Gradio Demo

voice-isolate 的 demo 支援兩種 backend：

- `PyTorch offline`：載入 `.ckpt`、`.pt` 或 `.pth` checkpoint，跑原本的
  offline 模型路徑。
- `ORT streaming`：選一個 `.onnx` 模型，跑 streaming 的 ONNX Runtime 路徑。

啟動 demo：

```bash
uv run python <your_recipe>/demo.py \
  --config_path <your_recipe>/config/dparn.yaml
```

用 **Refresh checkpoints** 掃描設定好的 `work_folder` 與 `exp` 目錄，接著
選 `ORT streaming` backend，再挑一個 export 出來的 `.onnx` 模型。

## Notes

- ONNX 模型只涵蓋 feature-frame 這一段；音訊的 STFT 與 iSTFT 刻意留在
  graph 之外。
- 現有的 checkpoint 應該要用上面那組固定的 Hann 前端設定去訓練或微調。
  可學習的 STFT kernel 不在 v1 streaming 合約的涵蓋範圍內。
- 目前這個 runtime 只支援 batch size 1 的 waveform streaming。
