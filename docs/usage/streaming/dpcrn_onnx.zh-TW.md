# DPCRN 串流 ONNX

English: [dpcrn_onnx.md](dpcrn_onnx.md)

DPCRN 串流推論使用逐幀的 ONNX 模型。Python（或可攜式 SDK）負責音訊緩衝、固定的 Hann
STFT、overlap-add iSTFT 與狀態張量；ONNX Runtime 一次只跑一個 DPCRN 特徵幀。這是每一顆
已發布 checkpoint 的部署路徑——人聲隔離（`egs/voice_isolate/pretrained_ckpt/streaming/`）
與噪音抑制（`egs/noise_suppression/pretrained_ckpt/streaming/`）都是。

DPCRN 與 DPARN 繼承同一個 `Unet` 基底，所以 encoder/decoder 的逐步推進、狀態佈局、
特徵/遮罩/iSTFT 前端、匯出與 manifest 都是共用的（`puresound/streaming/base.py`）。
不同的是 bottleneck：DPCRN 的 intra 路徑沿頻率軸運算，不帶跨幀狀態；沿時間軸的 inter
路徑是唯一狀態要跨幀保留的運算子。

## 匯出、驗證、執行

`egs/voice_isolate/scripts/streaming_onnx.py` 替任何一個任務的 DPCRN checkpoint 驅動
函式庫：

```bash
# 把 checkpoint 匯出成逐幀 ONNX + JSON manifest
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_curriculum_v1.ckpt /path/to/model.onnx

# 離線與 ORT 串流的一致性（依延遲對齊、裁掉邊緣）
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_curriculum_v1.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_curriculum_v1.onnx \
    --input_audio speech.wav

# 檔案對檔案的串流推論，以及串流的 real-time factor
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer <onnx> in.wav out.wav
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark <onnx> --seconds 10
```

| 子指令 | 值得知道的旗標 |
| --- | --- |
| `export` | `--manifest_path`（預設放在 ONNX 旁邊）、`--opset`、`--dry-blend`（預設 0.9，人聲隔離的發布設定）、`--onset-guard` 搭配 `--guard-t-arm`、`--guard-t-forget`、`--guard-tau-dn`、`--guard-margin-db` |
| `verify` | `--input_audio`（預設：合成噪音）、`--seconds`、`--trim`、`--provider`、`--no-onset-guard` |
| `infer` | `--manifest_path`、`--provider`、`--no-onset-guard` |
| `benchmark` | `--seconds`、`--provider` |

**噪音抑制的 checkpoint 要用 `--dry-blend 1.0` 匯出**，那是它們的紀錄評分時的操作點；
預設的 0.9 會出貨一個閘門從沒量過的系統：

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/noise_suppression/config/infer_dpcrn.yaml \
    egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \
    /path/to/dpcrn_mamba_v2.onnx --dry-blend 1.0
```

判斷新的匯出要用真實語音（`verify --input_audio`）。預設的白噪音探針是壓力訊號：它的
一致性數字反映的是某版本調變遮罩有多激進，而不是串流 graph 是否正確。

## DPCRN 與 DPCRN-Mamba 的 CPU 優化

匯出時加上 `--optimization portable`，固定 batch=1，讓兩個 recurrent block
沿用 frequency-major 排列，整張圖仍使用標準 ONNX 運算子。`cpu` 會再匯出旁邊的
`model.native.onnx`，使用融合 FP32 Mamba state update／readout 的運算子。
兩條路徑沿用 checkpoint 與 state ports，不增加時間緩衝。標準主圖支援 intra LSTM
搭配單層 inter LSTM、`mamba` 或 `mamba_context`。單幀 inter LSTM 將所有頻率位置
一起計算，合併 input／hidden projection 並使用標準 ONNX gate 運算；intra 的
雙向頻率 recurrence 沿用原本訓練公式。native companion 選項適用於 Mamba inter。

在 Linux 上編譯 library，再匯出：

```bash
uv run python -m puresound.streaming.native.build \
    --output /path/to/libpuresound_ssm.so
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/noise_suppression/config/infer_dpcrn_mamba_wide.yaml \
    egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v3.ckpt \
    /path/to/v3.onnx --dry-blend 1.0 --optimization cpu \
    --native-library /path/to/libpuresound_ssm.so
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark \
    /path/to/v3.onnx --provider cpu --native-ssm required \
    --native-library /path/to/libpuresound_ssm.so \
    --input_audio puresound/web/static/samples/noisy-speech.wav \
    --warmup 2 --repeats 3
```

編譯需要 C++17 compiler（`g++` 或 `CXX`），不下載依賴。產物以 x86-64 基線指令集編譯，
載入時才依 CPU 選用 AVX2 或 AVX-512 kernel，同一份 build 可用於任何 C library 相容的
x86-64 Linux 主機。重新編譯以原子替換寫入，執行中的 session 仍使用原本載入的檔案。融合 kernel 會重排 FP32 加總，因此允許微小數值差異。
主 ONNX 不需要 compiler 或 custom library；瀏覽器／WASM 使用主圖，manifest
透過既有 web asset builder 準備。

```python
from puresound.streaming import StreamingOrt

runtime = StreamingOrt("/path/to/v3.onnx", provider="cpu",
                       native_library="/path/to/libpuresound_ssm.so",
                       native_ssm="required")
print(runtime.execution_path, runtime.native_ssm_enabled)
```

`native_ssm="auto"`（預設）在 CPU session 指定 library 後選用 native companion，
載入失敗便回退到主 ONNX；`off` 一律使用主圖，`required` 則在無法使用 native 時拋錯。
也可透過 `PURESOUND_ORT_SSM_LIBRARY` 指定 library，供 Python facade／web server 使用。
推論不會觸發編譯。部署時一起放置主 ONNX、JSON 與旁邊的 native ONNX；manifest
不記錄機器專屬的 library 絕對路徑。CPU 優化 manifest 建議單執行緒，可用
`intra_op_num_threads` 或 CLI `--threads` 覆寫。

benchmark 會列出實際 graph、native 狀態、執行緒數、中位 RTF 與各輪結果。
計時包含 STFT、模型、overlap-add 與 flush，排除模型載入與暖機。v3 的算法延遲仍為
52–62 ms。model zoo 的六個 DPCRN 串流發布模型都預設使用標準 ONNX 優化圖，三個 NS
Mamba 模型另外附有 native companion。`load_model` 未指定 library 時使用主圖，
透過 `PURESOUND_ORT_SSM_LIBRARY` 指向本機編譯產物後，CPU session 可選用 native。

重新匯出並驗證整個 catalog，同步更新 SHA256：

```bash
uv run python model_zoo/reexport.py --native-library /path/to/libpuresound_ssm.so
uv run python sdk/web/tools/build_assets.py
```

### int8 量化

`export --quantize int8`（需搭配 `--optimization portable` 或 `cpu`）會把主圖與 native
companion 中 LSTM 與矩陣乘法的權重存成 int8，activation 在執行時才量化。匯出仍先嚴格驗證
浮點圖；int8 圖只檢查輸出有限且在合理範圍內，量到的誤差寫進 manifest 的 `quantization`。
這個範圍不代表品質：int8 輸出會隨圖或 kernel 的任何變動而改變，所以每次 int8 匯出都要
重跑閘門。NS v3 以 `flash` variant 發布（`--variant flash`）。

## 支援的設定

`validate_streaming_dpcrn_config` 在匯出前檢查 recipe，任何不符都丟 `ValueError`：

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` encoder，Hann 窗、`sr: 16000`、`fmax: 8000`、`trainable: False`、
  `win_length <= fft_length`、`hop_length > 0`
- `features.feats_type: complex`、`drop_stft_first_bin: True`、`trainable: False`，
  不用 `include_specaug`
- `DPCRN` backbone，`input_dim == fft_length // 2`，`norm_type` 為 `cLN` / `iLN` /
  `bN2d` 之一，`skip_conv: False`，每個 down layer 都是 `stride_t: 1` 與
  `dilation_t: 1`。允許 `bN2d`，是因為 `BatchNorm2d` 在 `eval()` 下對每個（頻率、時間）
  位置套用固定的 running statistics，不帶跨幀狀態。

逐幀模型另外要求複數遮罩。它支援 inter 路徑為 LSTM 或 Mamba 區塊（`inter_type: mamba`
或 `mamba_context`）、bottleneck 的感知分帶、attention intra 路徑，以及 deep-filter
殘差頭。它拒絕 `inter_type: lstm+mamba`——那個並聯分支需要 manifest 佈局裡沒有的狀態
埠——而不是只匯出 LSTM 分支。

有兩種情況只會警告：非零的 `delay`（look-ahead，有支援，見下），以及
`transpose_delay: True`，它讓 decoder 用到未來的幀而破壞串流一致性。請維持
`transpose_delay: False`。

## Look-ahead：future-buffering 怎麼運作

**因果（`delay=[0,0,0]`）**：逐幀 graph 與離線模型一致，**零**額外延遲——除了 down/up
卷積快取與 inter 路徑的狀態，沒有別的狀態。

**Look-ahead（某些 down layer 的 `delay > 0`，例如 `[1,1,1]`）**：離線 DPCRN 透過右側
補零讓每個 down layer 看到 `delay[i]` 個未來幀。因果的逐幀模型不能往前看，所以它把**自己
的輸出延後**同樣的幀數，並把離線路徑「在未來」看到的東西存成額外狀態。串流輸出就是延後了
的離線結果。

`DpcrnStreamingState` 在因果的 `down_caches` / `up_caches` / `h_states` /
`c_states` 之外，用三塊額外狀態實現這件事：

- **`skip_caches`**——需要的 U-Net skip 連接各有一條 FIFO。第 `k` 個 down layer 的累積
  look-ahead 是 `cum[k] = sum(delay[:k+1])`，主路徑最後延後 `D = cum[-1]` 幀。從第 `k`
  層分出來的 skip 在分岔處只延後了 `cum[k]` 幀，所以它要先經過額外 `D - cum[k]` 幀的
  FIFO 才和主路徑相會；否則兩者指的是不同的離線幀。`D - cum[k] == 0` 的層不需要
  （`skip_delay_layers` 列出需要的層）。
- **`noisy_cache`**——遮罩所作用的 noisy 頻譜的 `D` 幀延遲線，讓（晚了 `D` 幀才出來的）
  遮罩與它相乘的頻譜指的是同一幀。
- **`counter`**——一個幀索引，在最初 `D` 幀期間閘住 inter 路徑的狀態
  （`warmup_frames = D`）。因果 down 路徑最初 `D` 個輸出幀是沒有離線對應的啟動暫態，所以
  模型在管線把它們排空前把下一個 inter 狀態歸零；少了這道閘，第一次真正的更新會從被污染的
  狀態開始，毀掉整條串流。

有 deep-filter 殘差頭時，`df_cache` 保存濾波器要讀的、已對齊的 noisy 低頻帶最近幾幀。

演算法延遲是 `model.streaming_delay`（等於 `model.bottleneck_delay`）幀，記在 manifest
的 `streaming_delay_frames`：已發布的 `[1,1,1]` look-ahead 是 3 幀，hop 160 下為 30 ms。
**任何離線與串流的比較都必須先依這個幀數重新對齊、並裁掉兩端**，否則延遲本身會被讀成誤差；
`verify` 兩件事都會做。`test/streaming/test_dpcrn_streaming.py` 收著因果、look-ahead、
Mamba、分帶、attention 與 deep-filter 各變體的一致性測試。

這個數字是 graph 加上的 look-ahead。一個樣本還得等它周圍的分析窗到齊，它所在的 hop 才會
輸出，所以逐幀 runtime 的輸入到輸出延遲介於 look-ahead + 窗長 − hop 到 look-ahead + 窗長
之間：已發布的幾何（30 ms look-ahead、32 ms 窗、10 ms hop）是 52–62 ms。裝置預算請以上限
為準。

## ORT runtime

`StreamingDpcrnOrt` 就是共用的 `puresound.streaming.StreamingOrt`：runtime 完全由
manifest 驅動（`processor: stft_frame_ort`，每個狀態張量都依名稱讀回），所以一個類別就
服務兩種 backbone 與每一種 inter 路徑。

```python
from puresound.streaming import StreamingOrt

runtime = StreamingOrt("model.onnx", provider="auto")   # manifest：旁邊的 model.json
runtime.reset()
out = runtime.process_samples(chunk)      # 任意 chunk 大小
tail = runtime.flush()
```

- **`reset(batch_size=1)`** 依 `manifest["state_shapes"]` 把每個狀態張量填零，並清空輸入
  緩衝與 overlap-add 累加器。波形串流只支援 batch size 1。
- **`process_samples(samples)`** 把樣本接到輸入緩衝後面，只要有完整的一個窗，就取一個
  STFT 幀、連同狀態送進 session、存下新狀態、對該幀做 iSTFT 並 overlap-add，每消耗一幀
  輸出一段 `hop_length`。任意 chunk 大小的輸出都與一次呼叫相同。非有限值的輸入樣本視為靜音：
  狀態是遞迴的，一個 NaN 會讓之後每個樣本都壞掉，直到 `reset`。
- **`flush()`** 持續餵零幀直到模型的 look-ahead 排空，串流總長恰為
  `輸入長度 + streaming_delay_frames * hop_length`：完整的輸入、晚這麼多樣本。之後
  overlap-add 只有部分覆蓋的剩餘部分對應到補零，直接丟棄。
- `provider` 可以是 `auto`（CUDA、再 CoreML、再 CPU）、`cpu`、`cuda`、`coreml`，或
  `mps`（CoreML 的別名）。

## runtime 套用的 graph 後處理

追蹤出來的 graph 只含模型，後面什麼都沒有。所以兩個會形塑部署輸出的階段是**記在 manifest
裡、由 runtime 套用**的，讓只跑 graph 的部署跑的就是被評測過的系統。沒有該鍵就表示該階段
關閉。

### `recommended_inference`：dry blend

```json
"recommended_inference": {
  "dry_blend": 0.9,
  "suppression_ceiling_db": -20.0,
  "note": "out = 0.9*enhanced + 0.1*input, with the input latency-aligned ..."
}
```

`out = dry_blend * enhanced + (1 - dry_blend) * input`，輸入延後 `streaming_delay_frames`
讓兩者指同一幀。它不增加延遲，並把衰減上限封在 `20*log10(1 - dry_blend)`——0.9 時是
−20 dB——所以一個回報在這附近的殘留量到的是 blend，不是模型。
`StreamingOrt(..., postprocess_overrides={...})` 可以逐請求更改它。

### `onset_guard`：保護講話者的第一秒

```json
"onset_guard": {
  "t_arm_s": 1.0, "t_forget_s": 5.0, "tau_up_s": 0.05, "tau_dn_s": 2.0,
  "margin_db": 8.0, "floor_win_s": 2.0, "floor_rise_db_per_s": 3.0,
  "init_s": 0.2, "hangover_s": 0.2, "min_run_s": 0.1, "snap": 0.001,
  "note": "out = input, bit for bit, until a talker has been heard for 1 s ..."
}
```

由 `export --onset-guard` 寫入。串流狀態會把最後聽到的人當成前景，所以第一位講話者的第一
秒——以及停頓後下一位近講者的第一秒——可能被當成前景更換而遭衰減。在講話者被聽到持續
`t_arm_s` 的語音（高於追蹤中的噪音底 `margin_db`）之前，輸出是
`g*input + (1 - g)*enhanced` 且 `g = 1`，也就是逐位元的輸入；之後以時間常數 `tau_dn_s`
釋放給模型，安靜 `t_forget_s` 後重新上膛，保護下一次起音。它用第一秒的壓制深度換取更少的
漏字；要停在這個取捨的哪裡是產品決策（`puresound/system/onset_guard.py`）。

執行時每個 hop 的成本是一個幀能量、一個移動最小值的底噪追蹤與一步一階積分器：沒有 FFT、
沒有模型狀態、沒有任何學習出來的東西。

**一個 hop 的延遲規則。** 幀能量橫跨兩個 hop，所以 guard 的第 `t` 幀要等輸入 hop `t + 1`
到達才決定，而輸出 hop `m` 需要輸入幀 `m - streaming_delay_frames` 的增益。輸出 hop `m`
時 runtime 已收到 `m*hop_length + win_length` 個樣本，所以只要

```
streaming_delay_frames + win_length//hop_length - 2 >= 0
```

套用的增益就**正好**是離線 guard 算出的那一個，而且不增加延遲；runtime 以
`runtime.onset_guard_lookahead_hops` 暴露這個值。在 512/160 的幾何下，look-ahead 與因果的
匯出都成立——光是分析窗就涵蓋了這個延遲。窗長短於兩個 hop 又是零延遲 graph 時，會以
`ValueError` 拒絕，而不是給出錯的那一幀的增益。

guard 套用在 dry blend **之後**，因為它必須能還原整個輸入，而不只是 `1 - dry_blend` 的
輸入——這也是 `SISO.forward` 離線時的順序。請求可以拒用 manifest 記錄的 guard
（`StreamingOrt(..., onset_guard_overrides={"enabled": False})`，或 `infer` / `verify`
的 `--no-onset-guard`），也可以替換個別旋鈕。`verify` 會對它的離線參考套用同一個 guard，
所以它的一致性數字仍是 graph 自己的。

## 輔助頭

帶 `vad_head` 與/或 `background_vad_head` 的 backbone 會把它們匯出成旁路輸出
（`vad_logit`、`background_vad_logit`），列在 manifest 的 `extra_output_names`；每個頭各自
增加狀態埠。啟用它們不會改變音訊。logit 描述的是計算它的那個 bottleneck 幀，它**領先**輸出
音訊 `streaming_delay_frames`；要把它對齊到輸出的使用者必須自己算進這一段。
`StreamingOrt(..., collect_extras=True)` 會保留它們，`drain_extras()` 回傳並清空，讓長串流
維持有界。recipe `egs/voice_isolate/config/infer_dpcrn_heads.yaml` 用來載入帶這些頭的
checkpoint。

`dist_head` 刻意不匯出：它是整句 pooling 的，沒有逐幀的意義，仍只是訓練用的輔助項。

## 函式庫 API

```python
from puresound.streaming import (
    StreamingDpcrnOrt,              # 共用的 ORT runtime
    export_streaming_dpcrn_onnx,    # config + ckpt -> onnx + manifest
    load_streaming_dpcrn_model,     # torch 逐幀模型（StreamingDpcrnFrameModel）
    validate_streaming_dpcrn_config,
)
```

`export_streaming_dpcrn_onnx(config, ckpt, onnx, manifest=None, opset_version=17,
postprocess=IDENTITY, onset_guard=None)` 用 TorchScript exporter 追蹤逐幀模型，拿 ONNX 的
輸出（以及每個旁路輸出）對照 PyTorch 逐幀模型，並寫出 manifest：取樣率、FFT/窗/hop、輸入
輸出與狀態名稱、狀態形狀、`streaming_delay_frames`、`extra_output_names`、偏好的
provider，以及上述的 graph 後處理區段。

## 可攜式 SDK

只需要推論的部署使用 [`sdk/python`](../../../sdk/python/README.zh-TW.md) 裡的獨立 SDK，
它只依賴 NumPy 與 ONNX Runtime，可直接載入這些匯出（`processor: stft_frame_ort`）：

```python
from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")
enhanced = runtime.process_samples(audio_float32)
tail = runtime.flush()
```

它帶有 dry blend 與 onset guard 的 NumPy 版本；`test/streaming/test_sdk_runtime.py`
把它與函式庫的 runtime 釘成逐位元相同。

## 附註

- ONNX 模型只處理特徵幀；音訊的 STFT 與 iSTFT 刻意放在 graph 之外。
- runtime 的波形串流只支援 batch size 1。
