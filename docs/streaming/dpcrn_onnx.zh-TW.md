# DPCRN Streaming ONNX Runtime

English version: [dpcrn_onnx.md](dpcrn_onnx.md)

DPCRN 的 streaming 推論用的是一個逐 frame 的 ONNX 模型。Python 端負責音訊
緩衝、固定的 Hann STFT、overlap-add iSTFT，以及 ONNX Runtime 狀態管理；
ONNX Runtime 每次只跑一個 DPCRN feature frame。這是已發佈的 voice-isolate
checkpoint 所用的部署路徑（見
`egs/voice_isolate/pretrained_ckpt/streaming/`）——**是 production 路徑**，
不是 legacy，所以這篇文件寫得比對應的 DPARN 那篇更深入。

`puresound/streaming/dpcrn.py` 自己的 module docstring 就把這兩個 backbone
的關係講得很清楚：DPCRN 跟 DPARN 都繼承自同一個 `Unet` base，所以
encoder/decoder 的逐步驟邏輯、狀態排列、feature/mask/iSTFT 前端、export，
以及 manifest，在這兩個 streaming module 之間都是一樣的。**唯一**的結構性
差異在 bottleneck block：DPCRN 的 intra path 是沿頻率軸的 bidirectional
LSTM（跟時間無關，不帶跨 frame 的狀態），而 DPARN 用的是 self-attention。
Inter path（沿時間軸的單向 LSTM）兩者都是同一個 `SingleRNN`，也是唯一一個
`(h, c)` 狀態必須跨 frame 保留的 operator。

## Supported Configuration

`validate_streaming_dpcrn_config` 會在 export 前檢查 recipe，只要不符合下面
任何一項就會直接以 `ValueError` 失敗：

- `dataset.target_sample_rate: 16000`、`ConvEncDec` 前端（`sr: 16000`、
  Hann window、`win_length <= fft_length`）、凍結的 encoder/features
- `features.feats_type: complex`、`drop_stft_first_bin: True`
- DPCRN backbone，每個 down layer 都要 `stride_t: 1` 且 `dilation_t: 1`；
  `norm_type` 只能是 `cLN` / `iLN` / `bN2d` 其中之一（這裡允許 `bN2d`——
  跟 [DPARN 的 validator](dparn_onnx.zh-TW.md) 不同——因為 `BatchNorm2d`
  在 `eval()` 模式下是用固定的 running stats、依 `(freq, time)` 位置套用，
  因此不帶跨 frame 的狀態）

`transpose_delay` 必須維持 `False`：`transpose_delay=True` 會讓 decoder
變成 anti-causal（需要用到未來的 frame），破壞 streaming 的一致性。這項
檢查只會發出 `warnings.warn`，不是硬性失敗——自己寫 recipe 時務必再三確認。

## Look-ahead: how future-buffering actually works

**Causal（`delay=[0,0,0]`）**：逐 frame 的 graph 以**零**額外延遲做到
bit-exact streaming——除了 down/up conv 的 cache 跟 inter-LSTM 的
`(h, c)` 之外，不需要任何額外狀態。

**Look-ahead（某些 down layer 的 `delay > 0`，例如已發佈版本用的
`[1,1,1]`）**：offline 版的 DPCRN 讓每個 down layer 透過向右 padding 偷看
`delay[i]` 個未來的 frame。逐 frame 的 causal 模型沒辦法往前偷看，所以改成
**把自己的輸出延後**同樣多個 frame，把 offline 路徑「看到的未來」用額外的
持久狀態緩衝起來。這跟 offline DPCRN 是 bit-exact 的——只是延後了——
`test/test_utils/test_dpcrn_streaming.py` 裡的整段 utterance 一致性測試
驗證了這點（streaming 輸出對齊 bottleneck delay 之後，相對誤差 < 1e-3）。

除了 causal 情境下就有的 `down_caches` / `up_caches` / `h_states` /
`c_states`，`DpcrnStreamingState` 上還多追蹤了三塊額外狀態來實現這件事：

- **`skip_caches`** —— 給每個需要它的 U-Net skip connection 各一條 FIFO
  延遲線。Down layer `k` 累積的 look-ahead 是 `cum[k] =
  sum(delay[:k+1])`；bottleneck（連帶整條主路徑）因此延後了
  `D = cum[-1]` 個 frame。從 down layer `k` 分支出去的 skip，在分支當下
  *自己*只延後了 `cum[k]` 個 frame，所以必須再送過一條額外
  `D - cum[k]` frame 長的 FIFO，才能在對應的 up layer 跟主路徑拼接——否則
  skip 跟主路徑指的會是 offline 時間軸上不同的時間點。`D - cum[k] == 0`
  的 skip layer 完全不需要延遲線（`skip_delay_layers` 只列出真正需要的
  那些）。
- **`noisy_cache`** —— 給要套 mask 的 noisy 頻譜
  （`features_for_enhanced`）用的一條 `D` frame 延遲線，這樣 mask（本身
  出來時就已經延後了 `D` 個 frame）跟它要相乘的頻譜，指的才是 offline
  時間軸上同一個 frame。
- **`counter`** —— 一個 frame 索引，唯一的用途是在 stream 開始的前 `D`
  個 frame（`warmup_frames = D`）期間 gate 住 inter-LSTM 的狀態：causal
  down-path 輸出的前 `D` 個 frame 是沒有 offline 對應版本的暫態起始假
  frame（畢竟這時候還沒有「過去」可以偷看），所以模型會把
  `next_h`/`next_c` 歸零（`keep = counter >= D`），直到這段 pipeline
  把它們沖刷完為止。少了這個 gate，inter-LSTM 第一次真正的更新就會是從
  一個被污染的狀態開始，而不是乾淨的狀態，殃及的不只是開頭那段暫態，
  而是整個 stream。

演算法延遲是 `model.streaming_delay`（等於 `model.bottleneck_delay`）個
frame——對已發佈的 checkpoint 而言是 `3` 個 frame（hop 160 下約
30 ms）。**任何 offline↔streaming 的比對，都必須先按這個 frame 數對齊、
兩端都裁掉**，否則這段延遲本身就會被誤判成誤差。下方的 `verify` CLI 指令
會自動做這件事；對齊搜尋的邏輯可參考
`test/test_utils/test_dpcrn_streaming.py` 裡的 `_offline_vs_streaming_rel`
（`test_dpcrn_streaming_matches_offline_for_lookahead_model` 會 assert
搜出來最匹配的 delay 剛好等於 `model.bottleneck_delay`）。

如果想直接檢視這兩種形態，repo 裡有兩份現成可用的 recipe：
`egs/voice_isolate/config/exp/train_dpcrn_wide_causal.yaml`
（`delay=[0,0,0]`）與
`egs/voice_isolate/config/exp/train_dpcrn_wide_antisup.yaml`
（`delay=[1,1,1]`，look-ahead）——streaming 測試套件自己載入的也正是這
兩份 config。

## ORT streaming state handling

`StreamingDpcrnOrt` 並不是另外獨立的實作——它就是
`puresound.streaming.dparn.StreamingDparnOrt`、只是用 DPCRN 的名字重新
export 出來而已（`from puresound.streaming.dparn import
StreamingDparnOrt as StreamingDpcrnOrt`）。這個 runtime 完全是
manifest-driven 的（由 `manifest["processor"] == "stft_frame_ort"` 決定
分派，每個狀態 tensor 都是照名字從 JSON 讀回來的），所以同一個 ORT class
原封不動同時服務兩種 backbone——runtime 裡完全沒有任何 DPCRN 專屬的東西。

這個 runtime 每次呼叫實際做的事：

- **`reset(batch_size=1)`** —— 依 `manifest["state_shapes"]` 把每個狀態
  tensor 清成全零，並清空內部的 `input_buffer`，以及 `ola`/`ola_norm`
  這組 overlap-add 累加器。waveform streaming 只支援
  `batch_size=1`（其他值會丟出 `ValueError`）。
- **`process_samples(samples)`** —— 把 `samples` 接到 `input_buffer`
  後面，接著只要湊到一個完整 `win_length` 長度的 window，就：取一個 STFT
  frame，連同完整的狀態 dict 一起丟進 ONNX session
  （`session.run(...)`）跑，用輸出覆寫 `self.state`，把增強後的 frame
  做 iSTFT，再 overlap-add 進 `ola`/`ola_norm`（用累積的 window 平方能量
  做 normalize，下限 `1e-8`），每消耗一個輸入 frame 就吐出剛好一個
  `hop_length` 長度的區塊。輸入區塊大小可以任意——
  `test_streaming_ort_is_chunk_invariant_with_identity_session` 驗證了
  分成 3 段任意大小餵進去，跟一次全部餵進去，輸出完全一致。
- **`flush()`** —— 把 `input_buffer` 裡剩下的部分補零並清空，加上
  `ola` 累加器裡剩下的尾段（用同樣的方式 normalize）。

## Export / Verify / Run

```bash
# 把 checkpoint export 成逐 frame 的 ONNX + JSON manifest
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt /path/to/model.onnx

# offline 跟 ORT streaming 的一致性比對（已對齊、已裁邊）
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.json

# 檔案對檔案的 streaming 推論 / RTF benchmark
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer  <onnx> in.wav out.wav
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark <onnx>
```

## Library API

```python
from puresound.streaming import (
    StreamingDpcrnOrt,              # ORT runtime：process_samples()/flush()
    export_streaming_dpcrn_onnx,    # config + ckpt -> onnx + manifest
    load_streaming_dpcrn_model,     # torch 逐 frame 模型（StreamingDpcrnFrameModel）
    validate_streaming_dpcrn_config,
)
```

JSON manifest 裡記錄了 fft/hop、狀態名稱/shape、演算法延遲
（`streaming_delay_frames`），以及——對已發佈的 checkpoint 而言——一個
`recommended_inference` block。舉例來說，已發佈的 `dpcrn_v8.json` 裡這個
block 長這樣：

```json
"recommended_inference": {
  "dry_blend": 0.9,
  "note": "out = 0.9*enhanced + 0.1*input, with the input latency-aligned to the enhanced stream (algorithmic latency 3 frames / 30 ms). Bounds attenuation at any point to -20 dB, which trades a little residual interferer for far fewer deletions on capture chains the model was not trained on."
}
```

`dry_blend` 是由*呼叫端*套用的，並沒有烤進匯出的 graph 裡：在推論時把
增強後的 frame 跟已做過延遲對齊的輸入 frame 混合（不會增加額外延遲，因為
增強後的 frame 出來時，對齊過的輸入早就已經拿得到了）。這是一個部署時期
的安全開關，跟上面講的 look-ahead 狀態沒有關係。

Manifest-driven 的 portable SDK（`sdk/python/puresound_streaming`、
`processor: stft_frame_ort`）會直接載入這些 export 出來的產物。

## Gap: `vad_head` / `dist_head` are silently dropped

`DPCRN`（`puresound/nnet/dpcrn.py`）在 bottleneck 之外支援兩個可選的輔助
head，各自由自己的 config block 控管（`backbone_args.vad_head.enabled` /
`backbone_args.dist_head.enabled`）：`self.vad_head`/`self.dist_head`，
它們會在 offline `forward()` 呼叫的副作用裡填入
`self.last_vad_logits`/`self.last_dist_preds`。目前 production 用的
recipe `egs/voice_isolate/config/train_dpcrn.yaml` 就有開
`dist_head.enabled: True`（一個 utterance 層級的
distance/DRR regression 輔助任務，用 `DistHeadRegressionLoss` 訓練）——
而且那份 config 自己的註解就已經點出這一節要記錄的這個落差：
*「Training-only: inference and the streaming export never read it.」*

`StreamingDpcrnFrameModel._forward_feature_frame`（以及它在 DPARN 那邊的
對應版本）完全不會呼叫 offline 的 `backbone.forward()`——它是直接對著
`backbone.cnn_down` / DPRNN blocks / `backbone.cnn_up` 逐 frame 重新實作
down/bottleneck/up 這條路徑，到此為止。`puresound/streaming/dpcrn.py`
裡完全沒有任何程式路徑會去用到 `vad_head`、`dist_head`、
`last_vad_logits`，或 `last_dist_preds`。

**實務上的影響**：如果你 export 的 checkpoint 在 backbone config 裡開了
`vad_head` 和/或 `dist_head`，streaming 的 ONNX graph 依然會 export
成功、依然會輸出正確的增強後音訊 frame——只是完全沒有 VAD 或 distance
的輸出，就算 checkpoint 的權重裡明明有訓練過的 head 也一樣。Export 當下
不會有任何警告。如果下游有人需要在 streaming 時拿到這些訊號，目前只能
用別的方式算（例如對緩衝好的音訊跑一次完整的 offline 模型)——逐 frame 的
ONNX 路徑並不會 expose 它們。

## Notes

- ONNX 模型只涵蓋 feature-frame 這一段；音訊的 STFT 與 iSTFT 刻意留在
  graph 之外。
- 目前這個 runtime 只支援 batch size 1 的 waveform streaming。
