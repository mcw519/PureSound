# voice_isolate — 預訓練 checkpoint

English: [`README.md`](README.md)

近場（<1 m）前景人聲隔離，單聲道、不需 enrollment：保留近距講話者，壓制遠距／競爭
講話者與噪音。

**`dpcrn_curriculum_v1.ckpt` 是 model zoo 的預設**，`dpcrn_curriculum_v2.ckpt`
與 `dpcrn_v8.ckpt` 在旁邊作為候選；這三個是登錄在 [`model_zoo/catalog.yaml`](../../../model_zoo/catalog.yaml)
裡的版本。三者都是 DPCRN（complex ratio mask、16 kHz、30 ms look-ahead）。`dpcrn_v8` 與
`dpcrn_curriculum_v1` 寬度相同，用 `config/infer_dpcrn.yaml` 載入；`dpcrn_curriculum_v2`
較寬，要用 `config/infer_dpcrn_wide.yaml` 載入：

```bash
cd egs/voice_isolate
uv run python scripts/demo.py --config_path config/infer_dpcrn.yaml        # dpcrn_v8、dpcrn_curriculum_v1
uv run python scripts/demo.py --config_path config/infer_dpcrn_wide.yaml   # dpcrn_curriculum_v2
```

不論傳哪一份 config，下拉選單都會列出這個目錄裡的每一個 checkpoint；請選寬度與該 config
相符的那一個。

## 已發布的 blend：`dry_blend = 0.9`

runtime blend 是每個現役版本**已發布設定的一部分**，不是可有可無的附加選項：

```python
enhanced = model(wav, dry_blend=0.9)      # out = 0.9 * enhanced + 0.1 * input
```

它把任何一點的衰減都限制在 −20 dB 以內。代價是留下一點殘餘干擾者，換來的是訓練資料以外
的收音鏈上 deletion 大幅下降：加上它，模型在真實錄音上降低了 ASR 錯誤率；不加它，同一個
checkpoint 會在其中一個殘響測試集上提高 WER。評測腳本都有 `--dry-blend`；串流 manifest 把
這個值放在 `recommended_inference` 底下。

在 `dpcrn_curriculum_v2` 上實測（ep39、完整 benchmark，兩列用同一版程式）：

| 關卡 | `dry_blend 1.0` | `dry_blend 0.9` |
|---|---|---|
| moderate 殘響 WER（主要閘門） | 0.416 | **0.386** |
| Dawn WER / 刪字（不處理 0.184 / 0.082） | 0.215 / 0.132 | **0.174 / 0.084** |
| BUT-OFFICE WER（不處理 0.563） | 0.597 | **0.528** |
| BUT 極端殘響 WER（不處理 0.658） | 0.739 | **0.658** |
| turn-taking KEEP 合格 / 違規 | 88 / 12 | **90 / 10** |
| turn-taking SUPPRESS（中位數） | **−26.0 dB** | −18.6 dB |

只有在遠處人聲必須盡量清除乾淨、近講者的字詞沒那麼重要時才用 `dry_blend 1.0`；每一項錯字率
在那個設定下都比較差。curriculum-v1 的方向相同（Dawn WER 在 1.0 是 0.232、0.9 是 0.172）。

已知限制：0.9 的 blend 無法產生完全靜音（依構造就有 −20 dB 底線），而且透過與訓練語料非常
不同的收音鏈錄下的遠場語音，其壓制深度會比 in-domain 淺。要做到硬性靜音需要 gate，而不是
blend。

## 版本

| 版本 | Zoo id / 角色 | 訓練 | 什麼時候用 |
|---|---|---|---|
| **`dpcrn_curriculum_v1.ckpt`** | `voice-isolate-dpcrn-curriculum-v1`，**預設** | `config/train_dpcrn_curriculum_v1.yaml`，從冷啟動 recipe 的產出 warm-start | 收音硬體就是模型調校時的那一種 |
| `dpcrn_curriculum_v2.ckpt` | `voice-isolate-dpcrn-curriculum-v2`，候選 | 同樣兩步換成加寬的模型（先 `train_dpcrn_curriculum_v2_base.yaml`，再 `train_dpcrn_curriculum_v2.yaml`）；用 `config/infer_dpcrn_wide.yaml` 載入 | 更低的 WER 與更深的壓制，值得 curriculum-v1 的 1.4 倍 CPU |
| `dpcrn_v8.ckpt` | `voice-isolate-dpcrn-v8`，候選 | 多階段 warm-start 階梯；recipe 未公開 | 收音硬體未知，或與訓練語料差很多 |
| `dpcrn_curriculum_v0` | 不散布 | `config/train_dpcrn.yaml`，從零開始，取 ep99 | 不用於部署；它是 curriculum-v1 與 -v2 的 warm start |

## 量測

全部在 `dry_blend 0.9` 下量測。以下集合都讀固定音訊，所以數字可以跨版本比較。

| 集合 | 指標 | 不處理 | v8 | curriculum-v1 | curriculum-v2 |
|---|---|---|---|---|---|
| moderate 殘響（主要 WER 閘門） | WER | 0.582 | 0.411 | 0.427 | **0.386** |
| Dawn Chorus | WER / deletion | 0.184 / 0.082 | 0.180 / 0.094 | **0.172** / 0.089 | 0.174 / **0.084** |
| BUT-OFFICE（監看） | WER | 0.563 | 0.539 | 0.547 | **0.528** |
| real-RIR turn-taking | SUPPRESS 中位數（合格 / 失敗） | -- | −16.14 dB（87 / 13） | −17.10 dB（94 / 6） | **−18.60 dB（95 / 5）** |
| | KEEP 合格 / 違規 | -- | **94 / 6** | 89 / 11 | 90 / 10 |
| -- | 串流 CPU RTF，單執行緒 | -- | 0.26 | 0.26 | 0.36 |

在一組內部田野錄音（不散布）上，curriculum-v1 與 -v2 能壓下「在任何近場講話者之前出現的孤立
遠處講話者」（−7.1 與 −12.4 dB），v8 不行。在收音鏈與訓練資料不同的錄音上，兩者都可能衰減
近講者，v8 不會。

BUT-OFFICE 是一個 200 句的集合，它的 bootstrap 區間通常太寬，分不出模型與不處理之間的差別，
所以它是監看；主要的 WER 閘門是 `wer_set_moderate_test`（[`../scripts/WER_SETS.md`](../scripts/WER_SETS.md)）。

## 如何選版本

選 `dpcrn_curriculum_v1.ckpt` 搭配 `dry_blend 0.9`——它是 model zoo 的預設，在任何近講者
開口之前就能壓下遠場人聲。收音鏈未知、或與訓練語料差很多時選 `dpcrn_v8.ckpt`：它放棄這份
冷啟動增益，但在這類錄音上不會讓近講者 keep 退步，而那正是 curriculum-v1 尚未解決的缺陷。
CPU 預算容得下 curriculum-v1 的 1.4 倍時選 `dpcrn_curriculum_v2.ckpt`，並用
`config/infer_dpcrn_wide.yaml` 載入：benchmark 每一關都贏過或打平 v1，但跨鏈 keep 的缺陷
跟 v1 一樣還在。

**它們都做不到的事。** 透過與訓練語料非常不同的收音鏈錄下的遠場語音，仍然幾乎壓不下去。
這個落差是錄音鏈本身的特性，與距離無關，這裡沒有任何版本能補上它。

**判準慣例。** scheduler（`CosineAnnealingWarmRestarts`，`T_0=20`）每 20 epoch 重啟一次，
所以 checkpoint 只能在 cosine 谷底——ep19 / ep39 / ep59——互相比較。上面每一個數字都來自
谷底 epoch。

## `streaming/` —— 逐幀 ONNX 匯出

`dpcrn_v8`、`dpcrn_curriculum_v1` 與 `dpcrn_curriculum_v2`（`.onnx` + `.json`），登錄在案的版本，以
`../scripts/streaming_onnx.py export` 建置。全部都因 look-ahead 而帶有 **30 ms（3 幀）演算法
延遲**（加上 32 ms 分析窗，輸入到輸出為 52–62 ms），由 graph 內部的 future-buffering 處理，
而且都在 `recommended_inference` 記錄 `dry_blend 0.9`。用 `puresound.streaming.StreamingOrt`
或 SDK 的 `PureSoundStreamingRuntime`（`processor: stft_frame_ort`）載入；runtime 都在輸出端套用
blend，輸入延後 `streaming_delay_frames`，不增加延遲。

`dpcrn_v8` 與 `dpcrn_curriculum_v1` 的匯出採用 batch=1 的 frequency-major 佈局與向量化的單步
inter-LSTM 更新，把訓練好的 input/hidden 投影合成一次涵蓋所有頻率位置的矩陣乘法；訓練好的
intra 雙向 LSTM 與所有 state port 都保留。在同一份 5 秒噪音語音樣本上，交錯量測的單執行緒
CPU RTF 每個版本都從 **0.366 降到 0.258**。[重新匯出報告](../../../model_zoo/benchmarks/onnx_reexport_20261004.json)
附有新舊雜湊與 30 秒的串流一致性檢查。`dpcrn_curriculum_v2` 之後用同樣方式匯出：在一支真實
錄音上離線對串流的一致性 95.2 dB（curriculum-v1 為 95.7），串流 RTF **0.362**，
curriculum-v1 在同一份樣本上是 0.256（單執行緒、交錯量測）。

任何離線與串流的比較都**必須**依回報的延遲對齊並裁掉邊緣，否則延遲會被讀成誤差；`verify`
會做這件事。判斷新匯出要用真實語音（`--input_audio`）：預設的白噪音探針是壓力訊號，它的
一致性數字反映的是某版本調變遮罩有多激進，而不是串流是否正確。

```bash
cd egs/voice_isolate
uv run python scripts/streaming_onnx.py export \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_curriculum_v1.ckpt /tmp/model.onnx \
    --optimization portable --dry-blend 0.9
uv run python scripts/streaming_onnx.py verify \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_v8.ckpt \
    pretrained_ckpt/streaming/dpcrn_v8.onnx --input_audio speech.wav --provider cpu
```

重新訓練：[`../README.zh-TW.md`](../README.zh-TW.md#訓練)。這裡的 checkpoint 都是各自訓練 run
中經判準的谷底 epoch。
