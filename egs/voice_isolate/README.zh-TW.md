# voice_isolate — 近場前景人聲隔離

English: [`README.md`](README.md)

單聲道、**不需 enrollment 的近場**人聲隔離：保留距離麥克風約 1 m 以內的講話者，壓掉更遠
的講話者與噪音。沒有 enrollment，也沒有第二支麥克風，所以模型必須學會的線索是距離對人聲
造成的差異——最主要的是近場與遠場的直達對殘響比（DRR）對比。

每一顆已發布的模型都是 DPCRN（複數比例遮罩、16 kHz），帶 30 ms（3 幀）look-ahead，透過
逐幀 ONNX 匯出串流執行：`dpcrn_v8` 與 `dpcrn_curriculum_v1` 是 `channels [2,32,64,128]`、
`rnn_hidden 96`，較寬的 `dpcrn_curriculum_v2` 是 `channels [2,48,96,128]`、`rnn_hidden 128`。

`main.py` 是這個 recipe 的訓練進入點（`VoiceIsolationDataset`，帶有其他 recipe 沒有的真實
錄音列型別）。dataset 以下的一切都透過 `puresound/system/runner.py` 與
[`egs/noise_suppression`](../noise_suppression/README.zh-TW.md) 共用；那份 README 說明了命令
列旗標、續跑與 warm start 的差別（包括 `--pretrained_allow_reshaped`）、DDP、精度與 VAD
標記，這些在這裡完全適用。

## 已發布的模型

| Checkpoint | zoo id / 角色 | 什麼時候用 |
|---|---|---|
| **`pretrained_ckpt/dpcrn_curriculum_v1.ckpt`** | `voice-isolate-dpcrn-curriculum-v1`，**預設** | 收音路徑就是模型調校時的那一種；它對遠處講話者壓得最深，包括冷啟動時 |
| `pretrained_ckpt/dpcrn_curriculum_v2.ckpt` | `voice-isolate-dpcrn-curriculum-v2`，候選 | 負擔得起 v1 的 1.4 倍 CPU：同一套 recipe 的加寬模型，殘響 WER 集上 WER 較低、冷啟動壓得更深；在陌生收音硬體上保留近講者的表現沒有比 v1 好 |
| `pretrained_ckpt/dpcrn_v8.ckpt` | `voice-isolate-dpcrn-v8`，候選 | 收音硬體未知、或與訓練語料差很多：它壓得較淺，但在那種硬體上不會衰減近講者 |

三者出貨時都帶 **runtime dry blend 0.9**（`out = 0.9 * enhanced + 0.1 * input`），把衰減
上限封在 −20 dB，用一點殘留干擾換取在陌生收音鏈上大幅減少的漏字。串流 manifest 把它記在
`recommended_inference`，由 runtime 套用。

```bash
uv run puresound infer voice-isolate-dpcrn-curriculum-v1 \
    --input audio=in.wav --output audio=out.wav --provider auto

# 涵蓋每一顆 checkpoint 與匯出的 Gradio demo（從 recipe 目錄執行；
# 要載入 dpcrn_curriculum_v2 時改傳 config/infer_dpcrn_wide.yaml）
cd egs/voice_isolate && uv run python scripts/demo.py --config_path config/infer_dpcrn.yaml
```

各版本的判定與怎麼選：[`pretrained_ckpt/README.zh-TW.md`](pretrained_ckpt/README.zh-TW.md)。

## 它們是怎麼訓練出來的

兩條血統，同一個架構、兩種寬度。

| 血統 | Recipe | Warm start | 產出 |
|---|---|---|---|
| curriculum | `config/train_dpcrn.yaml`，120 epoch | 從零開始 | `dpcrn_curriculum_v0`（ep99；不隨 repo 發佈） |
| | `config/train_dpcrn_curriculum_v1.yaml`，40 epoch | `dpcrn_curriculum_v0` | **`dpcrn_curriculum_v1`** |
| curriculum，加寬 | `config/train_dpcrn_curriculum_v2_base.yaml` 80 epoch，再接 `config/train_dpcrn_curriculum_v2.yaml` 40 epoch | 從零開始，再從前一步的 ep79 | `dpcrn_curriculum_v2` |
| 階梯 | `dpcrn_v8` 來自一條未公開的多階段 warm-start 階梯 | 各自從前一階 | **`dpcrn_v8`** |

`config/train_dpcrn.yaml` 把階梯在 run *之間*改的東西，寫成在一次 run *之中*移動的
`curriculum` 排程：房間池逐步放寬、抗過度抑制 loss 逐步加入、收音真實感與真實錄音列在合成的
決策學會之後才進來，距離 loss 跟著它們一起進來。`config/train_dpcrn_curriculum_v1.yaml`
把那份資料配方固定在終點值，並依排程加入一條軸：多輪對話的 session 列、供一致性項使用的
配對收音 view，以及逐幀的遠近與在場頭。兩個 `curriculum_v2` recipe 就是這兩步換成加寬的
模型（channels `[2, 48, 96, 128]`、`rnn_hidden` 128），每個 epoch 抽 1.5 倍的列。每個選擇
的理由寫在 recipe 的檔頭。

## 訓練

第一次執行前，先把 recipe 的語料與 RIR bank 路徑指向你自己的資料：
[`DATA_SETUP.zh-TW.md`](DATA_SETUP.zh-TW.md)。**從這個目錄**執行——config 的 metafile 與
work-folder 路徑是相對於它的：

```bash
cd egs/voice_isolate

# 檢查 recipe 實際會訓練的資料，以及這條管線到底學不學得起來
uv run python scripts/check_training_data.py config/train_dpcrn.yaml --n 64 --dump 8
uv run python scripts/overfit_check.py config/train_dpcrn.yaml --steps 800 --device cuda

# 預設 recipe：一次從零開始的 curriculum run
uv run python main.py config/train_dpcrn.yaml --training

# 第二步，從你自己第一步 run 的 ep99 checkpoint warm-start
# （dpcrn_curriculum_v0 就取在這裡；它不隨 repo 發佈）
uv run python main.py config/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path exp/dpcrn_curriculum/lightning_logs/version_0/checkpoints/epoch=99-*.ckpt

# 加寬血統：同樣兩步，換成加寬的模型
uv run python main.py config/train_dpcrn_curriculum_v2_base.yaml --training --set_seed 1234
uv run python main.py config/train_dpcrn_curriculum_v2.yaml --training --set_seed 1234 \
    --pretrained_ckpt_path exp/dpcrn_curriculum_v2_base/lightning_logs/version_0/checkpoints/epoch=79-step=60000.ckpt
```

第二步要用 `--pretrained_ckpt_path`，不是 `--ckpt_path`：模型多了頭，所以 checkpoint 以非
嚴格方式載入，優化器重新開始。任一個 run 被*中斷*時，用 `--ckpt_path` 指向它自己最新的
checkpoint 續跑。

scheduler（`CosineAnnealingWarmRestarts`、`T_0=20`）每 20 epoch 重啟一次，所以只在 cosine
谷底比較 checkpoint——epoch 19、39、59……——而且相信兩個 run 的差異之前，要先評分末尾的
一組 checkpoint。兩份 recipe 都為輔助頭設了 `trainer.find_unused_parameters: true`，以及
`precision: bf16-mixed`。

這些 config 共用的設計（early target、困難 SIR、實測收音真實感、不用合成的 target-absent
列、抗刪字 loss）：[`config/README.zh-TW.md`](config/README.zh-TW.md)。

## Benchmark

```bash
cd egs/voice_isolate
bash run_full_benchmark.sh <ckpt> <tag> [device] [dry_blend] [presence_readout]
```

傳 `dry_blend 0.9` 就是以部署的方式評分（預設 1.0 表示不做 blend）。必須先停掉訓練；各關會
用到 GPU。`BLOCK_CKPTS="a.ckpt b.ckpt ..."` 會加上對一個 run 末尾幾顆 checkpoint 的 block
協定；`CFG_*` 覆寫則讓各關改用與非預設 backbone 相符的 recipe
（由 `scripts/make_arch_eval_configs.py` 產生）。

| # | 關卡 | 角色 |
|---|---|---|
| 0 | preflight：每一關的 recipe 都能完整載入 checkpoint | 中止 |
| 1 | 真實錄音田野計分卡，含跨鏈參考 clip（私有錄音） | gate |
| 1b | 對 `BLOCK_CKPTS` 的田野 block 協定（選用；私有錄音） | 田野差異要這樣才可信 |
| 2 | 依 bucket 的 in-domain SI-SDRi，含單獨干擾洩漏 | 合成，只與同一條合成鏈比 |
| 3–5 | 合成的純遠場探針：已見距離、未見的高殘響、未見的邊界距離 | 合成的洩漏檢查 |
| 6 | Dawn Chorus WER | 刪字護欄 |
| 7a | 中度殘響 WER | 主要的 WER gate |
| 7b | BUT-OFFICE 實測 RIR 的 WER | monitor：句數太少，分不出模型與不處理 |
| 8 | BUT 高殘響 WER | do-no-harm monitor，遠在訓練域之外 |
| 9 | 真實 RIR 的輪流說話：保留近場、壓掉單獨遠場 | 保留 / 壓制計分卡 |

第 1 與 1b 關讀的是一組私有錄音的田野集（`data_report/field_cases/test_vector_cases`，跨鏈
參考 clip 也在裡面）。它不隨 repo 發佈，所以請略過這兩關：沒有這組資料時，第 1 關會回報失敗，
其餘各關照常執行。

第 2–5 關在評測時才合成音訊，所以只能與同一條合成鏈上做出的紀錄比（摘要會印出 commit）；
其餘各關讀的是磁碟上固定的音訊。評測集放在不進版控的 `data_report/`。工具、建集方式與每一關
呼叫的腳本：[`scripts/README.zh-TW.md`](scripts/README.zh-TW.md) 與
[`scripts/WER_SETS.md`](scripts/WER_SETS.md)。

## 串流部署

`pretrained_ckpt/streaming/` 放著三顆已發布 checkpoint 的逐幀 ONNX 匯出，以
`scripts/streaming_onnx.py` 建出（`dpcrn_curriculum_v2` 要用 `config/infer_dpcrn_wide.yaml`
匯出）。30 ms 的 look-ahead 由 graph 內部當作額外狀態的
future-buffering 處理，manifest 驅動的 runtime——`puresound.streaming.StreamingOrt` 與可攜式
SDK——可直接載入，並在 graph 之後套用記錄下來的 dry blend：

```bash
uv run python scripts/streaming_onnx.py export \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_curriculum_v1.ckpt /path/to/model.onnx
uv run python scripts/streaming_onnx.py verify \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_curriculum_v1.ckpt \
    pretrained_ckpt/streaming/dpcrn_curriculum_v1.onnx --input_audio speech.wav
```

`export` 預設記錄 `--dry-blend 0.9`，也就是發布設定。細節——look-ahead 狀態、onset guard、
輔助頭輸出、一個 hop 的延遲規則：
[`docs/usage/streaming/dpcrn_onnx.zh-TW.md`](../../docs/usage/streaming/dpcrn_onnx.zh-TW.md)。

## 文件

| 檔案 | 內容 |
|---|---|
| [`DATA_SETUP.zh-TW.md`](DATA_SETUP.zh-TW.md) | 從公開語料到預設 recipe 讀的那些路徑 |
| [`config/README.zh-TW.md`](config/README.zh-TW.md) | 各份 config，以及它們共用的設計 |
| [`pretrained_ckpt/README.zh-TW.md`](pretrained_ckpt/README.zh-TW.md) | 版本判定表、怎麼選、串流匯出 |
| [`scripts/README.zh-TW.md`](scripts/README.zh-TW.md) | 資料準備、訓練中檢查、benchmark 與推論工具 |
| [`docs/architecture/task/voice_isolation.zh-TW.md`](../../docs/architecture/task/voice_isolation.zh-TW.md) | `VoiceIsolationDataset`、真實錄音列型別與它輸出的標籤 |
