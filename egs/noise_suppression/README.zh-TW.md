# 噪音抑制

English: [`README.md`](README.md)

單聲道語音增強：從乾淨語音語料、噪音語料與房間脈衝響應 bank 即時合成 (noisy, clean) 配對，
由一個以遮罩為基礎的 encoder / backbone / decoder 模型學習去除噪音與殘響，同時保留每一個
人聲。已發布的模型是 16 kHz、可串流、不需 enrollment 的 DPCRN-Mamba checkpoint。

`main.py` 是這個 recipe 的訓練進入點。dataset 以下的一切——sampler、dataloader、CLI、
Lightning 串接、DDP、評分與推論——都在
[`puresound/system/runner.py`](../../puresound/system/runner.py)，與
[`egs/voice_isolate`](../voice_isolate/README.zh-TW.md) 共用；後者有自己的進入點與 dataset。

## 檔案

| 檔案 | 用途 |
|---|---|
| `main.py` | 訓練、評分與推論進入點（`NoiseSuppressionDataset`） |
| `config/train_dpcrn_mamba_s0.yaml`、`_s1.yaml`、`_activebin_ft.yaml`、`_metricgan_ft.yaml` | 已發布的血統，依序執行（見下） |
| `config/infer_dpcrn.yaml` | v1/v2 只含模型的 config（閘門、串流匯出、單次增強） |
| `config/infer_dpcrn_mamba_wide.yaml` | v3 加寬模型的推論 config |
| `config/eval/ns_testset.yaml` | 閘門的凍結合成集是用這份 recipe 建的 |
| `config/dpcrn.yaml`、`config/dparn.yaml` | 起始範本：32 kHz 的 DPCRN，以及帶 `FrequencyEQLayer` 前端的 16 kHz 離線 DPARN。不是已發布的 recipe |
| `run_full_gate.sh` | benchmark 的關卡清單，見 [`benchmarks/stages.zh-TW.md`](benchmarks/stages.zh-TW.md) |
| `benchmarks/` | 關卡定義與評分紀錄，見 [`benchmarks/README.zh-TW.md`](benchmarks/README.zh-TW.md) |
| `pretrained_ckpt/` | 已發布的 checkpoint 與其串流匯出，見 [`pretrained_ckpt/README.zh-TW.md`](pretrained_ckpt/README.zh-TW.md) |

## 1. 準備資料

語料準備是每個 recipe 共用的函式庫程式碼：
[`docs/usage/data_preparation.zh-TW.md`](../../docs/usage/data_preparation.zh-TW.md)。v1/v2 的
recipe 讀的是 16 kHz 的 DNS Challenge 語音與噪音：

```bash
# 乾淨語音 -> 語者互斥的 train/valid metafile，一次轉成 16 kHz
uv run python -m puresound.dataset.corpus.dns_challenge speech /path/to/audio/dns-5 \
    --output-dir egs/noise_suppression/data/dns5 \
    --subset read_speech --valid-ratio 0.05 \
    --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/clean_fullband

# 噪音 -> augmentation_noise.noise_folder 要指向的資料夾
uv run python -m puresound.dataset.corpus.dns_challenge noise /path/to/audio/dns-5 \
    --output-dir egs/noise_suppression/data/dns5 \
    --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband
```

recipe 還會讀預先產生的 RIR bank 的兩個 view
（`augmentation_reverb.simulator.pregenerated.banks`：一個合成的 `wide` view 與一個實測
房間的 view），建法見 [`egs/voice_isolate/DATA_SETUP.zh-TW.md`](../voice_isolate/DATA_SETUP.zh-TW.md)
§3–4。把 recipe 裡的絕對路徑指向你自己的副本。

其他語音與噪音來源——VCTK、LibriLight、FSD50K、CochlScene、MUSAN、語音形狀噪音、經過
checkpoint 清理的目標——以及混合它們的旋鈕（`trainer.speaker_source_weights`、
`augmentation_noise.noise_sources`、`augmentation_noise.snr_bands`），都在同一頁。

`--scoring` 與 `--inference` 讀的 `dataset.test_folder` 是另一種格式：一個目錄，內有
`wav2scp.txt`（每行 `<uttid> <path>`），`--scoring` 另需放乾淨參考的 `wav2ref.txt`。

## 2. 訓練

### 已發布的血統

已發布的 checkpoint 來自依序執行的四份 recipe，每一份都以 `--pretrained_ckpt_path` 從前一
份的最後一顆 checkpoint warm-start。各階段之間的模型、優化器、資料與其他 loss 都相同；改變
的只有下表列的項目。在一個階段內，`curriculum:` 區塊會把增強機率從前一階的值往上拉，所以
交接是一道斜坡而不是懸崖。

| Recipe | Warm start | 這一階設定了什麼 | 產出 |
|---|---|---|---|
| `train_dpcrn_mamba_s0.yaml` | 從零開始 | SNR [0, 20]、殘響機率 0.45、20 epoch | 第 0 階 |
| `train_dpcrn_mamba_s1.yaml` | s0 的 epoch 19 | SNR [-5, 15]、殘響機率 0.65、20 epoch | 第 1 階 |
| `train_dpcrn_mamba_activebin_ft.yaml` | s1 的 epoch 19 | `+ ActiveBinLogMagLoss`，LR 1e-4 跑 2 epoch | `dpcrn_mamba_v1` |
| `train_dpcrn_mamba_metricgan_ft.yaml` | activebin_ft 的 epoch 1 | `+` MetricGAN PESQ critic，LR 1e-4 跑 4 epoch | `dpcrn_mamba_v2` |

從 repo 根目錄執行（recipe 裡的路徑相對於它）：

```bash
uv run python egs/noise_suppression/main.py \
    egs/noise_suppression/config/train_dpcrn_mamba_s0.yaml --training --set_seed 1234

uv run python egs/noise_suppression/main.py \
    egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml --training --set_seed 1234 \
    --pretrained_ckpt_path egs/noise_suppression/exp/ns_dpcrn-mamba_curriculum_s0/lightning_logs/version_0/checkpoints/epoch=19-<step>.ckpt
```

兩個短的微調階段依此類推。每個 20 epoch 的階段跑一個 cosine 週期，所以最後一個 epoch（19）
就是要拿來 warm-start 與判讀的谷底。校準與 MetricGAN 兩階刻意設計成短的低學習率微調；把
任一個併進完整的一階都重現不了它。

在 GPU 上訓練 Mamba 時間路徑時，若 `mamba_ssm` 套件可匯入且是針對已安裝的 torch 編譯的，
就用它的融合 selective-scan kernel；否則模型退回精確的純 PyTorch scan，結果正確但慢很多。

### 加寬模型候選（v3）

`noise-suppression-dpcrn-mamba-v3` 使用較寬模型與獨立訓練鏈：
`train_dpcrn_mamba_capW_s0.yaml` → `train_dpcrn_mamba_capWL3_s1.yaml` →
`train_dpcrn_mamba_capWL3_ft.yaml` → `train_dpcrn_mamba_capWL3_mg.yaml`。
配方快照、warm-start checkpoint 與發布量測見
[`pretrained_ckpt/README.zh-TW.md`](pretrained_ckpt/README.zh-TW.md#重現-v3)。
v3 匯出使用 `config/infer_dpcrn_mamba_wide.yaml`，`infer_dpcrn.yaml` 繼續供 v1/v2 使用。
預設維持 v2。

### 命令列旗標

`config_path` 是位置參數，放在最前面。階段旗標是單純的開關（寫 `--training`，不是
`--training True`）。

| 旗標 | 作用 |
|---|---|
| `--training` | 訓練 |
| `--scoring` | 對 `dataset.test_folder` 評分（PESQ-WB/NB、STOI、ESTOI、SI-SNR、BSS-SDR、DNSMOS） |
| `--inference` | 把 `dataset.test_folder` 增強到 `dataset.proc_output_folder` |
| `--dump_training_samples` | 把 3 個合成 batch 寫到 `./dummy_samples/` |
| `--ckpt_path PATH` | 搭配 `--training` 是續跑；搭配 `--scoring` / `--inference` 是要執行的 checkpoint |
| `--pretrained_ckpt_path PATH` | 搭配 `--training`，從另一顆 checkpoint 的權重 warm-start |
| `--pretrained_allow_reshaped` | 搭配 `--pretrained_ckpt_path`，形狀改變的參數改為在初始化時重建，而不是拒絕 |
| `--set_seed N` | 固定所有隨機種子 |
| `--inference_sr HZ` | 以這個取樣率執行評分 / 推論 |

`--dump_training_samples` 每一列寫一個 `batch_XX-YY.wav`，是疊了
`[noisy, clean, noisy - clean]` 的三聲道檔；用能分開顯示聲道的編輯器打開，就能在投入一次
訓練之前先聽過增強的效果。

### 續跑、warm start、重新載入

`--ckpt_path` 與 `--pretrained_ckpt_path` 都吃一個 `.ckpt`，但它們是三種不同的機制：

| 旗標 | 階段 | 還原什麼 |
|---|---|---|
| `--ckpt_path` | `--training` | Lightning 自己的續跑：模型、優化器、scheduler、epoch 與 callback 狀態。在**同一份** config 下繼續**同一個** run |
| `--pretrained_ckpt_path` | `--training` | 只有權重；優化器、scheduler 與 loss 都依目前的 config 重新開始。血統的每一階用的就是這種 warm start |
| `--ckpt_path` | `--scoring` / `--inference` | 只有權重，依名稱複製進剛建好的模型 |

warm start 對**名稱寬鬆、對形狀嚴格**。目前模型有而 checkpoint 沒有的參數——新的頭、
critic——維持初始化；checkpoint 有而模型沒有的參數則忽略；兩者都會記進 log
（`[pretrained] N new param(s) kept at init: ...`、
`[pretrained] N ckpt param(s) ignored: ...`）。**形狀**改變的參數代表這顆 checkpoint 不是它
要載入的那個模型，載入會拒絕。唯一刻意的情況是改了 STFT 窗長：固定的 STFT kernel、分帶
矩陣與輸入正規化跟著頻率長度變，而每個卷積與 RNN 的形狀都不變。這時傳
`--pretrained_allow_reshaped`，那些參數會在初始化時重建（log 為
`[pretrained] N param(s) rebuilt at init`）。

評分與推論時的重新載入會報出模型沒有的 checkpoint 參數（`... is not in the model.`）與
checkpoint 沒提供的模型參數（`Needed param name but missing: [...]`）；兩者都以
`L.Trainer(inference_mode=True)` 單一行程執行。

### 訓練環境

- **後端。** `runner.configure_torch_backends()` 在匯入時開啟 cuDNN 自動調校與 TF32。每個
  batch 的訓練列長度固定，所以自動調校每種形狀只付一次代價。
- **DDP。** `trainer.num_gpus > 1` 時使用帶 `gradient_as_bucket_view=True` 的
  `DDPStrategy`，否則用 Lightning 的 `auto`。`n_spk_per_batch` 是每張卡的量。模型若有某些
  batch 從不碰到的參數群——只在部分列上被讀的輔助頭——`trainer.find_unused_parameters`
  （預設 false）就必須設 true；每份人聲隔離 recipe 都有設。batch 由 recipe 自己的
  `SpeakerSampler` 組成，所以 Lightning 的 distributed sampler 是關掉的，`sync_batchnorm`
  則開啟。
- **精度。** `trainer.lightning_trainer_args` 原封不動交給 `lightning.Trainer`，所以任何
  Trainer 參數都能用。預設是全精度；加上 `precision: bf16-mixed` 可用一點精度換速度與記憶體。
  用 bf16 而不是 fp16，是因為複數頻譜的幅度與除法運算需要 fp32 的範圍，也不必用 gradient
  scaler。
- **VAD 標籤。** `vad_label.backend: energy`（預設）在 dataloader worker 裡標記。
  `backend: silero` 會從 worker 移出，在 batch 搬到裝置後於 GPU 上批次執行；它需要
  `silero-vad` 套件，而且 labeler 不會進 `state_dict()`，因此永遠不會落進 checkpoint。

## 3. 閘門

benchmark 是一支腳本，而且在模型存在之前就能跑：

```bash
bash egs/noise_suppression/run_full_gate.sh baseline
bash egs/noise_suppression/run_full_gate.sh <tag> <ckpt> <recipe>
```

怎麼建它的測試集、它的變數、彙整怎麼判定：
[`docs/usage/evaluation.zh-TW.md`](../../docs/usage/evaluation.zh-TW.md)。每一關能決定什麼：
[`benchmarks/stages.zh-TW.md`](benchmarks/stages.zh-TW.md)。結果是 `benchmarks/records/` 底下的
一份紀錄；有 gate 關卡失敗或分辨不出時以非零碼結束。

## 4. 使用已發布的模型

```bash
# 透過 model zoo（ONNX、串流 runtime）
uv run puresound infer noise-suppression-dpcrn-mamba-v2 \
    --input audio=in.wav --output audio=out.wav --provider auto

# 或直接用串流匯出
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer \
    egs/noise_suppression/pretrained_ckpt/streaming/dpcrn_mamba_v2.onnx in.wav out.wav
```

重新匯出串流模型時要傳 `--dry-blend 1.0`——每一份噪音抑制紀錄都是在這個操作點評分的；
exporter 預設的 0.9 是人聲隔離的設定：

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/noise_suppression/config/infer_dpcrn.yaml \
    egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \
    /path/to/dpcrn_mamba_v2.onnx --dry-blend 1.0
```

見 [`docs/usage/streaming/dpcrn_onnx.zh-TW.md`](../../docs/usage/streaming/dpcrn_onnx.zh-TW.md)。
要選哪一顆 checkpoint：[`pretrained_ckpt/README.zh-TW.md`](pretrained_ckpt/README.zh-TW.md)。

## 這個 recipe 還是 `voice_isolate`

兩個 recipe 是同一條合成管線上的兩個獨立進入點。`VoiceIsolationDataset` 繼承
`NoiseSuppressionDataset` 並覆寫它的列型別 hook；它保留近講者、壓掉遠講者，而這個 recipe
保留每一個人聲。最上層的 `task:` 說明一份 config 屬於哪個 recipe，每個進入點只載入自己的
任務，所以 `voice_isolation` 的 config 在這裡會於載入任何資料前就失敗。人聲隔離的區塊
（`augmentation_realfar`、`augmentation_realnear`、`augmentation_session_rows`、
`augmentation_speech.mix_mode`）在噪音抑制 recipe 裡是未知欄位。

## 函式庫參考

| 文件 | 涵蓋 |
|---|---|
| [`docs/architecture/task/ns.zh-TW.md`](../../docs/architecture/task/ns.zh-TW.md) | `NoiseSuppressionDataset` 與這個 recipe 驅動的合成骨架，以及每一個 `augmentation_*` 區塊 |
| [`docs/architecture/task/voice_isolation.zh-TW.md`](../../docs/architecture/task/voice_isolation.zh-TW.md) | 人聲隔離的特化 |
| [`docs/architecture/system/siso.zh-TW.md`](../../docs/architecture/system/siso.zh-TW.md) | config 用的 Lightning module `EncDecMaskBase`，以及 encoder → features → backbone → mask → decoder 的路徑 |
| [`docs/usage/configuration.zh-TW.md`](../../docs/usage/configuration.zh-TW.md) | recipe 驗證、sampler 旋鈕與 `curriculum` 區塊 |
