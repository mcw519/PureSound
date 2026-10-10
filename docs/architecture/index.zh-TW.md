# 架構

English version: [index.md](index.md)

各 package 如何組合在一起，以及資料如何從磁碟上的語料一路走到在別人的 process 裡
執行的模型。每個階段計算什麼，請看[演算法](../index.zh-TW.md#演算法)；怎麼執行，
請看[使用方式](../index.zh-TW.md#使用方式)。

儲存庫裡有三種程式碼：

- **`puresound/`**，函式庫。所有邏輯都在這裡：語料準備、合成、模型、訓練 driver、
  評測、匯出與推論。
- **`egs/<task>/`**，recipe。每個都是薄薄的 driver：一個指定自己 dataset class、
  再交給 `puresound.system.runner` 的 `main.py`、recipe YAML 檔、評測 stage 清單，
  以及已發布的 checkpoint。規則與理由請看[儲存庫佈局](../repository_layout.zh-TW.md)。
- **`sdk/python/`**，獨立的串流 runtime，不 import 任何 `puresound` 的東西，只需要
  NumPy 與 ONNX Runtime。

## 資料流

```text
 磁碟上的語料
     │  puresound.dataset.corpus         掃描 · 重取樣 · 切分 · 寫出
     ▼
 metafile（7 欄 CSV）+ JSONL inventory
     │  SpeakerSampler                   一個 batch 的 item key
     │  puresound.task.* datasets        每個 key 即時合成一列：
     │    （建構於 DynamicBaseDataset）     語音 + 房間 + 噪音 + 裝置鏈
     ▼
 batch：noisy、clean、VAD / 輔助目標
     │  puresound.system.EncDecMaskBase  encoder → features → backbone → mask → decoder
     │  puresound.system.runner          Lightning Trainer、DDP、checkpoint
     ▼                                   （recipe YAML 由 puresound.config 載入）
 checkpoint
     │  puresound.evaluation             配對評分、統計、紀錄
     │  egs/<task>/run_full_*.sh         stage 清單
     ▼
 評測紀錄 ──► 發版決定
     │  puresound.streaming              逐 frame 的 ONNX graph + JSON manifest
     ▼
 ONNX artifact + model_zoo/catalog.yaml
     ├─► puresound.inference             ModelZoo、load_model、`puresound infer`
     │     └─► puresound.web             本機 HTTP playground
     └─► sdk/python/puresound_streaming  獨立的 ORT runtime
```

### 1. 語料準備

`puresound.dataset.corpus` 把磁碟上的語料轉成 **metafile**：每個 utterance 一列，
`uttid, spkid, gender, path, length, sample rate, channels`。語料額外知道的東西
（噪音類別、收音裝置）寫進旁邊的 JSONL inventory，metafile 因此永遠不會多長一欄。
切分以 speaker 為單位，同一位說話者不會同時出現在兩邊；音訊只重取樣一次、寫進鏡像
目錄樹，而不是每次讀取時才做。各語料的特定佈局（DNS Challenge、VCTK-DEMAND、
LibriLight、Kaldi 清單）都是帶 `python -m puresound.dataset.corpus.<name>` 入口的
小模組。請看[資料準備](../usage/data_preparation.zh-TW.md)。

### 2. 動態合成

沒有任何東西是事先混好的。一列訓練資料在被要求時才組出來：

- `puresound.task.sampler.SpeakerSampler` 是 `batch_sampler`。它從 dataset 的
  metadata 抽 speaker，產出 **item key**
  `(speaker, sample_rate[, seed[, seconds[, epoch]]])`：per-item seed 讓 validation
  set 可重現，列長用於混合長度的 batch，epoch 給 curriculum 用。
- `puresound.dataset.dynamic_base.DynamicBaseDataset` 解析 metafile、驗證
  augmentation block、擁有 `AudioEffectAugmentor`（噪音資料夾、RIR bank 或房間
  模擬器），並把一個 key 轉成該列的長度、該 epoch 的旋鈕與它的 seed。
- 任務 dataset（`puresound.task.ns.NoiseSuppressionDataset`、
  `puresound.task.voice_isolation.VoiceIsolationDataset`）實作 `__getitem__`：
  選語音、把每個聲源放進房間、決定誰在何時說話（`overlap_gating`）、加入非語音的
  聲音（`noise_stage`）、讓混音經過收音與傳輸鏈（`device_chain`），然後輸出 noisy
  輸入以及它的乾淨參考與標籤。Session 列（`session_rows`）是另一種列型別；
  `paired_views` 為同一個混音再加一個收音視角，供一致性 loss 使用。
- collate function 把各列疊成一個 batch。

逐筆合成讓訓練分布是一組 recipe 旋鈕，而不是磁碟上的一份資料集；這也是 curriculum
能在 run 進行中移動這些旋鈕的原因。請看 [Datasets](dataset/index.zh-TW.md)、
[Tasks](task/index.zh-TW.md) 與[資料增強](../algorithms/augmentation/index.zh-TW.md)。

### 3. 模型

`puresound.system.siso.EncDecMaskBase` 是每個增強任務訓練的 Lightning module：
波形 → encoder（STFT 或可學習）→ 特徵轉換（`puresound.nnet.FeatureEncoder`）→
backbone → mask（`puresound.nnet.masker`）→ decoder → 波形。backbone 可以是
`puresound.nnet` 裡的任何模型（`DPCRN`、`DPARN`、`DPRNN`、`TFGridNet`、`SkiM`、
`ConvTasNet`、U-Net 家族），由 `puresound.nnet.lobe` 的可重用區塊組成；loss 在
`puresound.nnet.loss`。`puresound.system.base.BaseLightningModule` 負責每個 module
共用的部分：加權 loss 清單、optimizer 與 scheduler 管線、warm-up，以及 GPU 上的
批次 VAD 標註。推論時的後處理（`postprocess`、`onset_guard`，以及只用於離線的
`presence_gate`）放在 module 旁邊，讓離線評測與串流匯出套用同一份程式碼。請看
[訓練系統](system/index.zh-TW.md)與[模型](../algorithms/models/index.zh-TW.md)。

### 4. 訓練 driver 與 recipe

一個 recipe 是一個 YAML 檔，由 `puresound.config.load_recipe` 載入成型別化的
model（見[設定系統](#設定系統)）。`egs/<task>/main.py` 把它連同自己的 dataset 與
collate class 交給 `puresound.system.runner.build_dataloaders`，再呼叫
`runner.run_stages`，依要求執行 `--dump_training_samples`、`--training`、
`--scoring`、`--inference` 其中幾項。runner 建立 sampler 與兩個 dataloader、模型
（`puresound.recipes.init_model_for_task`）、loss、optimizer 與 scheduler、DDP
strategy 與 checkpoint callback，並處理 warm start（`--pretrained_ckpt_path`）。
把這些放在函式庫裡，兩個 recipe 才能共用同一個訓練迴圈，而不是各留一份、逐漸走岔。
請看 [Recipes](../usage/recipes.zh-TW.md)。

### 5. 評測閘門

checkpoint 依評測紀錄發版，不是依 validation loss。`puresound.evaluation` 放的是
協定：每個 stage 都同時評候選模型**與**不處理的輸入（`systems.Passthrough`），
並回報配對差值及其信賴區間（`statistics`），因為單一測試集上的絕對分數本身說明
不了什麼。各 stage 是以 `python -m puresound.evaluation.tools.<name>` 執行的函式庫
工具：`preflight`（checkpoint 能完整載入閘門用到的每個 recipe）、
`build_eval_set`（凍結一份合成測試集）、`reference`（PESQ、STOI、SI-SDR）、
`noreference`（DNSMOS）、`wer`（辨識器下的刪字）、`rtf`（CPU 即時率）與
`collect`——把各 stage 合併成一份紀錄（`records`）並寫出閘門的判定。recipe 的
`run_full_gate.sh`（人聲分離是 `run_full_benchmark.sh`）只是 stage 清單。請看
[評測](../usage/evaluation.zh-TW.md)與[指標](../algorithms/metrics.zh-TW.md)。

### 6. 匯出

`puresound.streaming` 把訓練好的離線模型轉成**逐 frame** 的模型：一次吃一個 STFT
frame，把 recurrent 與卷積狀態當成顯式的輸入與輸出，並設計成逐 frame 重現離線
forward 的結果。它被 trace 成 ONNX（`export_streaming_dpcrn_onnx`、
`export_streaming_dparn_onnx`）；匯出時會在 ONNX Runtime 下執行一次 graph、與
PyTorch frame model 比對，並在 graph 旁寫一份 JSON manifest：STFT 幾何、狀態 port
的名稱與形狀、串流延遲，以及建議的推論設定（`dry_blend`、onset guard）。後處理
記錄在 manifest 裡、不燒進 graph，runtime 因此能重現評測時的設定，不必靠人記得
再下一次 flag。請看 [Streaming](../usage/streaming/index.zh-TW.md)。

### 7. 部署

- **Model zoo** —— `model_zoo/catalog.yaml` 列出已發布的模型（見
  [Model zoo](#model-zoo)）。
- **`puresound.inference`** —— `ModelZoo` 讀 catalog；`load_model(id)` 解析出
  artifact，交給 catalog 指定的 processor（增強用 `stft_frame_ort`、語者 embedding
  用 `waveform_embedding_ort`）。CLI（`puresound models`、`puresound infer`、
  `puresound providers`、`puresound web`）是它上面薄薄的一層，不會載入訓練堆疊
  （Lightning、dataset、模型）。
- **`puresound.web`** —— 以標準函式庫寫的 HTTP server 加靜態瀏覽器 client，建立在
  `puresound.inference` 之上，以 `puresound web` 啟動：執行模型、比較模型、量測音訊、
  匯出比較結果。它沒有驗證機制，只適合本機或可信任的網路。請看
  [Web playground](../usage/web.zh-TW.md)。
- **`sdk/python/puresound_streaming`** —— `PureSoundStreamingRuntime` 依 manifest
  逐 frame 執行匯出的 ONNX graph，並套用 manifest 的後處理。它自行重寫需要的少數
  部分而不 import `puresound`，產品只裝 NumPy 與 ONNX Runtime 就能嵌入；重寫的部分
  （onset-guard 旋鈕、後處理）由測試釘住、與函式庫版本一致。

## Package 分工

| package | 職責 | 細節 |
| --- | --- | --- |
| `puresound.audio` | 音訊 I/O（`AudioIO`）、DSP、噪音、音量、頻譜、VAD labeler、`AudioEffectAugmentor`、房間模擬；`audio.rir` 離線建立與稽核 RIR bank，只有它的 bank loader 在訓練路徑上 | [Audio](../algorithms/audio/index.zh-TW.md) |
| `puresound.dataset` | 語料準備、metafile 解析、動態合成的 base dataset | [Datasets](dataset/index.zh-TW.md) |
| `puresound.task` | 任務 dataset、collate function、speaker sampler，以及它們組合的合成階段 | [Tasks](task/index.zh-TW.md) |
| `puresound.nnet` | 模型、可重用 layer（`lobe`）、masker、特徵 encoder、loss | [模型](../algorithms/models/index.zh-TW.md)、[Loss](../algorithms/losses/index.zh-TW.md) |
| `puresound.system` | Lightning module、訓練 driver（`runner`）、curriculum callback、MetricGAN critic、推論後處理 | [訓練系統](system/index.zh-TW.md) |
| `puresound.config` | 型別化的 recipe schema 與 `load_recipe` | [設定](../usage/configuration.zh-TW.md) |
| `puresound.recipes` | 把驗證過的 recipe 所指名的模型與 loss 建成物件 | [下方](#設定系統) |
| `puresound.evaluation` | 評測協定、配對統計、紀錄、轉錄器；stage 工具在 `evaluation.tools` | [評測](../usage/evaluation.zh-TW.md) |
| `puresound.streaming` | 逐 frame 模型、ONNX 匯出與 manifest、套件內的 ORT runtime（`StreamingOrt`） | [Streaming](../usage/streaming/index.zh-TW.md) |
| `puresound.inference` | catalog schema、`ModelZoo`、provider、processor、`load_model` | [Model zoo](#model-zoo) |
| `puresound.web` | 本機 HTTP playground | [Web](../usage/web.zh-TW.md) |
| `puresound.cli` | 建立在 `puresound.inference` 上的 `puresound` 指令；不載入訓練堆疊 | [Web](../usage/web.zh-TW.md) |
| `puresound.metrics`、`puresound.utils`、`puresound.logging_setup` | 客觀指標、共用小工具、函式庫的 logging handler | [指標](../algorithms/metrics.zh-TW.md)、[Utilities](utils.zh-TW.md) |
| `puresound.third_party` | PureSound 可選用但不隨附的第三方程式碼說明 | — |
| `sdk/python` | 獨立串流 runtime 套件 `puresound_streaming` | [Streaming](../usage/streaming/index.zh-TW.md) |
| `egs/` | recipe driver：`noise_suppression`、`voice_isolate`、`speaker_embedding`、`target_speaker_extraction`，以及建立 RIR bank 的 `rir_generation` | [Recipes](../usage/recipes.zh-TW.md) |

## 設定系統

recipe 是一個帶三個判別欄位的 YAML mapping：`schema_version`（2）、`purpose`
（`train` 或 `inference`）與 `task`（`noise_suppression`、`voice_isolation`、
`speaker_embedding`、`target_speaker_extraction`）。
`load_recipe(path, expected_task=..., expected_purpose=...)` 以 safe YAML loader
讀取，並驗證成該任務的 schema（`puresound.config.recipe` 的 `TASK_SCHEMAS`），
model-only 的 recipe 則驗證成 `InferenceRecipe`；任何不符都會丟出
`RecipeConfigError`。recipe driver 會傳入它預期的任務，所以把人聲分離的設定檔交給
降噪 driver，會在載入時就失敗，而不是跑到一半。

- 每個 config model 都繼承 `StrictConfig`：未知欄位是錯誤，model 是 frozen 的。
  `with_overrides` 產生改了部分欄位、且經過驗證的副本。
- 管線控制流程——dataset、trainer、optimizer、scheduler、augmentation、VAD 與
  curriculum block——完全型別化。`model` block 與每個 loss 的 `args` 維持開放的
  mapping，由接收它們的 constructor 驗證。
- `delegated_kwargs` 只把 recipe 實際寫出的 key 轉交給自帶預設值的元件（房間
  模擬器、RIR bank loader、VAD labeler），預設值因此只定義在一個地方。
- `curriculum` block 在載入時檢查：每條 track 都必須指向 recipe 真的有的 block、
  bank 成員或 loss。
- `puresound.recipes` 負責建立物件：encoder 與 backbone 用
  `getattr(puresound.nnet, type)`，Lightning module 用
  `getattr(puresound.system, type)`，每個 loss 用
  `getattr(puresound.nnet.loss, type)`；`MODEL_FACTORY_FOR_TASK` 依任務選擇單輸入
  或條件式（legacy 的目標語者萃取）模型形狀。

欄位參考：[Recipe 設定](../usage/configuration.zh-TW.md)。

## Model zoo

`model_zoo/catalog.yaml` **只放 metadata**：id、任務、lifecycle 與 role、音訊
契約、參數及其範圍，以及每個 artifact 的相對路徑、manifest、processor 名稱與
SHA-256。權重與 ONNX 檔留在各 recipe 的 `pretrained_ckpt/`。catalog 在載入時由
`puresound.inference.schema` 驗證。

- **Id** 是 `<task>-<arch>-<version>`，會被使用者釘在程式碼裡。改名的 id 會把舊
  拼法留在 `ModelZoo.get` 的 alias 表裡。
- **Role** 標出每個任務該用哪個模型：每個任務恰好一個可執行的 `default`
  （`ModelZoo.default_model`），旁邊可有 `candidate`、`alternative` 等其他 role。
- **位置** —— `ModelZoo.default()` 讀儲存庫內的 catalog，或 `PURESOUND_MODEL_ZOO`
  指定的檔案。
- **驗證** —— `puresound models validate` 檢查每個路徑、hash、manifest 與 ONNX
  輸入/輸出名稱；測試套件也以同樣方式驗證儲存庫內的 catalog。

## Legacy

目標語者萃取與語者 embedding 的任務 dataset 已凍結：`puresound.task.tse`、
`puresound.task.sv`、`puresound.system.miso` 與 `puresound.dataset.kaldi_base`
為既有 recipe 保留，不再開發。runner 在 `--scoring` 與 `--inference` 時透過
`KaldiFormBaseDataset` 讀取 recipe 的 `test_folder`，所以這個模組仍在評分路徑上。

## 細節頁面

| 頁面 | 涵蓋 |
| --- | --- |
| [Datasets](dataset/index.zh-TW.md) | metafile parser、`DynamicBaseDataset`、Kaldi 格式 dataset |
| [Tasks](task/index.zh-TW.md) | 降噪與人聲分離 dataset、sampler、合成階段 |
| [訓練系統](system/index.zh-TW.md) | Lightning module、optimizer 工廠、logger |
| [Utilities](utils.zh-TW.md) | `puresound.utils` 工具函式 |
