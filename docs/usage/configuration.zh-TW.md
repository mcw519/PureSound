# Recipe 設定

English: [configuration.md](configuration.md)

PureSound 透過 `puresound.config.load_recipe` 載入 YAML recipe。Pydantic 在建立任何
dataset、模型或 trainer 之前就驗證完整份 recipe，所以打錯字或缺欄位會在載入時就失敗，
而不是跑了幾個小時才出事。

## 必填欄位

每份 recipe 都要宣告 schema 版本、用途與任務：

```yaml
schema_version: 2
purpose: train
task: voice_isolation
```

| 欄位 | 值 |
| --- | --- |
| `purpose` | `train`（完整的訓練 recipe）或 `inference`（只含模型：`dataset.target_sample_rate`、`trainer.work_folder` 與 `model`，給 demo、benchmark 與串流匯出用） |
| `task` | `noise_suppression`、`voice_isolation`、`speaker_embedding`、`target_speaker_extraction` |

未知欄位與型別不符的純量都是錯誤。`speaker_embedding` 與 `target_speaker_extraction`
是凍結的 legacy 任務。

## 載入 recipe

```python
from puresound.config import load_recipe

recipe = load_recipe(
    "egs/voice_isolate/config/train_dpcrn.yaml",
    expected_task="voice_isolation",
    expected_purpose="train",
)
```

進入點會兩個預期值都傳，所以一份合法的 config 不會被錯的 runner 拿去用：
`egs/noise_suppression/main.py` 會拒絕 `voice_isolation` 的 recipe，反之亦然。

驗證檢查的是結構與欄位之間的關係。音檔、manifest、CUDA 與選用的 backend，要等到建構
管線時才檢查。

## 任務專屬欄位

recipe 的模型只暴露該任務會用到的區塊；別的任務的區塊就是未知欄位。

| 任務 | 變速設定 | 任務專屬區塊 |
| --- | --- | --- |
| 噪音抑制 | 連續的 `augmentation_speed.speed_range` | `augmentation_codec`、`augmentation_packet_loss`、`augmentation_target_absent`、`augmentation_row_initial_ambient` |
| 人聲隔離 | 連續的 `augmentation_speed.speed_range` | 同上，再加 `augmentation_speech.mix_mode`、`augmentation_realfar`、`augmentation_realnear`、`augmentation_session_rows` |
| 語者嵌入 | 離散的 `augmentation_speed.speed_change` | `treat_as_new_speaker` |
| 目標語者擷取 | 連續的 `augmentation_speed.speed_range` | `enroll_speech`、`signal_loss_func`、`class_loss_func` |

`model`、`loss_func`、`optimizer`、`scheduler` 底下的建構子參數，由被選中的元件在建構時
自行驗證（見 [recipes.zh-TW.md](recipes.zh-TW.md)）。

## Dataset 角色

`dataset_role` 標示這是 train、validation 還是 test dataset。
`dataset.train_pipeline_role` 與 `dataset.validation_pipeline_role` 選擇隨階段而異的
資源，例如 RIR bank 的切分。

驗證預設使用 validation 角色。要對著訓練時的房間分佈驗證，就明確設定：

```yaml
dataset:
  validation_pipeline_role: train
```

## Sampler 旋鈕

`trainer` 區塊有兩個會改變訓練 batch 組成的旋鈕；驗證兩者都不理會，所以驗證 loss 跨
epoch 仍可比。

```yaml
trainer:
  n_spk_per_batch: 4          # 驗證用，也是沒有 length_schedule 時的 batch
  length_schedule:            # 訓練：每個 batch 抽一組（長度, batch 大小）
    - {seconds: 6,  n_spk: 4, prob: 0.6}
    - {seconds: 12, n_spk: 2, prob: 0.3}
    - {seconds: 30, n_spk: 1, prob: 0.1}
  speaker_source_weights: {dns5_: 0.7, ll_: 0.3}
```

- `length_schedule` 讓模型在多種上下文長度下受監督，而不是只有一種。batch 大小跟著長度
  走，因為 activation 記憶體就是跟著長度走；機率總和必須是 1。長的列需要語句本身就那麼
  長的語料。
- `speaker_source_weights` 依 spkid 前綴決定每個語料佔 batch 的比例，而不是交給語者數
  決定。它的規則，以及噪音與 SNR 旋鈕 `augmentation_noise.noise_sources`、`snr_bands`，
  見 [data_preparation.zh-TW.md](data_preparation.zh-TW.md#這些工具餵給哪些-recipe-旋鈕)。
- `train_sampler: coverage`（預設 `speaker`）讓訓練 batch 從洗牌後的佇列依序取：每個
  來源的語者都輪過才會重複，每位語者也輪完該列長的合格句子才會重複，長列只取夠長的
  句子。Checkpoint 保存抽樣位置，可在 epoch 邊界 resume 或接續 warm-start 的下一階；
  epoch 中途的 checkpoint 須用 warm start。Resume 要求階段設定與合格佇列不變。
  需要 `n_utt_per_speaker: 1` 且設了 `target_sample_rate`。見
  [task.sampler](../architecture/task/sampler.zh-TW.md#class-coveragesampler)。

## Curriculum

選用的 `curriculum` 區塊依 epoch 改變指定的值，讓一次 run 就能表達原本要靠一串
warm-start run 才做得到的事：

```yaml
curriculum:
  used: true
  tracks:
    - path: "loss:OverSuppressionLoss"
      interp: linear
      points: [[0, 0.0], [30, 3.0]]
    - path: "aug:augmentation_noise.prob"
      interp: step
      points: [[0, 0.0], [40, 0.8]]
    - path: "bank:wide"
      interp: linear
      points: [[0, 0.1], [40, 0.5]]
```

| 前綴 | 目標 |
| --- | --- |
| `aug:` | 允許清單上的增強欄位（`puresound/config/curriculum.py` 的 `SCHEDULABLE_AUGMENTATION_PATHS`） |
| `bank:` | union RIR bank 中一個具名成員；權重是相對的，會重新正規化 |
| `loss:` | 已註冊的 loss 型別，或 `loss_func` 的 `#index` |

`linear` 在點之間內插。`step` 維持前一個值直到下一個點。超出點範圍的值取最近的端點。

限制：

- 排程的是區塊的機率，不是它的 `used` 旗標。合成會在隨機抽取**之前**就跳過停用的區塊，
  所以切換 `used` 會讓之後的每一次抽取都錯位。由排程引入的區塊要設 `used: true`，機率
  設在排程的第一個值。
- 路徑、bank 佈局、快取大小，以及其他在建構時讀取的東西都不能排程；允許清單會拒絕它們。
- 驗證資料不跟隨訓練 curriculum：它讀檔案裡的常數。
- 續跑的訓練從還原的 epoch 繼續。
- 權重為零的 loss 仍會被計算，讓有狀態的 loss 保持最新。

不合法的 curriculum 目標——停用的區塊、不在 union 裡的 bank 成員、沒註冊的 loss——會在
載入 recipe 時失敗。訓練開始時，run 會把每條 track 在 epoch 0 的值寫進 log。

## 新增一個欄位

1. 在 `puresound.config` 的 capability model 加上欄位。
2. 只在用得到它的任務裡納入該 capability。
3. 跨欄位驗證放在 capability 旁邊。
4. 同一次變更裡更新現役 YAML 與測試。
