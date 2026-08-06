# Noise Suppression（雜訊抑制）

通用的單聲道語音增強（speech-enhancement）recipe：從乾淨語音語料即時合成 (noisy, clean) 配對，
訓練一個 mask-based 的 encoder/backbone/decoder 模型。這是整個 repo 裡最活躍的共用進入點——
`main.py` 同時也是另一條產品線 `egs/voice_isolate`（近場前景語者分離）的訓練進入點，純粹透過
`dataset.task` 這個 config 開關切換。細節見下方「`dataset.task` 切換」一節。

English version: [`README.md`](README.md)

## 此 recipe 中的檔案

| 檔案 | 用途 |
|---|---|
| `prepare_metafile.py` | 把 Kaldi 風格的 `wav.scp` / `utt2spk`（可加選 `utt2gender`）轉成 dataset 類別會讀取的 CSV metafile。 |
| `main.py` | 訓練／scoring／inference 的進入點。同時也是 `egs/voice_isolate/main.py` 背後真正執行的程式（見下方）。 |
| `config/dpcrn.yaml` | 範例 config：DPCRN backbone、32 kHz、complex ratio mask。 |
| `config/dparn.yaml` | 範例 config：DPARN backbone、16 kHz（路徑預設指向 VCTK+DEMAND），多加了一層 `FrequencyEQLayer` 前端。 |

這個 recipe 裡沒有 `demo.py`。

## 1. 準備 metafile

`prepare_metafile.py` 讀入三個 Kaldi 風格的文字檔（`<key> <value...>`，預設以空白分隔），輸出
一份固定表頭的 CSV metafile：

```
uttid, spkid, gender, path, length, sample rate, channels
```

```bash
uv run python egs/noise_suppression/prepare_metafile.py \
    data/train_metafile.csv \
    data/train_wav2scp.txt \
    data/train_utt2spk.txt \
    --utt2gender_path data/train_utt2gender.txt \
    --insert_root_path /corpus/root \
    --separator " "
```

| 參數 | 是否必填 | 預設值 | 說明 |
|---|---|---|---|
| `output_path` | 是（位置參數） | — | 輸出 CSV metafile 的路徑 |
| `wav2scp_path` | 是（位置參數） | — | 每行一筆 `<uttid> <wav_path>` |
| `utt2spk_path` | 是（位置參數） | — | 每行一筆 `<uttid> <spk_id>` |
| `--utt2gender_path` | 否 | `None` | 每行一筆 `<uttid> <gender>` |
| `--separator` | 否 | `" "` | 只用來分隔**輸入**檔案的欄位；輸出的 CSV 一律以逗號分隔 |
| `--insert_root_path` | 否 | `None` | 補在從 `wav2scp` 讀到的每個路徑前面的字首；如果 `wav2scp` 裡已經是可以直接使用的路徑就不用給 |

補充說明：

- 這支程式要跑兩次——train split 一次、valid split 一次——再把 `dataset.train_metafile` /
  `dataset.valid_metafile` 分別指到這兩份輸出 CSV。
- 如果沒給 `--utt2gender_path`，每一列的 gender 欄位都會是字面上的 `None`。下游
  （`DynamicBaseDataset`）會把它歸進「性別未知」這一類，不會因此被拒絕——gender 只是用來做
  性別平衡取樣，不是必要條件。
- `dataset.test_folder`（給 `--scoring` / `--inference` 用，見下方）是完全不同的格式：它是一個
  由 `KaldiFormBaseDataset` 讀取的資料夾，不是這支程式產生的 metafile。裡面要放 `wav2scp.txt`
  （`--scoring` 還要多放 `wav2ref.txt`），格式跟 `prepare_metafile.py` 讀進去的**輸入**檔一樣是
  `<uttid> <path>`——不要把它指向這支程式輸出的 CSV。

## 2. 訓練

```bash
# 從頭開始訓練
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training

# 續訓一個中斷的 run（會還原 model + optimizer + scheduler + epoch）
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training \
    --ckpt_path exp/work/lightning_logs/version_0/checkpoints/epoch=12-step=13000.ckpt

# 用另一個 checkpoint 的權重 warm-start 一個新 run（optimizer/scheduler/loss 全部重新開始）
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training \
    --pretrained_ckpt_path some_other_run/epoch=49.ckpt

# 開一個長 run 之前先檢查 augmentation pipeline（會寫出 ./dummy_samples/*.wav）
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --dump_training_samples

# 固定 random seed
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training --set_seed 1234
```

`config_path` 是位置參數（放在所有旗標之前的第一個參數）。跟 `egs/speaker_embedding` 與
`egs/target_speaker_extraction`（用 `--training True`／`type=str2bool`）不同，**這個 recipe 的
`--training`、`--scoring`、`--inference`、`--dump_training_samples` 都是單純的 `argparse`
`store_true` 旗標**——只要寫 `--training` 就好，不要寫成 `--training True`（後者會失敗：`True`
沒有任何東西可以吃掉它，argparse 會丟出 `unrecognized arguments` 錯誤）。

`--dump_training_samples` 會在 `./dummy_samples/` 底下寫出 3 個 batch；每個
`batch_XX-YY.wav` 是一個**三聲道**檔案，把該筆資料的 `[noisy_speech, clean_speech,
consistency_noise]`（`noisy - clean`）疊在一起——用能顯示逐聲道波形的工具打開（例如
Audacity），在真正跑一個長 run 之前先耳測一下 augmentation 的效果。

`config/` 裡有兩份範例 config，兩者都可以當起點——複製一份、把 `dataset.*` 指到你自己的
metafile、再調整 `augmentation_*` 各區塊：

| | `config/dpcrn.yaml` | `config/dparn.yaml` |
|---|---|---|
| backbone | DPCRN | DPARN（+ `FrequencyEQLayer` 前端） |
| sample rate | 32000 | 16000 |
| mask | complex | complex |
| `vad_label` | `silero`，frame 800 / hop 320 | `silero`，frame 400 / hop 160 |
| 範例語料 | （路徑留空、未指定特定語料） | VCTK（語音）+ DEMAND（噪音） |

這兩份範例並沒有用到 schema 裡的每一個區塊：`augmentation_codec`、
`augmentation_packet_loss`、`augmentation_target_absent`、`augmentation_realfar`、
`augmentation_realnear` 都是真實存在、有支援的區塊，只是這兩份範例都沒開。完整的區塊清單與
語意見 [`docs/task/ns.md`](../../docs/task/ns.md) 與
[`docs/task/voice_isolation.md`](../../docs/task/voice_isolation.md)。

## 3. Inference 與 scoring

這兩個階段都是把 `dataset.test_folder` 當成一個 Kaldi 風格的資料夾來讀（`KaldiFormBaseDataset`）：
`wav2scp.txt` 是必要的；`--scoring` 還需要 `wav2ref.txt`（乾淨參考音訊）才能算出分數；
`--inference` 則完全不需要參考音訊。

```bash
# 只做語音增強 -> 輸出寫到 dataset.proc_output_folder 底下
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml \
    --inference --ckpt_path exp/work/.../epoch=49.ckpt

# 計算分數（pesq_wb/nb、stoi、estoi、sisnr、bss_sdr、dnsmos_p835）
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml \
    --scoring --ckpt_path exp/work/.../epoch=49.ckpt

# 覆寫整個流程使用的取樣率（例如 ckpt 是在 32k 訓練的，但測試集是 16k）
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml \
    --inference --ckpt_path exp/work/.../epoch=49.ckpt --inference_sr 16000
```

`--ckpt_path` 在這兩個階段其實都是必要的（它會被原封不動丟給 `torch.load`）。這兩個階段建立的
都是一個乾淨的 `L.Trainer(inference_mode=True)`——下方「DDP、precision 與效能相關旗標」一節
提到的 DDP strategy、`precision`，以及 `trainer.lightning_trainer_args` 裡的其他設定，全部
**只**套用在 `--training`；`--scoring`/`--inference` 完全不會讀 `trainer.num_gpus`，一律以單一
process 執行。

## `dataset.task` 切換：`noise_suppression` 與 `voice_isolation`

`egs/voice_isolate/main.py` 並不是另一份獨立實作——它只是一個三行的轉接殼（shim）：

```python
runpy.run_module("egs.noise_suppression.main", run_name="__main__")
```

每一次 `voice_isolate` 的訓練／inference，實際執行的都是**這一支** `main.py`。兩條產品線的差別
只在於指向哪一份 config。`init_dataloader` 會從 config 讀 `dataset.task`（沒寫這個 key 時預設是
`"noise_suppression"`——這個 recipe 裡的兩份範例 config 都沒寫），並依此切換 dataset/collate：

| `dataset.task` | dataset 類別 | collate 類別 | 額外傳入的參數 |
|---|---|---|---|
| `noise_suppression`（預設） | `NoiseSuppressionDataset` | `NoiseSuppressionCollateFunc` | — |
| `voice_isolation` | `VoiceIsolationDataset` | `VoiceIsolationCollateFunc` | `augmentation_realfar_args`、`augmentation_realnear_args` |

這兩個類別都定義在 `puresound.task.ns` / `puresound.task.voice_isolation`；`VoiceIsolationDataset`
繼承自通用版本，只覆寫了它的資料列類型 hook（`_plan_row`、`_prepare_foreground`、
`_sample_interferers`……）——底層其實只有一條合成 pipeline，不是兩條。完整細節見
[`docs/task/ns.md`](../../docs/task/ns.md)（共用骨架）與
[`docs/task/voice_isolation.md`](../../docs/task/voice_isolation.md)（特化的部分）。

有兩道防呆機制在 `init_dataloader` 裡守著這個配對關係，資料還沒開始載入就會擋下來：

- `dataset.task: noise_suppression`（或沒寫）**同時** `augmentation_realfar.used: True` 或
  `augmentation_realnear.used: True` → 丟出 `ValueError("augmentation_realfar/realnear need
  dataset.task: voice_isolation")`。這兩個區塊只有在 voice-isolation 的資料列類型下才有意義。
- 其他任何 `dataset.task` 的值 → 丟出 `ValueError("Unsupported dataset.task: <value>")`。

要把一份 `noise_suppression` 的 config 改成 `voice_isolation`，加上：

```yaml
dataset:
  task: voice_isolation

augmentation_realfar:
  used: True
  prob: 0.20
  lone_far_prob: 0.15          # 完全沒有近場語者的列所佔比例（target 全為靜音）
  pool_manifest: data/realfar_pool/voices.train.jsonl
  turn_taking_prob: 0.3

augmentation_realnear:
  used: True
  prob: 0.15
  turn_taking_prob: 0.5
  pool_manifest: data/realfar_pool/voices.near.train.jsonl
```

`augmentation_realfar` 插入的是**真的喇叭→空氣→麥克風收音**當作遠場干擾者（不會再疊加 RIR——
本來就已經是收音鏈的一部分，疊加的話等於重複加了一次 LTI 效果）；`augmentation_realnear` 則是
把前景換成**真正的近距離收音**，讓混音的近場／keep 這一側也是真的。pool manifest 是每行一個
JSON object，由 `egs/voice_isolate/scripts/build_real_recording_pool.py` 產生。現役的參考 recipe
是 `egs/voice_isolate/config/train_dpcrn.yaml`；在這裡對 `init_dataloader`、DDP/precision 設定、
VAD labeler 接線方式、或 checkpoint 載入邏輯做的任何修改，會同時套用到兩條產品線上。

## DDP、precision 與效能相關旗標

在檔案最上面、任何東西執行之前就無條件設定好：

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

訓練用的是固定長度的樣本（`dataset.training_length_seconds`），所以每一步看到的 tensor 形狀都
一樣——`cudnn.benchmark` 可以放心地把每種 conv 形狀 benchmark 一次、選出最快的演算法後重複使用
（預設的啟發式演算法會給 dilated encoder 的 conv 選到一個特別慢的 `dgrad` 演算法；實測整步下來
慢了大約 3.6 倍）。TF32 則是讓 attention/linear 的矩陣乘法在 Ampere 以上的顯卡吃到 tensor core
的加速，精度代價幾乎可以忽略。

### Strategy：DDP 或 auto

```python
strategy = (
    DDPStrategy(
        gradient_as_bucket_view=True,
        find_unused_parameters=trainer_dict.get("find_unused_parameters", False),
    )
    if trainer_dict["num_gpus"] > 1
    else "auto"
)
```

- `trainer.num_gpus > 1` → 用 `DDPStrategy`；`<= 1` → 用 Lightning 的 `"auto"`（單一裝置或
  CPU）。
- `gradient_as_bucket_view=True` 在 DDP 下一律開啟：它讓 all-reduce 直接就地讀 bucket 裡的
  梯度，可以消掉一個「grad strides do not match bucket view strides」的警告（cuDNN 的 1x1 conv
  weight-grad layout 觸發的），同時降低記憶體用量。
- `find_unused_parameters` 是 config 驅動的，key 是 `trainer.find_unused_parameters`（預設
  `False`）。`config/dpcrn.yaml` 跟 `config/dparn.yaml` 都沒設這個 key——單純的雜訊抑制模型每一
  步都會用到全部參數。當模型裡有**依資料而定**、某些 batch 根本不會碰到的參數群組時才需要打開
  它——例如一個 cold-start 的 VAD gate head，或是只有特定列才會算的 distance 輔助 head。
  `egs/voice_isolate` 底下每一份訓練 config 都因為這個理由設了
  `trainer.find_unused_parameters: true`。該開的時候沒開，DDP 會直接報錯；不需要的時候開了，
  則是每一步白白多走一次 autograd 圖的走訪。

### Precision

`trainer.lightning_trainer_args` 會直接展開丟進
`lightning.Trainer(**trainer_dict["lightning_trainer_args"], ...)`，所以它接受**任何**
Lightning 的 `Trainer` 參數，不只是兩份範例 config 用到的那幾個（`max_epochs`、
`gradient_clip_val`、`accumulate_grad_batches`）。precision 預設是 Lightning 的全精度
`32-true`。想用一點點精度換速度/記憶體的話：

```yaml
trainer:
  lightning_trainer_args:
    precision: bf16-mixed
```

（`egs/voice_isolate/config/train_dpcrn.yaml` 就是這樣設的。）這裡選 bf16 而不是 fp16，是因為
mask/loss 路徑裡的 complex-spectral 的振幅/除法運算需要 fp32 的數值範圍、也不需要
`GradScaler`；實測開啟後每一步快了大約 1.57 倍，Ampere 上的 activation 記憶體用量少了大約
40%。

還有兩個 `Trainer` 參數是寫死的、不是 config 驅動：`sync_batchnorm=True` 跟
`use_distributed_sampler=False`（batch 已經由自訂的 `SpeakerSampler` 處理好了，不能再讓
Lightning 額外包一層 distributed sampler）。

## GPU 端的 Silero VAD 標註

```yaml
vad_label:
  used: True
  backend: silero
  args:
    frame_length: 800     # 每個標註 frame 的取樣數 -- 要跟 encoder 的 hop_length 對齊
    hop_length: 320
    threshold: 0.5
    min_overlap: 0.5
    model_sample_rate: 16000
```

兩份範例 config 都開了這個功能（`dpcrn.yaml` 用 800/320，`dparn.yaml` 用 400/160——各自對齊
自己 encoder 的 `hop_length`）。有兩種 backend：

- `backend: energy`（`vad_label.backend` 沒寫時的預設值）——直接在 dataset 裡、於 DataLoader
  worker process 上，用 CPU 標註。
- `backend: silero`——一個真的神經網路 VAD，在 CPU worker 裡逐筆跑太重，所以整個標註流程被搬出
  dataset，改成在 batch 已經搬到 device 之後，**整批、在 GPU 上**跑：

```python
if vad_label_dict.get("backend", "energy").lower() == "silero":
    from puresound.audio.vad import BatchedSileroVADLabeler
    lightning_model.register_gpu_vad_labeler(
        BatchedSileroVADLabeler(**vad_label_dict.get("args", {}))
    )
```

`register_gpu_vad_labeler`（在 `puresound/system/base.py`）把這個 labeler 包在一個普通的
Python list 裡存起來（`self._gpu_vad_labeler = [labeler]`），就是刻意不讓 `nn.Module` 把 Silero
的 TorchScript 模型註冊成 submodule——它絕對不能跑進 `state_dict()`／checkpoint 裡，它只是個
標註工具，不是要訓練的權重。Lightning 的 `on_after_batch_transfer` hook 之後每一步都會呼叫
`ensure_vad_targets(batch)`：如果 batch 裡還帶著原始的 `vad_reference`（或
`background_vad_reference`）波形——只要這個 backend 有開，dataset 就不會自己算標註、而是把這
一步留到這裡——labeler 就會對整個 batch 跑一次，生出 `vad_target`（／
`background_vad_target`）。這需要額外裝 `silero-vad` 這個 optional 套件
（`uv pip install silero-vad`）；沒裝的話會丟出清楚的 `RuntimeError`，告訴你要裝它或是改回
`backend: energy`。算出來的標註會餵給 `VADActivityLoss`（兩份範例 config 的 `loss_func` 裡都
有），在 `voice_isolation` 的 config 裡還會餵給 background-VAD head。

## Checkpoint 載入：兩個旗標背後的三種機制

`--ckpt_path` 跟 `--pretrained_ckpt_path` 都是接一個 `.ckpt` 檔的路徑，但兩者不能互換
——而且 `--ckpt_path` 本身的意義還要看它跟哪個階段旗標一起用：

| 旗標 | 階段 | 機制 | 還原了什麼 |
|---|---|---|---|
| `--ckpt_path` | `--training` | `trainer.fit(..., ckpt_path=...)`（Lightning 原生的 resume） | model + optimizer + scheduler + epoch/global step + callback 狀態 |
| `--pretrained_ckpt_path` | `--training` | `lightning_model.load_state_dict(state_dict, strict=False)` | 只還原名字對得上的權重；optimizer/scheduler/loss 全部照目前的 config 重新開始 |
| `--ckpt_path` | `--scoring` / `--inference` | `lightning_model.reload_checkpoint(state_dict)` | 只還原名字對得上的權重，一個一個 key 複製過去；沒有其他東西（inference 階段本來就沒有 optimizer） |

### Warm start（`--pretrained_ckpt_path`，只在訓練階段）

```python
state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = lightning_model.load_state_dict(state_dict, strict=False)
```

`strict=False` 代表形狀或名字對不上不會讓載入直接爆掉：**目前**模型有、但 checkpoint 裡沒有的
參數，會靜靜地保持隨機初始化的狀態（`missing`）；checkpoint 裡有、但目前模型沒有的參數，則會
被靜靜地丟掉（`unexpected`）。`main.py` 只要這兩個數量不是零就會印出來（還會附上幾個名字當
樣本），這樣 config 設錯的時候才不會悄悄過去——例如 warm-start 進一份只是「多加了一個
head」的 config，實際上只會印出：

```
[pretrained] 2 new param(s) kept at init: ['backbone.vad_head.weight', 'backbone.vad_head.bias']
```

（「ckpt param(s) ignored」那一行在 `unexpected` 是空的時候完全不會印；只有當 checkpoint 裡
還帶著目前模型已經沒有的參數時才會出現，例如某個 head 被改名或移除。）

當目前 config 的模型比你要接續的那個 checkpoint「長大了」時就該用這個——例如 warm-start 進一份
多加了新輔助 head（VAD gate head、distance head……）的 config，而來源 checkpoint 從來沒有這些
head；或者更廣義地說，任何你只想把舊權重當成一個初始化 prior、其餘（新的 loss 權重、新的
optimizer/scheduler、新的 curriculum 階段）都想重新設定的場合都適用。`egs/voice_isolate` 那條
分階段訓練的 pipeline 就是這樣運作的：每個階段的 config 都是用 `--pretrained_ckpt_path` 從上一
階段的 checkpoint warm-start，而 loss/augmentation 的設定則在底下持續改變。

### Resume（`--training` 時的 `--ckpt_path`）

直接傳給 `trainer.fit(..., ckpt_path=...)`——這是 Lightning 自己的機制，用來在**同一份** config
下延續**同一個** run：同樣的 optimizer 狀態、同樣的 scheduler 進度、同樣的 epoch 數。這是用來
接續一個中斷的 run，不是用來換一套新的訓練配方。

### Scoring/inference 的重新載入（`--scoring`/`--inference` 時的 `--ckpt_path`）

```python
state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
lightning_model.reload_checkpoint(state_dict)
```

這是第三種、完全不同的機制（`BaseLightningModule.reload_checkpoint`，不是
`nn.Module.load_state_dict`）：它把載入的每個參數依名字複製進剛建立好的模型裡，複製不了的會印出
`"{name} is not in the model."`；最後如果目前模型有哪些參數 checkpoint 完全沒提供，會印出
`"Needed param name but missing: [...]"`。這裡沒有 optimizer/scheduler 的問題
——`--scoring`/`--inference` 建立的本來就是一個乾淨的 `L.Trainer(inference_mode=True)`。

## 函式庫參考文件

| 文件 | 內容 |
|---|---|
| [`docs/task/ns.md`](../../docs/task/ns.md) | `NoiseSuppressionDataset` / `NoiseSuppressionCollateFunc` / `RowPlan`——這個 recipe 驅動的合成骨架，以及每個 `augmentation_*` 區塊的意義。 |
| [`docs/task/voice_isolation.md`](../../docs/task/voice_isolation.md) | `VoiceIsolationDataset` / `VoiceIsolationCollateFunc`——`voice_isolation` 這個 task 的特化部分、`augmentation_realfar`/`augmentation_realnear` 的 schema、輸出的逐筆標籤。 |
| [`docs/system/siso.md`](../../docs/system/siso.md) | `EncDecMaskBase` / `EncPredClassBase`——兩份範例 config 都用到的 Lightning module 型別（`model.lightning_module.type`），以及 encoder → features → backbone → mask → decoder 的架構。 |
