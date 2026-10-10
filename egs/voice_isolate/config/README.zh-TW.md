# voice_isolate 設定檔

English version: [`README.md`](README.md)

這裡的每個設定檔都可以直接拿來用。

| config | 用途 |
|---|---|
| `train_dpcrn.yaml` | **預設訓練 recipe。**單一 curriculum run 從零訓練，不需要 warm-start checkpoint。 |
| `train_dpcrn_curriculum_v1.yaml` | 這條血統的第二步；從 `train_dpcrn.yaml` 那次 run 的 ep99 checkpoint（`dpcrn_curriculum_v0`，不隨 repo 發佈）warm-start，加入 session 列、成對擷取視角，以及逐幀 proximity／presence 頭。產出已發布的 `dpcrn_curriculum_v1`。 |
| `train_dpcrn_curriculum_v2_base.yaml` | 換成加寬模型、每個 epoch 1.5 倍列數的 `train_dpcrn.yaml`，從零訓練 80 epoch；它的 ep79 是下一列的 warm-start 起點。 |
| `train_dpcrn_curriculum_v2.yaml` | 換成加寬模型、每個 epoch 1.5 倍列數的 `train_dpcrn_curriculum_v1.yaml`，從 base run 的 ep79 warm-start。產出已發布的 `dpcrn_curriculum_v2`。 |
| `infer_dpcrn.yaml` | 預設推論設定；載入 `dpcrn_v8` 與 `dpcrn_curriculum_v1`（`../pretrained_ckpt/` 底下除了 `dpcrn_curriculum_v2` 以外的每一個 checkpoint） |
| `infer_dpcrn_wide.yaml` | 加寬模型的推論設定：載入 `dpcrn_curriculum_v2` |
| `infer_dpcrn_heads.yaml` | 同 `infer_dpcrn.yaml`，用於會輸出 VAD 側頭的 checkpoint |
| `eval/` | `../run_full_benchmark.sh` 驅動的評測設定 |

第一次執行前，請先把語料與 RIR bank 路徑指到你自己的資料：
[`../DATA_SETUP.zh-TW.md`](../DATA_SETUP.zh-TW.md)。

```bash
# 訓練 config 裡的資料與 work folder 路徑相對於 recipe 目錄
cd egs/voice_isolate

# 從零訓練
uv run python main.py config/train_dpcrn.yaml --training

# 接著（選用）從那次 run 的 ep99 checkpoint 跑 curriculum 的第二步
uv run python main.py config/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path exp/dpcrn_curriculum/lightning_logs/version_0/checkpoints/epoch=99-*.ckpt

# 推論／demo
uv run python scripts/demo.py --config_path config/infer_dpcrn.yaml
```

用 `--ckpt_path <ckpt>` 取代 `--pretrained_ckpt_path` 才是真正的 resume（還原
optimizer/scheduler/epoch）。已發布的推論設定包含 `dry_blend 0.9`——見
`../pretrained_ckpt/README.md`。

## 共用設計（這裡的所有 config 皆適用）

- **Backbone DPCRN**，complex ratio mask，16 kHz 原生：`channels [2,32,64,128]`，
  `rnn_hidden 96`，約 0.8 M 參數；兩個 `curriculum_v2` recipe 與 `infer_dpcrn_wide.yaml`
  把它加寬為 `channels [2,48,96,128]`、`rnn_hidden 128`，約 1.2 M 參數。
- **30 ms look-ahead**：`backbone.delay=[1,1,1]`（3 幀），inter-RNN 為
  unidirectional，因此 look-ahead 有上限。streaming ONNX 匯出用烘進 graph 的
  future-buffering 處理它（`../scripts/streaming_onnx.py`）。
- **Early target**（`target_rir_type: early`）：target 是去混響後的近場語音，
  所以 passthrough 無法直接命中它，分離仍是一個真正的目標。
- **Hard SIR** `[-10, 10]` 加上 `mix_mode`：前景可能比干擾者還安靜最多 10 dB，
  因此光靠音量無法解這個任務。
- **Measured-capture realism 開啟**（僅預設 recipe）：噪音與語音卷積同一個
  房間、絕對 dBFS 麥克風底噪，以及讓 `mix_mode distance_level` 從場景幾何
  抽出 SIR。沒有這些的合成資料，乾淨程度會不切實際地遠勝真實錄音，這樣只會
  教會模型「乾淨就等於可壓制」而不是「遠就等於可壓制」。
- **合成 `target_absent`：關閉。** 強制靜音的列會教模型「不確定時就輸出
  靜音」，這在訓練域之外會誤發。絕對抑制的監督改由真實錄音列提供
  （`augmentation_realfar.lone_far_prob`、turn-taking）。
- **Anti-deletion 損失**：`OverSuppressionLoss` + 兩個 `ASRFeatureLoss` 項
  （HuBERT + WavLM，cosine）+ `SDRLoss` + `MultiResolutionSTFTLoss` +
  `ResidualReferenceLoss`。
- **Scheduler `CosineAnnealingWarmRestarts T_0=20`**——每 20 epoch 重啟一次，
  所以只在 cosine 谷底（ep19/ep39/ep59）比較 checkpoint。optimizer 只訓練
  backbone；encoder/features 維持凍結。

## `eval/`

僅供 eval 的 config 用相同的 augmentation pipeline，只換掉 RIR bank 或單一 flag，讓任何
checkpoint 都能被 benchmark。從不用於訓練。

| config | 用途 |
|---|---|
| `eval_but_real.yaml` | 實測 RIR benchmark，RT60 1.15–1.84（遠超訓練域）；WER 各關用它的 `model:` 建模型 |
| `eval_targetabsent_probe.yaml` | far-only／noise-only 洩漏探針（強制開啟 `augmentation_target_absent`） |
| `eval_indomain_phase1.yaml` | in-domain SI-SDRi + solo-leakage + turn-taking buckets |
| `eval_targetabsent_probe_high.yaml` | 未見高殘響房間上的 far-only 探針 |
| `eval_targetabsent_probe_boundary.yaml` | 未見邊界距離上的 far-only 探針 |

最後三個是**逐位元組凍結的 fixture**：它們透過 `../run_full_benchmark.sh` 定義了
每一次留存判準所用的分布。不可編輯。工具說明：`../scripts/README.md`。
