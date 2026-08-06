# voice_isolate 設定檔

English version: [`README.md`](README.md)

這裡只有兩個檔案是**預設值**：可直接拿來用的調校結果。`exp/` 底下的一切都是
recipe 的歷史紀錄——較早的 pipeline 階段、消融實驗與僅供 eval 用的 fixture。

| config | 用途 |
|---|---|
| `train_dpcrn.yaml` | 預設訓練 recipe（產出已發布的 `dpcrn_v8` checkpoint） |
| `infer_dpcrn.yaml` | 預設推論設定；可載入 `../pretrained_ckpt/` 底下任何一個 checkpoint |

```bash
# 訓練（從 repo root），從前一個發布版本 warm-start
uv run python egs/voice_isolate/main.py egs/voice_isolate/config/train_dpcrn.yaml --training \
    --pretrained_ckpt_path egs/voice_isolate/pretrained_ckpt/dpcrn_v7.ckpt

# 推論／demo
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/infer_dpcrn.yaml
```

用 `--ckpt_path <ckpt>` 取代 `--pretrained_ckpt_path` 才是真正的 resume（還原
optimizer/scheduler/epoch）。已發布的推論設定包含 `dry_blend 0.9`——見
`../pretrained_ckpt/README.md`。

## 共用設計（這裡的所有 config 皆適用）

- **Backbone DPCRN**，complex ratio mask，`channels [2,32,64,128]`，
  `rnn_hidden 96`，約 0.8 M 參數，16 kHz 原生。
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

## `exp/`

Recipe 歷史紀錄，保留下來是為了讓任何一個階段都能被重現、重新判準或重新
warm-start。訓練或執行預設模型並不需要它。

**Pipeline 各階段**（每階段皆從前一階段的 checkpoint warm-start；
`../pretrained_ckpt/README.md` 的版本表對應了 recipe → checkpoint → 結果）：
`train_dpcrn_curriculum_core.yaml` → `train_dpcrn_curriculum_expand.yaml` →
`train_dpcrn_antisup_w1.yaml` → `_w2` → `_w3` → `train_dpcrn_wide_antisup.yaml`，
接著是真實錄音回合 `train_dpcrn_realE2E.yaml` → `_v2` → `_v2b` → `_v2c`
（產出 `dpcrn_v7`），再來是升格為預設的真實化回合
`../train_dpcrn.yaml`（`dpcrn_v8`）。

**其他分支與旁支：**

| config | 說明 |
|---|---|
| `train_dpcrn_realE2E_v2c.yaml` | 產出 `dpcrn_v7` 的 recipe；已被真實化設定取代預設地位，但保留可重現 |
| `train_dpcrn_wide_causal.yaml` | 全 causal 變體（`delay=[0,0,0]`），零 look-ahead；未訓練，因為 future-buffering 已在不靠它的情況下解決 streaming 問題，故只作為留存的 fallback 方案 |
| `train_dpcrn_gate.yaml` | 凍結分離器，只訓練 causal 逐幀 VAD gate head |
| `train_dpcrn_v2_sepgate.yaml` | 分離器＋gate head 在障礙物豐富的 RIR bank 上聯合訓練 |

Gate 系列的 recipe 在模擬資料上能把 gate 學得很好，但這個 gate 在真實錄音上
關不起來，而且透過聯合訓練硬推遠場壓制會提高真實音訊的 deletion；把模擬 gate
分數當工程訊號看待即可，不要當作可部署的依據。

**僅供 eval 的 config**（`eval_*.yaml`）用相同的 augmentation pipeline，只換掉
RIR bank 或單一 flag，讓任何 checkpoint 都能被 benchmark。從不用於訓練。

| config | 用途 |
|---|---|
| `eval_but_real.yaml` | 實測 RIR benchmark，RT60 1.15–1.84（遠超訓練域） |
| `eval_heldout.yaml` | 未見房間泛化，分布與 expand 階段相同 |
| `eval_targetabsent_probe.yaml` | far-only／noise-only 洩漏探針（強制開啟 `augmentation_target_absent`） |
| `eval_indomain_phase1.yaml` | in-domain SI-SDRi + solo-leakage + turn-taking buckets |
| `eval_targetabsent_probe_high.yaml` | 未見高殘響房間上的 far-only 探針 |
| `eval_targetabsent_probe_boundary.yaml` | 未見邊界距離上的 far-only 探針 |

最後三個是**逐位元組凍結的 fixture**：它們透過 `../run_full_benchmark.sh` 定義了
每一次留存判準所用的分布。不可編輯。工具說明：`../scripts/README.md`。

`exp/backup/`——僅為可重現性保留的已取代 config（死路實驗、pre-DPCRN 的
query/FiLM/VAD 變體、ASR-loss 消融、已結案的回合）。conformer/distance-query
這條軸線的 library 程式碼已從 `puresound/` 移除，所以這些檔案只記錄歷史，
已不是可執行的 recipe。
