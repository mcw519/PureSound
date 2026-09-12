# voice_isolate — 近場前景語音分離

English version: [`README.md`](README.md)

單聲道、**免註冊（enrollment-free）、近場（<1 m）前景**語音分離：保留近距語者，
壓制遠距／競爭語者與噪音。唯一可用的線索是**近／遠 DRR (direct-to-reverberant
ratio) 對比**。目標效果 = ai-coustics Voice Focus 2.0。

**Backbone：DPCRN**（complex ratio mask、16 kHz、約 0.8 M 參數、30 ms look-ahead）。
之所以選它，是因為 overfit 測試加上全量訓練比較顯示：原本的 TS-Conformer
（mapping head）無法分離困難的近／遠案例，而 DPCRN/DPARN 可以——見下方
「Pre-DPCRN 沿革」。

## v1–v8 階梯的終點：`pretrained_ckpt/dpcrn_v8.ckpt` + `dry_blend 0.9`

> model zoo 現在的預設是另一條 curriculum 血統的 `dpcrn_curriculum_v1.ckpt`；`dpcrn_v8`
> 在旁邊登錄為跨錄音鏈的保守替代方案。本節描述的是下方那條階梯，v8 是它的最後一階。
> 兩個現役版本與已封存版本：[`pretrained_ckpt/README.zh-TW.md`](pretrained_ckpt/README.zh-TW.md)。

以 `config/train_dpcrn.yaml`（下方第 8 階段）訓練，發布時**把 runtime blend 當成
設定的一部分**——`out = 0.9 * enhanced + 0.1 * input`，把任何一點的衰減量都限制在
−20 dB 以內。第 7 階段是第一個能在**真實錄音**中壓制遠場語音、同時保留近場語音
的版本；第 8 階段在相同錄音上把這個壓制再加深 **2 dB**（>1 m 中位數 −15.91 對
−13.72 dB），並補平了讓第 7 階段距離響應變非單調的 2–3 m 凹陷，同時近場語音維持
不變，且對真實錄音的 ASR 錯誤率是**降低**而非提高（Dawn Chorus WER 0.180 對未
處理的 0.184，deletion 0.094）。Streaming 匯出已驗證
（`pretrained_ckpt/streaming/dpcrn_v8.{onnx,json}`，30 ms 延遲）。

有一個取捨要留意：在遠超出訓練域的殘響（RT60 > 1 s）上，第 8 階段的 WER 比中性
的第 7 階段高 0.020。極高殘響的部署場景原本建議選 `dpcrn_v7.ckpt`，但它已封存在
`pretrained_ckpt/backup/` 而不隨 repo 發佈。

真正造成差異的是**決策兩端都用上真實錄音**，加上 **turn-taking 監督**，而不是
更多或更好的 RIR。在此之前試過的每一個模擬遠場 rung（邊界距離、訓練中加入實測
RIR、只練 gate 的 VAD head、分離器＋gate 聯訓）都能把 RIR-convolution 這個域學得
任意好，卻不會遷移到真實錄音；而且在那個域裡把遠場壓制推得更狠，會讓真實聲學的
deletion 爆掉（reverberant-office WER 0.663 對 0.529）。這些 rung 都已結案；
這些路線已結案。

`dpcrn_v6.ckpt` 原本是 fallback（沒有 runtime 旋鈕、對遠場壓制最保守），同樣已封存。各版本的
細節、結果與部署說明見 `pretrained_ckpt/README.md`。

## Pipeline（9 個階段，每階段皆從前一階段 warm-start）

所有階段共用**同一套 DPCRN 架構**（`channels [2,32,64,128]`、`rnn_hidden 96`、
look-ahead `delay=[1,1,1]`）——只有 RIR bank、augmentation 與 loss 權重會變。只在
`CosineAnnealingWarmRestarts` 的 cosine 谷底（ep19/ep39/ep59）判準；週期中段的
epoch 會被 LR 擾動。

| # | 階段 | config | warm-start 來源 | 判準 ckpt | 主要結果 |
|---|---|---|---|---|---|
| 1 | curriculum core（RT60 0.20–0.45，DRR gap≥6dB） | internal | cold | `dpcrn_v1.ckpt` | in-domain SI-SDRi 中位數 **+6.98**（ep19 +5.47 → ep39 +6.98） |
| 2 | curriculum expand（RT60 ≤0.65，gap≥3dB） | internal | stage 1 ep39 | `dpcrn_v2.ckpt` | in-domain **+7.99**（ep19 +7.40→ep39 +7.75→ep59 +7.99）；BUT real-RIR **+5.46**；第一次非中性的 real-WER 勝出：BUT enh **0.777 < mix 0.796** |
| 3 | anti-suppression weight 1.0（`OverSuppressionLoss`） | internal | stage 2 ep59 | `dpcrn_v3.ckpt` | in-domain **+8.21**；large-v3 BUT deletion **0.308→0.291**（方向確認，幅度不大） |
| 4 | anti-suppression weight 2.0 | internal | stage 3 ep19 | `dpcrn_v4.ckpt` | in-domain **+8.32**；BUT deletion **→0.276**；兩個域中最穩健 |
| 5 | anti-suppression weight 3.0 | internal | stage 4 ep19 | `dpcrn_v5.ckpt` | in-domain **+8.45**；**部署級殘響上表現最佳**（moderate enh 0.372，全體最佳）但**極端 OOD 上表現最差**（BUT enh 0.692，比 w2 的 0.676 還退步）——「domain split point」 |
| 6 | wide-domain 部署（RIR 0.20–0.85 + media_voice/hpf 真實化） | `test/fixtures/recipes/train_dpcrn_wide_antisup.yaml` | stage 5 ep19 | `dpcrn_v6.ckpt` **(fallback)** | held-out unseen-room **+8.06**（歷來最佳）；部署 hard-gate 通過（moderate enh 0.373 ≈ w3）；BUT 仍未突破（0.680，目標是 <0.676）——5 項判準過 4 項 |
| 7 | 決策兩端都用真實錄音 + turn-taking + distance 輔助 head + channel consistency | internal | stage 6 ep19 | `dpcrn_v7.ckpt` | held-out 真實遠場 **−9.45 dB，依距離分級**；keep 側持平（−0.11）；**Dawn WER 0.174 < 0.184 raw**，deletion 0.088；reverberant-office WER −0.024 vs mix；in-domain +8.18——需要 `dry_blend 0.9` |
| 8 | 合成中的量測式擷取真實化（room-colored 噪音、絕對 dBFS 底噪、幾何驅動的 SIR） | internal | stage 7 ep19 | `dpcrn_v8.ckpt` | 真實遠場 **−15.91 dB**，優於第 7 階段在同一批檔案上的 −13.72（排除洩漏子集為 −17.74 對 −15.68）；補平 2–3 m 凹陷（−4.50 → −16.61）使分級單調；keep 側持平（0.00），最差個案 −5.45 → −1.06；**Dawn WER 0.180 < 0.184 raw**，deletion 0.094；reverberant-office WER −0.024 vs mix；in-domain +7.99——代價是極端殘響下 WER +0.020；需要 `dry_blend 0.9` |
| 9 | DRR 對比，接著冷啟動課程（訓練列以真實遠場獨白開場、沒有近場錨） | internal | stage 8 ep19 | `dpcrn_v9.ckpt`、`dpcrn_v10.ckpt` | **兩者都沒有取代 stage 8。** v9 是 ASR 閘最優點：reverberant-office WER −0.050 vs mix（v8 的兩倍）、Dawn 0.172 / deletion 0.086、極端殘響中性——代價是真實遠場壓制淺 3.5 dB。v10 是唯一推動「機器閒置冷啟動」的版本（10 個孤立遠場 clip 中 5 個跨過 −6 dB，v8 只有 1 個），但為此在部署殘響範圍的 WER 上付出代價（配對比 v8 差 0.024，區間不跨過 0），且 turn-taking KEEP 違規 6 → 10。|

第一次執行前，請先把 recipe 裡的語料與 RIR bank 路徑指到你自己的資料：
[`DATA_SETUP.zh-TW.md`](DATA_SETUP.zh-TW.md)。

執行方式：**從本目錄執行**——config 裡的 metafile 與 work folder 路徑都是相對於它的：
```bash
cd egs/voice_isolate

# 預設 recipe：單一 curriculum run 從零訓練，不需要 warm start
uv run python main.py config/train_dpcrn.yaml --training

# 選用的第二步，從第一步 warm-start
uv run python main.py config/train_dpcrn_curriculum_v1.yaml --training \
    --pretrained_ckpt_path pretrained_ckpt/dpcrn_curriculum_v0.ckpt
```
用 `--ckpt_path <ckpt>` 取代 `--pretrained_ckpt_path` = 真正的 resume（還原
optimizer/scheduler/epoch）。
設定細節（共用設計、僅供 eval 的變體、推論設定）：`config/README.md`。

## 為什麼 weight-3.0 不是全面勝出（「domain split point」）

把 `OverSuppressionLoss` 權重從 1.0→2.0→3.0 逐步拉高，deletion（過度抑制）單調
下降，且 in-domain SI-SDRi 每一步都在*提升*——但到權重 3.0 時，兩個評估域開始
分歧：**部署級殘響**（rt60 0.44，in-domain）持續變好，而**極端 OOD 殘響**（BUT
real RIR，rt60 1.15–1.84）卻*變差*（substitution/insertion 上升得比 deletion
下降還快）。若只看 SI-SDRi 會說「繼續推」——但 WER（真實可懂度）說該停手了。
第 6 階段的解法是拓寬訓練用的 RIR 域（0.20→0.85），把 OOD 邊界往外推，而不是
繼續調 loss 權重。

## Streaming 部署（ONNX、real-time）

`pretrained_ckpt/streaming/dpcrn_v8.{onnx,json}`（加上 `dpcrn_curriculum_v1`）——
用 `scripts/streaming_onnx.py` 建置的逐幀 streaming 匯出。look-ahead
（`delay=[1,1,1]`，30 ms）由**烘進 ONNX graph 當成額外 state 的 future-buffering**
處理（inter-LSTM warmup gate + U-Net skip 延遲線 + noisy-spectrum 延遲）——不需要
改動任何 runtime 程式碼；既有的 manifest-driven SDK runtime
（`sdk/python/puresound_streaming`，`processor: stft_frame_ort`）可以直接載入它。
對齊 30 ms 延遲後與離線模型比對驗證（SI-SDR：v6 88–105 dB、v7 63 dB）；CPU
real-time factor 0.43。使用 v7 時，在輸出端套用發布用的 blend——把 enhanced
frame 與延遲 `streaming_delay_frames` 的 input frame 混合；這不會增加額外延遲。
機制細節見 `pretrained_ckpt/README.md` 與 `puresound/streaming/dpcrn.py`。

## Pre-DPCRN 沿革（精簡版）

在上述 pipeline 之前：TS-Conformer（mapping-head）在困難的近／遠案例上一直卡在
接近 passthrough 的水準（`mix_mode` curriculum 有幫助但無法突破；額外加一個
far-decoder 分支（P1）也沒用）。`overfit_check.py` 的容量測試把成因定位到
backbone 本身，而非資料或 loss——TS-Conformer 無法 overfit 最難的 batch
（enh→target 卡在 −2.2 dB），而 DPARN/DPCRN 可以（+4–5 dB）。換成 DPCRN
（complex ratio mask）之後立刻清掉了每一個 in-domain bucket。後續一次
「冷啟動 + 全量 RIR bank + `target_absent`」的組合，在真實域引發了過度抑制的
災難性結果（Dawn WER 0.626，SI-SDRi −9.53）——這就是目前 pipeline 採用分級
RIR curriculum、且 `target_absent: OFF` 的原因。已被取代且無法執行的 config 已從
現行設定樹移除；原始內容仍可由版本控制歷史取得。

## 評估

工具與用法：`scripts/README.md`。

- **主要判準（synthetic、in-domain）：** `scripts/eval_indomain.py --by-bucket`
  （或 `check_training_run.sh`）。以 early-reverb target 為基準算 SI-SDRi；要看
  困難的 bucket（counter_level / 1N+0F / overlap），不要只看 aggregate。
- **真實聲學、部署級殘響（rt60 0.44）：** `config/eval/eval_but_real.yaml` 的
  姊妹集合，建在中等 RT60 上——WER **0.723→0.487（−32%）**，do-no-harm 已確認。
  這是產品實際部署的域。
- **真實聲學、極端 OOD（BUT real-RIR，rt60 1.15–1.84）：** `scripts/build_wer_set.py`
  （建一次）→ `scripts/eval_wer.py`（SI-SDRi + 對照真實 LibriTTS 逐字稿的
  WER），config 為 `config/eval/eval_but_real.yaml`。刻意設得比訓練域更難；用來
  追蹤過度抑制在多大程度上是 OOD-殘響現象（結果是——見上文）。
- **未見房間泛化：** internal（seed-2026、與 expand 同難度
  分布的獨立房間 bank）。每個階段 seen→unseen 的落差都持續很小（≤0.4 dB）——
  代表模型是靠 DRR／幾何泛化，不是背房間。
- **洩漏探針（far-only/noise-only）：** `config/eval/eval_targetabsent_probe.yaml`
  ——強制每一列都 target-absent；檢查模型不會洩漏／幻覺出一個近場語者。所有
  pipeline checkpoint 都通過（power reduction ≤ −24.8 dB，false-near ≤3%），
  而且從未針對這個情境訓練過。
- **合成 vs 真實的 domain-gap 分解：** `scripts/eval_domain_gap.py`——在同時擁有
  真實錄音與實測脈衝響應的匹配 VOiCES (room, mic) 三元組上，把 synthetic-to-real
  的落差拆解成 LTI-convolution 天花板（real vs measured-IR 的 fit）與 RIR-bank
  保真度（measured vs synthetic-bank-IR 的 fit）。產出 `data_report/domain_gap_v7.jsonl`，
  正是促成第 8 階段量測式擷取真實化修正（`dpcrn_v8`）的證據。
- **Dawn Chorus = 參考／do-no-harm 專用。** 它**沒有近／遠 DRR 對比**，且 78%
  是窄頻 GSM，因此線索／頻寬都與這個任務不匹配；表現正常的模型在它上面應該
  ≈passthrough。

## 文件

| 檔案 | 內容 |
|---|---|
| `config/README.md` | config 排版：`train_*` / `eval_*` / `infer_dpcrn.yaml`、共用設計。 |
| `pretrained_ckpt/README.md` | checkpoint 沿革表 + streaming 匯出用法。 |
| `scripts/README.md` | 所有工具（資料準備、訓練期驗證、benchmark、推論）。 |

資料 pipeline（動態混音、預先生成的 RIR bank 近／遠對比、overlap gating）是
`puresound` 這個 library 的一部分，由 `augmentation_*` 這些 config block 驅動。
