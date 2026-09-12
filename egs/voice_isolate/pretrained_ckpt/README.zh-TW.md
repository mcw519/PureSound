# voice_isolate — 預訓練 checkpoint

English version: [`README.md`](README.md)

近場（<1 m）前景語音分離，單聲道，免註冊：保留近距語者，壓制遠距／競爭語者
與噪音。

版本依訓練順序編號。**`dpcrn_curriculum_v1.ckpt` 是 model zoo 的預設**，`dpcrn_v8.ckpt`
在旁邊作為保守替代方案；這兩個是[`model_zoo/catalog.yaml`](../../../model_zoo/catalog.yaml)
裡僅存的兩個版本。下表用來與它們對照的其他版本已封存——見「封存版本」。每個版本都共用同一套架構
（DPCRN、complex ratio mask、16 kHz、約 0.8 M 參數、30 ms look-ahead），所以
一份推論設定就能載入全部版本：

```bash
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/infer_dpcrn.yaml     # 下拉選單列出每個版本
```

## 已發布的 blend：`dry_blend = 0.9`

Runtime blend 是兩個現役版本**已發布設定的一部分**，不是可有可無的附加選項。下面的數字
出自 v8，也就是當初論證這個 blend 的地方；curriculum 這條線沿用同一個設定與同一個
manifest key：

```python
enhanced = model(wav, dry_blend=0.9)      # out = 0.9 * enhanced + 0.1 * input
```

它把任何一點的衰減都限制在 −20 dB 以內。代價是留下一點殘餘干擾者，換來的是
訓練資料以外的擷取鏈上 deletion 大幅下降——加上它，模型在真實錄音上*降低*了
ASR 錯誤率（Dawn Chorus WER 0.180 對未處理的 0.184）；不加它，同一個 checkpoint
會在其中一個殘響測試集上提高 WER。Eval script 都有 `--dry-blend` 參數；streaming
manifest 把這個值放在 `recommended_inference` 底下。

已知限制：0.9 的 blend 無法產生完全靜音（依構造就有 −20 dB 底線），而且透過與
訓練語料非常不同的擷取鏈錄下的遠場語音，其壓制深度會比 in-domain 淺。若要做到
硬性靜音，mask 本身可達約 −38 dB，但那條路需要 gate，而不是 blend。

## 版本

| 版本 | recipe | warm-start | 這個階段新增了什麼 | 判準結果 |
|---|---|---|---|---|
| `dpcrn_v1.ckpt` | internal | cold | curriculum core：RT60 0.20–0.45，近／遠 DRR gap ≥ 6 dB | in-domain SI-SDRi 中位數 **+6.98**（ep39） |
| `dpcrn_v2.ckpt` | internal | v1 | 拓寬 curriculum：RT60 ≤ 0.65，DRR gap ≥ 3 dB | in-domain **+7.99**；measured-RIR 集合 +5.46；第一次 real-WER 勝出（BUT enh 0.777 < mix 0.796）（ep59） |
| `dpcrn_v3.ckpt` | internal | v2 | `OverSuppressionLoss` 權重 1.0（anti-deletion） | in-domain **+8.21**；BUT deletion 0.308 → 0.291（ep19） |
| `dpcrn_v4.ckpt` | internal | v3 | anti-suppression 權重 2.0 | in-domain **+8.32**；BUT deletion → 0.276；兩個域中最平均（ep19） |
| `dpcrn_v5.ckpt` | internal | v4 | anti-suppression 權重 3.0 | in-domain **+8.45**；部署級殘響上最佳，極端殘響上最差——domain split point（ep19） |
| `dpcrn_v6.ckpt` | `test/fixtures/recipes/train_dpcrn_wide_antisup.yaml` | v5 | 拓寬 RIR 域（RT60 0.20–0.85）+ 擷取真實化（media-voice 干擾者、HPF） | held-out unseen-room **+8.06**；**第一個經 streaming 驗證**的版本（詳見下文）（ep19） |
| `dpcrn_v7.ckpt` | internal | v6 | 決策兩端都用真實錄音（真實遠場干擾者 + 真實 <1 m keep 列）、turn-taking far-solo 監督、distance/DRR 輔助 head、channel-perturbation mask consistency | held-out 真實遠場 **−9.45 dB，依距離分級**（1–2 m −3 → 5 m+ −38；v6：−1.76 持平），近場 keep 持平（−0.11），**Dawn WER 0.174 < 0.184 raw**，deletion 0.088 ≈ raw 底線，reverberant-office WER −0.024 vs mix，in-domain +8.18（ep19，搭配 `dry_blend 0.9`） |
| **`dpcrn_v8.ckpt`** | **`config/train_dpcrn.yaml`** | v7 | 合成中的量測式擷取真實化：噪音與語音卷積同一個房間、絕對 dBFS 麥克風底噪、部分合成混音的 SIR 取自場景幾何 | 真實遠場壓制 **−15.91 dB**，優於 v7 在同一批檔案上的 −13.72（排除洩漏子集為 −17.74 對 −15.68），且補平了 v7 距離響應中的 2–3 m 凹陷（−4.50 → −16.61）使分級單調；近場 keep 持平（0.00），最差個案有改善（−5.45 → −1.06）；Dawn WER 0.180 < 0.184 raw，deletion 0.094；reverberant-office WER −0.024 vs mix（兩種干擾者數量皆然）；turn-taking KEEP 6 違規；in-domain +7.99。代價：極端殘響監看上 +0.020 WER，而 v7 在此為中性（ep19，搭配 `dry_blend 0.9`） |
| `dpcrn_v9.ckpt` | internal | v8 | DRR 對比增強：每個新取出的 RIR 通道（機率 0.4）重縮殘響尾——前景通道 DRR 最多 +4 dB、遠場通道最多 −4 dB | ASR 閘最優工作點：reverberant-office WER **0.513，−0.050 vs mix**（v8 為 0.539、−0.024；兩種干擾者數量皆改善——1 itf 0.480、2 itf 0.576），**Dawn WER 0.172 / deletion 0.086**（皆為歷來版本最佳），極端殘響監看 **−0.003**（v8 的 +0.020 代價歸零），turn-taking KEEP 94/6，in-domain +8.10。代價：真實遠場壓制 −12.44 對 v8 的 −15.91（逐檔配對，24/71 檔淺 >3 dB）、turn-taking SUPPRESS 80/20 @ −13.43 對 v8 的 87/13 @ −16.14（ep19，搭配 `dry_blend 0.9`） |
| `dpcrn_v10.ckpt` | internal | v8 | 自我校準課程：讓部分訓練列在任何近場錨出現**之前**就以真實遠場獨白開場（列首遠場曝光 ~10% → ~23%、完全無錨的真實 lone-far 3% → 7.5%），教模型拿現場既有的任何參考——包含從第零幀就存在的噪音底——來校準 | **冷啟動軸動了，但部署閘門沒撐住**（ep39，完整關卡）。贏的部分：機器閒置時的遠場壓制 10 個孤立 clip 中 5 個跨過 −6 dB（v8 為 1、v9 為 2），並產生全集最深的單一結果；近場保留歷來最佳（最差 −0.34 dB）；turn-taking SUPPRESS 89/11 @ −15.98（ok 率最佳）；Dawn WER 0.173 / deletion 0.088 ≈ v9。輸的部分：**在部署殘響範圍上比 v8 差 0.024 WER**（同一批語句配對，95% CI [+0.008, +0.041]）；turn-taking KEEP 違規 6 → 10；in-domain +7.56（v8 +7.99、v9 +8.10）；極端殘響回到 +0.017（v9 −0.003）。在 BUT-OFFICE 上讀到 −0.004 對 v8 的 −0.024，但該集合分辨不出這兩個數字（n=200，兩個區間都跨過 0）——它是監看，不是閘門。冷啟動的增益侷限在 200 cm。**非部署候選**——留作冷啟動參考與 warm-start 起點 |
| `dpcrn_v16_ep19.ckpt` | `test/fixtures/recipes/train_dpcrn_v16_lengthmix.yaml` | v8 | 長度排程：每個 batch 從 {3, 6, 12, 30 s} 抽一個列長（章節語料＋真實池接續，不補零） | field block 協定：keep 違規嚴重度優於 v8、session 壓制較淺（block −9.73 vs v8 −11.78 dB）；Azure WER：四集 deletion 全部最低（Dawn 0.198 vs v8 0.226）。ASR 用途的部署候選；壓制深度仍以 `dpcrn_v8` 為預設。 |
| `dpcrn_curriculum_v0.ckpt` | `config/train_dpcrn.yaml` | **冷啟動** | 把整條血統寫成**一次 run**：上面各階在 run 之間改的旋鈕，改成 `curriculum` 排程在訓練中移動（房間池加寬、反刪除 0→3、ep20 進 capture realism、ep30/40 進真實錄音列、距離損失權重隨之進場），混長列全程開 | 單一 120 epoch 冷啟動達到階梯的**約七成**（ep99）：moderate 殘響 WER **−0.108**（階梯 −0.171）、turn-taking 壓制 **−15.12** 84/16（−16.14 87/13）、field session **−8.28**（−11.78）、in-domain +6.87。兩關持平或勝出：Dawn WER **0.178**／deletion 0.089（v8 0.180／0.094），以及 27 個 near clip 的保留（配對中位 +0.04 dB，p=0.044）。QVF 跨鏈遠場較深（`qvf_price_far1` −17.2 vs −3.6、`qvf_price_far2` −30.1 vs −18.5，n=6）但 **RealMAN 上兩者無差**（>1 m 中位 −7.37 vs −7.21）——是 QVF 特有現象，不是跨鏈軸。**非部署候選**——field session 少 3.5 dB，且 `dry_blend 1.0` 會崩（Dawn 0.234）。三個負面實驗把剩餘缺口釘在「warm-start 鏈本身」，而非預算／房間配重／列長。 |
| `dpcrn_curriculum_v1.ckpt` | `config/train_dpcrn_curriculum_v1.yaml` | `dpcrn_curriculum_v0` | 這條血統的第二個排程步驟：對話式 **session 列**（使用者與旁人輪流說話，含空檔、再次進場、換座位）、同一列的**成對擷取視角**做一致性項，以及兩個**逐幀頭**——由每個 turn 的算繪距離監督的相對 proximity，以及 presence。三者各自 ramp 進場；檔案常數＝最終值，所以 validation 量的是末端分佈 | **冷啟動遠場那道牆動了，在內部 device 鏈上**（ep39，5-ckpt block）：無近場錨的孤立遠場 **−7.07 dB**（階梯 −0.96，p=0.000）、有房間音時 **−28.27**（−14.51，p=0.031），而該鏈的近場保留完全沒退（24 clip 中位 −0.16 對 −0.19）。turn-taking 壓制 −17.10 @ **94/6**，超過階梯的 −16.14 @ 87/13；Dawn WER **0.172**／deletion 0.088，所有版本最佳；moderate 殘響 WER −0.156（達階梯 −0.171 的 91%，上一步只有 63%）；in-domain +7.32。**代價**：QVF 跨鏈 clip 的 keep 退步，最壞 `qvf_keep_in_touch_near1` −32.8 dB——那是**逐錄音的校準失準，不是傷語音**（諧波對比不變、位準降 19.7 dB；模型自己的 proximity 讀數把那支 clip 放在 −0.59，介於健康近場 +1.35 與真遠場 −3.00 之間）。keep 側加噪護欄沒有傾斜（五個 s2f 檔位對階梯配對皆 p>0.05）。**跨鏈部署非候選**；ep19 併存為較溫和的工作點（最壞跨鏈 keep −12.6、合成遠場 probe 深 2.6–5.9 dB、ASR 關略遜）。|

> **v13–v18（2026-08/09）沒有鑄成版本的回合。**設定與判定屬於內部紀錄，判定屬於內部紀錄，不隨本次發佈：v13 inter-LSTM→Mamba（最後一階課程的重初始化債）、v14 零初始化並聯
> Mamba 分支（`v14_VERDICT.md`，只多了一顆免費的 `dry_blend` 旋鈕）、v15 只用 30 s 列（場景多樣性被餓死）、
> v17 房間音前導（`v17_round_design.md`；冷啟動是缺房間參照，要在部署端解不是訓練端）、v18 損失地板
> （`v18_VERDICT.md`；relative inactive-SDR 封頂了壓制誘因）。這些 ckpt 只留在 `exp/`。

> **關於上表的 reverberant-office 數字。** 表中每一個 `reverberant-office WER`（v7 −0.024、
> v8 −0.024、v9 −0.050、v10 −0.004）都來自一個 200 句的集合，其 bootstrap 區間約 ±0.03。
> 2026-08 重新量測後，v8 與 v10 在該集合上**都無法證明比不處理更好**，所以它隱含的版本排序
> 並未成立。主要 WER 閘門已改為 `wer_set_moderate_test`，它能乾淨地分辨同一組比較。細節與
> 配對數字見 [`../scripts/WER_SETS.md`](../scripts/WER_SETS.md)。

## 封存版本

把 model zoo 縮到兩個現役版本時，`dpcrn_v6`、`dpcrn_v7`、`dpcrn_v9`、`dpcrn_v10`、
`dpcrn_v11_ep19`、`dpcrn_v16_ep19` 與 `dpcrn_curriculum_v0` 已移出版本控制。它們的
`.ckpt` 與 `streaming/` 匯出檔現在放在本目錄下的 `backup/` 與 `backup/streaming/`，該路徑
被 gitignore：檔案還在磁碟上、仍可用路徑載入，但不再隨 repo 發佈、也不再登錄在 catalog 裡。
上表仍以它們為對照；需要時可從 git 歷史取回單一檔案
（`git show <rev>:egs/voice_isolate/pretrained_ckpt/<name>.ckpt`）。

`dpcrn_v1`–`dpcrn_v5`、`dpcrn_v6_gate` 與 `dpcrn_v11_ep16` 仍留在版本控制裡：它們是訓練
歷史階段與參考檔，本來就沒有登錄進 catalog。

`dpcrn_curriculum_v0.ckpt`——以另一種意義不在主線上：架構與任務都相同，但它是這裡
唯一**沒有**從別的版本 warm-start 的版本。它是可復現的單run基線（一份設定、一道指令），
也是「排程能取代什麼、不能取代什麼」的參照點；部署選項仍是上面那幾版。

`dpcrn_v6_gate.ckpt`——不在主線上：internal 凍結 v6，
只訓練一個 causal 逐幀近／遠 VAD gate head（98,689 參數）。它在模擬資料上達到
0.90+ 的 balanced accuracy，但這個 gate 在真實錄音上關不起來，所以它是 gate
這條路的工程參考，不是可部署的模型。它的分離器權重與 v6 完全相同；那次訓練中
只有 10 個 BatchNorm running-statistic buffer 產生漂移，所以除了這些 buffer
以外，它的 mask 輸出就是 v6 的輸出。

**如何選版本。** 這個目錄裡僅存的兩個版本中，選 `dpcrn_curriculum_v1.ckpt` 搭配
`dry_blend 0.9`——它是 model zoo 的預設，也是唯一能在內部裝置鏈上壓下「沒有近場錨的
孤立遠場人聲」的版本。當部署是跨錄音鏈時選 `dpcrn_v8.ckpt`：它放棄冷啟動的增益，但不會
在訓練語料以外的錄音上讓 keep 退步，而那正是 curriculum-v1 未解的缺陷。這段以下描述的
是已封存的版本，保留是因為上表每個版本都以它們為對照。
`dpcrn_v9.ckpt` 位在不同的工作點：當目標是近場語者的下游 ASR 品質時選它
（它的 reverberant-office WER 增益是 v8 的兩倍、deletion 是歷來最低），
代價是遠場壓制淺約 3.5 dB——殘留音裡遠場人聲會比 v8 更聽得見。當部署場景的殘響遠超過
訓練域（RT60 > 1 s）時，`dpcrn_v7.ckpt` 是替代方案：它在極端殘響 WER 監看上是
中性的（v8 在此要付出 +0.020 的代價，v9 在此亦為中性），代價是放棄約 2 dB 的真實遠場壓制。
`dpcrn_v6.ckpt` 是保守 fallback：沒有 runtime 旋鈕，而且對遠場語音大致上是
放行的。**`dpcrn_v10.ckpt` 不是部署選項**：它是唯一能推動「機器閒置冷啟動壓制」的版本，但完整
關卡顯示它為此在部署殘響範圍的 WER 上付出代價：同一批語句配對比 v8 差 0.024，bootstrap 區間
不跨過 0。把它當冷啟動的參考，或當作「要把 office 閘門
補回來」那個 recipe 的 warm-start 起點。v1–v5 是訓練歷史階段，保留下來是為了讓任何階段都能被重新
判準或重新 warm-start；它們不是部署候選。

**它們都做不到的事。** 透過與訓練語料非常不同的擷取鏈錄下的遠場語音，仍然幾乎
壓不下去（跨鏈參考片段上約 −1 dB，而某商用參考系統能達到 −44 dB）。這個落差是
錄音鏈本身的特性，與距離無關，這裡沒有任何版本能補上它。

**判準慣例。** Scheduler（`CosineAnnealingWarmRestarts`，`T_0=20`）每 20 epoch
重啟一次，所以 checkpoint 只能在 cosine 谷底——ep19/ep39/ep59——互相比較。上面
每一個數字都來自谷底 epoch。

## `streaming/` —— 逐幀 ONNX 匯出

`dpcrn_v8.{onnx,json}` 與 `dpcrn_curriculum_v1.{onnx,json}`——兩個註冊在案的版本；
已封存版本的匯出檔隨其 checkpoint 一起搬走了。全部用 `../scripts/streaming_onnx.py export` 建置。全部都因 look-ahead 而帶有
**30 ms（3 幀）演算法延遲**，由烘進 graph 當成額外 state 的 future-buffering
處理（`puresound/streaming/dpcrn.py`），而且全部都在對齊該延遲後與離線模型比對驗證過。
`verify` 預設用**白噪音**探針，那是壓力訊號而非部署訊號——v6 88–105 dB、v7 63 dB、
v8 49.7 dB、v9 48.2 dB、v10 24.6 dB SI-SDR，這條下降趨勢反映的是各版本調變 mask 的
激進程度，不是串流正確性。用 `--input_audio` 餵真實語音，同一批 graph 幾乎位元精確：
在田野 benchmark 90D session 的 30 秒上，v8 124.4 dB、v9 120.2 dB、v10 116.1 dB。
判斷新匯出是否合格要看語音那個數字，白噪音數字只用來做版本之間的相對比較。CPU RTF 0.43。用
`puresound.streaming.StreamingDpcrnOrt` 或 SDK 的 manifest-driven
`PureSoundStreamingRuntime`（`processor: stft_frame_ort`）載入。

任何離線↔streaming 的比較都**必須**依回報的延遲對齊並剪掉邊緣，否則延遲會被
誤讀成誤差；`streaming_onnx.py verify` 會做這件事：

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.json --provider cpu
```

用 v7 或 v8 做 streaming 時，blend 套用在輸出端：把 enhanced frame 與延遲
`streaming_delay_frames` 的 input frame 混合。這不會增加額外延遲；兩份 manifest
都把這個值放在 `recommended_inference` 底下。

## 重新匯出／重新訓練

```bash
# 從這裡任何一個 checkpoint 匯出 streaming ONNX
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt /tmp/model.onnx

# 執行預設 recipe，像 v8 當初那樣從 v7 warm-start
# （從 recipe 目錄執行——config 裡的 metafile 路徑是相對於它的）
cd egs/voice_isolate && uv run python main.py config/train_dpcrn.yaml --training \
    --pretrained_ckpt_path pretrained_ckpt/backup/dpcrn_v7.ckpt
```

這裡的 checkpoint 都是從 `../exp/` 底下完整訓練歷史中取出的判準谷底（該目錄
symlink 到工作用的 volume，已 gitignore，保留每個 epoch）。各版本量測數字的
脈絡：`../EXPERIMENT_LOG.md`。

## 檔名沿革

較早的 log 與報告使用版號制之前的名稱：

| 舊名稱 | 版本 |
|---|---|
| `dpcrn_curriculum_core_ep39.ckpt` | `dpcrn_v1.ckpt` |
| `dpcrn_curriculum_expand_ep59.ckpt` | `dpcrn_v2.ckpt` |
| `dpcrn_antisup_w1_ep19.ckpt` | `dpcrn_v3.ckpt` |
| `dpcrn_antisup_w2_ep19.ckpt` | `dpcrn_v4.ckpt` |
| `dpcrn_antisup_w3_ep19.ckpt` | `dpcrn_v5.ckpt` |
| `dpcrn_wide_antisup_ep19.ckpt` | `dpcrn_v6.ckpt` |
| `dpcrn_gate_synth_ep7.ckpt` | `dpcrn_v6_gate.ckpt` |
| `dpcrn_realE2E_v2c_ep19.ckpt` | `dpcrn_v7.ckpt` |
| `dpcrn_realism_0729_ep19.ckpt` | `dpcrn_v8.ckpt` |
| `dpcrn_drrcontrast_ep19.ckpt` | `dpcrn_v9.ckpt` |
| `dpcrn_coldstart_ep39.ckpt` | `dpcrn_v10.ckpt` |
| `exp/dpcrn_v16_lengthmix/epoch=19*.ckpt` | `dpcrn_v16_ep19.ckpt` |

訓練執行的目錄仍保留原本的名稱（`exp/dpcrn_wide_antisup_0702`、
`exp/dpcrn_realE2E_v2c_0722`、`exp/dpcrn_realism_0729`、……）。
