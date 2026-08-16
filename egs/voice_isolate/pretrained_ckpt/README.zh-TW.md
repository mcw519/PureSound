# voice_isolate — 預訓練 checkpoint

English version: [`README.md`](README.md)

近場（<1 m）前景語音分離，單聲道，免註冊：保留近距語者，壓制遠距／競爭語者
與噪音。

版本依訓練順序編號——**`dpcrn_v8.ckpt` 是現行預設**。每個版本都共用同一套架構
（DPCRN、complex ratio mask、16 kHz、約 0.8 M 參數、30 ms look-ahead），所以
一份推論設定就能載入全部版本：

```bash
uv run python egs/voice_isolate/scripts/demo.py \
    --config_path egs/voice_isolate/config/infer_dpcrn.yaml     # 下拉選單列出每個版本
```

## 預設：`dpcrn_v8.ckpt` + `dry_blend = 0.9`

Runtime blend 是**已發布設定的一部分**，不是可有可無的附加選項：

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
| `dpcrn_v1.ckpt` | `config/exp/train_dpcrn_curriculum_core.yaml` | cold | curriculum core：RT60 0.20–0.45，近／遠 DRR gap ≥ 6 dB | in-domain SI-SDRi 中位數 **+6.98**（ep39） |
| `dpcrn_v2.ckpt` | `config/exp/train_dpcrn_curriculum_expand.yaml` | v1 | 拓寬 curriculum：RT60 ≤ 0.65，DRR gap ≥ 3 dB | in-domain **+7.99**；measured-RIR 集合 +5.46；第一次 real-WER 勝出（BUT enh 0.777 < mix 0.796）（ep59） |
| `dpcrn_v3.ckpt` | `config/exp/train_dpcrn_antisup_w1.yaml` | v2 | `OverSuppressionLoss` 權重 1.0（anti-deletion） | in-domain **+8.21**；BUT deletion 0.308 → 0.291（ep19） |
| `dpcrn_v4.ckpt` | `config/exp/train_dpcrn_antisup_w2.yaml` | v3 | anti-suppression 權重 2.0 | in-domain **+8.32**；BUT deletion → 0.276；兩個域中最平均（ep19） |
| `dpcrn_v5.ckpt` | `config/exp/train_dpcrn_antisup_w3.yaml` | v4 | anti-suppression 權重 3.0 | in-domain **+8.45**；部署級殘響上最佳，極端殘響上最差——domain split point（ep19） |
| `dpcrn_v6.ckpt` | `config/exp/train_dpcrn_wide_antisup.yaml` | v5 | 拓寬 RIR 域（RT60 0.20–0.85）+ 擷取真實化（media-voice 干擾者、HPF） | held-out unseen-room **+8.06**；**第一個經 streaming 驗證**的版本（詳見下文）（ep19） |
| `dpcrn_v7.ckpt` | `config/exp/train_dpcrn_realE2E_v2c.yaml` | v6 | 決策兩端都用真實錄音（真實遠場干擾者 + 真實 <1 m keep 列）、turn-taking far-solo 監督、distance/DRR 輔助 head、channel-perturbation mask consistency | held-out 真實遠場 **−9.45 dB，依距離分級**（1–2 m −3 → 5 m+ −38；v6：−1.76 持平），近場 keep 持平（−0.11），**Dawn WER 0.174 < 0.184 raw**，deletion 0.088 ≈ raw 底線，reverberant-office WER −0.024 vs mix，in-domain +8.18（ep19，搭配 `dry_blend 0.9`） |
| **`dpcrn_v8.ckpt`** | **`config/train_dpcrn.yaml`** | v7 | 合成中的量測式擷取真實化：噪音與語音卷積同一個房間、絕對 dBFS 麥克風底噪、部分合成混音的 SIR 取自場景幾何 | 真實遠場壓制 **−15.91 dB**，優於 v7 在同一批檔案上的 −13.72（排除洩漏子集為 −17.74 對 −15.68），且補平了 v7 距離響應中的 2–3 m 凹陷（−4.50 → −16.61）使分級單調；近場 keep 持平（0.00），最差個案有改善（−5.45 → −1.06）；Dawn WER 0.180 < 0.184 raw，deletion 0.094；reverberant-office WER −0.024 vs mix（兩種干擾者數量皆然）；turn-taking KEEP 6 違規；in-domain +7.99。代價：極端殘響監看上 +0.020 WER，而 v7 在此為中性（ep19，搭配 `dry_blend 0.9`） |
| `dpcrn_v10.ckpt` | `config/exp/train_dpcrn_coldstart.yaml` | v8 | 自我校準課程：讓部分訓練列在任何近場錨出現**之前**就以真實遠場獨白開場（列首遠場曝光 ~10% → ~23%、完全無錨的真實 lone-far 3% → 7.5%），教模型拿現場既有的任何參考——包含從第零幀就存在的噪音底——來校準 | **冷啟動**工作點，目前只判過田野 benchmark（ep39）。機器閒置時的遠場壓制：10 個孤立遠場 clip 中有 5 個跨過 −6 dB，v8 只有 1 個、v9 只有 2 個；並產生全集最深的單一結果（−11.95 dB，殘留高於底噪 12.6 dB）。近場保留是歷來最佳（最差 −0.34 dB）。增益侷限在 200 cm——300 cm 並不優於 v9。代價是串流模式壓制比 v8 淺約 1 dB。**尚非部署候選**：只跑過田野這一軸 |
| `dpcrn_v9.ckpt` | `config/exp/train_dpcrn_drrcontrast.yaml` | v8 | DRR 對比增強：每個新取出的 RIR 通道（機率 0.4）重縮殘響尾——前景通道 DRR 最多 +4 dB、遠場通道最多 −4 dB | ASR 閘最優工作點：reverberant-office WER **0.513，−0.050 vs mix**（v8 為 0.539、−0.024；兩種干擾者數量皆改善——1 itf 0.480、2 itf 0.576），**Dawn WER 0.172 / deletion 0.086**（皆為歷來版本最佳），極端殘響監看 **−0.003**（v8 的 +0.020 代價歸零），turn-taking KEEP 94/6，in-domain +8.10。代價：真實遠場壓制 −12.44 對 v8 的 −15.91（逐檔配對，24/71 檔淺 >3 dB）、turn-taking SUPPRESS 80/20 @ −13.43 對 v8 的 87/13 @ −16.14（ep19，搭配 `dry_blend 0.9`） |

`dpcrn_v6_gate.ckpt`——不在主線上：`config/exp/train_dpcrn_gate.yaml` 凍結 v6，
只訓練一個 causal 逐幀近／遠 VAD gate head（98,689 參數）。它在模擬資料上達到
0.90+ 的 balanced accuracy，但這個 gate 在真實錄音上關不起來，所以它是 gate
這條路的工程參考，不是可部署的模型。它的分離器權重與 v6 完全相同；那次訓練中
只有 10 個 BatchNorm running-statistic buffer 產生漂移，所以除了這些 buffer
以外，它的 mask 輸出就是 v6 的輸出。

**如何選版本。** 選 `dpcrn_v8.ckpt` 搭配 `dry_blend 0.9`——它仍是預設。
`dpcrn_v9.ckpt` 位在不同的工作點：當目標是近場語者的下游 ASR 品質時選它
（它的 reverberant-office WER 增益是 v8 的兩倍、deletion 是歷來最低），
代價是遠場壓制淺約 3.5 dB——殘留音裡遠場人聲會比 v8 更聽得見。當部署場景的殘響遠超過
訓練域（RT60 > 1 s）時，`dpcrn_v7.ckpt` 是替代方案：它在極端殘響 WER 監看上是
中性的（v8 在此要付出 +0.020 的代價，v9 在此亦為中性），代價是放棄約 2 dB 的真實遠場壓制。
`dpcrn_v6.ckpt` 是保守 fallback：沒有 runtime 旋鈕，而且對遠場語音大致上是
放行的。`dpcrn_v10.ckpt` 是冷啟動階段：只有當你要優化的軸就是機器閒置時的行為
（使用者還沒開口、旁人先講話）才選它，而且必須先把其餘閘門跑完——目前它只判過
田野 benchmark 這一軸。v1–v5 是訓練歷史階段，保留下來是為了讓任何階段都能被重新判準或重新
warm-start；它們不是部署候選。

**它們都做不到的事。** 透過與訓練語料非常不同的擷取鏈錄下的遠場語音，仍然幾乎
壓不下去（跨鏈參考片段上約 −1 dB，而某商用參考系統能達到 −44 dB）。這個落差是
錄音鏈本身的特性，與距離無關，這裡沒有任何版本能補上它。

**判準慣例。** Scheduler（`CosineAnnealingWarmRestarts`，`T_0=20`）每 20 epoch
重啟一次，所以 checkpoint 只能在 cosine 谷底——ep19/ep39/ep59——互相比較。上面
每一個數字都來自谷底 epoch。

## `streaming/` —— 逐幀 ONNX 匯出

`dpcrn_v6.{onnx,json}` 到 `dpcrn_v10.{onnx,json}`，用
`../scripts/streaming_onnx.py export` 建置。全部都因 look-ahead 而帶有
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
    --pretrained_ckpt_path pretrained_ckpt/dpcrn_v7.ckpt
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

訓練執行的目錄仍保留原本的名稱（`exp/dpcrn_wide_antisup_0702`、
`exp/dpcrn_realE2E_v2c_0722`、`exp/dpcrn_realism_0729`、……）。
