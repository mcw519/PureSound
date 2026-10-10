# noise_suppression — 預訓練 checkpoint

English: [`README.md`](README.md)

單聲道 16 kHz 語音增強，可串流、不需 enrollment。各版本都是 DPCRN，時間路徑使用
24 個 ERB 頻帶上的 Mamba。關卡見 [`../benchmarks/stages.zh-TW.md`](../benchmarks/stages.zh-TW.md)。

```bash
uv run puresound infer noise-suppression-dpcrn-mamba-v3 \
    --input audio=in.wav --output audio=out.wav --provider cpu
```

v1/v2 使用 [`../config/infer_dpcrn.yaml`](../config/infer_dpcrn.yaml)。
v3 必須使用 [`../config/infer_dpcrn_mamba_wide.yaml`](../config/infer_dpcrn_mamba_wide.yaml)：
通道 `[2,48,96,128]`、hidden size 128；v1/v2 則為 `[2,32,64,96]`、96。
串流匯出皆為 `dry_blend: 1.0`，不使用 onset guard 或 graph 後緩解。
重新匯出要傳 `--dry-blend 1.0`。

## 版本

| 版本 | zoo id / 角色 | 訓練 | 適用情況 |
|---|---|---|---|
| `dpcrn_mamba_v3.ckpt` | `noise-suppression-dpcrn-mamba-v3`，已發布候選 | 加寬模型、多來源 coverage 訓練、ActiveBin、MetricGAN | 在意較高 PESQ 與 STOI；DNSMOS、辨識與 CPU 取捨見下 |
| **`dpcrn_mamba_v2.ckpt`** | `noise-suppression-dpcrn-mamba-v2`，**預設** | v1 加 PESQ critic 微調 4 epoch | 平衡音質與 CPU 成本 |
| `dpcrn_mamba_v1.ckpt` | `noise-suppression-dpcrn-mamba-v1`，候選 | 小模型 curriculum 加 ActiveBin 校準 | 最在意乾淨朗讀語音的刪字率 |
| `dpcrn_mamba_v0.ckpt` | 已退役（`backup/`，不進版控） | 小模型 curriculum | -- |

v3 另有 int8 的 `flash` variant，適合 CPU 預算吃緊時使用，見 [v3 flash](#v3-flashint8)。

## 量測

皆使用 `dry_blend: 1.0`。欄位是相同測試集上的平均值，不代表版本之間的配對信賴區間。

| 資料集 | 指標 | 未處理 | v1 | v2 | v3 |
|---|---|---|---|---|---|
| VCTK-DEMAND（n=824） | PESQ-WB | 1.968 | 2.551 | 2.567 | **2.631** |
| | STOI | 0.9210 | 0.9213 | 0.9231 | **0.9310** |
| | SI-SDR（dB） | 8.45 | **18.38** | 18.04 | 17.13 |
| | 刪字率（Whisper large-v3） | 0.00744 | **0.00643** | 0.00870 | 0.00825 |
| 凍結合成集（500 段；PESQ n=499） | PESQ-WB | 1.471 | 2.354 | 2.409 | **2.449** |
| DNS-5 dev（n=921） | DNSMOS SIG / BAK / OVR / P.808 | 2.99 / 2.56 / 2.21 / 2.91 | 3.18 / 3.82 / 2.81 / 3.38 | 3.19 / 3.85 / 2.83 / 3.42 | 3.19 / 3.80 / 2.81 / 3.41 |
| 困難 WER 集（n=500） | 原始 WER，含辨識器重複轉寫 | 0.12251 | -- | 0.16006 | 0.17651 |
| | 刪字率 | 0.04235 | -- | 0.04728 | 0.04838 |
| -- | CPU RTF，單執行緒（checkpoint 關卡） | -- | 0.171 | 0.164 | 0.242 |
| -- | CPU RTF，單執行緒（原始串流 ONNX，10 秒探測） | -- | -- | 0.408 | 0.570 |
| -- | CPU RTF，單執行緒（目前最佳化 ONNX，5 秒語音） | -- | 0.299 | 0.311 | 0.446 |
| -- | CPU RTF，單執行緒（目前 native SSM companion，5 秒語音） | -- | 0.257 | 0.266 | 0.390 |
| -- | Graph look-ahead（ms） | -- | 30 | 30 | 30 |
| -- | 串流演算法緩衝延遲（ms，不含運算） | -- | 52–62 | 52–62 | 52–62 |

紀錄：[v3](../benchmarks/records/ns_dpcrn-mamba_capWL3_mg_ep3.json)、
[v2](../benchmarks/records/ns_dpcrn-mamba_metricgan_ft_ep3.json)、
[v1](../benchmarks/records/ns_dpcrn-mamba_activebin_ft_ep1.json)。

## 發版決定

- v3 發布為可選候選，v2 維持預設。v3 的兩個 PESQ 關卡與 checkpoint CPU 預算通過，
  但兩個刪字護欄皆為 `no-resolution`（信賴區間跨零）；這不能證明沒有退步或保字能力改善。
- v3 困難集有 7 段辨識器重複轉寫，未處理輸入有 1 段。任一側的假設字數超過參考字數
  1.5 倍就排除該配對，留下 493 段：模型 WER **0.144154**、輸入 **0.120800**、
  差 **+0.023354**。這個子集與先前版本不同，不能當成 v3 對 v2 的配對排名；
  重複轉寫失敗也應另外保留為部署指標。
- v2 曾在 VCTK 刪字護欄勉強未過時發布，接受的取捨記在其 benchmark lineage。
  v1 繼續提供較低的 VCTK 刪字率。

## v3 flash（int8）

`flash` 是 v3 的 artifact variant，不是新 checkpoint：權重相同，LSTM 與矩陣乘法的權重
存成 int8，activation 在執行時量化（`export --quantize int8`）。使用 `--variant flash` 選用。
下表兩欄都是同一批輸入的串流 ONNX 輸出，以逐句配對差比較，所以 v3 欄與 checkpoint 紀錄
略有不同。

| 量測 | v3 | v3 flash | 配對差（95% CI） |
|---|---|---|---|
| CPU RTF，單執行緒，標準 / native 圖 | 0.454 / 0.387 | 0.370 / 0.304 | −18.5% / −21.5% |
| frozen 合成集 PESQ-WB | 2.508 | 2.504 | −0.0044（−0.0054, −0.0035） |
| VCTK-DEMAND PESQ-WB | 2.629 | 2.624 | −0.0053（−0.0065, −0.0041） |
| VCTK-DEMAND SI-SDR（dB） | 17.14 | 17.10 | −0.038（−0.047, −0.028） |
| DNS-5 dev DNSMOS OVR | 2.808 | 2.805 | −0.0024（−0.0035, −0.0013） |
| VCTK-DEMAND 刪字率 | 0.0086 | 0.0084 | −0.0002（−0.0019, +0.0013） |
| hard 集刪字率（排除迴圈） | 0.0434 | 0.0434 | +0.0000（−0.0025, +0.0025） |

所有品質指標都可解析地較低，幅度約為 v3 相對 v2 PESQ 進步的十分之一；刪字沒有可解析的
變化。適合 CPU 預算吃緊時使用。計時是在 AVX-512 Xeon 上以單執行緒處理 5 秒含噪語音、
九次交錯測量的中位數；請看比例，且 int8 kernel 的效率因指令集而異，請在目標 CPU 上實測。
瀏覽器／WASM 仍使用浮點圖。刪字率使用 `int8_float16` 的 Whisper large-v3，不能與上方表格
直接比較。

## 重現 v3

使用 seed 1234 依序執行，每階用前一階最後一顆 checkpoint warm-start。
run 保留實驗名稱，v3 只用於已發布的模型。

| Recipe | 起點 | Epoch 數／最後 checkpoint |
|---|---|---|
| [`train_dpcrn_mamba_capW_s0.yaml`](../config/train_dpcrn_mamba_capW_s0.yaml) | 從零開始；DNS-5、SNR [0,20] | 20／ep19、step40000 |
| [`train_dpcrn_mamba_capWL3_s1.yaml`](../config/train_dpcrn_mamba_capWL3_s1.yaml) | capW_s0 ep19；多來源語音／噪音、coverage sampler、SNR [-10,40]、4 秒片段 | 20／ep19、step80000 |
| [`train_dpcrn_mamba_capWL3_ft.yaml`](../config/train_dpcrn_mamba_capWL3_ft.yaml) | capWL3_s1 ep19；ActiveBin 校準 | 2／ep1、step4000 |
| [`train_dpcrn_mamba_capWL3_mg.yaml`](../config/train_dpcrn_mamba_capWL3_mg.yaml) | capWL3_ft ep1；PESQ MetricGAN | 4／ep3、step8000 |

配方保存這次使用的設定；語料與 RIR 路徑須指向自己的副本，短階段使用已移除全靜音檔的池。
`target_clipping: own_quantile` 保留訓練時的值。發布的 checkpoint 包含推論權重與來源資訊，
移除 critic、loss、optimizer 與 sampler 狀態；推論權重與 `ns_dpcrn-mamba_capWL3_mg` ep3 相同。

## 匯出驗證

隨機輸入、凍結 NS 混音、困難集混音三段測試中，發布模型的推論輸出與來源 checkpoint
逐位相同。CPU ONNX 串流對齊 480 樣本的 look-ahead 延遲後，一致性為
92.2／116.7／111.6 dB，最大樣本誤差低於 5e-7。

較早一次發布環境測量使用原始串流 ONNX runtime：CPU 單執行緒、10 秒輸入、兩次暖機、
三次交錯測量，v2 RTF **0.408**、v3 **0.570**。當時的原始 v3 匯出超過 **0.5** 的串流預算。
v3 全句 checkpoint 在此測得 **0.270**，原始 checkpoint 關卡則為 **0.242**；
兩者與串流 ONNX 是不同執行路徑，checkpoint 關卡不能代表串流 runtime 的成本。
完整驗證結果保存在 v3 紀錄的 release lineage。

三個 NS 發布版本之後都以固定 batch=1 與 frequency-major 排列重新匯出。標準 ONNX 圖現為
model zoo 預設；sidecar 另外引用經 SHA256 驗證的 native SSM companion。以同一段 5 秒含噪
語音交錯測量新舊匯出：v1 **0.393 → 0.299 / 0.257**、v2 **0.406 → 0.311 / 0.266**、
v3 **0.572 → 0.446 / 0.390**（標準 / native）。v3 在這顆 CPU 上已符合 0.5 預算。
[重新匯出報告](../../../model_zoo/benchmarks/onnx_reexport_20261004.json)記錄了樣本、計時重複、
雜湊值與 30 秒有狀態檢查。這組 5 秒語音計時與較早的 10 秒探測是兩個不同的 benchmark。
函式庫選擇見 [CPU 設定](../../../docs/usage/streaming/dpcrn_onnx.zh-TW.md)。

checkpoint 關卡用一次 PyTorch 呼叫處理完整 10 秒音訊。串流 runtime 每 10 ms hop
處理一次，10 秒約需 1,000 次 ONNX 呼叫，並逐幀交換狀態、執行 STFT／iSTFT 與重疊相加。
兩者的批次方式、後端與呼叫成本不同；目前量測沒有拆分各項成本。
RTF 是運算耗時除以音訊長度：0.570 表示 10 秒音訊約需 5.7 秒運算，
不衡量等待未來輸入的時間，因此須與延遲分開呈現。

v3 的 hop 為 160 樣本（10 ms）、分析窗為 512 樣本（32 ms）、look-ahead 為 3 幀（30 ms）。
串流演算法緩衝延遲為 `lookahead + 窗長 − hop` 到 `lookahead + 窗長`，
即 **52–62 ms**，依樣本在 hop 中的位置而異。裝置預算以 **62 ms** 為基準，
再加運算時間、額外輸入分塊緩衝、音訊裝置緩衝與傳輸。v1/v2 的幾何與演算法延遲相同。
波形對齊仍使用 480 樣本（30 ms）的 graph look-ahead，這與完整串流緩衝延遲不同。
這些推導值已保存於 v3 ONNX manifest 與發布驗證紀錄的 `algorithmic_latency`。

瀏覽器用的產物另通過 Node 上 ONNX Runtime Web 1.24.3 WASM 驗證：
與原生串流相比，正規化 RMS 誤差為 3.30e-7；不規則分塊與重設串流後的輸出一致。
