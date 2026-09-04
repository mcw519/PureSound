# PureSound Web 推論工作區

專案內含一個 Model Zoo 與統一 ONNX facade 的瀏覽器介面。它刻意只使用
Python 標準函式庫，以及原生 HTML、CSS、JavaScript，方便稽核與部署。

在 repository 根目錄啟動：

```bash
puresound web
# 開啟 http://127.0.0.1:7860
```

IP 與 port 都由 CLI args 控制：

```bash
puresound web --ip 0.0.0.0 --port 8080
# --host 與 --ip 相同
```

綁定 `0.0.0.0` 會把推論服務開放給周圍網路。服務本身沒有驗證機制，因此
只應在可信任網路使用，或放在有身份驗證的 reverse proxy 後方。

介面包含三個畫面：

- **Model Zoo**：列出 catalog 中可執行的模型，顯示與 `puresound models
  list` 相同的 artifact、lifecycle 與 preprocessing metadata。
- **Voice Isolation**：上傳單一音檔，透過 `stft_frame_ort` 的具名 `audio`
  input 執行推論。輸出可直接播放或下載 WAV；manifest delay alignment 與
  dry-blend override 仍由 processor 負責。完成後也會顯示輸出音量指標與在 scorer
  可用時的 reference-free DNSMOS，因此一次 Playground 執行就能先做基本品質檢查。
- **Speaker Verification**：上傳 `enrollment` 與 `test` 音檔，呼叫兩次
  waveform embedding processor，顯示 cosine score 與 threshold verdict。
- **Measurements**：提供 probe 音檔、可選的 clean reference，以及一個或
  多個 Voice Isolation model。會回報可重現的音量／頻譜指標（RMS、peak、
  clipping、silence、zero-crossing rate、spectral centroid），並可計算不需
  reference 的 DNSMOS；有 reference 時再計算 SI-SDR、SNR、correlation、
  STOI 與 PESQ。頁面會以三步驟引導操作，Model Zoo integrity check 則收在
  獨立的技術檢查區。設定集中在右側抽屜，報表可使用完整工作區寬度。成功的
  model output 會先依 streaming latency 對齊、存入有上限的記憶體 RunStore，
  再使用與 Playground 相同的 waveform、spectrogram、播放 boost 與 limiter
  控制顯示。若所有模型都失敗，comparison 會直接失敗；只有部分失敗時則顯示警告。

瀏覽器、CLI 與 legacy streaming runtime 共用相同的 provider 選擇：`auto` 依序
偏好 CUDA、Apple `CoreML`、CPU；`cpu` 強制 CPU；`cuda` 要求 NVIDIA CUDA；
`coreml` 要求 Apple CoreML execution provider。因為 ONNX Runtime 沒有原生的
MPS execution provider，`mps` 會作為 `coreml` 的使用者介面 alias。`/api/health`
會回報目前 ONNX Runtime wheel 註冊的 providers，Playground 也會停用不可用的
明確選項。即使 provider 已註冊，若 driver 或 shared library 載入失敗仍可能
fallback 到 CPU；每次 inference result 都會回報實際 provider list。

完成的 Measurements report 可直接下載 JSON（包含 request、input/reference
metrics 與各模型結果）或 CSV（每個模型一列）。檔案在瀏覽器端產生，不會在
server 上持久化資料。

左側主導航可收合，瀏覽器會記住下次開啟時的選擇。Playground 的 Voice
Isolation 與 Speaker Verification 設定改為各自的右側抽屜；抽屜關閉時，主要
工作區可使用完整寬度。

所有上傳或產生的音檔都使用相同的檢視 panel，提供共同時間軸、同步的
waveform 與 0–8 kHz spectrogram、點擊定位播放，以及 Wave／Both／Spec
顯示模式。播放 boost 範圍為 0 至 +48 dB；**Auto** 會依 peak 自動保留
-1.5 dBFS headroom，並另有 safety limiter。Boost 只影響瀏覽器監聽，
不會改變模型輸入或下載的 WAV。

API 維持精簡：

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | runtime 與 catalog 摘要 |
| GET | `/api/models?task=voice_isolation` | catalog entries |
| GET | `/api/models/{model_id}` | 單一模型 contract |
| GET | `/api/validate` | 路徑、sidecar、hash 與 graph 檢查 |
| POST | `/api/infer` | 以 JSON named inputs 執行模型，可選擇附加輸出量測 |
| POST | `/api/jobs` | 啟動非同步推論或量測並取得 job id |
| GET | `/api/jobs/{job_id}` | 讀取進度、狀態與結果 |
| GET | `/api/jobs?limit=20` | 列出最近推論與量測 job |
| POST | `/api/jobs/{job_id}/cancel` | 要求取消 queued／running job |
| POST | `/api/measure` | 測量 probe、比較 Voice Isolation model，並可選 DNSMOS |
| GET | `/api/runs/{run_id}/{output}` | 下載產生的音檔 |

非同步 job 會回報 `queued`、`running`、`succeeded`、`failed` 或
`cancelled`。Voice Isolation 每個 streaming frame 都會更新進度；Speaker
Verification 則回報音訊載入、enrollment embedding、test embedding 與
scoring 階段。量測可在相同 endpoint 帶入 `"kind": "measurement"`，進度
會包含目前比較到的模型以及該模型的 frame progress。取消會在下一個
streaming frame 或兩次 embedding 呼叫之間生效；ONNX Runtime 無法強制
中斷單次已開始的 session call。

上傳資料使用下列 JSON descriptor：

```json
{
  "filename": "speech.wav",
  "data": "data:audio/wav;base64,..."
}
```

預設停用本機檔案路徑。受信任的本機 client 可使用
`puresound web --allow-local-paths` 開啟相容格式
`{"path": "/absolute/path/input.wav"}`。產生的輸出放在有上限的記憶體
store，process 結束後即清除。

要在推論或量測中要求不需 reference 的品質指標，可加入
`"measurements": {"include_dnsmos": true}`；瀏覽器 client 會自動帶入。
DNSMOS 是 optional scorer；若套件或模型資產不可用，response 仍保留音量等
基本指標，並回報 DNSMOS error，不會讓整次執行失敗。
