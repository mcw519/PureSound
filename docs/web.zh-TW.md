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
  dry-blend override 仍由 processor 負責。
- **Speaker Verification**：上傳 `enrollment` 與 `test` 音檔，呼叫兩次
  waveform embedding processor，顯示 cosine score 與 threshold verdict。
- **Measurements**：提供 probe 音檔、可選的 clean reference，以及一個或
  多個 Voice Isolation model。會回報可重現的音量／頻譜指標（RMS、peak、
  clipping、silence、zero-crossing rate、spectral centroid）；有 reference
  時再計算 SI-SDR、SNR、correlation、STOI 與 PESQ。

左側主導航與 Playground 的右側設定欄可獨立收合；瀏覽器會記住下次開啟時
的選擇。

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
| POST | `/api/infer` | 以 JSON named inputs 執行模型 |
| POST | `/api/jobs` | 啟動非同步推論並取得 job id |
| GET | `/api/jobs/{job_id}` | 讀取進度、狀態與結果 |
| GET | `/api/jobs?limit=20` | 列出最近推論結果 |
| POST | `/api/jobs/{job_id}/cancel` | 要求取消 queued／running job |
| POST | `/api/measure` | 測量 probe 並比較選定的 Voice Isolation model |
| GET | `/api/runs/{run_id}/{output}` | 下載產生的音檔 |

非同步 job 會回報 `queued`、`running`、`succeeded`、`failed` 或
`cancelled`，並提供粗略 phase／progress。取消採 cooperative 語意：若
processor 已進入 ONNX 呼叫，背景工作會完成但丟棄結果，job 最終標記為
cancelled。

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
