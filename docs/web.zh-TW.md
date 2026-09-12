# Web UI 與推論 API

English: [web.md](web.md)

PureSound 的本機 Web UI 與 CLI 共用同一套 Model Zoo runtime。

## 啟動

```bash
puresound web
# http://127.0.0.1:7860
```

變更綁定位置：

```bash
puresound web --host 0.0.0.0 --port 8080
```

服務沒有身份驗證。只有在可信任網路或已配置驗證的 reverse proxy 後方，才應綁定
`0.0.0.0`。

## 畫面

- **Model Zoo：**查看模型 metadata 與 artifact 驗證。
- **Voice Isolation：**上傳音訊、執行模型、播放並下載 WAV。
- **Speaker Verification：**比較 enrollment 與 test 錄音。
- **Measurements：**比較多個 voice-isolation 模型，可附 reference metrics 與
  DNSMOS。

播放 gain 與瀏覽器 limiter 不會改變模型輸入或下載檔案。

## Providers

`auto` 依序嘗試 CUDA、CoreML、CPU。也可指定 `cpu`、`cuda`、`coreml`
或 `mps`；`mps` 是 CoreML 的 alias。Response 會回報實際執行模型的 provider。

## API

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | Runtime 與 provider 狀態 |
| GET | `/api/models` | 列出 catalog 模型 |
| GET | `/api/models/{model_id}` | 讀取模型 contract |
| GET | `/api/validate` | 驗證 artifacts 與 ONNX graph |
| POST | `/api/infer` | 執行同步推論 |
| POST | `/api/measure` | 比較模型與 metrics |
| POST | `/api/jobs` | 啟動非同步 job |
| GET | `/api/jobs/{job_id}` | 讀取狀態與結果 |
| POST | `/api/jobs/{job_id}/cancel` | 要求取消 job |
| GET | `/api/runs/{run_id}/{output}` | 下載生成音訊 |

非同步 job 狀態為 `queued`、`running`、`succeeded`、`failed` 或
`cancelled`。

## 音訊輸入

瀏覽器上傳使用 data URL：

```json
{
  "filename": "speech.wav",
  "data": "data:audio/wav;base64,..."
}
```

本機路徑預設停用。受信任的 local client 可用 `--allow-local-paths` 啟動服務，
再傳送：

```json
{"path": "/absolute/path/input.wav"}
```

生成檔案放在有容量上限的記憶體 store，服務停止時會清除。

## 音訊量測

加入 `"measurements": {"include_dnsmos": true}` 可要求 DNSMOS。若 optional
scorer 不可用，其他 metrics 仍會正常回傳。

有 clean reference 時可計算 SI-SDR、SNR、correlation、STOI 與 PESQ；沒有
reference 時仍可取得位準與頻譜量測。

## Onset guard

Voice-isolation request 可覆寫：

- `onset_guard`
- `onset_guard_t_arm_s`
- `onset_guard_t_forget_s`
- `onset_guard_tau_dn_s`
- `onset_guard_margin_db`

Guard 在偵測到持續語音前直接輸出原始輸入，之後逐步切換至模型輸出。它能保護語句
開頭，但會降低初期抑制。省略 override 時使用模型 manifest。
