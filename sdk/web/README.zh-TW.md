# PureSound Web SDK

[English](README.md)

獨立的 TypeScript 串流 runtime，包含 periodic Hann STFT、重疊相加、模型狀態、
對齊的 dry blend 與 onset guard。固定使用 ONNX Runtime Web 1.24.3。網頁介面用它執行
試聽台的 *這台裝置*（見[網頁介面說明](../../docs/usage/web.zh-TW.md#在這台裝置上執行)）：
WASM 在專用 Worker 中執行；頁面沒有跨來源隔離時使用單執行緒，有隔離時最多四執行緒。
`puresound web` 提供的每個頁面都有隔離。

## 建置

在此目錄執行：

```sh
npm ci
npm run build    # src/ -> dist/
npm run assets   # -> puresound/web/static/device/
```

`npm run assets`（`tools/build_assets.py`）把編譯後的 runtime、ONNX Runtime Web，以及每個
已發布模型的 ONNX 檔與嚴格 JSON manifest 複製到 `puresound/web/static/device/`，並寫入
`catalog.json`（每個檔案的 SHA-256，頁面據此驗證下載）與第三方授權聲明。工具只複製既有
發布模型，不重新訓練或匯出。完成後在專案根目錄執行 `.venv/bin/python -m puresound web`，
在試聽台的設定面板選 *這台裝置*。

## Runtime API

```ts
import { PureSoundStreamingRuntime } from '@puresound/streaming';
const runtime = await PureSoundStreamingRuntime.create(modelBytes, manifest);
const chunk = await runtime.processSamples(float32Mono16k);
const tail = await runtime.flush();
runtime.reset();
await runtime.dispose();
```

依序呼叫處理方法。輸出與 Python SDK 相同，比輸入晚
`streaming_delay_frames * hop_length` 個 samples；`flush` 後串流長度恰為輸入長度加上
這段延遲，檔案播放時從開頭丟掉這麼多 samples 即可。
`flush` 可重複呼叫；`reset` 開始獨立串流；`dispose` 釋放模型 session。
預設依賴由 bundler 解析；靜態 Worker 可傳入 `{backend: ort}`，參考
`puresound/web/static/device/worker.js`。不支援的 processor 或後處理設定會明確拒絕。

## 在網頁介面中：`window.PureSoundDevice`

`puresound/web/static/device/device.js` 在 Worker 中執行 runtime，是網頁介面各畫面在
裝置上執行模型的唯一途徑；試聽台的檔案、錄音與即時都用它。頁面以
`<script src="/device/device.js" defer>` 載入。

| 呼叫 | 結果 |
| --- | --- |
| `status()` | `{ready, models, threads, isolated, reason}`：能否在這裡執行、有裝置版本的模型 id、執行緒數，以及不能執行時的原因（頁面語言） |
| `decode(blob, {sampleRate = 48000})` | `{samples, sampleRate}`：瀏覽器能解碼的音訊，轉成單聲道 `Float32Array`；傳入檔案本身的取樣率可避免重取樣兩次 |
| `process(samples, options)` | 整段錄音離線通過模型（見下文） |
| `liveLink({model, dryBlend, onsetGuard})` | 給 `PureSoundCapture.LiveSession`（`capture.js`）用的即時連結；麥克風、監聽與統計由 LiveSession 負責 |

`process(samples, {model, sampleRate = 16000, dryBlend, onsetGuard, onProgress, signal})`
在 worker 中把單聲道 `Float32Array` 重取樣到 16 kHz、串流通過模型並 flush。結果包含
`input`（16 kHz 輸入）、`output`（去掉模型前瞻、與 `input` 等長的串流）、`stream`（原始
輸出）、`removed`（`input − output`）、`latencySamples`、`latencyMs`、`rtf`、`threads`、
實際套用的 `dryBlend` 與 `onsetGuard`，以及 `seconds`。`dryBlend` 介於 (0, 1]；
`onsetGuard` 為 `false`（關閉）、`true` 或覆蓋 manifest 設定的參數物件（`t_arm_s`、
`t_forget_s`、`tau_dn_s`、`margin_db`……），省略則依 manifest。`onProgress(fraction, phase)`
先回報 `"load"`（fraction 為 `null`）再回報 `"run"`。中止 `signal` 會以 `cancelled` 為
`true` 的錯誤結束並終止 worker，下一次呼叫會重新載入模型。呼叫依序執行，最後用的模型
保持載入，超過十分鐘的輸入會被拒絕。

```js
const run = await PureSoundDevice.process(samples, { model: "voice-isolate-dpcrn-curriculum-v1", sampleRate: 16000, signal });

const session = new PureSoundCapture.LiveSession({ onStats, onLevel, onStatus });
await session.start({ link: PureSoundDevice.liveLink({ model, dryBlend: 0.9 }), monitor: "off" });
const { input, output, removed } = await session.stop();
```

即時連結開啟時會先測量裝置：以 20 ms 分塊處理一秒音訊的 RTF 必須不超過 0.7，否則拒絕
並建議改為錄音。工作階段落後一秒（50 個分塊）就會結束，保留已串流的部分。
`PureSoundCapture.serverLink` 是連到伺服器 `/api/live` 的同一種連結。

## 驗證

先在專案根目錄產生 Python 對照，再在此目錄執行測試：

```sh
# 在專案根目錄
.venv/bin/python sdk/web/tools/generate_fixtures.py
# 在 sdk/web
npm test
npx playwright install chromium firefox webkit
npm run test:browser
npm run test:microphone
```

數值測試涵蓋每個有裝置版本的模型、短檔、靜音、語音、onset guard 啟動及重新保護、任意分塊、重複
flush 與 reset，要求 NRMS ≤ 1e-4。

瀏覽器與麥克風測試以 *這台裝置* 模式操作試聽台，需要已建置裝置資源並在 7861 埠執行的
`puresound web`；`PURESOUND_WEB_URL` 可改位址，`PURESOUND_BROWSERS=chromium` 可限縮為單一
引擎，`PURESOUND_CHROMIUM_PATH` 可指定非 Playwright 安裝的 Chromium。瀏覽器測試檢查頁面
已跨來源隔離、每個有裝置版本的模型都能處理範例、僅限伺服器的模型仍列出並標示、模型下載
失敗與取消及執行取消後可恢復、裝置輸出與伺服器對同一範例的輸出一致（16-bit 匯出後
NRMS < 2e-3），以及語言切換。頁面網址加 `?threads=1` 可強制單執行緒基線。麥克風測試使用
模擬收音裝置：錄音並在裝置上執行、即時串流（或確認太慢的裝置被拒絕），並檢查麥克風
權限被拒；`PURESOUND_LIVE_SECONDS=600` 可執行十分鐘串流。正式發布前仍需在 macOS 驗證實際
Safari（Playwright WebKit 為另一項引擎檢查），並另行驗證實際音訊裝置。

架構參考：[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/build-web-app.html)、
[WASM 執行緒限制](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)。

## 授權

SDK 自有程式碼採用 [Apache-2.0](LICENSE)，排除項目請見 [NOTICE](NOTICE)。
PureSound 自行訓練的模型（包含 model zoo 的 checkpoint 與 ONNX 匯出）同樣採用
Apache-2.0。相依套件與第三方模型保留各自的授權。`licenses/` 內是 ONNX Runtime
的授權與第三方聲明。將 SDK 複製到其他專案時，請一併保留 LICENSE、NOTICE
及適用的第三方聲明。
