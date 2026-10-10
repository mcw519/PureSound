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

你在別處開啟的網頁，仍可能對本機的服務送出請求。因此服務會拒絕來自其他 origin 的寫入或即時
工作階段；綁定在 loopback 位址的服務只回應 `localhost`、IP 位址與它的綁定位址，網頁就無法
用自己的網域名稱連到它（DNS rebinding）。若要用別的名稱連到它——例如 reverse proxy 的對外
名稱——請傳入 `--allow-host NAME`（可重複），或把該名稱列在 `PURESOUND_ALLOWED_HOSTS`
（以逗號分隔；在 Python 中是 `create_server(..., allowed_hosts=[...])`）。

瀏覽器只在 HTTPS 或 `localhost` 下允許網頁使用麥克風。要從別台機器錄音或即時串流，
請用 HTTPS：

```bash
puresound web --host 0.0.0.0 --https          # 自簽憑證，用 openssl 產生一次
puresound web --host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem
```

自簽憑證存在執行紀錄旁的 `tls/`；瀏覽器第一次會警告，接受後即可使用。即時模式會改走
`wss://`。

## 工作台

每個畫面都是同一個框架。**側欄**把畫面分組——「執行」（試聽台、語者驗證、比較）、
「資料」（標記、資料管線）、「資源庫」（模型庫、紀錄）——並放著執行環境狀態、語言切換
（EN / 中）、主題（自動＝跟隨系統、淺色、深色）與版本。每個畫面的**頁首**有標題、一行
用途說明、次要動作、*說明*（這個畫面做什麼、步驟與快捷鍵）、*設定*，以及最右邊唯一的
主要動作（*執行*、*語者驗證*、*執行比較*、*下載匯出檔*、*建立並追蹤*）。工作執行時，頁首
下方的狀態列會顯示正在做什麼、進行到哪裡；結束時顯示「完成」或失敗原因。

畫面的所有設定都在右側的**設定面板**。*設定* 可隱藏或顯示它，每個畫面各自記得。視窗窄於
1100 px 時，設定面板改為覆蓋在頁面上的抽屜（Esc 或點外面關閉）；窄於 760 px 時，側欄變成
頂端列加選單。整個介面有英文與繁體中文：數字、單位、模型名稱、檔名與指標名稱維持原樣。

## 畫面

| 畫面 | 網址 | 用途 |
| --- | --- | --- |
| **試聽台** | `#/playground` | 用檔案、範例、麥克風錄音或即時串流，在伺服器或瀏覽器中（*執行端*，見下文）跑 voice-isolation 或 noise-suppression 模型（可選 *先執行* 串接另一個模型，例如先降噪再隔離）；聆聽、比較並下載。 |
| **語者驗證** | `#/verify` | 判斷 enrollment 與 test 錄音是否為同一位語者（餘弦相似度對門檻）。 |
| **比較** | `#/compare` | 多個增強模型跑同一批錄音，與未處理的輸入並列，可附 reference metrics 與 DNSMOS。每一列是一個模型加它自己的設定；*+* 可為同一模型再加一組設定，因此 dry blend 或 onset guard 的掃描就是一次比較。 |
| **標記** | `#/annotate` | 在錄音上標出 keep / suppress 區段並匯出給評估使用（見下文）。 |
| **聲學世界** | `#/world` | 講者在房間中移動，渲染成一支麥克風的收音，經模型處理並評分；極限地圖掃描兩個參數（[world.zh-TW.md](world.zh-TW.md)）。 |
| **資料管線** | `#/pipeline` | 用自己的音訊（或內建樣本）搭配某個已發布 recipe 的增強旋鈕合成一列訓練資料，逐階段檢視（見下文）。 |
| **模型庫** | `#/models` | 模型 metadata、catalog 為每個模型列出的 release notes 與閘門紀錄（*詳細資料*）、多個模型的閘門紀錄並排（*比較閘門紀錄*），以及目錄完整性檢查。 |
| **紀錄** | `#/history` | 所有試聽台執行、語者驗證、比較與即時工作階段；*開啟* 可放回它原本的畫面。 |

網址會跟著畫面，重新整理或分享連結都會回到同一個畫面；頁面會開在上次使用的畫面，第一次
開啟時是試聽台。

播放 gain 與瀏覽器 limiter 不會改變模型輸入或下載檔案。

### 逐字檢查

每個比較播放器下方的 *Which words survived?* 會轉寫各軌並逐字對齊：刪除線＝被刪掉、底線＝
多出來、標色＝被改掉。點一下某個字，播放器會選到那一軌並循環播放該字附近。

- **辨識器：**本機 Whisper（faster-whisper，需安裝 `asr` extra：`uv sync --extra asr`；預設
  `large-v3`，因為弱的辨識器會掩蓋過度抑制）、Azure Speech（key + region）或 ElevenLabs
  Scribe（API key，預設模型 `scribe_v2`）。
- **Key** 在網頁上輸入，只存在該分頁的記憶體裡，只隨轉寫請求送出：server 不寫 log、不寫進
  執行紀錄、不回傳，錯誤訊息中也會遮蔽；瀏覽器也不儲存。關掉分頁就沒了。
- **有參考逐字稿**（近講者實際說的話）時，每一軌都算出對它的錯誤率。**沒有時**，各軌會和某一軌
  （預設是輸入）的轉寫比較；這是「差異」不是錯誤率——voice isolation 本來就該拿掉遠場干擾者
  的字，那些字也會顯示為缺少。
- 中文、日文、韓文按字元計（CER），其他語言按字計，正規化方式與
  `puresound.evaluation.tools.wer` 相同。
- 轉寫長度超過參考 1.5 倍的軌會標示為辨識器迴圈。

逐字檢查不會進入執行紀錄。

### 麥克風與即時模式

*Record* 以 16 kHz 擷取麥克風，並關掉瀏覽器自己的回音消除、降噪與自動增益，讓模型聽到
收音鏈原本的樣子；錄好後就像檔案一樣執行。*Live* 把麥克風每 20 ms 一段串流給選定的
模型。*Listen to* 只決定串流時你聽到什麼：*Nothing*（不播放）、*Model output*（模型輸出，
請戴耳機）、*Raw microphone*（未處理的麥克風，說話時與 Model output 切換即可即時 A/B）；
無論選哪個，串流與保存的內容都一樣。頁面會顯示每段的運算時間、來回延遲、演算法延遲（分析視窗＋look-ahead）以及麥克風
到耳朵的估計延遲。停止後整段會開在比較播放器裡。麥克風需要安全來源：請在 `localhost`
或 HTTPS 下開啟。

### 在這台裝置上執行

試聽台設定面板的 *執行端* 決定模型在哪裡跑：*伺服器*，或 *這台裝置*——在瀏覽器中以
worker 裡的 ONNX Runtime Web（WebAssembly）與 [Web SDK](../../sdk/web/README.zh-TW.md)
的串流 runtime 執行，音訊不會上傳。檔案、錄音與即時都能在這裡執行，結果開在同一個比較
播放器；同一個檔案先在伺服器、再在這台裝置上跑，伺服器的輸出會保留為 *Previous*，方便
並排聆聽。

- 只有具備裝置版本的模型能在這裡執行。在 `sdk/web` 執行一次
  `npm ci && npm run build && npm run assets` 即可建置；其他模型仍留在清單中，標示為
  *僅限伺服器*。
- *先執行*、成品變體與運算後端是伺服器的設定。在裝置上執行沒有 DNSMOS，也不會留在
  紀錄中；乾濕比與起音保護的效果和伺服器相同。
- `puresound web` 讓每個頁面都跨來源隔離（COOP `same-origin`、COEP `require-corp`），
  WebAssembly 因此最多可用四個執行緒。瀏覽器只隔離安全來源的頁面：從另一台機器以一般
  HTTP 開啟時只用單一執行緒（要更多請用 `--https`）。
- 即時模式會先測量裝置速度：處理一秒音訊若超過 0.7 秒，會建議改為錄音。工作階段落後
  一秒就會停止，並開啟已處理的部分。

其他畫面也透過同一個模組 `window.PureSoundDevice` 在裝置上執行模型；呼叫方式見
[Web SDK README](../../sdk/web/README.zh-TW.md#在網頁介面中windowpuresounddevice)。

Playground 內建三段增強範例與兩組語者配對（LibriSpeech 語音、合成的房間與噪音；
`tools/build_web_samples.py`）。

*ZIP* 與 *Report*（結果區、比較報告與每筆 History 都有）可把一次執行帶走：所有音訊輸出
加 `report.json`，或一頁內嵌音訊播放器、設定與分數的 HTML，給沒有 PureSound server 的人
看。報告內嵌音訊超過 40 MB 就不再嵌入並註明。

*清除* 會把畫面清回初始狀態：試聽台的輸入與結果、語者驗證的兩段錄音，或比較報告
（設定裡選好的錄音與列會保留）。執行中或即時串流時無法清除。

### 管線檢視器

Pipeline 畫面一次回答一列的問題：「訓練資料管線對這一列做了什麼？這對模型提出了什麼要求？」

- **合成一列。**選一個已發布的訓練 recipe（`--pipeline-root` 底下的
  `egs/*/config/*.yaml`）、seed、列的用途（training 或 validation），recipe 有 curriculum
  時還可選 epoch。音訊由你提供：一段前景語音、其他講者、噪音片段與房間——附幾何的內建
  樣本房間、上傳的脈衝響應（套用到整個混音）或不加殘響。內建兩位 LibriSpeech 講者、六種
  合成噪音，以及 PureSound 自有合成 RIR bank 的四間房（`tools/build_pipeline_samples.py`）；訓練語料一律不讀。需要檢視器沒有
  之語料的 block（真實錄音池、對話列）會被關閉，並列在「What the inspector changed in this
  recipe」。
- **階段。**依序列出合成的每個階段，依作用對象分組。實線表示在這一列上有作用；虛線表示
  沒有觸發（附其機率）；點線表示 recipe 沒有這一段或被檢視器關閉。
- **有效 SNR。**每個階段之後，target 能量對混音中其他所有成分的比值——噪音、不屬於
  target 的講者、`early` target 之後的晚期殘響、傳輸損傷：也就是留給模型移除的東西。
- **模型。**選了模型時，每個階段的 pair 都當成「訓練到此為止」來評分：混音與模型輸出
  各自相對於該階段 target 的 SI-SDR、STOI、PESQ；沒有 target 的列則看輸出的音量變化。
  超過滿刻度的 pair（A/D 階段之前）會先像轉換器那樣做增益分級。
- **階段詳情。**這一步做了什麼、對模型的意義（跟隨頁面語言）、這一列抽到的參數、
  前後 pair、模型分數，以及一個 deck：該階段前後的混音、target、其他講者、模型輸出與
  這一步本身造成的改變。
- **房間。**樣本房間的 3D 視圖（three.js，已內附；沒有 WebGL 時改為平面圖）：牆、
  障礙物、麥克風與所有聲源，這一列用到的聲源以其角色的顏色標示；以及每個脈衝響應的
  包絡、能量衰減與 target 視窗。

合成一列需要幾秒，有模型評分時每個階段再加幾秒；一次只合成一列。

### 標記

標記畫面用來在錄音上標出帶標籤的時間區段，供評估視窗與試聽筆記使用。它完全在瀏覽器裡
執行：音訊在瀏覽器解碼，不會上傳。

1. 開啟一個或多個音檔或一個資料夾（*開啟音檔*、設定面板的「檔案」，或直接拖放到波形區）。
   音檔旁的區段檔——`<name>.spans.json`、`.spans.csv`、`windows.json`、Audacity 標籤——會
   自動讀入；*匯入區段* 可手動讀取。
2. 在波形或頻譜圖上拖曳選取範圍、選標籤（1–9，或設定面板的「新區段的標籤」），按 Enter。
3. 在波形下方的軌道調整區段（拖曳邊緣或本體），在表格中更換標籤或加上說明文字。
4. *下載匯出檔*，或 *存回資料夾* 把 `<name>.spans.json` 寫在每個檔案旁邊（以可讀寫方式
   開啟的資料夾，限 Chromium 系瀏覽器）。

| 格式 | 內容 |
| --- | --- |
| JSON | 檔案 metadata 與區段；多個檔案時合在一起 |
| CSV | 開始、結束、長度、標籤與說明文字（多個檔案時加上檔名） |
| `windows.json` | 每個檔案給評估用的 `keep` 與 `suppress` 區間 |
| Audacity 標籤 | 目前檔案的開始、結束與說明文字，以 tab 分隔 |

`windows.json` 中，標籤含 `keep`、`near` 或 `double` 的成為 `keep` 區間，含 `sup` 或 `far`
的成為 `suppress`；其他標籤不寫入。

每個比較播放器（試聽台、比較、資料管線）都有 *標記*，會把正在聆聽的音軌當成新檔案在這裡
開啟。標記畫面的快捷鍵（空白鍵、Enter、1–9、S / E、Backspace、[ / ]、+ / − / 0、
Ctrl/⌘ + O / I / S / E；*說明* 裡有完整列表）只在這個畫面上、且沒有在欄位中輸入時作用；
播放器的快捷鍵在這裡會讓開。

### 試聽與比較

增強結果會開在比較播放器裡：模型實際聽到的輸入、輸出、被移除的部分（輸入減輸出），
共用同一條時間軸與同一組播放控制。所有軌同時播放、只有選中的那軌出聲，因此切換
無縫且逐 sample 對齊。同一個檔案再跑一次時，上一次的輸出會保留成第四軌，方便對照
設定改動前後。*Match level* 會把每軌調到與輸入相同的響度，避免「比較大聲的聽起來
比較好」。

| 按鍵或手勢 | 動作 |
| --- | --- |
| Space | 播放／暫停 |
| 1–9 | 切換出聲的軌 |
| 拖曳 | 選取區段並循環播放 |
| 拖曳區段邊緣 | 調整區段 |
| Shift + 點擊 | 把區段延伸到點擊處 |
| L | 開關循環 |
| Z | 放大到區段／還原 |
| + / − / 0 | 時間放大／縮小／時間與頻率回到全部 |
| Ctrl/⌘ + 滾輪 | 以游標為中心縮放時間 |
| Shift + 滾輪 | 時間平移（或拖曳尺規下方的捲軸） |
| Alt + 滾輪 | 以游標為中心縮放頻率（或用右側的 + / − 與捲軸） |
| Esc | 清除區段與縮放 |
| ← / → | 跳 1 秒（Shift：5 秒） |
| Ctrl+Enter | 執行目前的任務 |

*Export* 會下載目前出聲那軌的選取區段，以檔案原取樣率切出；*Annotate* 會把整軌開到標記畫面。*View ⚙* 可設定頻譜（FFT
大小、頻率範圍可到 Nyquist、線性或對數頻率軸、底限與動態範圍、色表）與波形（線性或 dB
振幅、增益）；設定套用到頁面上所有播放器並記在瀏覽器。檔案預覽也使用同一個播放器（單軌）。

*Δ vs input* 會加兩條與輸入對照的分析軌：每 20 ms 的位準差（總能量——語音、噪音、
殘響加在一起，所以下降本身不代表壓對了東西），以及逐頻帶的頻譜差（冷色＝被移除、
暖色＝被加入）。*dB* 把波形改成 0 到 −60 dBFS 的刻度，讓安靜的殘留看得見。滑鼠移到
圖上會顯示時間、頻率與位準。有輔助 head 的模型可用 `collect_extras` 回傳，會以曲線軌
顯示。


## Providers

`auto` 依序嘗試 CUDA、CoreML、CPU。也可指定 `cpu`、`cuda`、`coreml`
或 `mps`；`mps` 是 CoreML 的 alias。Response 會回報實際執行模型的 provider。

## API

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | Runtime 與 provider 狀態 |
| GET | `/api/models` | 列出 catalog 模型 |
| GET | `/api/models/{model_id}` | 讀取模型 contract |
| GET | `/api/models/{model_id}/benchmarks` | 解析後的該模型 benchmark 參考資料 |
| GET | `/api/validate` | 驗證 artifacts 與 ONNX graph |
| POST | `/api/infer` | 執行同步推論 |
| POST | `/api/measure` | 比較模型與 metrics |
| GET | `/api/jobs` | 最近的 job（`?limit=`，預設 20） |
| POST | `/api/jobs` | 啟動非同步 job |
| GET | `/api/jobs/{job_id}` | 讀取狀態與結果 |
| POST | `/api/jobs/{job_id}/cancel` | 要求取消 job |
| GET | `/api/runs/{run_id}/{output}` | 下載生成音訊 |
| POST | `/api/uploads` | 上傳檔案原始 bytes（`X-Filename` header），回傳 `upload_id` |
| GET | `/api/uploads/{upload_id}` | 確認上傳檔仍在 |
| GET (WebSocket) | `/api/live` | 逐幀串流音訊經過模型 |
| GET | `/api/asr` | 可用的逐字檢查辨識器、已快取的 Whisper 模型 |
| GET | `/api/pipeline` | 管線檢視器：可用的訓練 recipe、內建樣本、是否可用 |
| GET | `/api/jobs/{job_id}/outputs.zip` | 已完成執行的所有音訊輸出，附 `report.json` |
| GET | `/api/jobs/{job_id}/report.html` | 單一自足 HTML 頁面、內嵌音訊（`?tz=` 指定時區） |

非同步 job 狀態為 `queued`、`running`、`succeeded`、`failed` 或
`cancelled`。

## 音訊輸入

瀏覽器用 `POST /api/uploads` 把每個檔案送一次（body 為原始 bytes，檔名放在
`X-Filename`），之後以 id 引用：

```json
{"upload_id": "3f2a...", "filename": "speech.wav"}
```

因此重跑或加比較模型都不必再傳一次。上傳檔有上限（64 個、1 GiB），保留到服務停止；
引用到已被淘汰的檔案時會回錯誤請重新上傳。單一解碼後的音訊輸入最多
`--max-upload-mb` MiB（預設 64），其取樣點以 16-bit 計最多為該值的 16 倍，因此壓縮檔不會
無限膨脹。也仍可使用 data URL：

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

生成檔案與執行紀錄有上限（32 次）。`puresound web` 會把它們存在
`$XDG_CACHE_HOME/puresound/web`（否則 `~/.cache/puresound/web`），重啟後仍在；
`--history-dir` 可改位置，`--no-history` 只放記憶體、服務停止即清除。只保留已完成的
執行。

## 增強輸出

增強類 request（voice isolation 或 noise suppression）會回傳多個 `output_urls`：

| 輸出 | 內容 |
| --- | --- |
| `audio` | runtime 原樣輸出的串流：前面帶串流延遲，之後是完整輸入、不多不少。與 CLI 寫出的內容相同。 |
| `aligned` | 去掉延遲並裁到輸入長度的 `audio`。瀏覽器播放與匯出用這個。 |
| `input` | 模型實際聽到的輸入：單聲道、模型取樣率。 |
| `removed` | `input - aligned`：模型拿掉的部分。 |

`alignment` 記錄移除的延遲量。

## 音訊量測

加入 `"measurements": {"include_dnsmos": true}` 可要求 DNSMOS。若 optional
scorer 不可用，其他 metrics 仍會正常回傳。

有 clean reference 時可計算 SI-SDR、SNR、correlation、STOI 與 PESQ；沒有
reference 時仍可取得位準與頻譜量測。

比較時會用同一組評分器替未處理的輸入打分，以 `baseline` 回傳（附音檔）；頁面上每個
分數都與它並列。每一列是 `candidates` 裡的一項：一個模型加它自己的參數（疊在共用的
`parameters` 上）：

```json
{
  "candidates": [
    {"model_id": "voice-isolate-dpcrn-curriculum-v1", "label": "release"},
    {"model_id": "voice-isolate-dpcrn-curriculum-v1", "parameters": {"dry_blend": 1.0}}
  ]
}
```

舊的 `models`（每個模型一列、共用參數）仍可使用。每個 candidate（以及一般的
`/api/infer` request）可加 `"stages": [{"model_id": ...}]`：先跑的模型，每個都吃前一個
的輸出（已對齊回輸入時間軸）。回應的 `pipeline` 列出各段與延遲；RTF 與延遲涵蓋整條串接。

多段錄音放在 `inputs.clips`（`[{"audio": ..., "reference": ...}]`，最多 50 段）。回應的
`clips` 是每段的報告；`aggregate` 則是每個 candidate 相對各段未處理輸入的變化平均、95%
bootstrap 區間、變好的段數，以及 `resolved`——只有區間不含 0 且至少 5 段時才為 true。

RTF 只計入逐幀處理；建立 stream 與第一次建立 ORT session 的時間另外記在
`metadata.setup_seconds`。剛載入的增強模型會先跑一次不計時的 0.5 秒暖機。單次
RTF 仍只是一次抽樣（同一台共用主機上相同的執行會有明顯落差），因此比較時可設
`"measurements": {"timing_repeats": 3}`（1–5）回報中位數，每次結果列在
`rtf_runs`。回應附上 `host`（1 分鐘 load average 與 CPU 數），方便對照當時主機
負載判讀數字。

## 逐字檢查 job

`POST /api/jobs`，`"kind": "transcription"`：

```json
{
  "kind": "transcription",
  "backend": "whisper",
  "model": "large-v3",
  "language": "zh-TW",
  "credentials": {},
  "reference_text": "",
  "reference_track": "input",
  "tracks": [
    {"id": "input", "label": "Input", "source": {"url": "/api/runs/<run>/input"}},
    {"id": "output", "label": "Output", "source": {"upload_id": "<id>"}}
  ]
}
```

`backend` 為 `whisper`、`azure`（`credentials: {"key", "region"}`）或 `elevenlabs`
（`credentials: {"key"}`）。`language` 用 BCP-47，空字串為自動偵測（Azure 此時用 `en-US`）。
結果包含每軌的文字、帶時間的字、評分 token，以及對參考的 `alignment`
（`{"op": "hit" | "sub" | "del" | "ins", "ref", "hyp"}`）與 `counts`（`error_rate`、`del_rate`
等）。這類 job 只放記憶體，不會出現在 `/api/jobs` 列表。

## 管線追蹤 job

以 `"kind": "pipeline_trace"` 呼叫 `POST /api/jobs`，會用 `GET /api/pipeline` 列出的
recipe 合成一列訓練資料，並逐階段追蹤：

```json
{
  "kind": "pipeline_trace",
  "recipe": "egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml",
  "role": "train",
  "seed": 1234,
  "epoch": null,
  "seconds": 6.0,
  "foreground": {"upload_id": "3f2a..."},
  "talkers": [{"sample": "reader-1272"}],
  "noises": [{"sample": "babble"}, {"upload_id": "9c1d..."}],
  "rir": {"kind": "samples", "rooms": ["room_000020_000000"]},
  "model_id": "noise-suppression-dpcrn-mamba-v2"
}
```

每個來源都是上傳檔或內建樣本；訓練語料、RIR bank 與錄音池一律不讀。recipe 只提供
它的增強旋鈕：語料路徑改指向一個存放所選音訊的暫存 workspace，需要 workspace 無法
替代之語料的 block（真實錄音池、對話列）會被關閉並列在報告的 `notes`。`rir.kind`
可為 `samples`（附幾何的內建房間）、`upload`（套用到整個混音的脈衝響應）或 `none`。

完成的 job 其 `result` 是追蹤報告（`schema: puresound.pipeline-trace/1`）：依合成
順序列出每個階段抽到的參數、該階段之後 pair 的位準與有效 SNR，指定 `model_id` 時還有
模型在該 pair 上的分數；另附脈衝響應、房間幾何與 `audio_urls`。音訊為 32-bit float
WAV，因為轉換器之前的階段可能超過滿刻度。一次只合成一列。提供的 recipe 來自
`--pipeline-root`（預設：目前目錄若有 `egs/` 資料夾即用它）底下的
`egs/*/config/*.yaml` 訓練 recipe；管線 job 與最近六次追蹤的音訊只放在記憶體中，和執行紀錄分開。

## 即時協定

`/api/live` 是 WebSocket。client 第一則訊息為 JSON：`{"model_id", "variant",
"provider", "parameters"}`；server 回 `{"type": "ready", "sample_rate",
"hop_length", "win_length", "latency_samples", ...}`。之後每則 binary 訊息是 little-endian
`uint32` 序號加上模型取樣率的 float32 samples（最多 16000）；server 以同一序號、float32
處理毫秒數與 stream 輸出的 float32 samples 回覆。回覆的串流與離線 runtime 一樣晚
`latency_samples`。server 約每秒音訊送一次 `{"type": "stats"}`；`{"type": "stop"}` 結束
連線。單次連線上限 30 分鐘。

## Onset guard

Voice-isolation request 可覆寫：

- `onset_guard`
- `onset_guard_t_arm_s`
- `onset_guard_t_forget_s`
- `onset_guard_tau_dn_s`
- `onset_guard_margin_db`

Guard 在偵測到持續語音前直接輸出原始輸入，之後逐步切換至模型輸出。它能保護語句
開頭，但會降低初期抑制。省略 override 時使用模型 manifest。

## 聲學世界

`#/world` 畫面、其 API 與場景包說明見[world.zh-TW.md](world.zh-TW.md)。
