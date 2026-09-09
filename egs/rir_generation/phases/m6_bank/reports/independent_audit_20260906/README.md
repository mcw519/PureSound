# M6 RIR bank independent audit — 2026-09-06

本次目的：重新判斷目前磁碟上的 M6 是否適合做訓練素材，不把舊實驗結論、程式測試 PASS 或 config 註解當成實證。未訓練模型、未使用 GPU、未修改 bank 或訓練設定。

## 結論

**現有 M6 主／boundary release 可以列為受控訓練實驗的候選素材；本次沒有發現足以全面否定它的檔案／幾何標籤問題。但「已經證實物理正確、完整涵蓋部署場景、優於舊 bank」都不成立。**

這三個判定要分開：

1. 素材是否可讀、標籤是否一致、split 是否隔離：本次查到的正面證據較強，抽樣與範圍見下。
2. 是否重現真實房間／設備的條件分佈：未獲充分證實，且存在距離與衰減分佈限制。
3. 是否改善目前 separator：本次沒有新訓練證據；既有實驗有混淆因素及追溯缺口，不宜據此宣布 M6 有效或無效。

`candidate`／缺 production certificate 是證據狀態，並不等於 WAV 不可用；反過來，QC 通過也不等於下游模型一定更好。

## 直接檢查的實體資料

根目錄：`/work/any_exp_link/puresound_exp/hybrid_rir_16k_m6_20260804`。

| release | release SHA-256 | 全部 items | train / validation / test | train 聲學空間 |
|---|---|---:|---|---:|
| path-events-m4_release | `09e2f5d22a0bf48e4a1daf347890c603aad8685cdeb518b479aa89570d03688e` | 50,000 | 39,560 / 5,540 / 4,900 | 1,978 |
| boundary_release | `49eb723a4dcf58be8e4ac8459c112be25e57a516dc1ec2a8a544182f3e9b3c10` | 10,000 | 8,180 / 960 / 860 | 409 |

兩者 `release_status=candidate`。ready recipes 是 `synthetic_calibrated` 與 `synthetic_peak_normalized`；`real_native` 與 `mixed_calibrated_real` 都 blocked。實測素材另存在 legacy `real_rir_16k_train_view`，不是這兩個 release 已包含的 variant。

另核對 `rir_release_m6_v0/hybrid_rir_16k_m6_20260804` 內兩個 release 的宣告 SHA 與上述一致。`rir_release_m6_v0` 是合成主 bank、boundary、實測 bank 等素材的包裝目錄，不能與「單獨的 M6 合成 renderer」混稱。

- 解析兩個完整 manifest，共 60,000 items。每個 release 內 train/validation/test 的 acoustic_space_id 交集為 0；room_id 沒有跨 split；兩個 release 之間各 split 的 acoustic_space_id 交集也是 0。這驗證宣告的識別資料，沒有對所有 WAV 做重複波形搜尋。
- 重算 release 指向的兩個 variant manifest、QC summary、distribution 與所有 ready recipe split index 的檔案 SHA，均符合 release 描述。
- 固定 seed `20260906`，各 bank 隨機抽取 128 個不同 train+pass 聲學空間，每個空間抽一個 item、讀取全部 5 個通道：共 256 items／1,280 channels。
- 抽中 item 的 WAV 與 metadata SHA 全部符合 manifest；聲音數值有限且能量非零；channel_map 距離與 source／receiver 三維座標重算距離一致。
- M6 抽樣中，第一個超過相對峰值 1e-7 的 sample 與幾何到達時間的差均在 1 ms 內。這只是到達時間 sanity check，不是全套物理驗證。
- 額外各抽 128 個舊 hybrid synthetic、legacy measured items，共 1,081 通道作探索性比較；這兩組沒有做 manifest hash 驗證，也不是依物理房間均衡抽樣。

## 新量到的限制

### 1. 主 bank 不涵蓋 1–2 m；boundary 必須另計

抽樣主 bank 近通道距離 0.350–0.938 m、遠通道 2.052–5.426 m，640 通道中 1–2 m 為 0。metadata 生成設定分別為 near `[0.35,0.95]`／far `[2.05,5.5]`。

boundary 抽樣遠通道 1.201–2.100 m，作用確實是補較近的旁人。兩者不是可互換的資料，也沒有覆蓋所有 1 m 附近或更遠的部署位置。

直接執行目前 loader `_pick` 的函式內容，用 v20 的 far `[1.5,4.0]`、空 used-channel set：主 bank 2/128、boundary 10/128 場景找不到範圍內通道，會回退到區間中心最近的現有通道。實際距離 metadata 仍正確；錯的是把 config 範圍當成硬性保證。這不是完整 row 分佈量測，多次使用通道後的候選集合還會改變。

程式：`puresound/audio/rir/bank/loader.py::_pick`。

### 2. 名義 RT60 範圍不是 WAV 實際分佈

主 bank metadata 的 `config.rt60_range=[0.25,0.8]`。但是 v2 material sampler 明確不繼承 legacy scene.rt60，而由抽樣材質推導新 scene.rt60。

本次獨立讀 WAV，用全頻 Schroeder EDC -5 到 -25 dB 線性擬合外推 RT60，保留 R²≥0.95：

| bank | 有效通道 | RT60 外推中位數 | p95 |
|---|---:|---:|---:|
| M6 main | 632/640 | 0.475 s | 2.484 s |
| M6 boundary | 633/640 | 0.477 s | 1.813 s |
| 舊 hybrid synthetic | 630/640 | 0.546 s | 0.831 s |

因此「大部分衰減集中在約半秒」與「有顯著長尾」可以同時成立。長尾不自動表示物理錯誤，也不自動表示對部署有幫助；這是比較 renderer 時必須控制的資料分佈變因。

以上 T20 是由 20 dB 衰減外推的 RT60，不是完整觀測 60 dB；M6 檔案只有 1.6 s，長衰減估計有截尾與模型假設限制。本次算法不含量測噪聲補償，因此不拿 measured 組的 T20 或總體斜率做「哪個 bank 更真實」的排名。

程式：`puresound/audio/rir/scene/sampling.py::upgrade_hybrid_scene_to_v2`、`puresound/audio/rir/scene/schema.py::rt60`。

外部方法依據：[Pyroomacoustics 官方 Room Simulation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html) 也區分理論／指定 RT60 與實際模擬量測值；此原則不是對 M6 本身的認證。

### 3. calibrated 不等於「已拿真實房間校準通過」

目前 manifest 的 renderer profile 是 `evidence_tier=development`，`calibration_report_sha256`／`approval_report_sha256` 均為 null；generator revision 以 `-dirty` 結尾。雖然檔案可以由 hash 鎖定，但僅憑 commit id 無法完整重建當時未提交的 generator。

抽樣 metadata 的 `output_calibration` 描述參考源聲壓與不做 item peak normalization。全部抽樣 M6 場景記錄 `crossover.policy=energy_rms_match`、`source_convention_preserved=false`。這表示本批次低高頻交界採能量匹配，不能把較新 renderer 支援的 source-convention preservation 或校準能力回推成這批 WAV 已具備的證據。

這些旗標本身不證明聲音錯誤；它們限制「已完成量測校準／絕對聲壓模型正確」的宣稱。

### 4. 訓練處理會改變磁碟上 RIR 的語意

`puresound/audio/impulse_response.py::wav_apply_rir` 逐通道以絕對峰值正規化，並把卷積結果對齊該通道的最大絕對峰值。故 bank 的絕對增益／跨通道聲級比與幾何傳播延遲不會原樣到達模型；DRR、頻谱與衰減形狀仍會影響訓練。其他 mix-mode 可能再明確加入距離聲級關係，但這是不同的處理步驟。

也直接計算 full／early 的裁切峰值與正規化：本次 2,361 通道未見兩者對齊／尺度不一致。未發現問題不等於所有 bank 通道都已驗證。

### 5. 舊「DRR 斜率更像實測，所以 M6 正確」結論過強

目前 DRR 函式以最大絕對峰值開始取 2.5 ms 窗，並忽略之前能量；它是 repo 的操作性指標，不保證該峰就是幾何直達聲。

主 M6 抽樣 93/640 通道的最大峰比幾何到達時間晚超過 6 ms；boundary 為 81/640。原因可能包含反射、方向性、遮蔽與濾波，不能直接判成違反物理，但足以說明 peak-based DRR 與幾何直達聲 DRR 不能混同。

舊 `compare_bank_acoustics.py` 的取樣也不是按房間、距离、混響、設備匹配後的 renderer 對照。新獨立測量依同一 item 去均值後的 DRR 對 log(distance) 斜率：M6 main 約 -9.96、boundary 約 -10.83、hybrid synthetic 約 -14.00 dB/decade；這支持「兩個生成分佈不同」，不支持「M6 因此較真實」。measured 組的房間、量測噪聲、設備與近遠通道組合並未匹配，不做正確性排名。

## 為何不能直接信任舊實驗裁決

- `phases/m6_bank/reports/m6_production_decision_report.json` 明示 `explicitly_not_production_evidence=true`，listening/downstream 是 `contract_fixture`、empirical false。它是在測 promotion gate，不是有人做完盲聽與多 seed 模型訓練的證据；不是報表造假，但不能誤讀其 PASS。
- 單獨 M6 synthetic 替換 hybrid+measured，會同時移除 measured RIR 分量；「只改了一個 YAML block」仍可能同時改 origin、room count、距離／RT60 分佈。
- `m6bank_real` 的 union 設計保留同一 measured 分量，比 pure-M6 換 bank 對照更接近公平；但仍需同初始化、同訓練預算、匹配聲學覆蓋與多 seed 的原始證據。
- 對 v8 checkpoint 再訓練 M6 後直接比未續訓 v8，差值包含續訓效果；需 old-bank 也同起點續訓的控制。from-scratch 也不是唯一能比較 bank 的方法，重點是各臂起點與預算一致。
- 四份目前 `train_dpcrn_m6bank*.yaml` 均明訂 `validation_pipeline_role: train`。其 validation 不證明未見房間泛化；獨立 real／room-held-out eval 仍可提供資訊，但不能用該 valid curve 替代。
- 四個歷史 run 的 `lightning_logs/version_0/hparams.yaml` 都是 `{}`；該層只看到 event、checkpoints、hparams，沒有完整 recipe snapshot。這不是證明當時一定用錯設定，是目前無法僅從這些記錄還原完整訓練條件。
- 一個具體文字矛盾：目前 `train_dpcrn_m6bank_v8recipe.yaml:37–38` 把 scratch 描述為 NO real rows；目前 scratch recipe 的 realfar／realnear 卻都是 used=true。可能是後續 config 變更，也可能是註解錯誤；未取得當時快照前，不把任一版本當作已確定的歷史事實。

因此保留 raw per-file eval 作為觀察，不採用「M6 已證實無效／RIR fidelity 絕不重要」等強因果結論。

## 對 v20 的建議

1. 現在的 v20 仍指向 `hybrid_rir_16k_realfar`。不要把現有 recipe 當成 M6 experiment，也不要因舊結論直接丟棄 M6。
2. 固定 bank 版本、room split、距離／實際衰減／來源分佈，先測生成後到模型輸入前的有效資料；包括 `_pick` fallback、正規化、SIR 與 early target。
3. 第一個公平對照：舊 synthetic + 固定 measured vs M6 main + 同一 measured，使用相同初始化、語音／噪聲來源、有效樣本預算、loss 與 optimizer，並匹配距離、衰減覆蓋。boundary 作後續獨立開關，避免混入另一個變因。
4. 評估固定的未見房間與獨立真實收音條件；matched far suppression 下看使用者開頭／回來保留與 ASR deletion。不能僅用總 loss 或 DRR 斜率選 bank。
5. 若要修 renderer，先做固定幾何的距離、朝向、遮蔽、材質與 crossover 單因素檢查，再以匹配条件實測 RIR 校準；目前抽樣不支持直接宣稱 renderer 哪個物理部件已錯。

## 重現與限制

在 repo root、既有 `.venv`：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  egs/rir_generation/phases/m6_bank/reports/independent_audit_20260906/audit.py
python egs/rir_generation/phases/m6_bank/reports/independent_audit_20260906/supplement.py
```

腳本輸出 `/tmp/m6_independent_audit_20260906.json`；本次結果存為同目錄 `measurements.json`。原始測量、sample file paths、release 描述、hash 檢查、來源程式 SHA 均保留。重新跑之前應注意 tmp 報表會被覆盖；bank 是唯讀。

本次是 512 items 的有限抽樣與 60,000 manifest entries 的結構檢查，沒有全量 waveform QC、盲聽、實測 room matching、模型重訓、或目標裝置部署測試。legacy measured 描述性統計含未作噪聲補償的估計，不能用它作正確性標準。沒有把本次結果包裝成 production certification。
