# RIR bank v2

English: [rir_bank_v2.md](rir_bank_v2.md)

v2 bank 格式把 RIR 音訊和 split、provenance、完整性、QC、release 資訊放在
同一份契約中。只有一個 WAV 目錄不足以支撐可重現的訓練 release。

## Bank 內容

Bank root 包含：

```text
bank/
├── rir_bank_manifest.json
├── rir_bank_generation_audit.json
├── indexes/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── scenes/
├── metadata/
└── audio/
```

Manifest 是唯一可信來源。每個 item 記錄：

- 穩定的 item、room 與 parent-room identity；
- train、validation 或 test split；
- scene schema 與 canonical scene hash；
- renderer profile、設定與 code revision；
- sample rate、channels、frames 與 level policy；
- 保留資產的路徑與 SHA-256；
- QC 狀態與 release eligibility。

同一實體房間，或同一 synthetic parent room 的衍生場景，不得跨 split。

## 產生 bank

查看完整參數：

```bash
python egs/rir_generation/phases/m6_bank/scripts/generate_m6_bank.py --help
```

執行時應明確指定 output directory、renderer backend、room count、seed、sample
rate 與 duration。這些值必須保留在 release recipe；任何會改變輸出的設定，都
代表另一個 bank。

產生流程會寫出 manifest、split indexes 與 generation audit，並支援安全續跑：

- identity、設定與 hash 都一致時才重用 item；
- 資產缺少或被修改時會重建；
- renderer、revision、duration 或其他生成條件改變時，不會混入舊 bank。

只要偵測到 v2 manifest 或完整 split indexes，reader 就會 fail closed，不會
退回 legacy directory scanning。

## 讀取 split

多 split v2 bank 必須為 `PreGeneratedRoomBank` 指定 split：

```python
from puresound.audio.rir.bank import PreGeneratedRoomBank

bank = PreGeneratedRoomBank(
    "/data/rir-bank",
    split="train",
)
```

訓練時不要只指向 bank root 而省略 split。

## Quality control

QC 有四種狀態：

| 狀態 | 意義 |
|---|---|
| `pass` | 可量測且在限制內 |
| `fail` | 違反 hard invariant，必須 quarantine |
| `not_evaluable` | 訊號或 metadata 不足以可靠估計 |
| `not_applicable` | 此檢查不適用於該 item |

QC policy 與其 canonical hash 會一起保存，因此 threshold 變更會形成不同證據。

結構檢查包含：

- 檔案存在且 SHA-256 正確；
- item identity 與 canonical scene hash；
- 解碼後 sample rate、frame count 與 channel count；
- 音訊皆為 finite、能量非零且符合 level policy；
- channel map 完整且沒有重複；
- renderer 與 provenance 欄位。

物理檢查依 item 適用範圍涵蓋 causality、arrival timing、decay、clarity、
spectrum、echo density 與 spatial behavior。不同步的 transfer paths 不能被
當成 microphone array。

執行 QC：

```bash
python egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank.py --help
```

失敗 item 應進 quarantine，不得出現在 release index。產生的 audit 與 QC
report 預設是本機產物，除非 release 流程明確要求保留。

## Release variant 與 recipe

Release recipe 定義訓練或評估實際使用的混合方式。Measured 與 synthetic 的
來源必須保留，不要合併成無標示的 WAV 目錄。

常見 variant：

- calibrated synthetic RIR；
- peak-normalized synthetic RIR；
- native measured RIR；
- 明確定義比例的 measured/synthetic mixture。

Recipe 應固定：

- bank 與 manifest identity；
- 允許的 split 與 variant；
- renderer profiles；
- sampling weights；
- level policy；
- QC policy 與必要狀態。

Measured data 只有在授權允許重新散布或衍生訓練用途時才能加入 release。

## 實測 RIR

Measured corpus 透過[實作指南](rir_realism_algorithm.zh-TW.md#實測-rir-匯入)
中的 ingest 流程進入 bank。量測鏈視為 renderer profile；alignment policy、
環境假設、原始 provenance 與拒絕原因都會附在 item 上。

## Production evidence

Renderer 通過技術檢查，不等於已獲 production approval。正式升級仍需要 unit
test 以外的證據，例如：

- 符合人數要求的 controlled listening；
- room-disjoint downstream training 與 evaluation；
- 可追溯 bank、recipe、結果與 code revision 的 evidence bundle。

Production certificate 會區分已實作檢查與 empirical evidence。不得用跳過證據
或重新命名 development profile 的方式通過 production-required 設定。

## 訓練設定

訓練設定必須明確引用已 release 的 bank 與 split。各 task 的欄位名稱可能不同，
但最後必須解析到 manifest-backed release recipe，而不是任意 WAV 目錄。

即使 train、validation、test 都放在同一 bank root，也要分開指定 recipe。

## Release checklist

發佈前確認：

1. 使用固定 recipe 與乾淨 code revision 重新產生或續跑。
2. 驗證 manifest 與所有保留資產的 hash。
3. 執行結構檢查與適用的物理 QC。
4. 確認 room 與 parent-room 的 split 不重疊。
5. Release index 只包含通過項目。
6. 稽核 measured sources 的 license 與 provenance。
7. Empirical evidence 與 implementation tests 分開記錄。
8. 從最終 bundle 實際載入每個公開 split。
