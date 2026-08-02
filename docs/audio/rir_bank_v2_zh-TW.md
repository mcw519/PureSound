# M6 Production RIR Bank v2 契約

狀態：M6.1–M6.6 implementation 已完成；M6.6 production promotion 因缺少
measured／listening／downstream／renderer approval evidence 而維持 BLOCKED
最後更新：2026-08-01

這份文件說明 PureSound 如何把已驗證的 RIR renderer 包裝成可重現、可稽核、
不會發生 room leakage 的訓練資料 bank。M6.1 解決的是「一個 bank 必須記住
哪些事，才有資格被比較或發布」，不是宣稱目前的 renderer 已通過 production
evidence。

## 1. 為什麼不能只保存 WAV

舊版 `PreGeneratedRoomBank` 使用同 stem 的 WAV／JSON pairs，適合快速載入，
但單靠目錄無法回答：

- 這個 item 屬於 train、validation 還是 test？
- 同一實體房間或同一 synthetic parent room 是否洩漏到另一個 split？
- 使用哪個 scene schema、renderer backend、設定與 code revision？
- WAV、metadata 或 scene 是否在產生後被修改？
- 使用 development renderer 的 bank 是否被誤標成 production？

M6.1 保留既有 WAV／JSON layout，另外在 bank root 加入
`rir_bank_manifest.json`。舊 reader 仍可使用檔案；release／evaluation 工具則以
manifest 作為唯一的 provenance contract。

## 2. 三層 identity

每個 item 同時保留：

- `item_id`：一個 WAV／JSON pair 的唯一名稱；
- `room_id`：同一組 room geometry/material realization；
- `acoustic_space_id`：split 的最小原子單位。

`acoustic_space_id` 比 `room_id` 更重要。真實量測時，它應表示 corpus namespace
加實體房間；synthetic bank 則表示 parent room realization。相同
`acoustic_space_id` 的位置、source/receiver pose、normalized variant 或 residual
variant 必須全部留在同一 split。

因此 audit 同時檢查：

\[
|\{s_i:a_i=a\}|=1,\qquad |\{s_i:r_i=r\}|=1,
\]

其中 \(a_i\) 是 acoustic-space identity，\(r_i\) 是 room identity，\(s_i\)
是 split。

## 3. Deterministic room-disjoint split

M6.1 凍結政策：

```text
puresound.m6_split.sha256_acoustic_space.v1
```

對 seed、policy id 與 `acoustic_space_id` 做 SHA-256，取前 64 bits 映到
\([0,1)\)，再依 manifest 的 train／validation／test fractions 決定 split。
同一 identity 在不同機器、worker 數與檔案列舉順序下都會得到相同結果。

這不是逐 item random split。若同房間有 1,000 個位置，1,000 個 item 仍會一起
進入同一 split。

## 4. Content-addressed provenance

每份 manifest 記錄兩層 hash：

1. 每個 RIR WAV、metadata JSON 與 canonical scene JSON 的 SHA-256；
2. 排除 `manifest_sha256` 本身後，對完整 manifest 做 sorted-key、無 NaN 的
   canonical JSON SHA-256。

因此下列修改都會被偵測：

- 替換或截斷 WAV；
- 修改 metadata 或 scene；
- 改 split、renderer profile、item id 或 asset hash 後沒有重新簽 manifest。

Manifest 也保存 WAV 的 sample rate、channel count 與 frame count。Audit 會用
實際 audio header 交叉檢查，不能只靠副檔名或 metadata 聲明。

## 5. Generator 與 renderer provenance

Bank-level generator provenance 包含：

- generator id 與版本；
- code revision；
- canonical generation-config SHA-256；
- generation seed。

Renderer profile 包含：

- renderer id／version；
- low／high backend；
- scene schema version；
- renderer-config SHA-256；
- calibration／residual／approval evidence hashes；
- evidence tier。

Evidence tier 只有：

```text
development
empirical_candidate
production_approved
```

M4/M5 implementation validators 雖然已 PASS，但 M5 empirical exit 仍 OPEN，
所以目前 M6.1 fixture 必須標為 `development`。

## 6. Release status 必須 fail closed

Release status 分為 `draft`、`candidate`、`production`。若要宣稱 production：

- 所有 renderer profiles 必須是 `production_approved`；
- 每個 profile 必須帶 approval-report hash；
- 每個 item 的 M6.3 QC 必須 PASS 並帶 QC-report hash；
- generator code revision 不得是 `dirty` 或 `unknown`；
- split、assets、audio、metadata 與 manifest audit 必須全部通過。

目前 fixture 的 `release_status=draft`、`qc.status=pending`，因此
`ready_for_m6_bank_generation=true` 與 `ready_for_production=false` 可以同時成立。
前者只表示契約和資產一致，後者才是發布判斷。

## 7. M6.1 正式結果

`validate_m6_bank_contract.py` 建立一個明確標記為 non-evidence 的三 split fixture，
並執行 12 個 gates：

- strict schema round-trip 與 deterministic manifest digest；
- train／validation／test 皆存在；
- deterministic acoustic-space assignment；
- acoustic-space／room split disjoint；
- asset、manifest、scene 與 audio-header integrity；
- unsafe relative path rejection；
- development renderer 的 false production claim rejection；
- 舊 `PreGeneratedRoomBank` layout 相容。

正式結果為 **M6.1 implementation PASS**；這不代表 production bank 已完成。

## 8. M6.2：接到真正的 generator

`generate_hybrid_rir.py --emit-m6-manifest` 現在會先 materialize 完整 task plan，
再開始 serial 或 parallel rendering。每個 item 都由 base seed 與 `sample_id`
導出獨立 seed，所以 worker scheduling 不會改變聲學 realization。

M6 resume 不再只檢查「WAV 和可解析 JSON 是否存在」，還會檢查：

- generation-config、renderer profile、split 與 task identity；
- code revision 與關鍵 runtime package versions；
- canonical scene hash；
- WAV SHA-256；
- sample rate、channel count 與 frame count。

全部一致才會 skip。若一個 WAV 被修改，只重建該 item；若 duration、renderer、
code revision 或其他 content-generating config 改變，舊 items 不會被混入新
bank。

生成完成後自動輸出：

- `rir_bank_manifest.json`；
- `indexes/train.jsonl`、`validation.jsonl`、`test.jsonl`；
- `rir_bank_generation_audit.json`。

`PreGeneratedRoomBank` 遇到 M6 multi-split root 時會強制要求明確
`split="train"`、`"validation"` 或 `"test"`；未指定 split 會 fail closed，
避免訓練程式因沿用舊目錄掃描而把 validation/test RIR 混入 train。若 manifest
缺席但目錄仍有 M6 metadata 或三份 split indexes，reader 會拒絕退回 legacy；只有
真正 legacy layout 才保留原行為。

正式 non-evidence fixture 使用實際 Pyroomacoustics high backend，同時固定 NumPy
與 libroom RNG，比較 serial、two-worker parallel、獨立 fresh rerun、tampered
resume 與 changed-revision/config resume，共 17/17 gates PASS。竄改一個 WAV
後只重建 `1/6` 並恢復原 hash；revision/config 改變時不會生成混血 bank。

這個功能是 opt-in；未傳 `--emit-m6-manifest` 時的既有生成流程與
Pyroomacoustics default 不變。

## 9. M6.3：逐 item 的物理 QC 與隔離

M6.3 不把「某個 metric 算不出來」一律當成零，也不把五個不同聲源的 channels
誤認為五支同步麥克風。每個 check 有四種狀態：

- `pass`：量得到，而且落在版本化 policy 的物理界線內；
- `fail`：違反 hard invariant，item 必須進 quarantine；
- `not_evaluable`：資料長度、noise floor 或 metadata 不足以可靠估計；
- `not_applicable`：metric 的物理前提不適用。

預設 policy id 是 `puresound.rir_bank_qc.physical.v1`。Policy 本身也做 canonical
JSON SHA-256，因此調整 threshold 會形成不同的 QC evidence，不會悄悄覆蓋舊結果。

### 9.1 Structural 與 provenance checks

每個 item 先檢查：

- WAV／metadata 存在，且 SHA-256 與 manifest 相同；
- metadata identity、canonical scene hash 與 manifest 相同；
- WAV 可解碼，sample rate、frame count、channel count 與 manifest 相同；
- 所有 samples finite、總能量非零、peak 符合 level policy；calibrated float 沒有
  人為的 unit-peak 上限，只有 peak-normalized variant 要求 peak 不超過 1；
- `channel_map` 對每個 audio channel 恰有一個唯一 index。

這一層失敗代表 asset 或契約本身不可信，不應繼續把聲學數字當成證據。

### 9.2 因果與 direct arrival

對距離 (d)、聲速 (c)、取樣率 (f_s)，幾何 direct arrival 是

\[
n_d = \frac{d}{c} f_s.
\]

在 `floor(n_d)` 之前若出現相對於 channel peak 過大的訊號，就標記
`prearrival_energy`；第一個可靠 onset 與 (n_d) 的誤差也必須在 policy 的
tolerance 內。onset 搜尋只在幾何 arrival 後的 local window 內進行，避免把
有限 modal／voxel 展開造成的極小數值殘留誤認成 direct；生成器本身也會在
crossover 前清除 `n < floor(n_d)` 的低頻樣本。DRR、C50/C80 與 decay analysis
使用幾何 arrival 附近的 local direct peak，不使用整段 RIR 的最大 peak，避免把
晚期 reflection 誤當 direct。

### 9.3 Early、decay、頻譜與 diffusion

每個 channel 記錄：

- peak、總能量、50 ms 後 tail-energy fraction；
- DRR、C50、C80；
- noise-aware EDT、T20、T30 與 fit \(R^2\)；
- broadband spectral tilt 與 valid octave-band metrics；當長度足夠時，octave
  T20／fit coverage 會參與 hard admission gate；
- Abel–Huang normalized echo density、mixing time 與 late median density。

不可靠的單一 decay fit 先標成 `not_evaluable`。只有當 RIR 長度已足夠、但整個
item 仍達不到最低可靠 T20 channel coverage 時，`decay_fit_coverage` 才是 hard
failure。這把「真的沒有合理 decay evidence」和「短檔案不能估」分開。

目前 generator 的 5 channels 是五個 source-to-one-receiver transfer paths，並非
one-source-to-five-synchronized-receivers。因此 IACC／array coherence 明確寫成
`not_applicable`；只有 metadata 表示同步 receiver pair/array 時，未來的 spatial
QC 才可執行這些指標。

### 9.4 Candidate indexes 與 quarantine

執行：

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/phases/m6_bank/scripts/run_m6_item_qc.py \
  --bank exp/my_m6_bank
```

會產生：

- `qc/items/<item_id>.json`：完整 per-item metrics、checks、failure reasons；
- `qc/candidate_indexes/{train,validation,test}.jsonl`：只含 QC PASS items；
- `qc/quarantine/index.jsonl`：QC FAIL items、report reference 與原因；
- `rir_bank_qc_summary.json`：policy、indexes、counts、release decision 與 hashes；
- 更新後的 `rir_bank_manifest.json`：每個 item 的 QC status、report path/hash。

QC 不刪除或修改原始 WAV／metadata。`PreGeneratedRoomBank` 預設永遠排除
`qc.status=fail`；candidate/production manifest 更只接受 `pass`。只有診斷工具明確
傳入 `include_failed_qc=True` 才能讀取隔離項目。若任何 candidate split 變空，
release 會保持 `draft`，而不是建立缺少 test 或 validation 的假 candidate。

正式 validator 使用真正的 M6.2 generator output。正常組 `6/6` 通過；四個負
控制分別注入 silent RIR、pre-arrival impulse、direct-only sparse late field 與
晚 3 ms 的 direct arrival，四者都只進 quarantine。共 13/13 gates PASS，且竄改 QC report 會使 content
audit FAIL。這仍是 synthetic implementation evidence，不是 measured acoustic
或 production evidence。

## 10. M6.4：凍結 distribution、variant 與 recipe

M6.4 的 release root 使用 `puresound.rir_bank_release.v1`，把三件事分開：

1. **variant**：一組訊號語意一致的 bank，例如 calibrated 或
   peak-normalized；
2. **distribution**：該 variant 的 scene 與 acoustic metrics 快照；
3. **recipe**：訓練時可選的 variant、origin 權重與三個 split indexes。

目前可發布的兩個 synthetic variants 是：

- `synthetic_calibrated`：M6.3 QC candidate 的 identity copy；
- `synthetic_peak_normalized`：每個 item 的所有 channels 只乘同一個 gain，令
  global peak 到 `0.98`。

「所有 channels 共用一個 gain」很重要。若逐 channel 正規化，channel 間的能量
關係會被破壞；共同 gain 則只改變 level-dependent peak，理論上保持

\[
\mathrm{DRR},\ C_{50},\ C_{80},\ T_{20}
\]

不變。Formal validator 實際重算上述 metrics，最大誤差皆在 `1e-4` 以內，也要求
parent/child 具有完全相同的 acoustic-space、room、scene 與 split identity。

Distribution 對 room volume、RT60、distance、peak、tail energy、DRR、C50/C80、
T20、spectral tilt、mixing time 與 late echo density 保存完整 rows、quantiles 與
SHA-256。這使 bank-level 比較可重算，而不是只保存一張圖或人工摘要。

Ready recipe 的 train／validation／test JSONL 都有 item count 與 file hash。
`PreGeneratedReleaseBank` 可直接消費：

```python
from puresound.audio.rir_bank import PreGeneratedReleaseBank

bank = PreGeneratedReleaseBank(
    "exp/my_m6_release",
    recipe_id="synthetic_calibrated",
    split="train",
)
scene = bank.sample_scene()
rir, metadata, sample_rate = bank.select_channel(scene, "foreground")
```

Reader 初始化時會先 audit 整個 release；它先依 recipe 的 `origin_weights` 抽
synthetic／real，再抽該 origin 的 variant，而不是依目錄中檔案數量形成隱性
權重。Metadata 會帶 `release_sha256`、recipe、variant 與 origin。

目前 `real_native` 與 `mixed_calibrated_real` recipe 是 `blocked`，因為沒有
QC-passed measured M6 variant。Blocked recipe 沒有可讀 index 且必須附原因；工具
不會用 synthetic 資料冒充 real。M6.4 正式結果為 **12/12 implementation
PASS**，但 measured／mixed recipe 仍是 OPEN。

### 10.1 Float WAV 為何需要 header canonicalization

libsndfile 寫 IEEE-float WAV 時通常加入 `PEAK` chunk，其中含寫檔當下的 Unix
timestamp。它不影響任何 audio sample，卻會讓相同 RIR 在不同秒寫出不同
SHA-256。M6.4 在寫檔後只把這個 non-acoustic timestamp 歸零，不改 chunk size、
peak value 或 waveform。這是 byte reproducibility 修正，不是聲音處理。

## 11. M6.5：評估的是證據，不只是能否執行

`puresound.m6_bank_evaluation.v1` 同時彙整四類 evidence：

- bank-level acoustic distributions；
- generator throughput、skip 與 failure counts；
- controlled listening；
- room-disjoint downstream tasks。

它分成兩個 exit：

- **implementation exit**：schema、重算、hash、負控制和 fail-closed 邏輯可用；
- **empirical exit**：真的有 measured reference、真人 responses 與 trained-model
  多 seed 結果。

兩者不能互換。目前 calibrated／normalized 的 scale-invariant metrics 比較已
PASS；因 release 沒有 real variant，synthetic-to-measured 距離明確回報
`not_evaluable`，不產生假數字。

Listening report 要求 randomized、double blind、共同 loudness master gain、
hidden reference、degraded anchor、room-disjoint stimuli 與 content-addressed
assignment/responses/analysis records；`empirical` 還要求至少 20 位參與者，並由
response records 重算 participant mean 與信賴區間。Downstream report 會由 recipe
index 重算 train/test acoustic-space hashes，至少需要三個 unique seeds、凍結的
training/model recipe，並從 paired per-seed improvement 重算 95% t confidence
interval；每個 primary metric 的 lower bound 都必須改善。

正式 M6.5 validator 的 14/14 implementation gates PASS，並確認五種負控制會被
拒絕：unblinded listening、竄改 test split identity、single-seed downstream、
偽造正 CI，以及把 non-human responses 改標 empirical。現有
listening/downstream 數字標為 `contract_fixture`，明確不是人類回覆
或已訓練模型的結果。所以 **M6.5 implementation PASS，empirical／production
OPEN**。

## 12. M6.6：promotion 是 certificate，不是改一個字串

直接把 `release_status` 從 `candidate` 改成 `production` 不是發布：那會使 release
content hash 失效，也沒有證明 M6.5 evidence 或 reviewer approval 存在。M6.6
因此保留整個 M6.4 candidate 不變，另外產生
`puresound.m6_production_decision.v1` certificate，綁定：

- release SHA 與 M6.5 evaluation SHA；
- synthetic calibrated／normalized、real native、mixed 四個 ready recipes；
- 所有 item 的 QC PASS 與 pinned generator revision；
- 每個 renderer profile 的 `production_approved` tier 與 approval-report hash；
- 真人 listening 與 trained downstream 使用的實際 files；
- acoustics、ML、release owner 三個角色的 approve records。

Evidence bundle 不是只填一串 hash。每個 artifact 必須使用安全的 bundle-relative
path、實際存在，且檔案 SHA-256 與 bundle 相同。Listening assignment／responses／
analysis、downstream training recipe／checkpoints／analysis 的 hashes 還必須與
M6.5 當時評估的 hashes 完全相同。Renderer approval files 也要涵蓋所有 profiles。

Certificate 只有兩種決定：`approved` 或 `blocked`。任何 check 為 false，
`production_ready` 就必須是 false，並保存 deterministic blocker list。Candidate
仍可研究使用；production consumer 必須明確要求：

```python
bank = PreGeneratedReleaseBank(
    "exp/my_m6_release",
    recipe_id="mixed_calibrated_real",
    split="train",
    require_production=True,
)
```

此時 reader 會驗證 certificate 的 schema、decision hash、release hash、固定的
canonical gate set 與 blocker consistency，並重新執行所綁定的 release／evidence
audits 及 decision recomputation。Blocked、竄改或不存在的 certificate 都會拒絕。

Formal validator 的 15/15 implementation gates PASS，包括 unsafe evidence path、
evaluation pass-flag/hash tamper、全 true 且重算外層 hash 的稱職偽造 certificate、
直接修改 candidate status 與 production-reader 負控制。目前真實決定仍為 **BLOCKED**；缺件正是 real/mixed
variants、renderer approval、M6.5 三類 empirical evidence、原始 evidence files
與三方 sign-off。這代表 M6.6 decision engine 已完成，不代表 production evidence
已憑空出現。

Content addressing 能證明「被 review 的 bytes 沒變」，不能單獨證明 reviewer
身份。正式部署仍應讓 sign-off records 由受控 CI／權限系統產生與保管；若威脅
模型包含能任意改寫 workspace 的攻擊者，還需在部署層加入組織的數位簽章。

### 12.1 接到現有 training augmentation

現有 `AudioEffectAugmentor` 已能由 YAML 直接建立 release reader：

```yaml
pregenerated:
  used: true
  bank_type: release
  folder: egs/rir_generation/exp/rir_realism/m6/training_pilot/pyroomacoustics_release
  recipe_id: synthetic_calibrated
  split: train
  usage_role: train
  require_production: false
```

Release mode 強制要求 recipe、split 與相同的 `usage_role`，dynamic dataset 也會
和自己的 train／validation／test role 交叉檢查；因此不會把 validation/test RIR
混入 train。舊設定未提供 `recipe_id` 時仍走 `PreGeneratedRoomBank`，維持
backward compatibility。正式 production deployment 才把
`require_production` 設成 true；目前 blocked certificate 會如預期拒絕。

建議先生成兩個 matched 4,000-item pilots，而不是立刻產生十萬筆：

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot \
  --backend pyroomacoustics --num-workers 8

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot_m4 \
  --backend path-events-m4 --num-workers 8
```

兩組固定相同 scene v1、low backend、seed、room/item count 與 calibrated level，
只改 high backend。Script 依序執行 resumable M6 generation、M6.3 QC 與 M6.4
release，且不會覆寫既有 release。若要快速 smoke test，可加上
`--n-rooms 6 --rir-per-room 1 --num-workers 1`（M6 必須同時涵蓋
train/validation/test 三個 split）。訓練範例在
`egs/rir_generation/phases/m6_bank/config/m6_release_training_example.yaml`。

### 12.2 強化後 matched backend preflight（2026-08-02）

在進行完整 4,000-item pilot 前，先完成一個可稽核的 30-room preflight：每個
room 生成 2 items，兩個 backend 各有 60 items／300 channels；scene `v1/mixed`、
seed `1337`、calibrated `16 kHz / 1.6 s` 與 GPU low backend
`pytard-cupy-material` 完全相同。Pyroomacoustics 與 PathEvents-M4 的 60/60
scene／room／acoustic-space／split／seed／shape identity 全部匹配，兩邊都是
60/60 QC PASS、0 quarantine、release audit PASS；兩個實際 release reader 也各
成功載入 56 個 train items。

配對聲學摘要：

| 指標中位數 | Pyroomacoustics | PathEvents-M4 | 解讀 |
|---|---:|---:|---|
| DRR | -4.83 dB | -3.24 dB | M4 早期／直達能量較強 |
| C50 | 6.10 dB | 10.28 dB | M4 +4.18 dB |
| C80 | 8.03 dB | 15.01 dB | M4 +6.97 dB |
| T20 | 0.99 s | 0.47 s | M4 短 0.52 s |
| `|T20 − scene RT60|` | 0.327 s | 0.080 s | M4 在此樣本較貼近材料目標 |

因果到達、exact-zero 尾端、390 Hz comb regression 與 M4 高頻尾場 coverage
均通過。結論是：M4 的 PathEvent + FDN 實作確實改變了 early／late 能量與衰減
機制，不是只調 Pyroomacoustics 參數；但這仍是 **candidate preflight PASS**，
不是 realism 或 production PASS。validation/test 各只有 2 items，generation
使用 dirty code revision；Pyroomacoustics 的 air absorption 雖在 renderer 內
套用，現行 high-band metadata 沒有像 M4 一樣完整序列化 policy／coefficients。
目前也沒有 measured reference、真人聆聽或 downstream model 結果，故 Pyroomacoustics
仍保持預設，完整 4,000-item matched pilot 與後續 empirical gate 尚未完成。

完整、可重算的結論與 hash 見
[`preflight_validation_summary.json`](../../egs/rir_generation/exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/preflight_validation_summary.json)。

## 13. 如何重現

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_contract.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_reproducible_generation.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_item_qc.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_variant_release.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_evaluation.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_production_decision.py

.venv/bin/pytest -q \
  test/test_rir_bank_manifest.py \
  test/test_m6_bank_contract_validator.py \
  test/test_generate_hybrid_rir_m6.py \
  test/test_m6_reproducible_generation_validator.py \
  test/test_rir_bank_qc.py \
  test/test_m6_item_qc_validator.py \
  test/test_rir_bank_release.py \
  test/test_m6_variant_release_validator.py \
  test/test_rir_bank_evaluation.py \
  test/test_m6_bank_evaluation_validator.py \
  test/test_rir_bank_production.py \
  test/test_m6_production_decision_validator.py \
  test/test_m6_release_training_integration.py
```

主要輸出：

- `egs/rir_generation/exp/rir_realism/m6/rir_m6_bank_contract/rir_bank_manifest.json`；
- `egs/rir_generation/phases/m6_bank/reports/m6_bank_contract_report.json`；
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/reproducible/`；
- `egs/rir_generation/phases/m6_bank/reports/m6_reproducible_generation_report.json`；
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/item_qc/`；
- `egs/rir_generation/phases/m6_bank/reports/m6_item_qc_report.json`；
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/variant_release/`；
- `egs/rir_generation/phases/m6_bank/reports/m6_variant_release_report.json`；
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/evaluation/`；
- `egs/rir_generation/phases/m6_bank/reports/m6_bank_evaluation_report.json`；
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/production/`；
- `egs/rir_generation/phases/m6_bank/reports/m6_production_decision_report.json`。

下一步不是再加一層 schema，而是取得缺少的外部 evidence。只有 measured
distribution、真人 controlled listening、room-disjoint downstream、renderer
approval files 與三方 sign-off 全部可稽核時，重跑同一個 M6.6 CLI 才能輸出
`approved` certificate。
