# M6 RIR Generation — 程式審查與建議

審查日期：2026-08-02
審查範圍：M6.1–M6.6 全鏈（契約、生成、QC、release、evaluation、production decision）＋
M6 實際使用的渲染演算法鏈（scene 抽樣、hybrid crossover、PathEvents/FDN/spatial）＋
讀取端與訓練整合
審查性質：唯讀。本次審查未修改任何程式碼或設定；所有驗證產物寫在 session scratchpad。

本文件記錄審查發現與建議。每條發現標註嚴重度、`file:line` 與判定依據，並區分
**實測驗證**（本次審查實際跑出數據）與**讀碼判定**（靜態分析，未實跑）。

---

## 0. 摘要

M6 的**工程契約層品質相當高**：三層 identity、SHA-256 deterministic split、
immutable candidate + certificate 晉升模型、QC 三態（拒絕把「算不出來」偽裝成
零）、雙軌 exit（implementation vs empirical）。這些不是形式主義，程式碼確實照做，
`decide_m6_production` 也如實輸出 `blocked` 而非粉飾。

但有兩個 critical 問題會直接影響資料品質與可信度，而且**兩者都落在 validator 的
覆蓋盲區**。

| # | 嚴重度 | 發現 | 影響範圍 | 驗證方式 |
|---|---|---|---|---|
| C1 | critical | 預設後端 pyroomacoustics 不可重現（libroom RNG 未播種） | 預設 bank 全部 | 實測 |
| C2 | critical | 低頻激勵 comb：每個 item 在 390 Hz 有 18.7 dB 固定凹口 | 兩個後端全部 | 實測 |
| A1 | major | FDN 晚場在 5.66 kHz 以上是空的（16 kHz 下） | `path-events-m4` | 實測 |
| A2 | major | 晚場能量錨定在 order-truncated tail，長 RT60 位準偏低 | `path-events-m4` | 讀碼＋agent 實測 |
| A3 | major | resume 不綁 `code_revision`，可產生混血 bank ＋ 假 provenance | 全部 | 讀碼 |
| A4 | major | production certificate 可被「重算 hash 的稱職偽造」通過 | 晉升流程 | 讀碼 |
| A5 | major | manifest 缺席時靜默退回 legacy，三個 split 全混 | 讀取端 | 實測 |
| A6 | major | 訓練 YAML 可誤拿 `split: test`，無 role 交叉檢查 | 訓練整合 | 讀碼 |
| A7 | major | downstream CI lower bound 不重算；listening 為申報制 | M6.5 | 讀碼 |
| A8 | major | QC `direct_arrival_timing` gate 是死碼 | M6.3 | 讀碼 |
| A9 | major | 低頻 RT60 被壓平成 mid-band 標量 | 預設後端 | 讀碼 |
| A10 | major | 兩後端物理不對稱（directivity、obstacle） | A/B pilot | 讀碼 |
| A11 | major | FOA per-channel 縫合破壞 diffuse isotropy | M4.5 spatial | 讀碼 |
| A12 | major | `simulated_rir` dict 無界成長 | 訓練整合 | 讀碼 |
| A13 | major | release/variant/split provenance 在 dataset 層被 drop | 訓練整合 | 讀碼 |

**最重要的一句話**：M6 宣稱的「deterministic bank」在預設後端上不成立，而且每個
item 都帶著同一個合成簽名凹口。

---

## 1. 審查方法

- 讀 `RIR_REALISM_PLAN.md`（1854 行）與 `docs/audio/rir_realism_algorithm_zh-TW.md`
  （4918 行）建立設計脈絡；
- 逐行精讀 M6 六層核心模組與六支 validator，以及 `hybrid_rir.py`（3003 行）、
  `rir_path_events.py`（2245 行）等渲染模組；
- 實跑契約文件宣稱的 13 個測試檔（26 tests，全過，82 秒）；
- 實跑三次完整 M6 pipeline（6 rooms × 1 RIR）對照決定性；
- 對實際生成的 bank 做頻譜量測與 reader 行為驗證。

重現指令見 [§7](#7-重現本次驗證)。

---

## 2. 驗證通過的部分

避免只列問題而失衡，以下是本次**實測確認為真**的宣稱：

- 契約文件列出的 13 個測試檔、26 個測試全數通過（82 秒）；
- 六份凍結報告的 gate 數與計畫書逐一相符（12/12、13/13、12/12、12/12、12/12、14/14）；
- `decide_m6_production` 如實輸出 `blocked` 與 6 個 blockers，`evidence_audit` 誠實
  回報 0/9，沒有偽造證據；
- 用真實 M6 bank 驗證：未指定 split 讀取確實 fail-closed；
- QC 在 production duration（1.6 s）下 6/6 pass，三個 split 皆非空；
- **scene 抽樣本身是決定性的**（三次執行的 `scene_sha256` 完全相同）——
  非決定性只發生在渲染層；
- canonical JSON hash 鏈同源、無重複實作（`canonical_json_sha256` 單一來源，
  `sort_keys` + 固定 separators + `allow_nan=False`）；
- libsndfile `PEAK` chunk timestamp 歸零邏輯正確（不動 chunk size、peak value 與
  waveform）；
- `not_evaluable` 語意 fail-closed 正確：缺 measured reference 時
  `empirical_exit` 不可能通過，不存在被計為 pass 的路徑；
- causality 契約全鏈成立：低頻帶 clip → causal LP4 保零；高頻帶對齊後逐通道清零 →
  causal HP4 保零；PathEvents 用 one-sided Lagrange kernel 構造性滿足；FDN coupling
  在 transition 前 sample-exact（實測誤差 0.0）；
- PathEvent 幾何核正確：fold/unfold 距離互檢 1e-11、reciprocity、Cayley filter 的
  passivity 與 pole < 1 檢查皆通過；
- 舊腳本搬遷（`tools/`、`phases/`、`examples/`）無殘留舊路徑引用。

---

## 3. Critical

### C1 — 預設後端 pyroomacoustics 不可重現

**位置**：[`generate_hybrid_rir.py:858-861`](generate_hybrid_rir.py)、
`generate_m6_bank.py:46-48`（預設 `--backend pyroomacoustics`）

`generate_m6_bank.py` 的 docstring 寫「Generate one **deterministic** M6 synthetic
RIR bank」。實際不成立。

pyroomacoustics 的 ray tracing（預設開啟、20000 rays、v1 材質 scattering 全非零）
使用 libroom 內部的 process-global `std::mt19937_64`。`_generate_task` 只播種三個
RNG：

```python
task_seed = int(task["m6"]["generation_seed"])
random.seed(task_seed)
np.random.seed(task_seed)
torch.manual_seed(task_seed)
```

預設鏈（modal 低頻帶 + pra 高頻帶）**沒有任何元件消耗這三個 seed**。全 repo 沒有
任何 `pra.libroom.set_rng_seed` 呼叫。

**實測**（相同參數、相同 seed 1337、6 rooms）：

| 對照 | WAV bytes | manifest_sha256 |
|---|---|---|
| A（1 worker）vs B（2 workers） | 6/6 全部不同 | 不同 |
| A vs C（**同樣 1 worker**，前後兩次執行） | 6/6 全部不同 | 不同 |
| scene_sha256（三次） | — | **完全相同** |

關鍵在第二列：**這不只是 worker 排程造成的差異，而是每次執行都不一樣。**
差異幅度不是浮點誤差——相對 L2 差 0.17–1.26，能量幾乎全在 300 Hz 以上（高頻帶），
與 pra ray tracing 的來源一致。

**後果**：

1. `--resume` 的 help 文字「the high-band acoustic realization is task-seeded when
   M6 emission is on」（`generate_hybrid_rir.py:100-107`）對預設後端是不實陳述；
   resume 重建損壞項目時無法還原原始 bytes，會產生高頻帶不一致的混血 bank。
2. manifest 是事後對實際輸出 hash，所以永遠自洽——這是 **silent 非決定性**，
   audit 抓不到。
3. M6.2 validator 的 13 個 gate（含「serial 與 parallel manifest hash 相同」）使用
   `--low-backend analytic --high-backend path-events-m3`
   （`phases/m6_bank/scripts/validate_m6_reproducible_generation.py:61-64`），
   兩者天生決定性——等於在唯一不會失敗的組合上驗證了這個性質。計畫書據此宣稱的
   「identical serial and two-worker parallel hashes」是以 fixture 特例代替通用性質。

`path-events-m4` 後端用 `blake2b(fdn_seed, scene_id, source_index)` 派生，不受影響。

**建議**：在 `_generate_task` 加 `pra.libroom.set_rng_seed(task_seed)`；修正
`--resume` help 文字；用**預設後端**重跑 M6.2 的 serial/parallel gate，並新增一個
「刪除輸出、同參數 fresh rerun、manifest_sha256 相等」的 gate——後者才是
"reproducible generation" 的直接定義，目前完全沒有測。

---

### C2 — 低頻激勵 comb：每個 item 都有 390 Hz 固定凹口

**位置**：[`puresound/third_party/pytARD/common/impulse.py:181-183`](../../puresound/third_party/pytARD/common/impulse.py)、
[`puresound/audio/hybrid_rir.py:325-330`](../../puresound/audio/hybrid_rir.py)

低頻帶激勵用 vendored pytARD 的 `Unit` 脈衝，它是 **bipolar** 的：

```python
self.filter_coeffs = firwin(filter_order, (cutoff_frequency / 2) * 0.95, fs=sim_param.Fs)
self.impulse[0: len(self.filter_coeffs)] = self.filter_coeffs
self.impulse[len(self.filter_coeffs): 2 * len(self.filter_coeffs)] = -self.filter_coeffs
```

傳遞函數因此含 `(1 − z^−L)`（L = filter_order = 41），零點固定落在
`k × low_fs / L`。`hybrid_rir.py:325-330` 只補償了 firwin 內部把 cutoff 砍半的行為，
**comb 從未被 deconvolve**，mic 訊號直接當成 RIR。

**實測**（本次生成的 M6 預設 bank，6 items × 5 channels 平均頻譜）：

| 頻率 | 凹口深度 |
|---|---|
| 390 Hz | **18.7 dB** |
| 780 Hz | 4.8 dB |
| 300 / 600 / 900 Hz（對照） | −3.4 / −1.5 / −1.5 dB（即無凹口） |

780 Hz 較淺是因為高頻帶洩漏部分回填；390 Hz 遠低於 crossover，HP4 在 0.39·fc
只有 −33 dB，填不回來。

**為什麼這件事重要**：凹口頻率只取決於 `low_sample_rate / filter_order`，所以
**全 bank 每一個 item 都在同一個位置**。這正是一個模型可以用來區分合成與真實資料的
簽名，而這條專案線的核心目標就是消除合成與真實的差距。它也直接汙染低頻 tilt 統計，
而 tilt 方向錯誤正是已量化的既有域差之一。

QC 攔不到：spectral gate 只查 octave 級 tilt（±18 dB/oct，`rir_bank_qc.py:63-66`），
單一窄凹口不影響 octave 中位數。

附帶影響：780.5 Hz 落在 crossover RMS match band `[700, 1300]` 內 → `low_rms` 被
低估、橋接 gain 系統性偏高（幅度小）。

**建議**：改用單極脈衝加顯式 DC 處理，或對激勵做 deconvolve；新增一個
「激勵頻譜平坦度」測試，避免這類問題再次無聲通過。

---

## 4. Major

### A. 決定性與 provenance

#### A3 — resume 不綁 `code_revision`

**位置**：[`generate_hybrid_rir.py:537`](generate_hybrid_rir.py)（`task["m6"]` 內容）、
`generate_hybrid_rir.py:800-803`（resume 比對）

`task["m6"]` 含 `bank_id`、`acoustic_space_id`、`split`、`scene_sha256`、
`generation_config_sha256`、`renderer_profile_id`、`generation_seed`、audio shape——
**唯獨沒有 `code_revision`**，而 `_task_is_complete` 只比對這些 key：

```python
recorded = metadata.get("m6")
if not isinstance(recorded, dict) or any(
    recorded.get(key) != value for key, value in expected.items()
):
    return False
```

manifest 卻寫入本次 run 的 revision。

**失效情境**：在 rev A 生成 3000 項後修改渲染程式碼（行為變了、config dataclass
不變），用 rev B `--resume` 補完剩餘 1000 項 → 舊項目全被判 complete 保留，最終
manifest 宣稱整個 bank 由 rev B 產生。混血 bank ＋ 假 provenance，15 個 audit
checks 全部照過。

**建議**：把 `code_revision` 納入 `task["m6"]`；或在 resume 時比對並明確拒絕
（至少警告）。

#### 環境 provenance 缺席（minor，但與 A3 同源）

`BankGeneratorProvenance` 只有六欄（generator id/version、code_revision、
config_sha256、task_plan_sha256、seed），`renderer_version` 硬編 `"M6.2"`，
`pyproject.toml` 對 pyroomacoustics 無版本 pin。numpy（NEP 19 不凍結 `Generator`
stream）、pra、torch、libsndfile 任一升級都可能改變輸出 bytes，而 manifest 記錄的
重現條件毫無變化。建議增加關鍵套件版本指紋。

---

### B. 訊號品質與物理

#### A1 — FDN 晚場在 5.66 kHz 以上是空的（`path-events-m4`）

**位置**：[`rir_metrics.py:786-790`](../../puresound/audio/rir_metrics.py)、
[`multiband_fdn.py:389`](../../puresound/audio/multiband_fdn.py)、
`hybrid_rir.py:1441`

`valid_octave_centers` 的條件是 `center * sqrt(2) < nyquist * 0.99`，16 kHz 下
8000 Hz octave 的上緣 11314 Hz > 7920 Hz 被剔除，FDN 只剩 {500, 1000, 2000, 4000}
四個 band，而輸出就是這四個 bandpass 的和：

```python
combined = np.sum(np.vstack(list(band_rirs.values())), axis=0)
```

**實測**（四個 4 階 Butterworth octave bandpass 合成響應，相對 1 kHz）：

| 頻率 | 5 kHz | 6 kHz | 7 kHz | 7.5 kHz |
|---|---|---|---|---|
| 晚場能量 | −0.3 dB | **−11.2 dB** | **−43.0 dB** | **−68.7 dB** |

而 coherent 早場是常數增益 taps，一路到 Nyquist 都有能量。結果是每條 RIR 在
`direct + 約 16 ms` 的 transition 處**高頻突然塌陷**。

更糟的是 energy-preserving gain 以「全頻寬 tail 能量」為 target，會把缺失頻帶的
能量硬塞回 353 Hz–5.66 kHz，額外造成 tilt 偏差。

附帶實測：octave 重組在各 crossover 有 **+2.7 dB 駝峰**（707 / 1414 / 2828 Hz），
tail 頻譜疊了週期性 ripple。`analyze_fdn_coloration` 只看帶內平坦度，看不到跨帶
這一層。

**建議**：給最高 band 加 shelf 延伸到 Nyquist，或改用 half-octave / highpass 尾帶；
新增「coupled tail 在 5.7–8 kHz 對早場的能量比」測試。

#### A2 — 晚場能量錨定在 order-truncated tail

**位置**：[`rir_late_coupling.py:118`](../../puresound/audio/rir_late_coupling.py)

```python
target = float(np.dot(original_tail, original_tail))
```

`original` 是 `max_order=12` 的 image render，有有限時間跨度（實測某
4.2×5.1×2.7 m 房間最遲事件在 184.9 ms，render 窗卻是 1600 ms），之後 target 積分
為零。FDN 的**斜率是對的**，但總能量被這個截斷上限鎖死，整條晚場往下平移。

數值實驗：RT60 ≈ 1.67 s（α=0.06）時晚場比物理外插低約 **3.5 dB**；RT60 = 0.33 s
（α=0.3）時只差 0.9 dB——**偏差隨 RT60 增長**。

方向與已量化的「合成 bank DRR 誇大 4 dB」域差一致。

**建議**：能量 target 改用 Sabine / 解析 tail 外插，而非截斷的 reference。

#### A9 — 低頻 RT60 被壓平成 mid-band 標量

**位置**：`hybrid_rir.py:387-399, 2175-2190`、`rir_scene.py:699-703`

M6 預設 `material_modal_damping=False, apply_rt60_decay=True`：整個 20–1000 Hz 帶
套用單一 `scene.rt60`（材質 Sabine 在 500/1000 Hz 的中位數）指數包絡。

- brickwork α(125)=0.01 vs α(1k)=0.03 → 低頻 RT60 實應約 3× mid-band（衰減被砍太快）
- plasterboard α(125)=0.15 > α(1k)=0.04 → 方向相反

無論哪個方向，低頻 RT60 對頻率恆平坦，且與高頻帶（pra per-band 材質）在 crossover
兩側衰減律不連續。metadata 有誠實記 `global_rt60_envelope_applied: true`，修正機制
（`pytard-material` per-mode damping）也存在，只是不是預設。

#### A10 — 兩個後端物理不對稱

計畫的下一步是 1000 rooms × 4 RIR 的 matched A/B pilot，「兩組固定相同 scene、
seed、level，只改 high backend」。但兩者的差異目前不只是預期中的物理模型差異：

| 面向 | `pyroomacoustics`（預設） | `path-events-m4`（候選） |
|---|---|---|
| Source directivity | scene 宣告 `speech_cardioid` 與隨機朝向，**實際渲染 omni**（`hybrid_rir.py:1310` 只做 `room.add_source(pos)`） | 真的渲染 cardioid |
| Obstacle | 整條 RIR（含殘響尾）乘上衰減 → **DRR 不變**（物理上應下降） | per-path visibility（正確） |
| 高頻晚場 | 完整 | 5.66 kHz 以上空洞（A1） |
| 晚場總能量 | — | 長 RT60 偏低（A2） |
| 距離慣例 | `1/(4πd)` | `1/r`（差約 22 dB，由 calibrated 層吸收；**兩種 bank 未經校準不可直接混用**） |
| 390 Hz comb | 有（共用低頻帶） | 有（共用低頻帶） |

也就是說，這場 A/B 目前量到的會是這些實作差異，而不是「coherent PathEvents + FDN
是否比 ISM 更接近真實」。**建議在跑 pilot 之前先處理 A1、A2 與 directivity/obstacle
的不對稱**，否則對照被汙染。

附帶：`obstacle_effects` metadata 對 `path-events-m4` 變體記錄了從未套用的模型
（`high_frequency_post_occlusion_scatter` 含逐事件 attenuation_db），m4 bank 的 JSON
消費者會讀到不存在的衰減事件。

#### A11 — FOA per-channel 縫合破壞 diffuse isotropy

**位置**：`spatial_late_field.py:418-445`、`rir_late_coupling.py:123-127`、
`spatial_rir.py:247-254`

`couple_receiver_array_early_late` 把 W/Y/Z/X 當四個獨立 receiver 逐一解
`energy_preserving_diffuse_gain`。晚場本是同一組 plane waves 的投影（isotropic 時
`E[Y²] ≈ E[W²]/3`），per-channel 縫合後 Y/Z/X 的 diffuse 位準改由「幾何 tail 的
方向能量」決定。

更麻煩的是該函式在 `target <= tiny` 時 `gain = 0.0`：coherent tail 近乎為零的
channel，其 diffuse 成分會被**整支歸零**——對稱場景的 Z 分量直接消失，晚場失去
垂直分量，diffuseness/IACC 統計變成幾何 artifact；FOA 晚場與 receiver-array 晚場也
不再是同一個場的兩個視圖。

#### 其他訊號面 design notes

- **Cayley boundary filter 在 M6 路徑上是死路徑**：
  `PathEventHighFrequencyBackend.simulate` 與 `render_path_events_ambisonic` 呼叫
  `render_path_events` 時都不傳 `surface_admittance_models`，`_gain_spectrum` 得到
  常數實 gain → 走 `constant_real_value()` 分支。M3.2 花大量篇幅驗證的 passive
  causal filter，**在實際 bank 生成裡沒有參與**。
- **材質頻變在早場被折疊成 1 kHz 單點**（`absorption.at(reference)`），吸收頻譜只
  影響 FDN 的 per-octave RT60。早／晚場看到兩套不同的材質視圖。
- **整條高頻鏈無空氣吸收**：4–8 kHz、長距離與長 tail 會偏亮。與 A1 方向相反但
  **無法互抵**（一個是頻帶消失、一個是帶內偏亮）。
- **scattering 只在一階折損**：僅 `len(surface_ids) == 1` 的 path 被拆
  `(1−s) + s`，≥2 階 path 等效 s=0 → diffuse 比例低估、多次反射 specular 偏亮。
- **RT60 長尾在 1.6 s 硬切無 fade**：classroom 類材質組合有明顯機率
  RT60(1k) > 2 s，尾端在 −30~−40 dB 處出現階梯，bank RT60 分佈右截尾。

---

### C. 證據鏈與 fail-closed

#### A4 — production certificate 可被稱職偽造

**位置**：[`rir_bank_production.py:423-432`](../../puresound/audio/rir_bank_production.py)

`validate_m6_production_certificate` 只做內部一致性檢查：

```python
valid = bool(
    parsed.get("schema_version") == M6_PRODUCTION_DECISION_SCHEMA_VERSION
    and claimed_hash == canonical_json_sha256(payload)      # 攻擊者重算即過
    and parsed.get("release_sha256") == release.release_sha256
    and checks_are_boolean                                   # key set 完全不驗
    and parsed.get("production_ready") is computed_approved
    and parsed.get("decision") == ("approved" if computed_approved else "blocked")
    and isinstance(blockers, list)
    and blockers == [name for name, passed in checks.items() if not passed]
)
```

它**不**重跑 `build_m6_production_decision`、**不**驗 checks 的 key set、**不**驗
`release_status == "candidate"`、**不**驗 evaluation 檔案存在。任何能寫檔的人用公開
的 `canonical_json_sha256` 重算一次即可鑄造 approved certificate，而
`PreGeneratedReleaseBank(require_production=True)` 會直接放行。

M6.6 validator 的 `forged_approved_certificate_is_rejected` gate 只測「翻 flag 但不
重算 hash」的懶惰偽造。同樣問題也在 evaluation 層（flip `empirical_exit.passed` 後
重算 `evaluation_sha256` 即可通過）。

**公允說明**：`docs/audio/rir_bank_v2_zh-TW.md` §12 末段自己講明了「Content
addressing 能證明被 review 的 bytes 沒變，不能單獨證明 reviewer 身份」——這個認識
是對的。問題在於 **gate 名稱與計畫書的「14/14 拒絕偽造」措辭比實際保證強**。

**建議**（在補簽章之前的零成本下限）：
1. 要求 checks key set 完全等於 canonical 的 12 個名稱；
2. 重跑 `audit_m6_variant_release` 並要求 `release_status == "candidate"`；
3. 驗 evaluation 檔案共存且 self-hash 等於 certificate 的 `evaluation_sha256`；
4. 補 negative control：「全 true checks + 重算 hash」的偽造（**必須先修再補測**）。

#### A7 — downstream CI 不重算、listening 為申報制

**位置**：[`rir_bank_evaluation.py:415-417`](../../puresound/audio/rir_bank_evaluation.py)（downstream）、
`rir_bank_evaluation.py:273-301`（listening）

per-seed 資料就在 report 裡，程式卻只重算 mean，CI 只檢查存在且為正：

```python
all_lower_bounds_positive = bool(
    all_lower_bounds_positive and ci_low is not None and ci_low > 0.0
)
```

填 `[+10, −9.99, +0.02]`、mean 填對（過 `isclose`）、`confidence_interval_low` 填
`0.001` 就通過，而真實 3-seed t-interval 下界深度為負。「CI lower bound 全升」的
統計判準**在本層沒有實作**。這是可以直接修的（用 `*_by_seed` 重算 t-interval 並與
申報值 `isclose` 比對），不像 A4 屬於信任邊界問題。

listening contract 同理：十項協議（randomized、double_blind、hidden_reference、
degraded_anchor…）全是「填 true 就過」的自我宣告布林；名為
`noninferiority_is_recomputed` 的檢查實際只是申報值的算術關係；
`responses_sha256` 只做 64-hex 格式檢查，內容從不解析。M6.5 validator 自產的
contract fixture 內有一個守門欄位 `explicitly_not_human_responses: True`，
**但沒有任何一行程式碼讀它**——把 `evidence_tier` 改成 `"empirical"` 即成「人聽
證據」。

#### A8 — QC `direct_arrival_timing` gate 是死碼

**位置**：[`rir_bank_qc.py:490-531`](../../puresound/audio/rir_bank_qc.py)、
policy `rir_bank_qc.py:52-54`

搜尋窗上界與允許誤差同為 **1.0 ms**（`direct_search_after_expected_ms = 1.0`、
`maximum_arrival_error_ms = 1.0`），而 `direct_index` 被夾在搜尋窗內：

```python
direct_index = first_physical + int(np.argmax(magnitude[first_physical:search_end]))
...
if arrival_error_ms is None or arrival_error_ms > policy.maximum_arrival_error_ms:
    channel_failures.append("direct_arrival_timing")
```

`arrival_error_ms` 數學上恆 ≤ 1.0 ms，`> maximum_arrival_error_ms` 分支**不可達**。
唯一能 fail 的路徑是 onset 為 None（窗內全部低於 `max(1e-12, peak·1e-8)`）。

**失效情境**：renderer regression 讓 direct 晚到 2–5 ms（例如 pra 40-sample 延遲
補償失效，這正是已知 gotcha），只要窗內有任何 −160 dB 以上的殘渣就照樣 pass。
負控制裡也**沒有 late-arrival 這一項**，所以死碼不會被發現。

#### QC 其他發現

- **`item_id` 未做路徑安全驗證即拼進報告路徑**（`rir_bank_qc.py:839` 先寫檔、
  `:843` 才 `relative_to` 拋錯）：對第三方 bank 跑 QC 時，
  `item_id="../../../..."` 可在 bank root 外寫檔（內容受限為 QC JSON）。
  release 端 `_materialize_peak_normalized_bank` 有同型問題。
- **lineage 的 `parent_rir_sha256` 記錄了但從未被驗證**：
  `audit_m6_variant_release` 的 lineage 檢查只比對 identity tuple 集合，沒有比對
  parent hash，也沒驗證 child 音訊 == parent × 標量。換掉 peak_normalized 的 WAV
  再自洽重算所有 hash，audit 全綠。
- **late echo density 在未截斷的尾巴上取「最後 25%」**：若有效尾巴 < 0.75×duration
  （乾房間、短 RT60），最後 25% 是精確零 → density 0 → 誤殺。本次 1.6 s smoke 未
  觸發（實測 last-25% max|x| = 1.1e-3，非零），但 8 kHz/0.2 s 的 validator fixture
  完全測不到這個 regime。
- **octave bands 有算沒 gate**：每 channel 算 4 個 band 的完整指標，`checks` 中沒有
  任何 gate 讀它。計畫文字把 octave bands 列為 QC 項目，實為 informational。
- **`maximum_peak_abs = 1.0` 對 calibrated 語意不成立**：float WAV 沒有 1.0 物理
  上限，近場物理校準 RIR 的 peak 可 > 1.0。

---

### D. 訓練整合與讀取端

#### A5 — manifest 缺席時靜默退回 legacy，三個 split 全混

**位置**：[`rir_bank.py:80`](../../puresound/audio/rir_bank.py)（`manifest_path.is_file()` 判定）、
`rir_bank.py:125-138`（legacy 掃描）

M6 的實體佈局（`room_X/room_X_N.wav` + same-stem json）**恰好完全符合 legacy 掃描
規則**。

**實測**：把 manifest 刪掉（模擬 `rsync room_*` 漏拷、或 YAML 打錯
`manifest_name`），reader 靜默載入全部 6 個房間——實際上那是 3 train + 1 validation
+ 2 test。**零錯誤、零警告，三個 split 全混。**

有 manifest 時的 fail-closed 是真的（實測確認會 raise），但整條 split 紀律建立在
一個檔案的存在上。

#### A6 — 訓練 YAML 可誤拿 test split

**位置**：`augmentation.py:93-97`、`dynamic_base.py:268-281`

release 模式只驗 split 非空字串，`split: test` 是合法值直接通過；
`DynamicBaseDataset` 不知道自己是 train 還是 valid dataset，兩者共用同一份
`augmentation_reverb` 設定塊，故無法在程式層擋「train dataset 配 test split」。
現有防線只有註解與文件。

#### A12 — `simulated_rir` dict 無界成長

**位置**：[`augmentation.py:327-332`](../../puresound/audio/augmentation.py)

```python
rir_id = f"bank-{next(self.simulated_rir_counter)}"
self.simulated_rir[rir_id] = {"impulse": impaulse, "sample_rate": sr, "metadata": rir_metadata}
```

每次走 bank 都以遞增 counter 塞入新的 impulse tensor，從無驅逐機制。每 sample 存
2–4 筆、每筆約 100 KB。目前 DataLoader 沒開 `persistent_workers`，worker 每 epoch
重生把洩漏截斷在 epoch 內——**是僥倖而非設計**。快取重用其實只需要上一個
`rir_id`。

#### A13 — provenance 在 dataset 層被 drop

bank → `apply_rir` 這段確實帶了完整 release/recipe/variant/split identity（有測試
涵蓋），但 `_emit_task_metadata` 在基底類別是 no-op，唯一的 override
（`voice_isolation.py`）只萃取 float scalar，字串全數丟棄，collate 也只收 scalar。

計畫書 `RIR_REALISM_PLAN.md:1552` 的「End-to-end augmentation preserves
release/variant/split provenance」只在 augmentor 回傳值層面為真；訓練 batch、log 與
checkpoint 一側拿不到「這個 run 用了哪個 release/variant」。

#### 讀取端其他發現

- `include_failed_qc=True` 同時解除 candidate/production bank 的 pass-only 規則
  （也放行 `pending`），flag 名稱只承諾「failed」；
- plain M6 room bank 模式下損壞 item 靜默消失，無 count 對帳（release bank 有）；
- channel index 越界時靜默 clamp 而非 raise（legacy bank 會產生標籤錯誤的訓練訊號）；
- `PreGeneratedReleaseBank` 每次建構都全量 re-audit（重 hash 全部 WAV/report + 重算
  distribution）。50k–200k item 規模下，每個 DataLoader worker 建構期是 O(bank)，
  可能是分鐘級；fail-closed 是正確取捨，但缺「audit 通過後發 token」的捷徑。

---

## 5. 效能觀察

- **FDN 是逐 sample Python 迴圈**（`multiband_fdn.py:356-375`），實測
  2.2 秒/channel。預設 1000 rooms × 4 RIR × 5 sources = 20000 channels，光 FDN 就
  約 **12 CPU-hr**（8 workers 約 1.5 小時 wall time）。可向量化（block processing
  或 per-band scipy）。
- **filtered boundary branch 對每 event × interaction 重建 filter**（含 `np.roots`
  穩定性檢查與 JSON canonicalization），實測同一 2580-event render **207 s vs
  0.83 s（約 250×）**。不在 M6 預設路徑上（見 Cayley 死路徑），但 M2/M3 validator
  會踩到。

---

## 6. 系統性模式：validator 的 fixture 選擇

有一個反覆出現的模式值得單獨指出：**validator 的 fixture 選擇系統性地避開了會失敗
的組合**。

| Gate 宣稱 | 實際 fixture | 結果 |
|---|---|---|
| serial/parallel manifest hash 相同 | `analytic` + `path-events-m3`（天生決定性） | 預設後端的非決定性被遮蔽（C1） |
| QC 負控制（silent / pre-arrival / sparse） | 無 late-arrival 控制 | timing 死碼不會被發現（A8） |
| forged certificate 被拒 | 只翻 flag、不重算 hash | 稱職偽造會通過（A4） |
| peak_normalized 共同 gain | 只驗整體 peak 命中 0.98 | 退化成 per-channel normalize 也會過 |
| manifest tamper 被拒 | 只測不重算 hash 的竄改 | 自我一致性 ≠ 防竄改 |
| QC fixture regime | 6 items、8 kHz、0.2 s | 1.6 s / 16 kHz 生產 regime 未測 |

每一個單看都合理，合起來讓「N/N gates PASS」的說服力低於字面。

**建議**：把 validator 的 fixture 選擇本身當成審查對象——每個 gate 都要能回答
「如果這個性質是壞的，這個 fixture 會不會抓到？」

---

## 7. 建議優先序

### P0 — 在跑 4,000-item pilot 之前必須處理

1. **`pra.libroom.set_rng_seed(task_seed)`**（C1）——30 秒的修改，解掉預設後端不可
   重現；同時修正 `--resume` help 文字，並用預設後端重跑 M6.2 gate ＋ 新增 fresh
   rerun gate。
2. **低頻激勵 comb**（C2）——影響兩個後端的每一個 item；加激勵頻譜平坦度測試。
3. **FDN 高頻帶覆蓋（A1）與能量錨定（A2）**——否則 matched A/B pilot 量到的是
   實作 artifact，不是物理差異。
4. **兩後端的 directivity / obstacle 不對稱（A10）**——同上理由。

### P1 — 證據鏈與 split 紀律

5. resume 綁 `code_revision`（A3）；provenance 加環境指紋。
6. certificate validator 下限拉到「重跑 decision build」（A4）；補「重算 hash 的
   偽造」negative control。
7. downstream CI 重算（A7）；listening 至少讓 `explicitly_not_human_responses`
   這類守門欄位真的被讀。
8. manifest 缺席時不得靜默退回 legacy（A5）——例如以目錄特徵偵測 M6 佈局並拒絕。
9. QC `direct_arrival_timing` 修成可達（A8），並加 late-arrival 負控制。

### P2 — 訓練整合與工程債

10. `simulated_rir` 改為有界快取（A12）。
11. train/test split 的 role 交叉檢查（A6）。
12. provenance 傳到 batch/log（A13）。
13. FDN 向量化（§5）；release audit 加通過後的快取 token。
14. `item_id` 路徑安全驗證前置於寫檔（QC 與 release 兩處）。

### 版控備註

審查開始時，整個 M6 實作（連同大部分 M0–M5 新模組）還是 untracked / modified
狀態。**這點已於本次 session 期間解決**：M6 核心模組現已全部進版控，working tree
相對 HEAD 乾淨。原本的顧慮——`code_revision` 記到 dirty 或未提交的樹會讓證據鏈
起點浮動——不再成立，但它仍是 A3（resume 不綁 `code_revision`）的前提條件：
revision 現在可信，resume 卻不會比對它。

---

## 8. 重現本次驗證

```bash
# 1) M6 測試套件（26 tests，約 82 秒）
PYTHONPATH=. .venv/bin/pytest -q \
  test/test_rir_bank_manifest.py test/test_m6_bank_contract_validator.py \
  test/test_generate_hybrid_rir_m6.py test/test_m6_reproducible_generation_validator.py \
  test/test_rir_bank_qc.py test/test_m6_item_qc_validator.py \
  test/test_rir_bank_release.py test/test_m6_variant_release_validator.py \
  test/test_rir_bank_evaluation.py test/test_m6_bank_evaluation_validator.py \
  test/test_rir_bank_production.py test/test_m6_production_decision_validator.py \
  test/test_m6_release_training_integration.py
```

```bash
# 2) 決定性對照（C1）：同參數跑三次，比對 WAV hash
for tag in a b c; do
  .venv/bin/python egs/rir_generation/generate_m6_bank.py \
    --output-dir /tmp/m6_smoke_$tag --n-rooms 6 --rir-per-room 1 \
    --num-workers $([ $tag = b ] && echo 2 || echo 1)
done
# a 與 c 參數完全相同（1 worker、seed 1337）；比對後應發現 6/6 WAV 不同
cd /tmp/m6_smoke_a/pyroomacoustics_bank && find . -name '*.wav' | sort | xargs sha256sum
```

```bash
# 3) 390 Hz comb 凹口（C2）：對生成的 bank 取平均頻譜
# 量測 390/780 Hz 相對鄰帶中位數的深度，並以 300/600/900 Hz 作對照
```

```bash
# 4) FDN 高頻空洞（A1）：檢查 16 kHz 下的有效 octave centers
PYTHONPATH=. .venv/bin/python -c \
  "from puresound.audio.rir_metrics import valid_octave_centers; print(valid_octave_centers(16000))"
# 應輸出 [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]，最高僅到 4000 Hz
```

```bash
# 5) manifest 缺席 fail-open（A5）
# 複製一份 bank、刪掉 rir_bank_manifest.json，再以 PreGeneratedRoomBank 讀取
# 應觀察到三個 split 被靜默混合載入，無任何警告
```

---

## 附註

本審查對「M6 是否是一套好的工程契約」與「M6 產出的 RIR 是否物理正確」分開評價：
前者水準之上，後者有兩個 critical 與數個落在域差軸上的 major。兩者的共通風險是
**validator 的證據力被 fixture 選擇稀釋**——這比任何單一 bug 都值得優先修正，
因為它決定了未來的問題會不會再次無聲通過。
