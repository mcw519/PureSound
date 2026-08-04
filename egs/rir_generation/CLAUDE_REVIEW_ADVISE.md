# M6 RIR Generation — 現況

**更新日期**：2026-08-03
**範圍**：M6.1–M6.6 全鏈（契約、生成、QC、release、evaluation、production decision）
＋ M6 實際使用的渲染演算法鏈（scene 抽樣、hybrid crossover、PathEvents/FDN/spatial）
＋ 讀取端與訓練整合

---

## 0. 這份文件是什麼

2026-08-02 的初版審查列了 2 個 critical、14 個 major。逐條實測覆核之後：

| | 數量 | 處置 |
|---|---|---|
| critical | 2 | 都是真的，**都已修復並實測確認** |
| major | 1 | 真實（晚場能量錨定），**已修復** |
| major | 12 | **實測不成立，已刪除** |
| major | 1 | 判準需要人為決定，**2026-08-03 已決定並實作**（[§3](#3-m53-收斂判準已改為擬合達到穩定極小)） |

被刪掉的條目不再列在這裡。**留著壞掉的發現比沒有發現更糟**——它會讓下一個人去修
一個不存在的問題，或是繞過一個其實好的機制。要查它們寫過什麼，看 git 歷史
（`git log -p egs/rir_generation/CLAUDE_REVIEW_ADVISE.md`）。

本文件保留：**現在為真且經量測的性質**（§2，pilot 的地基）、**M5.3 收斂判準的決定
與實作**（§3）、**100 房間 A/B pilot 的結果與由它挖出的問題**（§4）、**效能**（§5）、
**怎麼重跑這些量測**（§6）。

全套測試現況：`.venv/bin/python -m pytest -q test/` → **621 passed**，全綠。

> **2026-08-03 下半場更新。** A/B pilot 跑完了（100 房間 × 4 RIR × 兩臂），結果與
> 它挖出的問題寫在 [§4](#4-ab-pilot-的結果與由它挖出的問題)。兩個要點：
>
> - 找到並修好一個真的 physics bug——**ISO 9613-1 大氣吸收的濕度單位錯了**，4 kHz
>   少 5 倍（§4.3b）。舊測試只驗曲線上升、不驗上升幅度，所以它活了下來。
> - **我在 §4.1 記的兩條「合成 vs 實測」缺陷都已撤回**：tilt（§4.4）與噪音地板
>   （§4.5）。兩次都是同一個成因——**實測參照不是物理量，是五條量測鏈的加權平均**。
>   讀完語料自己的文件後定案：**ACE 是被處理得最多的那一個**（發佈未等化的 AIR、
>   噪音地板人工淡掉），而**沒有任何語料發佈喇叭已補償的 RIR**。**先驗證目標再追
>   它，救回了兩條錯誤的工作方向。**

> **有一條方法論結論值得留下。** 12 條被推翻的發現有同一個成因：**量測或引述
> 「我以為程式在做什麼」，而不是實作本身**。對照組很乾淨——直接對真實產物量測而
> 得的結論（兩個 critical）全部站得住。
>
> 凡是要據以行動的發現，先問一句：**我量的是程式，還是我對程式的複述？**
>
> §4.4 加了第二句，成因不同但代價一樣：**我拿來當目標的那個「實測值」，是一個物理
> 量，還是一堆量測鏈的平均？** 那條 tilt 發現是對真實 WAV 量出來的——所以躲過了第
> 一句話的檢查——但參照本身不成立。

---

## 1. 環境（先讀這段，否則會浪費時間）

**這個 repo 必須用 `.venv/bin/python`，不能用 PATH 上的 `python`。**

PATH 上是 conda 的直譯器，它缺 `pyroomacoustics` 與 `rir_generator`，而且 numpy 與
編譯過的相依套件 ABI 不合（`ValueError: numpy.dtype size changed`）。用它跑測試會
得到 18 個 collection error 與 5 個 M6 validator 失敗——**全部是環境假象，不是程式
問題**。

```bash
.venv/bin/python -c "import pyroomacoustics, rir_generator, numpy; print(pyroomacoustics.__version__, numpy.__version__)"
# 0.10.1 2.4.4
```

---

## 2. 已量測為真的性質

這一節是 pilot 可以站上去的地基。每條都標明**怎麼驗證的**，以及**去哪裡重跑**。

### 2.1 決定性與 provenance

| 性質 | 證據 |
|---|---|
| **預設後端已可重現** | C1 修復後兩次執行 6/6 WAV byte-identical、manifest hash 相同（實測） |
| scene 抽樣本身決定性 | 三次執行 `scene_sha256` 完全相同；非決定性只曾出現在渲染層（實測） |
| **resume 綁 `code_revision`** | `task["m6"]` 含 `code_revision`（[generate_hybrid_rir.py:572](generate_hybrid_rir.py)），`_task_is_complete`（`:834-838`）逐 key 比對 → 換 revision 續跑會重做，不會產生混血 bank |
| canonical JSON hash 鏈同源 | `canonical_json_sha256` 單一來源，`sort_keys` + 固定 separators + `allow_nan=False` |
| `item_id` 路徑安全 | manifest 層 `_safe_path_component("item_id", …)`（[bank/schema.py:345](../../puresound/audio/rir/bank/schema.py)）；`../` 進不了 manifest |
| libsndfile `PEAK` chunk timestamp 歸零 | 不動 chunk size、peak value 與 waveform |

**仍值得補**（minor，非阻礙）：`BankGeneratorProvenance` 只有六欄，
`renderer_version` 硬編 `"M6.2"`，`pyproject.toml` 對 pyroomacoustics 無版本 pin。
numpy（NEP 19 不凍結 `Generator` stream）、pra、torch、libsndfile 任一升級都可能改變
輸出 bytes，而 manifest 記的重現條件毫無變化。建議加關鍵套件版本指紋。

### 2.2 訊號與物理

| 性質 | 量測結果 |
|---|---|
| **低頻激勵平坦** | C2 修復後 390 Hz 凹口 18.7 → **0.9 dB**（對照頻率 ~0 dB） |
| **FDN 晚場覆蓋到 Nyquist** | `_fdn_partition_sos` 是 cascaded binary split，最高帶是到 Nyquist 的 highpass；實測 7.9 kHz **+0.02 dB**、帶內漣波 1.11 dB |
| **低頻走 per-mode 材質阻尼** | M6 預設 `--low-backend pytard-material`；實際 bank metadata `boundary_model: per_mode_surface_material_damping`、`global_rt60_envelope_applied: false` |
| **晚場能量錨定已用 RT60 外推** | `extrapolated_path_tail_energy_target`（[render/coupling.py:186](../../puresound/audio/rir/render/coupling.py)）以材質 RT60 衰減律把有限 PathEvent tail 積分到 render 邊界，不再鎖在截斷 tail 上 |
| **source directivity 真的生效** | `speech_cardioid` → `CardioidFamily(p=0.5)`，`simulate` 對 RoomSceneV2 傳 `directivity=`；cardioid vs 強制 omni 每 channel 差 **4.34–5.79 dB** |
| **obstacle 壓 direct、不動殘響** | `np.linspace(attenuation, 1.0, …)` 斜坡只作用在 `[direct_idx:recovery_end]`；實測 DRR −6.26 → −7.46 dB（**Δ −1.20**），recovery 之後的晚場 **Δ +0.000 dB** |
| **FOA 晚場 isotropy 保住** | 單一 shared gain（`one_shared_array_gain_preserves_spatial_ratios`）；純擴散區 Y/Z/X 距 SN3D isotropic 期望值 −4.77 dB 偏差 **0.23–0.91 dB**，diffuseness 0.919–0.967；array 與 FOA 兩次獨立求解的 gain 差 0.04–0.46 dB |
| Cayley boundary filter 在 M6 路徑上活著 | `PathEventHighFrequencyBackend` 傳 `surface_admittance_models`（[render/high_frequency/path_event.py:120](../../puresound/audio/rir/render/high_frequency/path_event.py)） |
| 空氣吸收在 M6 路徑上活著 | 同 backend `air_absorption: bool = True` 為預設，`apply_air_absorption` 實際被呼叫（`:133-169`）並寫進 metadata |
| **causality 契約全鏈成立** | 低頻帶 clip → causal LP4 保零；高頻帶對齊後逐通道清零 → causal HP4 保零；PathEvents 用 one-sided Lagrange kernel 構造性滿足；FDN coupling 在 transition 前 sample-exact（實測誤差 0.0） |
| PathEvent 幾何核正確 | fold/unfold 距離互檢 1e-11、reciprocity、Cayley filter 的 passivity 與 pole < 1 檢查皆通過 |

**已知設計極限**（不是 bug，但決定實驗邊界）：

- **晚場能量外推的前提是材質 RT60 衰減律**。`HybridRIRConfig.rt60_range = (0.25, 0.8)`
  內沒問題；若把上限調到 1.5 s 以上，外推段佔比會變大，值得重新量一次。
- **材質頻變在早場被折疊成參考頻率單點**，吸收頻譜的完整形狀只影響 FDN 的
  per-octave RT60。早／晚場看到兩套精細度不同的材質視圖。（未實測，讀碼觀察。）

### 2.3 QC 與證據鏈

| 性質 | 證據 |
|---|---|
| **`direct_arrival_timing` gate 是活的** | 搜尋窗 6.0 ms / 容許誤差 1.0 ms，且 `__post_init__` 明文禁止兩者相等；實測延後 direct arrival：0.9 ms 放行、**1.5 ms 起觸發** |
| **production certificate 不可用「重算 hash」偽造** | `validate_m6_production_certificate` 會 **重算** decision components（`_production_decision_components`）並要求 `checks == actual_checks`、key set 完全等於 `PRODUCTION_DECISION_CHECK_NAMES`，還驗 `release_audit`/`evidence_audit`/`evaluation_sha256`（[bank/production.py:477](../../puresound/audio/rir/bank/production.py)） |
| **downstream CI 下界是重算的** | `_paired_t_confidence_interval(improvement)` 由 `*_by_seed` 重算，申報值要與它 `isclose`，而 `all_lower_bounds_positive` 用的是**重算值** `interval[0]`（[bank/evaluation.py:543](../../puresound/audio/rir/bank/evaluation.py)） |
| **lineage 有真的驗** | audit 比對 `parent_rir_sha256`、重算 parent 檔案 hash，並要求 `np.array_equal(child_audio, parent × common_gain)`（[bank/release.py:912-930](../../puresound/audio/rir/bank/release.py)） |
| `not_evaluable` 語意 fail-closed | 缺 measured reference 時 `empirical_exit` 不可能通過，沒有被計為 pass 的路徑 |
| `decide_m6_production` 誠實 | 如實輸出 `blocked` 與 blockers，`evidence_audit` 回報 0/9，不偽造證據 |
| QC 在生產 regime 可用 | 本次實跑 6 rooms / 0.4 s / 16 kHz calibrated：**6/6 pass**，split 3 train / 1 validation / 2 test 皆非空 |

**已知取捨**（有意為之，記著即可）：

- `maximum_peak_abs = 1.0` **對 calibrated 明文豁免**
  （`item.level_policy == "calibrated" or peak_abs <= …`，[bank/qc.py:394](../../puresound/audio/rir/bank/qc.py)）——
  近場物理校準 RIR 的 peak 本來就可以 > 1.0，這是對的。
- **octave bands 有算沒 gate**：每 channel 算 4 個 band 的完整指標，`checks` 沒有任何
  gate 讀它。目前是 informational；若計畫書把它列為 QC 項目，措辭要跟著改。

### 2.4 讀取端與訓練整合

| 性質 | 證據 |
|---|---|
| **manifest 缺席不會靜默混 split** | `_manifestless_m6_metadata_detected()` 偵測 `indexes/*.jsonl` 或 item metadata 的 `m6` block；實測真實 bank 刪 manifest、再刪 indexes，**兩種情況都 raise** |
| 有 manifest 時未指定 split 即 fail-closed | 實測 raise |
| **train/test split 有 role 交叉檢查** | release 模式要求 `split == usage_role`，不符即 raise（[augmentation.py:110-114](../../puresound/audio/augmentation.py)） |
| **`simulated_rir` 是有界 LRU** | `OrderedDict` + `simulated_rir_cache_size = 32` + `popitem(last=False)` 驅逐（`augmentation.py:52-64`） |
| **provenance 有傳到 sample** | base `_emit_task_metadata`（[task/ns.py:1004](../../puresound/task/ns.py)）送出 `rir_release_id`、`rir_release_sha256`、`rir_recipe_id`、`rir_variant_id`、`rir_split`、`rir_origin`、`rir_renderer_profile_id`、`rir_production_certificate_sha256`、`rir_interferer_variant_ids` |

**仍值得看一眼**（未實測，讀碼觀察，非阻礙）：

- `include_failed_qc=True` 同時解除 candidate/production bank 的 pass-only 規則
  （也放行 `pending`），flag 名稱只承諾「failed」；
- `PreGeneratedReleaseBank` 每次建構都全量 re-audit（重 hash 全部 WAV/report）。
  50k–200k item 規模下每個 DataLoader worker 的建構期是 O(bank)。fail-closed 是正確
  取捨，但缺「audit 通過後發 token」的捷徑。

---

## 3. M5.3 收斂判準：已改為「擬合達到穩定極小」

**決定（2026-08-03）**：M5.3 的 `converged` 定義為**擬合達到穩定極小**，不是
「scipy 宣告了終止條件」。

**背景**：C2 的激勵修正改變了最佳化地貌。凍結報告（e454c08）記錄 `success: true`、
6 次評估、`ftol` 收斂、cost 0.040368；修正後同一個 fixture 變成 `success: false`、
40 次用盡、cost **0.033323**——成本更低，只是不再觸發終止條件。

**為什麼 `success` 是錯的判準**（實測，不是推論）：

| 量 | 值 | 讀法 |
|---|---|---|
| 參數數量 | 4 | |
| nfev / njev | 40 / 28 | 跑了 28 輪 Jacobian，不是「才剛起步」 |
| first-order optimality | 0.0619 | 梯度說「非駐點」 |
| 座標步進 1e-1 / 1e-2 / 1e-3（相對 bound span） | **改善 0** | 沒有任何方向能降低成本 |
| 座標步進 1e-4 | 相對改善 1.9e-4 | 只有最細尺度找得到一點 |

粗尺度全無改善、最細尺度才有一點——這是**目標函數局部粗糙**的簽名。`optimality`
量到的是粗糙度，不是下降方向。所以 `success` 與 `optimality` **都不能**當判準。

**實作**（[calibration/inverse_m4.py](../../puresound/audio/rir/calibration/inverse_m4.py)，
policy `puresound.m4_profile_convergence.stable_minimum.v1`）：直接問字面問題——
**再優化下去還會不會變好**。當 `least_squares` 用盡預算時，**從它自己的答案重啟**
（重設 trust region），若重啟無法把成本降低超過 **1e-3 相對容許值**，該點就是求解器
無法認證的極小；若能，原本的擬合是被截斷的，不算收斂。

容許值 1e-3 的依據：sum-of-squares 校準成本的 0.1% 遠低於任何聲學上有意義的差異
（約 0.004 dB），且明顯高於目標函數自身的數值粗糙度（實測 1.9e-4）。

**fixture 上的結果**——重啟後 scipy 自己就宣告收斂了，證實塌掉的是 trust region
而非目標函數：

| | 初解 | 重啟 | 相對改善 |
|---|---|---|---|
| 最佳 profile | status 0（預算用盡）cost 0.0333229 | status 3（xtol）cost 0.033314 | **2.69e-4** ✓ |
| 次佳 profile | status 0，cost 4.90678 | status 3，cost 4.90632 | 9.47e-05 ✓ |
| 第三 profile | status 2（ftol）——不需重啟 | — | — |

**這個判準還能說「不」**：`test_m4_profile_convergence_separates_a_stable_minimum_from_a_truncated_fit`
用 `maximum_evaluations=2` 造一個確實還在下降的擬合，判定為**未收斂**。沒有這個
負控制，新判準就只是把 gate 改成恆真。

報告裡記的是完整證據（`initial_solve` 與 `restart_solve` 各自的 status／cost／改善
幅度），不是一個布林值——讀的人能自己判斷邊際有多少。

---

## 4. A/B pilot 的結果與由它挖出的問題

**設定**：100 房間 × 4 RIR = 400 items/臂，兩臂唯一差異是高頻後端。配對契約由
`validate_m6_pilot_pair.py` 驗過：acoustic space 集合相同、每個 item 的
scene_sha256／split／seed／shape 逐一吻合、低頻帶同組態、release audit 皆過。
**比較可歸因於後端。**

### 4.1 對照 600 個真實 RIR channel 的成績

| bucket | | DRR | C50 | T30 | T30擬合% | tilt dB/oct |
|---|---|---|---|---|---|---|
| 0–1m | **實測** | **−1.83** | **13.90** | **0.33** | 60.2 | **−2.19** |
| | pyro | −0.06 | 8.54 | 0.99 | 68.2 | +0.81 |
| | m4 | +1.93 | 12.59 | 0.50 | 100 | +0.91 |
| 2–3.5m | **實測** | **−4.32** | **12.26** | **0.36** | 76.4 | **−2.74** |
| | pyro | −6.58 | 4.64 | 0.93 | 69.0 | +0.32 |
| | m4 | −5.16 | 8.66 | 0.55 | 100 | +0.10 |
| 3.5–6m | **實測** | **−7.85** | **5.70** | **0.31** | 52.1 | **−2.66** |
| | pyro | −7.91 | 3.41 | 0.96 | 75.0 | +0.37 |
| | m4 | −5.97 | 8.12 | 0.66 | 100 | +0.11 |

平均絕對誤差：**C50** pyro 5.09 / m4 **2.44**；**T30** pyro 0.63 s / m4 **0.24 s**；
**DRR** pyro **1.36** / m4 2.16。

> ~~**tilt** pyro 3.03 / m4 2.90（兩個都錯）~~ — **已撤回，見 [§4.4](#44-tilt-目標無效41-那條-tilt-發現撤回)**。
> tilt 那一欄的實測參照不是房間物理，是五條量測鏈的加權平均；其中 ACE 語料的
> +0.69 幾乎等於合成值。**不要拿這一欄當目標。**

QC 良率：pyro **399/400**（`room_000022_000001` 因 `decay_fit_coverage` 被隔離），
m4 **400/400**。那個 item 的 pyro 版 5 個 channel T20 是 0.45–2.85 s（6.4 倍離散），
m4 版是 0.21–0.35 s，而場景預測 RT60 是 0.32–0.44 s。QC 抓對了。

**兩個容易誤讀的地方：**

- **M4 貼合場景預測 RT60 不是獨立證據**——M4 的 FDN 就是照那個數字設計的，它在複述
  自己的輸入。打破循環的只有實測那一欄。
- **M4 的 T30 擬合覆蓋率 100% 不是優點**。實測只有 52–76% 能擬合單斜率；pyro 的
  68–75% 反而落在正確區間。per-band 指數衰減的 FDN 依定義就是單斜率，那是合成簽名。

~~**兩者共有、換後端不會改善的**：tilt 符號錯、完全沒有噪音地板。~~
**兩條都撤回**——tilt 見 [§4.4](#44-tilt-目標無效41-那條-tilt-發現撤回)，噪音地板見
[§4.5](#45-噪音地板那條也撤回而且它對訓練不構成缺陷)。**§4.1 表格裡「兩者共有的缺陷」
現在是空的。**

### 4.2 crossover 增益夾限太低（已修）

追 tilt 時發現的。`crossover_match_gain_range` 上限是 2.0，而**實際需求的中位數就在
它之上**：拿掉夾限後量 24 場景 × 5 source × 兩後端，需求是 0.76–5.27，中位數
2.0–2.4。所以 pilot 裡 **pyro 70.2% / M4 45.6% 的 channel 卡在 2.000**。

這比一個調參常數嚴重，因為低頻帶的位準是誰給的：

```python
# calibrate_pytard_signal
peak = float(np.max(np.abs(signal)))
calibrated = signal / peak * target_peak * (1.0 / distance_m)
```

`signal / peak` 把模態解的物理振幅**整個丟掉**，1/r 是手動補的。所以
**低頻帶沒有物理絕對位準，這個 crossover 匹配是它唯一的位準來源**，而
`target_peak / peak(房間模態響應)` 隨場景變動（1.5–3.2 倍散佈就是這樣來的）。
被夾住的 channel 因此帶著一個沒有任何地方記錄的位準誤差出貨。

**已修**：上限改 8.0（重測飽和率 0.4%），且 match 回傳夾限前的原始增益，metadata
新增 `low_band_gain_requested_by_channel` / `low_band_gain_range` /
`low_band_gain_clipped_channels`。無聲的夾限就是它能藏起來的原因。

**對 tilt 的效果，同場景配對量測**：六格裡五格往正確方向動 **約 0.5 dB/oct**。
原缺口約 3 dB/oct，所以這是**約五分之一，不是解決**。其他指標（C50/DRR/T30）在
n=15–27 的樣本下兩臂方向不一致，讀不出來。

**根因仍在**：pytARD 應該帶出物理位準，讓匹配從「修正」變成「驗證」（應 ≈1.0）。
M2 的 `rir_source_convention` 機器只接在 `analytic-impedance` 上，
`direct_path_source_convention_matched` 對 M6 預設的 `pytard-material` 永遠是 false，
所以 `preserve_source_convention_at_crossover` 雖然開著卻從不觸發。

### 4.3 材質抽樣有 25% 的房間 RT60 隨頻率上升（成因已查明，未修）

選 README 示意場景時發現的。100 房間 pilot 的分佈：

| | p05 | 中位數 | p95 | max |
|---|---|---|---|---|
| `scene.rt60`（標量） | 0.27 | **0.56** | **2.34** | **3.27** |
| predicted RT60 @250 Hz | 0.34 | 0.64 | 1.59 | 2.54 |
| predicted RT60 @4 kHz | 0.24 | 0.44 | **3.93** | **5.22** |

- **RT60(4k)/RT60(250) 中位數 0.76**（正確地隨頻率下降），但 **25% 的 item 大於 1**，
  p95 達 2.86。**真實房間做不到這件事**——光空氣吸收就禁止 RT60 隨頻率上升。
- **只有 80% 落在文件記載的 `rt60_range = (0.25, 0.8)` 內，12% 超過 1.5 s。**
  v1 場景會從抽到的材質重新推導 RT60，不會約束回設定的範圍。

seed 1337 的 `room_000000` 就是這個尾巴的樣本：2.82 s、每個表面都是硬材質、predicted
RT60 從 250 Hz 的 1.53 s **上升**到 4 kHz 的 4.37 s。它當了 README 示意圖很久，
Schroeder 面板幾乎不衰減。

**成因已查明（2026-08-03）。** RT60 的頻率斜率由「房間總吸收隨頻率上升或下降」決
定，而總吸收約 94% 來自六個邊界面。

**主因：recipe 允許「硬地板 × 硬天花」的組合，機率不低。** 目錄裡的硬表面吸收
**正確地**隨頻率下降（油漆石膏板靠板共振吸低頻、高頻幾乎全反射，這是真實行為）；
`ceiling_plasterboard` 0.200 → 0.020（比值 **0.13**，目錄裡最陡）。兩者相遇時，
房間裡沒有任何東西的吸收是隨頻率上升的：

| 地板 × 天花 | n | 比值中位數 | >1 |
|---|---|---|---|
| wood_16mm × ceiling_plasterboard | 99 | **2.32** | **100%** |
| linoleum × ceiling_plasterboard | 150 | **2.21** | **100%** |
| carpet_cotton × ceiling_plasterboard | 211 | 1.01 | 53% |
| linoleum × ceiling_fissured_tile | 239 | 0.64 | 0.4% |
| carpet × ceiling_fissured_tile | 224 | 0.67 | **0%** |

recipe 權重讓它常發生：`living_room` 給 `ceiling_plasterboard` 權重 **0.85** →
**62.6% 的客廳反向**；`classroom` 是 linoleum 0.80 × ceiling_plasterboard 0.45 →
38.9%。office 8.8%、meeting_room 23.9%（那兩個 recipe 偏好吸收性天花）。

**不是成因，但值得修的兩件事：**

- **家具完全不在 Sabine 總和裡。** `predicted_octave_rt60_s` 只加總
  `effective_boundary_materials()`，`self.objects` 缺席。加進去後反向率 33.6% →
  33.9%（**沒動斜率**），但 scalar RT60 的 p95 從 2.60 降到 **1.78**——所以它修的是
  「太殘響」不是「反向」。家具暴露面積中位數只佔邊界面積 **5.9%**。
- **而且家具在原理上也救不了**：`upholstered_furniture` 吸收 250 Hz 0.66 → 4 kHz
  0.70，**幾乎是平的**。要靠它壓平反向需要軟性面積達邊界面積的 **100%**（中位數）、
  p90 **207%**。目錄裡沒有任何強烈隨頻率上升的大面積吸收材（`curtain_cotton`
  0.30 → 0.71 是上升的，但只當 obstacle 材質用，不佔表面積）。

**一句話**：抽樣器產出的是**聲學上合法但不具代表性**的房間——空的硬盒子。真實有家
具的語音環境，高頻吸收來自窗簾、地毯、書櫃、衣物、人體，而模型既沒有隨頻率上升的
大面積材料，也把家具當成三個不進入殘響計算的小稜柱。

### 4.3b 順帶查出一個 physics bug：大氣吸收的濕度單位錯了（已修）

追 §4.3 時發現的，**這一條比 §4.3 本身更該先修**。

`atmospheric_absorption_db_per_m`（[physics/propagation.py:15](../../puresound/audio/rir/physics/propagation.py)）
宣稱實作 ISO 9613-1，公式逐項正確，但濕度單位錯了：

```python
humidity_fraction = float(relative_humidity_percent) / 100.0   # ← 這裡
molar_water_concentration = humidity_fraction * saturation_pressure_ratio / pressure_ratio
```

ISO 9613-1 的 `h`（水蒸氣莫耳濃度）定義是**百分比**，`h = h_rel · p_sat/p_a` 其中
`h_rel` 也是百分比。程式先除了 100，`h` 因此小 100 倍。

**後果**：氧氣鬆弛頻率算出 **60.5 Hz 而不是 35,414 Hz**（差 585 倍），分子鬆弛項在
整個音頻帶等於失效，只剩太小的古典 f² 項：

| Hz | 實作 | ISO 9613-1 | 倍數差 |
|---|---|---|---|
| 250 | 0.003181 | 0.001310 | 0.41×（**偏高**） |
| 1000 | 0.003498 | 0.004665 | 1.33× |
| 4000 | 0.005906 | 0.029666 | **5.02×** |
| 8000 | 0.013578 | 0.105291 | **7.75×** |

250 Hz → 8 kHz 的動態範圍：實作 **4.3×**，ISO **80.4×**。

**證明方式**：獨立照 ISO 9613-1 重寫一次。把 `h` 當分數傳時，逐位重現實作的數字
（0.003181 / 0.003498 / 0.005906 / 0.013578）；當百分比傳時給出公開表值。

**已修（2026-08-03）**：拿掉 `/ 100.0`，`AIR_ABSORPTION_POLICY` 升到 `…v2`（輸出變
了，provenance 必須能區分 v1 與 v2 的 bank）。全套 **621 passed**。

**為什麼舊測試沒抓到**：`test_iso_air_absorption_...` 只斷言「有限、0 Hz 為零、單調
遞增」。100 倍的單位錯誤三項全過。**曲線會上升不等於上升幅度正確**，而現在補上了
對照 ISO 公開表值與獨立轉寫的測試（後者刻意重寫一次標準公式，而不是呼叫 propagation
自己——單位錯誤會被每個內部呼叫端一起繼承，自己比自己抓不到）。

**修完的實際效果，比我預估的小得多。** 同場景配對量測（50 場景 × 兩後端）：

| bucket | pyro 舊→新 | m4 舊→新 |
|---|---|---|
| 0–1m | +0.21 → +0.21 | +0.55 → +0.58 |
| 2–3.5m | −0.17 → −0.25 | −0.30 → −0.43 |

**最大變化 0.13 dB/oct**，不是我先前估的約 1 dB/oct。**估算錯在**把 0.024 dB/m
乘上 0.5 s 尾巴走的 171 m 當成單次通過損失；實際上衰減尾巴的能量集中在前段，有效
路徑遠短於 171 m，而且空氣吸收是加在**衰減率**上（`60/rt60 + α·c`），4 kHz 只讓
RT60 縮 6–20%，對應 0.3–1 dB 的能量差。

**所以：bug 是真的、修正是對的（實作先前與標準不符，現在符合），但它沒有關掉 tilt
缺口。§4.1 那個約 2.2 dB/oct 仍然沒有解釋。** 唯一確定的改善是分佈層面——air-adjusted
的 RT60 反向率 32.9% → 26.7%，而那正是 M4 的 FDN 直接瞄準的量。

### 4.4 tilt 目標無效——§4.1 那條 tilt 發現撤回

**先驗證目標再追它，結果目標本身站不住。**

實測 RIR 除非語料做過反捲積，否則帶著量測喇叭的響應。判別法是**變異數分解**：同一
語料內的房間在尺寸、材質、家具上都不同，所以真實的房間物理會表現為**組內**離散；
量測鏈每個語料固定，會表現為**組間**偏移。

實測（`compare_measured_tilt_by_corpus.py`，3784 channel / 34 房間 / 5 語料）：

| 語料 | 房間數 | tilt 中位數 | 房間之間 sd |
|---|---|---|---|
| DIFFRIR | 11 | **−2.99** | 0.95 |
| dEchorate | 9 | **−2.54** | **0.14** |
| BRUDEX | 3 | **−1.60** | **0.14** |
| REVERB | 5 | −1.45 | 0.91 |
| **ACE** | 6 | **+0.69** | 0.36 |

| | |
|---|---|
| pooled 中位數 | −1.85 dB/oct |
| 組內（房間之間）平均 sd | **0.50** |
| 組間（語料之間）sd | **1.42（2.85×）** |
| 語料中位數全距 | **3.68 dB/oct** |

**組間離散比組內大 2.85 倍，而語料中位數的全距 3.68 dB/oct 比整個合成-實測缺口還
大。** 最有力的是組內的緊密度：dEchorate 9 個房間 sd 只有 **0.14**，而 dEchorate 的
設計就是可調牆板、各房間吸收特性刻意不同——若 tilt 是房間物理，它們該分散。

**而 ACE 的 +0.69 幾乎等於合成 bank 的 +0.2～+0.6。** 五個實測語料裡有一個和渲染器
一致。

**結論：§4.1 表格裡那個「tilt 符號錯、差約 3 dB/oct」的發現撤回。** 那個 −2.4 不是
房間的物理性質，是五條互相矛盾的量測鏈按檔案數加權後的平均值（DIFFRIR 22000 +
dEchorate 21600 兩個最負的語料主導了 pooled 值）。**照它調渲染器等於把別人的喇叭
烤進訓練語料。**

順帶：這也讓 §4.2 與 §4.3b 的「買到多少 tilt」變成無從評價——那兩項各自都是獨立成
立的修正（夾限確實截斷了唯一的位準機制、大氣吸收確實與標準不符），但它們「縮小了
tilt 缺口」這個說法失去了參照。

**要重建一個有效的 tilt 目標，需要**：一個有記錄且已反捲積量測鏈的語料，或是量出各
語料的鏈並校正回去。在那之前，**tilt 不該列在缺陷清單上**。

`compare_measured_tilt_by_corpus.py` 在組間離散超過 0.75 dB/oct 時 exit 1，所以參照
語料組成一變就會叫。

### 4.5 噪音地板那條也撤回，而且它對訓練不構成缺陷

同一套變異數分解，套在「合成沒有噪音地板」上。分兩層問，因為先驗答案不同：**存在性**
（每個真實量測都有噪音，合成完全沒有）該與鏈無關；**位準**由鏈的 SNR 決定，必然因語料
而異。

用 Lundeby 偵測器（`estimate_noise_floor_lundeby`）與一個無門檻的量——晚段 Schroeder
斜率 / 早段斜率，1.0 = 持續等速衰減，0 = 完全變平（＝有地板）：

| group | 偵測到地板 | dyn dB | **flatness** |
|---|---|---|---|
| DIFFRIR | 5.3% | 43.9 | **0.03** |
| dEchorate | 68.1% | 64.7 | **0.13** |
| BRUDEX | 1.8% | 85.1 | **0.15** |
| REVERB | 62.4% | 58.5 | **0.27** |
| **ACE** | 25.5% | 61.6 | **0.91** |
| SYNTH path-events-m4 | 0.0% | n/a | **0.70** |
| SYNTH pyroomacoustics | 0.0% | n/a | **1.40** |

**存在性不成立。** Lundeby 偵測率在語料之間從 1.8% 到 68.1%——BRUDEX 與 DIFFRIR 幾乎
和合成的 0% 分不出來。那個偵測器有 15 dB 的最小動態範圍門檻，所以「**語料發佈前就把
尾巴截掉了**」會被登記成「沒有地板」，與量測當時有沒有噪音無關。

無門檻的 flatness 乾淨得多：四個語料強烈變平（0.03–0.27），合成完全不變平
（0.70–1.40）。**但 ACE 的 0.91 落在合成那一側。**

**位準完全不能當目標**：dyn dB 從 43.9 到 85.1（相距 **20.8 dB**），組間 sd 9.19 vs
組內 3.50（**2.62×**）。

#### ACE 在兩條軸上都站在合成這邊

這是整場調查最有訊息量的一個觀察，而且不像巧合：

| | ACE | 合成 | 其餘四個語料 |
|---|---|---|---|
| tilt dB/oct | **+0.69** | +0.2 … +0.6 | −1.45 … −2.99 |
| flatness | **0.91** | 0.70 … 1.40 | 0.03 … 0.27 |

**已定案（2026-08-04，讀語料自己的文件）：ACE 是被處理得最多的那一個，不是最乾淨的。**
它兩條軸上的「像合成」都有各自記載明確的後處理成因：

| 觀察 | 成因（ACE 論文原文） |
|---|---|
| flatness 0.91（無地板） | 「The tail of each AIR was **faded down to zero** over 10,000 samples once the level fell below **−70 dB**」（§2.4）——**噪音地板是人工淡掉的** |
| tilt +0.69 | 「The location of the direct path was found by convolving the AIR with the equalisation filter for the source… Equation (3) was then applied to the **unequalized AIR**」（§2.6）——**等化濾波器存在，但發佈的 AIR 沒套用**，含 Fostex 6301B（100 mm 驅動器個人監聽）的響應 |

**而且沒有任何一個語料發佈喇叭已補償的 RIR：**

| 語料 | 喇叭補償 | 尾巴處理 | 對應觀測 |
|---|---|---|---|
| ACE | **未等化**（明文） | 低於 −70 dB 起淡到零 | tilt +0.69、flatness 0.91 |
| dEchorate | 直達音反捲積**只是標註工具**，發佈的是 ESS 估計的 RIR | 固定長度、無 level-triggered 淡出 | tilt −2.54、flatness 0.13 |
| BRUDEX | 只提 inverse sweep 卷積 | Hanning 淡出**只在檔尾** 2400 樣本 | tilt −1.60、flatness 0.15 |

dEchorate §3.1 給了 tilt 另一端的直接機制證據：

> For the octave bands centred at 125 Hz and 250 Hz, the measured RIRs did not exhibit
> sufficient dynamic range for a reliable estimation. This observation found confirmation
> in **the frequency response provided by the loudspeakers' manufacturer, which decays
> exponentially from 300 Hz downwards.**

作者用**喇叭自己的低頻滾降**解釋資料的限制——若喇叭已被反捲積，那個滾降不會在資料
裡。而我的 tilt 量測窗是 **200 Hz–4 kHz**，正好把低端放在那個滾降裡 → 200 Hz 能量被
壓低 → 擬合斜率變得更負。**dEchorate 的 −2.54 與 ACE 的 +0.69 因此各有記載明確的
喇叭成因，不是房間差異。**

**結論**：tilt 目標在這批語料上**無法驗證**，而且理由比 §4.4 原本寫的更強——不是
「語料互相矛盾所以不知道」，而是**各語料自己的論文都記載了喇叭仍在資料裡**。§4.4 的
撤回成立。

**但這不是死路。** 建構一個與鏈無關的參照是可行的：**對每一個 RIR 做直達音反捲積**
（把孤立出來的直達音當逆濾波器），同時除掉喇叭與麥克風響應，留下房間。dEchorate 已
經在標註流程裡這樣做過，ACE 也說了 source 的等化濾波器存在。這是標準做法，也是重建
tilt 參照唯一站得住的路。

#### 而且對這條訓練管線來說，缺噪音地板不是缺陷

`puresound/task/ns.py` 的順序是 **`apply_rir`（:535）→ `add_bg_noise`（:601）**：先卷
RIR，**再**加背景噪音。所以訓練混音的噪音地板是由顯式的噪音增強決定的，不是由 RIR
決定。**RIR 自己沒有地板，在這條管線裡本來就會被下游補上。**

**結論：噪音地板不列在缺陷清單上。** 存在性不普遍、位準不可用、而且就算成立也會被
管線的噪音增強蓋掉。

---

## 5. 效能實測

| 項目 | 實測 |
|---|---|
| M4 高頻帶（優化後） | **24.9 s** / item（優化前 60.1 s，**2.41×**，輸出 bit-identical） |
| ↳ 物件遮蔽 | 31.4 → **5.1 s**（AABB 預排除；526,304 次測試裡 94.7% 可用幾個浮點比較擋掉，真正相交 0.63%） |
| ↳ boundary 濾波器 | 16.1 → **7.5 s**（`(model, cosine, fs)` 記憶化，約半數呼叫重複） |
| pyroomacoustics 高頻帶 | 3.2 s / item |
| 低頻帶 pytARD | CPU **39.2 s** / GPU **6.5 s**（6.1×，GPU 跑三次 bit-identical） |
| `render_multiband_fdn` | 0.155 s / 1.6 s channel（**FDN 不是瓶頸，約 1%**） |
| 400-item 臂（GPU 16 workers） | pyro **9 min** / M4 **25 min**（優化前 44 min） |
| M6 契約測試（13 檔） | **30 passed / 133 s** |
| 全套測試 | **581 passed** |

**只有低頻帶能上 GPU，高頻帶永遠 CPU。** 所以兩臂受益差很多：pyro 93% 的成本在低頻
帶，M4 的主成本是 path event 生成。

---

## 6. 重現

```bash
# M6 契約測試（13 檔，30 tests，約 133 秒）— 注意 .venv
.venv/bin/python -m pytest -q \
  test/test_rir_bank_manifest.py test/test_m6_bank_contract_validator.py \
  test/test_generate_hybrid_rir_m6.py test/test_m6_reproducible_generation_validator.py \
  test/test_rir_bank_qc.py test/test_m6_item_qc_validator.py \
  test/test_rir_bank_release.py test/test_m6_variant_release_validator.py \
  test/test_rir_bank_evaluation.py test/test_m6_bank_evaluation_validator.py \
  test/test_rir_bank_production.py test/test_m6_production_decision_validator.py \
  test/test_m6_release_training_integration.py
```

```bash
# 端到端生成一個小 bank（pyroomacoustics 後端需已安裝；path-events-m4 不需要）
.venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir /tmp/m6_check --backend path-events-m4 \
  --n-rooms 6 --rir-per-room 1 --num-workers 1 --duration 0.4 --sample-rate 16000
# 應以 status=candidate 結束，QC 6/6 pass，三個 split 皆非空
```

```bash
# FOA diffuse isotropy（常駐 validator，四個場景含一個對抗性反例；PASS 為 exit 0）
.venv/bin/python \
  egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_foa_diffuse_isotropy.py
```

```bash
# 決定性：同參數兩次 fresh run，manifest_sha256 應相等
for tag in a b; do
  .venv/bin/python egs/rir_generation/generate_m6_bank.py \
    --output-dir /tmp/m6_det_$tag --backend path-events-m4 \
    --n-rooms 6 --rir-per-room 1 --num-workers 1 --duration 0.4
done
diff <(jq -S . /tmp/m6_det_a/*_bank/rir_bank_manifest.json) \
     <(jq -S . /tmp/m6_det_b/*_bank/rir_bank_manifest.json)
```

```bash
# A/B pilot 一臂（GPU；兩臂的 seed / low backend / GPU 設定必須完全相同）
PURESOUND_M6_PILOT_ROOMS=100 PURESOUND_M6_PILOT_RIR_PER_ROOM=4 \
PURESOUND_M6_PILOT_WORKERS=16 PURESOUND_M6_PILOT_QC_WORKERS=8 \
PURESOUND_M6_PILOT_LOW_BACKEND=pytard-cupy-material \
PURESOUND_M6_PILOT_GPU_DEVICES=0,1 \
  bash egs/rir_generation/phases/m6_bank/scripts/generate_m6_training_pilot.sh \
    path-events-m4 <pilot-root>
```

```bash
# pilot 配對契約 + 良率 + 分距離聲學（PAIRING HOLDS 為 exit 0）
.venv/bin/python egs/rir_generation/phases/m6_bank/scripts/validate_m6_pilot_pair.py \
  --pilot-root <pilot-root>
```

```bash
# 獨立量 WAV，三方對照（§4.1 那張表就是這樣來的）
.venv/bin/python egs/rir_generation/compare_bank_acoustics.py \
  pyro=<pilot-root>/pyroomacoustics_bank \
  m4=<pilot-root>/path-events-m4_bank \
  measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view \
  --per-bank 400
# pair validator 報的是 generator 自己記在 metadata 的 realized_acoustics；
# 這一支直接量 WAV。兩個來源獨立，對得上才算數。
```

---

## 附註：fixture 與判準的選擇

有一個模式值得單獨記著：**測試的 fixture 選擇會系統性地決定它能不能抓到東西**，
而 fixture 通常是照「方便建構」而不是照「會不會失敗」挑的。

兩個本次確認的實例：

- **M6.2 的 serial/parallel manifest hash gate 用
  `--low-backend analytic --high-backend path-events-m3`**
  （`phases/m6_bank/scripts/validate_m6_reproducible_generation.py:66-69`）——兩者
  天生決定性。等於在唯一不會失敗的組合上驗證了決定性，而預設後端當時**確實**是
  非決定性的（C1）。**這個 gate 應該改用預設後端重跑**，並新增一個「刪除輸出、
  同參數 fresh rerun、`manifest_sha256` 相等」的 gate——後者才是 "reproducible
  generation" 的直接定義。
- **`build_m6_bank_contract` 的 fixture item metadata 沒有 `m6` block**
  （只有 `fixture_scope`/`room_id`/`sample_id`/`scene`）。任何拿它來探測「M6 佈局
  偵測」邏輯的測試都會測錯東西——本次覆核 §2.4 那條時就先踩到，換成真實 bank 才
  得到正確結論。

**建議**：每個 gate 都要能回答一句話——**「如果這個性質是壞的，這個 fixture 會不會
抓到？」**

判準也一樣。§3 是這條原則的正面示範：判準確實改了，但改之前先量出
`success` 為什麼是錯的判準，改之後補上一個**確實會失敗**的負控制
（`maximum_evaluations=2` 的截斷擬合判為未收斂）。**沒有負控制的判準變更，跟
把 gate 改成恆真沒有分別。**
