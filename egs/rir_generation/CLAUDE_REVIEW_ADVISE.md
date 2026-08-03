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

本文件只保留三件事：**現在為真且經量測的性質**（§2，pilot 的地基）、**M5.3 收斂
判準的決定與實作**（§3）、**怎麼重跑這些量測**（§5）。

全套測試現況：`.venv/bin/python -m pytest -q test/` → **580 passed (326 s)**，全綠。

> **有一條方法論結論值得留下。** 12 條被推翻的發現有同一個成因：**量測或引述
> 「我以為程式在做什麼」，而不是實作本身**。對照組很乾淨——直接對真實產物量測而
> 得的結論（兩個 critical）全部站得住。
>
> 凡是要據以行動的發現，先問一句：**我量的是程式，還是我對程式的複述？**

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

## 4. 效能實測（pilot 規劃用）

| 項目 | 實測 |
|---|---|
| `render_multiband_fdn` | **0.155 s** / 1.6 s channel @ 16 kHz（`.venv`，3 次取最小） |
| 預設 bank 的 FDN 總量 | 1000 rooms × 4 RIR × 5 sources = 20000 channels → **約 0.86 CPU-hr** |
| M6 契約測試（13 檔） | **30 passed / 133 s** |
| 全套測試 | **580 passed / 326 s**（全綠） |
| filtered boundary branch | 對每 event × interaction 重建 filter（含 `np.roots` 與 JSON canonicalization），同一 2580-event render **207 s vs 0.83 s（約 250×）**。M2/M3 validator 會踩到 |

FDN 不是瓶頸。若要提速，`np.roots` 那條路徑的價值遠高於 FDN。

---

## 5. 重現

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
