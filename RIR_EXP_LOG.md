# RIR 實驗紀錄與計畫

> 本檔彙整 RIR 系統的**實驗紀錄、審查結論與工作計畫**，保留發現與撤回的完整歷史。
> 它不是產品文件：演算法與代碼的對應見
> [docs/audio/rir_realism_algorithm_zh-TW.md](docs/audio/rir_realism_algorithm_zh-TW.md)，
> 使用方式見 [egs/rir_generation/README.md](egs/rir_generation/README.md)。
>
> 三個部分原為獨立文件，2026-08-04 併入本檔（heading 各降一級；併入前的 git 歷史
> 在各自的舊路徑上）：
>
> | 部分 | 原文件 | 內容 |
> |---|---|---|
> | M6 審查與現況 | `egs/rir_generation/CLAUDE_REVIEW_ADVISE.md` | 審查結論、pilot 結果、逐項發現與撤回 |
> | Realism 計畫 | `RIR_REALISM_PLAN.md` | M0–M6 milestone 計畫與進度 |
> | 模組化重構 | `RIR_MODULARIZATION_PLAN.md` | R0–R7 重構計畫與結果 |

---

## M6 RIR Generation — 現況

**更新日期**：2026-08-04
**範圍**：M6.1–M6.6 全鏈（契約、生成、QC、release、evaluation、production decision）
＋ M6 實際使用的渲染演算法鏈（scene 抽樣、hybrid crossover、PathEvents/FDN/spatial）
＋ 讀取端與訓練整合

---

### 0. 這份文件是什麼

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
與實作**（§3）、**100 房間 A/B pilot 的結果與由它挖出的問題**（§4）、**real_native
解封與實測 RIR 的時間原點**（§5）、**M6.6 證據鏈**（§6）、**效能**（§7）、
**怎麼重跑這些量測**（§8）。

全套測試現況：`.venv/bin/python -m pytest -q test/` → **666 passed**，全綠。

> **2026-08-04 更新。** `real_native` / `mixed_calibrated_real` 兩條寫死 blocked 的
> recipe 已解封（[§5](#5-real_native-已解封實測-rir-的時間原點)）。實測 RIR 過不了
> M6 item QC 的原因是**時間原點不同**（40/40 全掛在 `prearrival_energy`），不是資料
> 壞掉；把語料移除掉的傳播延遲放回去之後 **313/313 channel 通過因果 gate**。
> BRUDEX 因為本來就帶著傳播延遲，成了選 onset 判準的**客觀校正標準**——ISO 3382-1 的
> peak−20 dB 讓它只差 1 樣本（2.1 cm、99% 在 ±2 內），而所有「噪音地板以上第一個
> 樣本」的變體都差 13–66 樣本。**這次的三個錯誤又是同一型**：掃描方向、淡入位置、
> 以及第三次的「量到的東西不是我命名的那個概念」（拿位準去判「更早的到達」）。
>
> 同日補上 M6.6 的**證據產生器**（[§6](#6-m66-證據鏈13-項檢查-10-項已過)）。M6 一直
> 有完整的驗證器卻沒有產生器，決策因此 blocked 在「做不出來的檔案」上。13 項檢查現在
> **10 項通過**，剩下三項全部指向真人聽測與下游訓練——兩者都在 codebase 之外。
> **沒有偽造任何聽測資料**：無真人回應時走 `contract_fixture` 乾跑，驗證器接受它是
> 格式正確的契約並正確拒絕當成 empirical 證據。
>
> 並且用**真實 pilot 實跑過一次完整證據鏈**（§6.4，400 合成 item + 1465 measured）。
> 舊 pilot 被 fail-closed 正當地拒收（沒計時、且早於兩個物理修正）。實跑揭出**合成 bank
> 也需要剪枝**（1/400 掛 `decay_fit_coverage` 就足以擋掉整個 release）。
> **§6.5 記了我第四次同型錯誤**：差點把「寬帶 T20 對中頻 Sabine 預測長 1.71×」寫成渲染器
> 超標，實際上那兩個量根本不同。逐 octave 對齊後真正的觀察是交越兩側的 0.75× / 2.07×。
>
> 那條線追到底了（[§6.6](#66-追查交越兩側的-rt60-差異sabine-目標不足但-pyro-過衝更嚴重)）。
> **「交越兩側」這個框架本身也是錯的**——問題在高頻帶隨頻率單調發散。追下去發現
> `predicted_octave_rt60_s` 是純 Sabine，而這批房間 3/4 的吸收分佈不均到讓 Sabine 差 2×
> 以上（天花板吸音、四面牆硬 → 水平掠射路徑避開吸音面），M4 的 FDN 精準命中這個目標
> （0.94–1.00×）而 pyro 的射線追蹤會回應那個不均勻。
>
> **然後對照實測，結論反轉（§6.6.6）。** 實測 RT60 的**頻率形狀**是可用參照（喇叭響應改
> 位準、不改帶內衰減率），而**五個語料一致同意衰減隨頻率下降**。M4 與實測的平均偏差
> **0.081**，pyro **0.628**——**M4 勝 8 倍**，pyro 的高頻殘響是實測的兩倍以上、方向就錯。
> **要擬真度就用 M4。**
>
> 這裡有第五與第六次同型錯誤。§6.6.4：Pearson r 對兩臂都給 +0.6~0.7，只看 r 會誤判「兩臂
> 都抓到了」，實際效應大小差 20 倍。§6.6.6 是新形狀——**每個量測都對，錯在由一連串正確
> 量測推出一個沒被量的結論**（從「Sabine 不足」跨到「pra 更完整」）。
>
> 依此，**M6 預設 backend 已切換為 path-events-m4**（§6.6.6 決策記錄）。速度依決定不重測，
> pyro 留作顯式 A/B 臂；底層 `generate_hybrid_rir` 的預設由 M4/M5 出口 gate 錨定、不動。
> 新預設組合的可重現性已補量（§2.1）。

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

### 1. 環境（先讀這段，否則會浪費時間）

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

### 2. 已量測為真的性質

這一節是 pilot 可以站上去的地基。每條都標明**怎麼驗證的**，以及**去哪裡重跑**。

#### 2.1 決定性與 provenance

| 性質 | 證據 |
|---|---|
| **預設後端已可重現** | C1 修復後兩次執行 6/6 WAV byte-identical、manifest hash 相同（實測；當時預設高頻帶為 pyroomacoustics）。2026-08-04 預設切為 path-events-m4 後**重量一次**：同參數兩次 fresh run `manifest_sha256` 相同、QC 6/6、status=candidate（實測，6 rooms × 1 item / 0.4 s / pytard-material + path-events-m4） |
| scene 抽樣本身決定性 | 三次執行 `scene_sha256` 完全相同；非決定性只曾出現在渲染層（實測） |
| **resume 綁 `code_revision`** | `task["m6"]` 含 `code_revision`（[generate_hybrid_rir.py:572](egs/rir_generation/generate_hybrid_rir.py)），`_task_is_complete`（`:834-838`）逐 key 比對 → 換 revision 續跑會重做，不會產生混血 bank |
| canonical JSON hash 鏈同源 | `canonical_json_sha256` 單一來源，`sort_keys` + 固定 separators + `allow_nan=False` |
| `item_id` 路徑安全 | manifest 層 `_safe_path_component("item_id", …)`（[bank/schema.py:345](puresound/audio/rir/bank/schema.py)）；`../` 進不了 manifest |
| libsndfile `PEAK` chunk timestamp 歸零 | 不動 chunk size、peak value 與 waveform |

**仍值得補**（minor，非阻礙）：`BankGeneratorProvenance` 只有六欄，
`renderer_version` 硬編 `"M6.2"`，`pyproject.toml` 對 pyroomacoustics 無版本 pin。
numpy（NEP 19 不凍結 `Generator` stream）、pra、torch、libsndfile 任一升級都可能改變
輸出 bytes，而 manifest 記的重現條件毫無變化。建議加關鍵套件版本指紋。

#### 2.2 訊號與物理

| 性質 | 量測結果 |
|---|---|
| **低頻激勵平坦** | C2 修復後 390 Hz 凹口 18.7 → **0.9 dB**（對照頻率 ~0 dB） |
| **FDN 晚場覆蓋到 Nyquist** | `_fdn_partition_sos` 是 cascaded binary split，最高帶是到 Nyquist 的 highpass；實測 7.9 kHz **+0.02 dB**、帶內漣波 1.11 dB |
| **低頻走 per-mode 材質阻尼** | M6 預設 `--low-backend pytard-material`；實際 bank metadata `boundary_model: per_mode_surface_material_damping`、`global_rt60_envelope_applied: false` |
| **晚場能量錨定已用 RT60 外推** | `extrapolated_path_tail_energy_target`（[render/coupling.py:186](puresound/audio/rir/render/coupling.py)）以材質 RT60 衰減律把有限 PathEvent tail 積分到 render 邊界，不再鎖在截斷 tail 上 |
| **source directivity 真的生效** | `speech_cardioid` → `CardioidFamily(p=0.5)`，`simulate` 對 RoomSceneV2 傳 `directivity=`；cardioid vs 強制 omni 每 channel 差 **4.34–5.79 dB** |
| **obstacle 壓 direct、不動殘響** | `np.linspace(attenuation, 1.0, …)` 斜坡只作用在 `[direct_idx:recovery_end]`；實測 DRR −6.26 → −7.46 dB（**Δ −1.20**），recovery 之後的晚場 **Δ +0.000 dB** |
| **FOA 晚場 isotropy 保住** | 單一 shared gain（`one_shared_array_gain_preserves_spatial_ratios`）；純擴散區 Y/Z/X 距 SN3D isotropic 期望值 −4.77 dB 偏差 **0.23–0.91 dB**，diffuseness 0.919–0.967；array 與 FOA 兩次獨立求解的 gain 差 0.04–0.46 dB |
| Cayley boundary filter 在 M6 路徑上活著 | `PathEventHighFrequencyBackend` 傳 `surface_admittance_models`（[render/high_frequency/path_event.py:120](puresound/audio/rir/render/high_frequency/path_event.py)） |
| 空氣吸收在 M6 路徑上活著 | 同 backend `air_absorption: bool = True` 為預設，`apply_air_absorption` 實際被呼叫（`:133-169`）並寫進 metadata |
| **causality 契約全鏈成立** | 低頻帶 clip → causal LP4 保零；高頻帶對齊後逐通道清零 → causal HP4 保零；PathEvents 用 one-sided Lagrange kernel 構造性滿足；FDN coupling 在 transition 前 sample-exact（實測誤差 0.0） |
| PathEvent 幾何核正確 | fold/unfold 距離互檢 1e-11、reciprocity、Cayley filter 的 passivity 與 pole < 1 檢查皆通過 |

**已知設計極限**（不是 bug，但決定實驗邊界）：

- **晚場能量外推的前提是材質 RT60 衰減律**。`HybridRIRConfig.rt60_range = (0.25, 0.8)`
  內沒問題；若把上限調到 1.5 s 以上，外推段佔比會變大，值得重新量一次。
- **材質頻變在早場被折疊成參考頻率單點**，吸收頻譜的完整形狀只影響 FDN 的
  per-octave RT60。早／晚場看到兩套精細度不同的材質視圖。（未實測，讀碼觀察。）

#### 2.3 QC 與證據鏈

| 性質 | 證據 |
|---|---|
| **`direct_arrival_timing` gate 是活的** | 搜尋窗 6.0 ms / 容許誤差 1.0 ms，且 `__post_init__` 明文禁止兩者相等；實測延後 direct arrival：0.9 ms 放行、**1.5 ms 起觸發** |
| **production certificate 不可用「重算 hash」偽造** | `validate_m6_production_certificate` 會 **重算** decision components（`_production_decision_components`）並要求 `checks == actual_checks`、key set 完全等於 `PRODUCTION_DECISION_CHECK_NAMES`，還驗 `release_audit`/`evidence_audit`/`evaluation_sha256`（[bank/production.py:477](puresound/audio/rir/bank/production.py)） |
| **downstream CI 下界是重算的** | `_paired_t_confidence_interval(improvement)` 由 `*_by_seed` 重算，申報值要與它 `isclose`，而 `all_lower_bounds_positive` 用的是**重算值** `interval[0]`（[bank/evaluation.py:543](puresound/audio/rir/bank/evaluation.py)） |
| **lineage 有真的驗** | audit 比對 `parent_rir_sha256`、重算 parent 檔案 hash，並要求 `np.array_equal(child_audio, parent × common_gain)`（[bank/release.py:912-930](puresound/audio/rir/bank/release.py)） |
| `not_evaluable` 語意 fail-closed | 缺 measured reference 時 `empirical_exit` 不可能通過，沒有被計為 pass 的路徑 |
| `decide_m6_production` 誠實 | 如實輸出 `blocked` 與 blockers，`evidence_audit` 回報 0/9，不偽造證據 |
| QC 在生產 regime 可用 | 本次實跑 6 rooms / 0.4 s / 16 kHz calibrated：**6/6 pass**，split 3 train / 1 validation / 2 test 皆非空 |

**已知取捨**（有意為之，記著即可）：

- `maximum_peak_abs = 1.0` **對 calibrated 明文豁免**
  （`item.level_policy == "calibrated" or peak_abs <= …`，[bank/qc.py:394](puresound/audio/rir/bank/qc.py)）——
  近場物理校準 RIR 的 peak 本來就可以 > 1.0，這是對的。
- **octave bands 有算沒 gate**：每 channel 算 4 個 band 的完整指標，`checks` 沒有任何
  gate 讀它。目前是 informational；若計畫書把它列為 QC 項目，措辭要跟著改。

#### 2.4 讀取端與訓練整合

| 性質 | 證據 |
|---|---|
| **manifest 缺席不會靜默混 split** | `_manifestless_m6_metadata_detected()` 偵測 `indexes/*.jsonl` 或 item metadata 的 `m6` block；實測真實 bank 刪 manifest、再刪 indexes，**兩種情況都 raise** |
| 有 manifest 時未指定 split 即 fail-closed | 實測 raise |
| **train/test split 有 role 交叉檢查** | release 模式要求 `split == usage_role`，不符即 raise（[augmentation.py:110-114](puresound/audio/augmentation.py)） |
| **`simulated_rir` 是有界 LRU** | `OrderedDict` + `simulated_rir_cache_size = 32` + `popitem(last=False)` 驅逐（`augmentation.py:52-64`） |
| **provenance 有傳到 sample** | base `_emit_task_metadata`（[task/ns.py:1004](puresound/task/ns.py)）送出 `rir_release_id`、`rir_release_sha256`、`rir_recipe_id`、`rir_variant_id`、`rir_split`、`rir_origin`、`rir_renderer_profile_id`、`rir_production_certificate_sha256`、`rir_interferer_variant_ids` |

**仍值得看一眼**（未實測，讀碼觀察，非阻礙）：

- `include_failed_qc=True` 同時解除 candidate/production bank 的 pass-only 規則
  （也放行 `pending`），flag 名稱只承諾「failed」；
- `PreGeneratedReleaseBank` 每次建構都全量 re-audit（重 hash 全部 WAV/report）。
  50k–200k item 規模下每個 DataLoader worker 的建構期是 O(bank)。fail-closed 是正確
  取捨，但缺「audit 通過後發 token」的捷徑。

---

### 3. M5.3 收斂判準：已改為「擬合達到穩定極小」

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

**實作**（[calibration/inverse_m4.py](puresound/audio/rir/calibration/inverse_m4.py)，
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

### 4. A/B pilot 的結果與由它挖出的問題

**設定**：100 房間 × 4 RIR = 400 items/臂，兩臂唯一差異是高頻後端。配對契約由
`validate_m6_pilot_pair.py` 驗過：acoustic space 集合相同、每個 item 的
scene_sha256／split／seed／shape 逐一吻合、低頻帶同組態、release audit 皆過。
**比較可歸因於後端。**

#### 4.1 對照 600 個真實 RIR channel 的成績

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

#### 4.2 crossover 增益夾限太低（已修）

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

#### 4.3 材質抽樣有 25% 的房間 RT60 隨頻率上升（成因已查明，未修）

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

#### 4.3b 順帶查出一個 physics bug：大氣吸收的濕度單位錯了（已修）

追 §4.3 時發現的，**這一條比 §4.3 本身更該先修**。

`atmospheric_absorption_db_per_m`（[physics/propagation.py:15](puresound/audio/rir/physics/propagation.py)）
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

#### 4.4 tilt 目標無效——§4.1 那條 tilt 發現撤回

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

#### 4.5 噪音地板那條也撤回，而且它對訓練不構成缺陷

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

##### ACE 在兩條軸上都站在合成這邊

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

##### 而且對這條訓練管線來說，缺噪音地板不是缺陷

`puresound/task/ns.py` 的順序是 **`apply_rir`（:535）→ `add_bg_noise`（:601）**：先卷
RIR，**再**加背景噪音。所以訓練混音的噪音地板是由顯式的噪音增強決定的，不是由 RIR
決定。**RIR 自己沒有地板，在這條管線裡本來就會被下游補上。**

**結論：噪音地板不列在缺陷清單上。** 存在性不普遍、位準不可用、而且就算成立也會被
管線的噪音增強蓋掉。

---

### 5. real_native 已解封：實測 RIR 的時間原點

`real_native` 與 `mixed_calibrated_real` 兩條 recipe 過去是**寫死 blocked** 的，理由是
「沒有通過 QC 的 measured M6 variant」。先量了一次到底差多遠：**把五個語料打包成 M6
item 直接跑真實 QC，40/40 全掛，而且掛在同一個地方**——每個語料 23–34 個 channel 的
`prearrival_energy`，外加 `sound_speed_is_physical: not_evaluable`。

**成因是時間原點不同，不是資料壞掉。** M6 定義 `t=0` 是聲源發聲時刻，所以
`floor(distance/c·fs)` 之前必須是零。發佈的 RIR 以自己的直達音為原點。判別兩者的量測：
**實測的 pre-arrival 能量比噪音地板高 44–74 dB**（p95 到 0.0 dB，即直達音峰值本身就落在
幾何窗內）——那是直達音掉進窗裡的簽名，不是噪音漏進窗裡。

所以 ingest 做的事是**把語料移除掉的傳播延遲放回去**：每個 channel 平移，使它的
ISO 3382-1 起點落在該 channel 距離對應的幾何到達樣本上，前面的區間靜音。房間響應本身
一個樣本都沒改。

#### 5.1 BRUDEX 是這件事的校正標準

BRUDEX 的 onset 隨距離的斜率是 **+1.00**（其餘語料 ≈ 0），即傳播延遲本來就在資料裡。
**正確的 onset 判準必須向 BRUDEX 要求零平移**，這給了一個客觀的選法：

| onset 門檻 | BRUDEX median \|shift\| | ±2 樣本內 |
|---|---|---|
| **peak−20 dB（ISO 3382-1）** | **1.0 樣本（2.1 cm）** | **99%** |
| peak−40 dB | 3.0 | 38% |
| noise+30 dB | 13.0 | 1% |
| noise+20 dB | 26.0 | 0% |
| noise+10 dB | 66.5 | 0% |

**每一個「噪音地板以上第一個樣本」的變體都差 13–66 樣本**，因為掃頻解捲積留下的
acausal pre-ringing 就坐在噪音地板之上、直達音之前。ISO 的 peak-relative 門檻贏。

實作上有兩個容易寫錯的地方，兩個都踩過並被上表抓出來：

- **掃描方向**。ISO 的規則是**從頭往前找第一次越過門檻**。從峰值往回找會得到「峰值前
  最後一個安靜的樣本」，那會把所有更早的真實響應留在「起點之前」，於是 muting 吃掉
  真訊號。改對之後 `removed_energy` 從 p95 ≈ 0.5 掉到 2–4e-3。
- **淡入位置**。ramp 必須落在起點**之前**的次門檻樣本上。原本讓它從幾何到達點開始，
  結果衰減掉直達音峰值本身，近場 channel 一半的能量就這樣沒了，而所有 gate 都還報成功。
  代價是刻意的 4 樣本（0.25 ms）到達偏移，均勻施加於全部 channel，且在 QC 的 1 ms 容忍內。

#### 5.2 拒收而不硬對

`earlier_arrival` 擋的是 ISO 門檻的盲區：直達音比某個晚期反射弱 20 dB 以上時，前向掃描
會跨過它。**判別依據是「間隔」，不是位準**——pre-onset 區間本來就有直達音自己的上升緣，
比噪音高 40 dB 以上，所以拿位準去比會 100% 誤判（這是第三次同型錯誤：**量到的東西不是
我命名的那個概念**）。改成「pre-onset 最大短窗 RMS 高於噪音，且與 onset 之間有空隙」。

#### 5.3 實測結果

`--limit-per-corpus 60`，300 個 item：

| 語料 | aligned | rejected | shift p50 | 拒收原因 |
|---|---|---|---|---|
| brudex | 60 | 0 | **+4**（= 淡入偏移，原始誤差 0） | — |
| reverb | 60 | 0 | −2111（拿掉它 131 ms 的前導 pad） | — |
| diffrir | 59 | 1 | +5 | earlier_arrival |
| ace | 56 | 4 | +5 | earlier_arrival 3、removed_energy 2 |
| dech | 28 | **32** | +2 | earlier_arrival |

- **313/313 channel 通過因果 gate**，arrival error 最大 0.19 ms（容忍 1.0 ms）。
- QC **217/263 pass**；被隔離的全是 DIFFRIR，掛在 `implausible_t20` /
  `decay_fit_coverage`——就是 §4.5 量到的 flatness 0.03、尾巴被淡掉。**那是真實資料的
  性質，不是對位問題，讓 QC 正當地隔離它們才對。**
- dech 拒收率最高（53%），與 dEchorate 自己文件承認的喇叭低頻滾降一致。

**沒有為了通過而放寬任何 QC 門檻。** release variant 需要 100% pass（production decision
逐 item 檢查），所以取一份**剪枝副本**；未剪枝的 bank 留著當「丟了什麼、為什麼」的紀錄。

兩個 production decision 檢查因此從 False 翻成 True：

```
                                    無 measured    有 measured
all_required_recipes_are_ready         False    →     True
real_and_mixed_recipe_semantics_are_valid False →     True
```

不給 `--measured-bank` 時行為與過去完全相同（blocked，blocker 文字不變）。

**`sound_speed_m_s` 是假設，不是量測**：沒有任何語料發佈氣溫濕度，所以用
`EnvironmentConfig(20 °C, 50% RH)` → 344.04 m/s，寫進每個 item 的 scene（QC 會用同一個
數字重算到達時刻，兩邊才對得上），並在 `provenance` 欄標明是假設。

---

### 6. M6.6 證據鏈：13 項檢查 10 項已過

M6 有完整的**驗證器**卻幾乎沒有**產生器**：`audit_m6_production_evidence` 會查 bundle、
三份角色簽核、每個 renderer profile 一份核准記錄，但 repo 裡沒有任何東西寫得出這些檔案。
所以那個檢查是永久 blocked 在「做不出來的檔案」上。現在補上了產生器。

現況（`build_m6_evidence.py --pass attest`）：

| | 檢查 | 說明 |
|---|---|---|
| PASS | candidate_release_audit_passed | |
| PASS | source_release_is_immutable_candidate | |
| PASS | all_required_recipes_are_ready | §5 measured variant |
| PASS | real_and_mixed_recipe_semantics_are_valid | §5 |
| PASS | all_variant_items_are_qc_passed | |
| PASS | all_generator_revisions_are_pinned | |
| **PASS** | **all_renderer_profiles_are_production_approved** | 新：核准記錄 |
| PASS | evaluation_schema_and_content_hash_match | |
| PASS | evaluation_targets_this_release | |
| **PASS** | **m6_5_implementation_exit_passed** | 新：throughput + listening 契約 |
| ---- | m6_5_empirical_exit_passed | **需真人聽測 + 下游訓練** |
| ---- | m6_5_production_enablement_passed | = implementation AND empirical |
| ---- | external_evidence_bundle_audits | **需下游三個 artifact** |

**剩下三項全部指向同兩件事**：真人聽測（≥20 人）與下游訓練。兩者都在這個 codebase 之外，
機制已建好、已驗證，資料一到就能算。

#### 6.1 核准必須在 QC 之前 → 兩趟流程

`manifest_hash_matches_summary` 把 bank 的 QC summary 綁在 manifest hash 上，而蓋
`production_approved` 會改變 manifest hash。**QC 之後才核准，會讓被核准的 release 自己
失效。** 所以：

1. **pass evaluate** — 生成 → QC → release → 評估。profile 停在 `development`，產出證據。
2. **pass approve** — 用第 1 趟的 `evaluation_sha256` 當核准依據 → 蓋章 → 重跑 QC → 重建
   release。核准因此在 hash 鏈**之內**，不是旁邊。
3. **pass attest** — 聽測 assignment、簽核、bundle、決策，全部綁到重建後的 release。

核准記錄引用的是第 1 趟的評估——那正是它據以決定的東西。

#### 6.2 不偽造：機制與資料分離

`validate_listening_report` 本來就很嚴：`evidence_tier: empirical` 時它會從逐受試者原始
記錄**重算** estimate 與 paired-t CI 並要求申報值吻合，且 <20 人不算 empirical。缺的只是
產生器——所以過去要滿足它只能手寫數字，也就是編造。

現在兩端都補上，且刻意分開：`build_listening_assignment` 設計實驗（room-disjoint 配對、
盲化標籤、hidden reference / degraded anchor），`ingest_listening_responses` 只讀真實回應
並計分。**這裡沒有任何函式會產生回應。** 無真人資料時走 `build_dry_run_report`，發出
`evidence_tier: contract_fixture` + `explicitly_not_human_responses: true`——驗證器接受它
是格式正確的契約，並正確地拒絕把它當 empirical 證據。

三個實作上的坑，都是實測踩出來的：

- **兩個不同的 `EVIDENCE_TIERS`**。renderer profile 用
  `(development, empirical_candidate, production_approved)`，listening 契約用
  `(contract_fixture, empirical)`。用錯會掛在 `evidence_tier_is_declared`，而錯誤訊息
  完全不提是哪個字彙表。兩個 tier 現在都有具名常數。
- **anchor 不能留在 `response_records` 裡**。驗證器從它拿到的**每一筆**記錄重算 estimate，
  所以 hidden reference / degraded anchor 若留在那個 list，就會被摺進主要 endpoint。
  它們現在放進 analysis 的 `validity_screening`，是受試者篩選而非 endpoint。
- **核准記錄要的是「檔案 hash」不是「內容 hash」**。bundle 驗 `sha256_file(path)` 並要求
  它等於 profile 的 `approval_report_sha256`；canonical-JSON 內容 hash 在縮排與換行之後
  就不一樣了，而這個不一致只會在最後決策時才浮現。

#### 6.3 產生器全部 fail-closed

- `build_throughput_report` 只從 generator 自己的 audit 取數，**沒有計時或沒有申報失敗數
  就拒絕產出**（缺量測不會變成預設值）。為此在 generator 補了 `elapsed_seconds`；
  `items_failed: 0` 是設計事實而非未驗證的宣稱——worker 例外會經 `future.result()` 中止
  整個 run，走不到寫 manifest。
- `RendererApproval` / `write_production_signoff` 沒有具名核准者與 scope 就 raise，並且
  **只抄它被交付的證據 hash，不自己算、不給預設**。
- `build_evidence_bundle` 逐檔案 hash，declared 但不存在就 raise，並把缺哪幾個 required
  kind 講出來（否則只會在 audit 裡靜靜地掛掉）。
- `approve_bank_renderer_profiles` 要求 bank 內**每一個** profile 都有核准，部分過只會把
  失敗推到更難查的地方。

#### 6.4 真實 pilot 實跑（100 房 × 4 = 400 item + 1465 measured）

`evidence_pilot_20260804`，用**當前 HEAD 重新生成**的 pilot 跑完整三趟。

**為什麼不能用舊的 pilot**（`pilot_100room_20260803`）：兩個獨立理由。

1. 它的 `generation_run` 沒有 `elapsed_seconds`（今天才加），所以
   `build_throughput_report` **直接拒絕**——這正是 fail-closed 設計在真實資料上生效：
   *缺量測不會變成預設值*。
2. 它的 `code_revision=a24691e9` **早於** crossover 夾限修正（`368a545`）與 ISO 9613-1
   濕度修正（`3f146da`）。拿它去核准，等於為已被取代的物理蓋章。

實跑結果：

| | 值 |
|---|---|
| 生成 | 400 item / 100 房，**1310.7 s**、**0.305 item/s**、失敗 0（GPU 低頻帶 + 16 workers） |
| 合成 QC | **399/400**（1 個掛 `decay_fit_coverage`） |
| measured ingest | 2000 抽樣 → 1762 對位 → QC **1465** pass |
| release | synthetic 399 / real 1465 / mixed 1864 |
| 決策 | **10/13 PASS**，exit 3 |

**新發現：合成 bank 也需要剪枝。** `all_variant_items_are_qc_passed` 檢查**每一個** item，
所以那 1/400 讓整個 release 過不了。`prune_bank_to_qc_passed` 因此不是 measured 專用的
——它是通用的 bank 手術，已從 `measured_ingest` **搬到 `release.py`**，並接進 evidence CLI
的 approve 趟。

BRUDEX 校正標準在 400 item/語料的規模下依然成立：shift p50 = **+4**（= 淡入偏移，原始誤差 0）。

#### 6.5 400 vs 1465 channel 的聲學分佈——以及我差點記錯的一條

`synthetic_to_measured` 的 normalized Wasserstein（距離 ÷ 實測 p05–p95 跨距）：

| 指標 | 距離 |
|---|---|
| **t20_s** | **4.581** |
| **mixing_time_s** | **1.685** |
| spectral_tilt_db_per_octave | 0.316 |
| c80_db / c50_db / drr_db | 0.312 / 0.266 / 0.248 |
| distance_m | 0.159 |
| late_median_normalized_density | 0.104 |

**我差點把 t20 記成「渲染器超出自己的目標 1.71×」。那是錯的，第四次同型錯誤。**
`scene.rt60` 的 `rt60_origin` 是 `surface_material_sabine_prediction`，而且它等於
**500/1k 中頻帶**預測的平均；realized `t20_s` 是**寬帶**。該 bank 的 octave 預測隨頻率
大幅上升（樣本：125 Hz 1.08 s → 8 kHz 4.46 s），所以寬帶 T20 比中頻預測長是**預期物理**，
不是超標。**又是拿兩個不同的量去比，然後給差值取了個名字。**

逐 octave 對齊之後（同頻帶、fit R²≥0.70，n≈1900–2000/帶）：

| 頻帶 | realized T20 / predicted RT60 p50 |
|---|---|
| 250 Hz | 0.78 |
| 500 Hz | 0.75 |
| 1000 Hz | **1.94** |
| 2000 Hz | **2.07** |

pooled median 0.93。**分界正好落在 1 kHz 的 hybrid crossover 上**：交越以下實現得比預測短
約 ¼，以上長約 2×。這是內部可查、與實測參照無關的觀察，**但成因尚未隔離**——crossover 只是
最可疑的嫌疑者，不是已證明的原因。列為待查，不是已知缺陷。

> **已追查完畢，見 §6.6。** 兩件事要回頭修正這一節的讀法：這些數字**只是 pyro 臂的**
> （這份 pilot 當時只有 pyro 臂），而且「交越兩側」的框架是錯的——真正的形狀是高頻帶隨
> 頻率單調發散。M4 臂在同樣的量測下與實測吻合得好得多。

至於 t20 那個 4.581 的**分佈**距離，主因是合成**族群**本身：`scene.rt60` 實際跨
**0.214–3.271 s**，完全不受 `rt60_range = (0.25, 0.8)` 約束（即先前已知未修的
「`rt60_range` 不 constraining v1 scenes」，現在有了規模化的量化）。合成 T20 p50 = 0.99 s
對實測 0.46 s。**這是抽樣問題，不是渲染器沒打中目標。**

`mixing_time_s` 的 1.685 要當心：實測 p05–p95 跨距只有 **0.043 s**，很窄的跨距會把
normalized 距離放大。合成 p50 0.058 對實測 0.022——差 36 ms，不是 1.7 個「單位」。

#### 6.6 追查交越兩側的 RT60 差異：Sabine 目標不足，但 pyro 過衝更嚴重

§6.5 記的「交越兩側 0.75× / 2.07×」被追到底了。**那個框架本身是錯的**——問題不在交越，
在高頻帶；而最終的缺陷不在渲染器，在**目標值**。

#### 6.6.1 分開量：高頻帶隨頻率發散，不是階梯

單獨渲染同一個 scene 的兩個帶（predicted 隨頻率下降的正常房間）：

| | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| predicted（Sabine） | 0.570 | 0.651 | 0.518 | 0.429 |
| 低頻帶單獨 | 0.450 | 0.402 | *(阻帶)* | *(阻帶)* |
| **高頻帶單獨** | **0.564** | 0.855 | 0.946 | **1.056** |

高頻帶在 250 Hz 幾乎完全命中預測（**0.99×**），然後單調發散到 2 kHz 的 **2.46×**。
它的衰減隨頻率**上升**，而它拿到的吸收係數說應該下降。交越只是把一個漸變的東西切成兩段，
讓它看起來像階梯。

沿路排除三個假設，**全部是量出來否證的**：

| 假設 | 否證方式 |
|---|---|
| pra 只拿到單一寬帶吸收值 | `_v2_materials` 傳完整頻譜；pra 存 7 個 band |
| 頻率格點不對齊 | 兩邊都是 `[125,250,500,1000,2000,4000,8000]` |
| 低頻帶模態只到 209 Hz | 那是 `material_modal_damping_metadata(max_modes=128)` 的**記錄**上限；solver 對整個 `omega[z,y,x]` DCT 格點施加阻尼 |

#### 6.6.2 成因：吸收分佈不均，Sabine 失效

同一房間的材質：**天花板 α 0.487→0.988（隨頻率升）、牆面 α 0.187→0.045（隨頻率降）**。
Sabine 假設擴散場均勻取樣所有表面（`Σ Sᵢαᵢ`），但**四面硬牆之間的水平掠射路徑幾乎碰不到
吸音天花板**——2 kHz 時牆的 α 只有 0.045，每次反射幾乎不損失能量。這是聲學上已知的
Sabine 失效條件，不是實作 bug。

399 item 在 2 kHz 的檢定（`mean_α / wall_α` 越大 = 牆比房間平均越硬）：

| 分佈不均程度 | n | **pyro 實現/Sabine** | **M4 實現/Sabine** |
|---|---|---|---|
| 0.68–4.18（最均勻） | 100 | 0.88 | 0.88 |
| 4.18–6.10 | 103 | 2.05 | 0.97 |
| 6.10–9.02 | 100 | 2.39 | 0.99 |
| 9.02–20.01（最不均） | 100 | **3.07** | **0.99** |

**Sabine 成立的那一組兩臂一致（0.88/0.88）**，這是假設成立的關鍵證據。母體
`mean_α/wall_α` p50 = **6.10**——3/4 的房間不均勻到讓 Sabine 差 2 倍以上。

#### 6.6.3 M4 arm：忠實命中一個錯的目標

M4 arm 用當前 HEAD 重新生成（400 item、1082 s、QC 400/400），**同 seed 同場景**，
所以可以逐 item 對照：

| | Sabine 預測 | M4 FDN 目標 | **M4 實現** | **pyro 實現** | M4/目標 | M4/pyro |
|---|---|---|---|---|---|---|
| 250 Hz | 0.648 | — | 0.500 | 0.499 | — | **1.001** |
| 500 Hz | 0.649 | 0.642 | 0.444 | 0.482 | 0.704 | 0.932 |
| 1000 Hz | 0.490 | 0.484 | 0.456 | 1.007 | **0.939** | **0.462** |
| 2000 Hz | 0.472 | 0.458 | 0.461 | 1.103 | **1.002** | **0.447** |

三件事同時成立：

1. **250 Hz 兩臂完全相同（1.001）**——共用低頻帶，這驗證了整套比較方法本身（場景真的
   配對、低頻帶真的相同）。
2. **M4 精準命中自己的 FDN 目標**（交越以上 0.939 / 1.002）。**FDN 沒有壞**，它做了被
   交代的事；目標由 `_target_rt60_s_by_hz` 取 `scene.predicted_octave_rt60_s()` 再做空氣
   吸收修正（[fdn.py:56](puresound/audio/rir/render/high_frequency/fdn.py:56)）。
3. **交越以上 M4 的尾巴只有 pyro 的 0.45 倍**，且 M4 的實現值在頻率上幾乎是平的
   （0.500/0.444/0.456/0.461）——因為 FDN 的目標是擴散場近似，**原理上無法回應吸收的
   空間分佈**。§6.6.2 那張表的 M4 欄（0.88→0.99 全平）就是這件事的直接證據。

#### 6.6.4 又挑錯統計量（第五次）

Pearson r：pyro **+0.730**、M4 **+0.626**。**只看 r 會得到「兩臂都捕捉到這個效應」的
錯誤結論。**

r 量的是**排序一致性**，不是**效應大小**。M4 的比值全距 0.88→0.99（13%），pyro 是
0.88→3.07（250%）——效應大小差 20 倍，r 幾乎一樣。

（第一個念頭是「M4 的 r 被離群值拉高」。**查了，不成立**：M4 比值 p5=0.79、p95=1.03、
max=1.08，沒有離群值。真正原因是 r 對小而一致的關係也給高分。連「為什麼統計量錯」都要
量，不能猜。）

#### 6.6.5 該修的是 `predicted_octave_rt60_s` 的第二個身份

它現在有兩個身份，其中一個站不住：

| 身份 | 狀態 |
|---|---|
| 元資料裡的房間描述 | 可以，但要標明是 Sabine 推估而非實測 |
| **M4 FDN 的晚場目標** | **不成立**——3/4 的房間不均勻到讓 Sabine 差 2× 以上 |

M4 的晚場因此在交越以上只有 pyro 實現值的 **0.45 倍**。

> **這一節下面的推論已被 §6.6.6 否證，保留原文以記錄推論是怎麼歪的。**
> 「M4 偏短」在**對照實測之後方向反了**：實測衰減隨頻率下降，M4 略偏平但很接近，
> pyro 才是離譜的那一個。所以上面那個 0.45 倍**不能讀成「M4 缺了 55%」**——它只是
> 「M4 是 pyro 的 0.45 倍」，而 pyro 本身高出實測兩倍以上。

**沒有證明 pyro 是對的。** 已證明的是：(a) Sabine 的前提在這批房間被違反、(b) pra 的
射線追蹤在方向與量級上都符合已知的 Sabine 失效機制、(c) FDN 完全無法回應。要斷定
pyro 的絕對正確性需要實測參照——**§6.6.6 補上了那個參照，結論是 pyro 更差。**

三個當時列的候選方向：

1. ~~**換 FDN 目標**——用射線/能量模擬的衰減取代 Sabine（pra 算得出來，可當 M4 的目標
   來源）~~ **已否證（§6.6.6）**：pra 的衰減比實測高兩倍以上，拿它當目標會把 pyro 的
   過衝搬進 M4。
2. **修材質取樣**讓吸收分佈不要這麼極端（`mean_α/wall_α` p50 = 6.10 很誇張）——這是改
   物理場景分佈，不是改渲染。仍然有效。
3. **兩臂都留但明確標註**各自的晚場語意——**已無必要**：§6.6.6 判定 M4 在擬真度上勝
   8 倍，不是「各有所長」。

順帶結論：**§6.5 提的「約束 `rt60_range`」應該延後。** 把 `predicted_octave_rt60_s`
夾進 (0.25, 0.8) 只會約束一個不代表實際衰減的數字——低頻帶偏快 0.62–0.79×、M4 高頻帶
與該預測值相當（0.94–1.00×）但整體形狀偏平、pyro 高頻帶偏長 2.2×。

#### 6.6.6 對照實測之後：§6.6.5 的言外之意被否證，pyro 才是差的那一個

**上面 §6.6.1–6.6.5 每一步都量對了，但我由它們暗示的結論是錯的。** 我寫「Sabine 是錯的
目標、pra 比較物理完整」——那一步沒有實測支撐，而現在有了，答案相反。

§4.4 判定實測語料不能當 **tilt** 的目標（喇叭響應烙在裡面）。但 **RT60 不受這個限制**：
喇叭的頻率響應改變各頻帶的**位準**，不改變頻帶**內部的衰減率**。而且只要每個 channel 用
自己的 500 Hz 值正規化，就得到一個**與房間族群無關**的量——衰減的頻率形狀。

先做 §4.4 教的檢查，**五個語料是否一致**：

| 語料 | ch | 250 Hz | 500 Hz | 1000 Hz | 2000 Hz | 2000/500 |
|---|---|---|---|---|---|---|
| ace | 1152 | 1.116 | 1.000 | 0.891 | 0.885 | FALLS |
| brudex | 1379 | 0.810 | 1.000 | 0.920 | 0.889 | FALLS |
| dech | 678 | 1.049 | 1.000 | 0.678 | 0.656 | FALLS |
| diffrir | 71 | 1.227 | 1.000 | 0.668 | 0.719 | FALLS |
| reverb | 1390 | 1.400 | 1.000 | 0.920 | 0.898 | FALLS |
| **pooled** | **4670** | **1.116** | **1.000** | **0.906** | **0.878** | |

**五個語料一致同意衰減隨頻率下降**（2000/500 全部 < 1，範圍 0.656–0.898）。語料間 sd
0.114，有離散但**方向無異議**——所以這個參照的**定性結論是鏈獨立的**。
（**250 Hz 那欄不能當目標**：語料間差 0.590，brudex 0.810 到 reverb 1.400。1k/2k 才一致。）

對照兩臂：

| | 250 Hz | 500 Hz | 1000 Hz | 2000 Hz | **與實測的平均絕對偏差** |
|---|---|---|---|---|---|
| 實測（1465 item / 5 語料） | 1.116 | 1.000 | 0.906 | 0.878 | — |
| **M4** | 1.114 | 1.000 | 1.017 | 1.089 | **0.081** |
| pyro | 1.012 | 1.000 | 1.994 | 2.199 | **0.628** |

**M4 的偏差是 pyro 的 1/8。** 真實房間衰減隨頻率下降，M4 大致持平（略偏高），
**pyro 上升到 2.2 倍——方向就錯了**，而且 2.199 是實測範圍上限（0.898）的 2.4 倍。

所以 §6.6.2 那張不均勻度表要重新解讀：**pra 確實捕捉到了 Sabine 失效那個機制，但嚴重
過衝**。真實房間即使吸收分佈不均，衰減仍隨頻率下降（空氣吸收與材質吸收上升共同作用）；
pra 衝到 +120%，實測是 −12%。「Sabine 不足」與「pra 正確」是兩件事，我把它們接在一起了。

**這是第六次同型錯誤，而且形狀不同**：前五次是量錯了東西；這次每個量測都對，錯在**由一連串
正確量測推出一個沒被量的結論**。三個前提（Sabine 前提被違反、pra 有反應、FDN 沒反應）全部
成立，卻推不出「pra 更接近真實」——那需要實測參照，而我當時明確知道自己沒有。
**應該當場停在「Sabine 不足」，不要多走那一步。**

判準結論：

| | M4 | pyro |
|---|---|---|
| 衰減形狀擬真度 | **0.081** | 0.628 |
| QC 良率（400 item） | **400/400** | 399/400 |
| 速度 | 慢 | **快** |

**要擬真度就用 M4。** pyro 的高頻殘響是實測的兩倍以上，拿它生訓練資料會系統性教模型
「高頻殘響很長」，而真實房間相反。

**速度不能用這次的數字下結論**：pyro 那趟是與 2000-item measured ingest 搶 CPU 跑的
（1.26 it/s 掉到 6.72 s/it），1310.7 s 是污染值；M4 獨跑 1082.4 s 是乾淨值。§7 表記
pyro 9 min / M4 25 min（約 2.7×）也與這次乾淨數字不合，**尚未重新驗證**。

兩條新線索（都未查）：

- **M4 為什麼還是太平**（2000/500 = 1.089，實測 0.656–0.898）。`_target_rt60_s_by_hz`
  **已經**做了空氣吸收修正，卻仍偏平——修正量不足或施加位置不對。後續用已有數字歸因：
  M4 在 2 kHz 命中目標（1.002）、在 500 Hz 只有 0.704，而 500 Hz 八度是**純低頻帶**——
  所以「偏平」多半是 §6.6.7 低頻帶偏快換座標的再現，歸併進那條，不另立案。
- **pyro 為什麼過衝**。曾懷疑 pra 的 `set_air_absorption()` 用它自己（較弱）的模型而非
  ISO 9613-1（兩臂確實用不同模型：pyro 走 pra 內建、M4 走我們的 ISO）。但用正確機制估
  （空氣吸收加在衰減**率**上）：2 kHz 約 3.4 dB/s，對 60 dB/s 的衰減率是 ~5% 的效應，
  **解釋不了 2.2×**；4–8 kHz 才放大。主因仍未知（scattering、ISM order、能量守恆方式都
  是候選），估算未經量測驗證。

**決策（2026-08-04）：M6 預設 backend 已切換為 path-events-m4。**

- 改的是 `generate_m6_bank.py --backend` 預設與 pilot script 的預設臂；依據即本節
  0.081 vs 0.628。README 兩份的表列與範例指令已同步。
- **速度依決定不重測**；pyro 保留為顯式選項與 A/B 對照臂，過衝成因列未解。
- **底層 `generate_hybrid_rir.py` 的預設沒動**（仍 pyroomacoustics）。三個既有 gate
  明文錨定那一層——M5 的 `production_default_remains_pyroomacoustics`（檢查預設 backend
  實例型別）、M4 coupling validator 的 `production_default_unchanged`（讀 fdn metadata）、
  以及 `test_m6_emission_is_opt_in_and_default_backend_is_unchanged`——而 M6 wrapper
  每次都顯式傳 `--high-backend`（[generate_m6_bank.py:186](egs/rir_generation/generate_m6_bank.py)）。
  改那一層等於改寫已結案里程碑的出口契約，不屬於這個決定的範圍。
- 新預設由新測試釘住：`test_m6_wrapper_defaults_to_the_higher_fidelity_backend`。
- fdn metadata 的 `opt_in: true / production_default_changed: false` 描述的是
  generate_hybrid_rir 那一層，仍為真；已加註解錨定語意，防止之後被「修正」成謊話或
  被誤讀成 M6 的狀態。

#### 6.6.7 還沒查的：低頻帶偏快

低頻帶在 250/500 Hz 實現 0.450/0.402 對預測 0.570/0.651（**0.79× / 0.62×**）。方向正確
但偏短，而它用的是 per-mode 表面參與阻尼（不是 Sabine），**與 §6.6.2 的成因無關，要分開
查**。兩臂共用低頻帶，所以這條同時影響兩者。

---

### 7. 效能實測

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
| 全套測試 | **666 passed** |

**只有低頻帶能上 GPU，高頻帶永遠 CPU。** 所以兩臂受益差很多：pyro 93% 的成本在低頻
帶，M4 的主成本是 path event 生成。

---

### 8. 重現

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

```bash
# §5：實測 RIR ingest。--limit-per-corpus 是跨房間取樣（不是取前 N 個，
# 否則抽到的會是同一個房間，room-disjoint split 會塌掉）。
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/ingest_measured_m6_variant.py \
  --source /work/any_exp_link/puresound_exp/real_rir_16k_train_view/items \
  --bank <out>/measured_bank --pruned-bank <out>/measured_pruned \
  --limit-per-corpus 60 --workers 8 --code-revision "$(git rev-parse --short HEAD)" \
  --report <out>/measured_ingest.json
# 盯兩個數字：brudex 的 shift p50 必須等於 fade_in_samples（4），
# 那是「onset 判準真的找到直達音」的校正標準；以及 audit=PASS。

# 把它接成 release，讓 real_native / mixed_calibrated_real 變 ready
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_variant_release.py \
  --source-bank <synthetic-qc-bank> --output-dir <out>/release \
  --measured-bank <out>/measured_pruned --mixed-synthetic-weight 0.5 --qc-workers 8
# 不給 --measured-bank 則兩條 recipe 維持 blocked，與過去行為完全相同。

# 對位與 recipe 的測試（19 tests；含 BRUDEX 不變量與前向掃描回歸守衛）
.venv/bin/python -m pytest -q test/test_rir_measured_ingest.py
```

```bash
# §6：M6.6 證據鏈，三趟。核准必須在 QC 之前，所以不能只跑一趟。
# 1) 評估：產 throughput + 第一趟 evaluation（核准的依據）
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py \
  --release <release_1> --evidence-root <ev> \
  --generation-audit <synth-bank>/rir_bank_generation_audit.json --pass evaluate

# 2) 核准：蓋章進 bank（會重跑 QC）→ 重建 release
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py \
  --release <release_1> --evidence-root <ev> --pass approve \
  --synthetic-bank <synth-bank> --measured-bank <measured_pruned> \
  --rebuild-release <release_2> --approver-id "<誰核准的>" --qc-workers 8

# 3) 見證：聽測 assignment、簽核、bundle、決策，全部綁到 release_2
#    有真人回應就加 --listening-responses <responses.jsonl>；沒有就是 contract_fixture
#    乾跑，controlled_listening_empirical_passed 會（正確地）維持 False。
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/build_m6_evidence.py \
  --release <release_2> --evidence-root <ev> --pass attest \
  --generation-audit <synth-bank>/rir_bank_generation_audit.json \
  --reviewer-id "<誰簽的>" --participants 24 --report <ev>/summary.json
# exit 0 = production_ready；exit 3 = 仍 blocked（會逐條印出 13 項檢查）

# 證據產生器與聽測的測試（26 tests）
.venv/bin/python -m pytest -q test/test_rir_m6_evidence.py
```

```bash
# §6.6：兩臂必須用同一個 seed 與同一個低頻帶，否則場景不配對、比較無效。
# 250 Hz 的 M4/pyro 比值應為 1.00（共用低頻帶）——那是這套比較方法自己的檢查。
for backend in pyroomacoustics path-events-m4; do
  PURESOUND_M6_PILOT_ROOMS=100 PURESOUND_M6_PILOT_WORKERS=16 \
  PURESOUND_M6_PILOT_QC_WORKERS=8 \
  PURESOUND_M6_PILOT_LOW_BACKEND=pytard-cupy-material \
  PURESOUND_M6_PILOT_GPU_DEVICES=0,1 \
  bash egs/rir_generation/phases/m6_bank/scripts/generate_m6_training_pilot.sh \
    "$backend" <pilot-root>
done
```

逐 octave 的 realized-vs-predicted 與吸收分佈不均的相關性，是用 bank 自己的
`predicted_octave_rt60_s`（scene metadata）、`octave_bands.t20_s`（QC report，篩
`decay_fit_r2.t20 >= 0.70`）、以及 `effective_boundary_materials()` 逐面積加權算出來的；
兩臂共用低頻帶這點讓 250 Hz 成為內建的方法學對照。**看效應大小（比值全距），不要看
Pearson r**——§6.6.4 說明了為什麼。

---

### 附註：fixture 與判準的選擇

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

---

## Physically Grounded RIR Realism Plan

Status: active  
Primary target: synthetic-to-real room acoustics for near/far speech training  
Secondary target: a reusable path toward microphone arrays, Ambisonics, and BRIR rendering

繁體中文的整體算法、物理解釋、驗證方法與目前結論見
[`docs/audio/rir_realism_algorithm_zh-TW.md`](docs/audio/rir_realism_algorithm_zh-TW.md)。

Experiment organization follows the milestone layout documented in
[`egs/rir_generation/phases/README.md`](egs/rir_generation/phases/README.md):
phase scripts and frozen evidence live under `phases/m0_baseline` through
`phases/m6_bank`, while `egs/rir_generation/` itself contains only stable
public commands.

### 1. Objective

PureSound already combines a low-frequency wave model with a high-frequency
geometric model. The next generation should make room-acoustic observables such
as RT60, DRR, early reflections, modal decay, and spatial coherence emerge from
the sampled scene rather than treating a broadband RT60 as the scene's primary
physical parameter.

The intended production model is:

```text
scene geometry + materials + environment + transducers
    -> low-frequency lossy wave response
    -> coherent direct and early path events
    -> spatial, multiband late field
    -> optional measured-room calibration / learned residual
    -> mono, array, Ambisonic, or binaural RIR
```

The project is not attempting a full-band brute-force wave solve for every
training sample. Wave, path, and statistical methods will be used where their
assumptions are valid and where their cost is justified.

### 2. Definition of success

No single waveform distance or RT60 number is sufficient. A release candidate
must pass all four gates below.

#### 2.1 Physical consistency

- Causal direct arrival consistent with source-receiver distance and sound speed.
- Stable relative gain when peak normalization is disabled.
- Reciprocity within the limits of source and receiver directivity.
- No unexplained energy gain at reflections or across solver crossovers.
- Smooth change of delay, energy, and direction across nearby positions.

#### 2.2 Acoustic distribution

Synthetic and measured banks are compared using at least:

- direct-to-reverberant ratio (DRR);
- EDT, T20, and T30 in octave bands;
- C50 and C80;
- spectral decay and coloration;
- low-frequency modal frequencies and Q;
- echo density;
- spatial/inter-channel coherence when applicable.

#### 2.3 Downstream synthetic-to-real performance

Every material generator change is evaluated with a fixed training recipe on
room-disjoint measured RIR test sets. A change is accepted only when it improves
the primary speech task consistently across multiple real-room subsets or when
it fixes a demonstrated physical defect without causing a statistically
meaningful regression.

#### 2.4 Operational quality

- Deterministic scene sampling for a fixed seed.
- Versioned metadata with complete provenance.
- Resume-safe dataset generation.
- Measured and generated RIRs share the same analysis and bank interfaces.
- Generation cost is reported together with acoustic quality.

### 3. Architecture principles

1. **Material first.** Sample geometry and frequency-dependent material
   properties; measure the realized decay afterwards. A requested RT range may
   constrain or reject scenes, but must not be imposed as a global envelope.
2. **Preserve causes.** Direct and early reflections should remain identifiable
   path events until the final RIR assembly.
3. **Separate acoustics from transducers.** Room propagation, source
   directivity, microphone response, HRTF, and device coloration are separate
   composable stages.
4. **Keep absolute and normalized modes distinct.** Dataset recipes may still
   request peak-normalized RIRs, but physically calibrated gain must remain
   available and recorded.
5. **Learn residuals, not basic physics.** Neural generation is introduced only
   after a measured baseline identifies error that the physical model cannot
   cover economically.
6. **Benchmark before replacing.** Every new backend runs beside the current
   hybrid generator until it passes the same benchmark.

### 4. Milestones

The estimates below are approximate single-engineer effort, not calendar
commitments.

#### M0 — Measurement and benchmark foundation (1–2 weeks)

Goal: turn the current generator into a frozen, reproducible baseline and make
the synthetic-to-real gap measurable.

Deliverables:

- a reusable `puresound.audio.rir_metrics` module;
- one analysis path for generated and measured bank WAVs;
- per-channel and per-distance-bucket JSON summaries;
- a frozen v0 generation configuration and seeds;
- a baseline report comparing synthetic, measured, and mixed training;
- tests using analytic decay signals and small fixture banks.

Initial metrics:

- direct sample and peak;
- DRR using the training pipeline's configurable direct window;
- C50 and C80;
- broadband EDT, T20, and T30 with fit quality;
- magnitude-response tilt;
- octave-band EDT/T20/T30/C50.

Exit gate:

- the benchmark is deterministic;
- invalid or insufficient decay fits are explicit rather than silently replaced;
- the largest measured/synthetic gaps can be ranked by metric and frequency
  band;
- one fixed downstream v0 result is recorded.

#### M1 — Material-first scene schema (completed 2026-07-30)

Goal: replace the single broadband room RT60 input with physical surface and
environment properties.

Introduce a versioned scene schema containing:

- named surfaces and meshes;
- octave or third-octave absorption, scattering, and transmission;
- optional complex surface impedance;
- correlated material families for walls, floor, ceiling, windows, doors, and
  furniture;
- temperature, humidity, pressure, and sound speed;
- source power, pose, and directivity identity;
- receiver pose, pattern, calibration, and array identity.

Implementation steps:

1. Add schema types and JSON serialization without changing the v0 generator.
2. Add a documented material catalog with provenance and uncertainty ranges.
3. Sample plausible material combinations by room type.
4. Add frequency-dependent materials to the high-frequency backend.
5. Record realized octave-band decay instead of copying a requested scalar RT60.
6. Add a calibrated output mode that does not normalize each item to a fixed
   peak.

Exit gate:

- material changes produce the expected band-dependent decay;
- metadata round-trips without loss;
- v1 and legacy banks remain readable;
- M0 shows an improvement over the broadband-material baseline.

Implementation:

- [x] Added `rir_scene.v2` types for named boundary meshes, material spectra,
  window/door patches, optional complex impedance, environment, poses, source
  power/directivity, receiver calibration, arrays, and interior objects.
- [x] Added lossless JSON serialization plus legacy `room_dim`, `mic_pos`,
  `source_pos`, `channel_map`, and material-derived `rt60` compatibility fields.
- [x] Added the documented `puresound-materials.v1` catalog and correlated
  office, meeting-room, classroom, and living-room sampling.
- [x] Passed per-boundary frequency-dependent absorption and scattering to
  Pyroomacoustics; temperature/humidity and scene sound speed now participate in
  rendering.
- [x] Separated material-predicted octave RT60 from realized broadband/octave
  metrics measured on the output RIR.
- [x] Added calibrated output that preserves source level, receiver gain,
  distance, and cross-room amplitude without per-item peak normalization.
- [x] Kept v0 as the CLI default and verified a generated v1 bank with the
  existing `PreGeneratedRoomBank` reader.

M1 acoustic probe:

- report: `egs/rir_generation/exp/rir_realism/m1/rir_benchmark_m1_probe100/acoustics.json`;
- 20 rooms × 5 positions = 100 v1 items/channels, seed 1337;
- analytic low-band probe backend, Pyroomacoustics high band, 16 kHz, 1.0 s;
- wall time: 156 s with four CPU workers (0.64 items/s on this host);
- measured comparison: 100 channels, seed 0.

| Distance | v0 tilt | M1 tilt | measured tilt | v0→M1 absolute-gap reduction |
|----------|---------|---------|---------------|------------------------------|
| 0–1 m | +1.12 | -0.25 | -2.51 | 3.63 → 2.26 dB/oct (38%) |
| 2–3.5 m | +2.50 | +0.44 | -3.32 | 5.82 → 3.76 dB/oct (35%) |
| 3.5–6 m | +3.31 | +0.62 | -2.52 | 5.83 → 3.14 dB/oct (46%) |

The comparison is directional rather than a downstream acceptance result
because v0 and M1 use different-sized samples and the probe uses the analytic
low backend. It nevertheless passes the M1 targeted acoustic gate: the
synthetic upward spectral tilt is reduced in every populated distance bucket.
The C50 gap did not improve (M1 medians 9.84/3.88/3.82 dB versus measured
17.16/9.31/6.02 dB), so early/late path energy remains an explicit next target
rather than being attributed to materials.

M1 audition acceptance:

- recipe: `egs/rir_generation/phases/m1_material/config/audition_m1.json`;
- artifacts: `egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1_bank` and
  `egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1`;
- 10 rooms × 5 positions = 50 items and 250 RIR channels;
- all WAV/metadata, finite-sample, calibrated-peak, tail-energy, and physical
  `distance/c` causality checks pass;
- the gate found and fixed Pyroomacoustics fractional-delay/ray-tail energy
  left at `t=0` after direct-path alignment; regenerated RIRs have exactly zero
  samples before each source's physical arrival;
- median near/far DRR gap is 10.63 dB and valid T20 coverage is 79.6% at
  `R² >= 0.9`;
- six RT60-quantile rooms have dry plus shared-gain `near_0`/`far_0` listening
  previews. This passes the local usability gate, not measured-room or M2
  acceptance.

#### M2 — Lossy low-frequency wave model (active since 2026-07-30)

Goal: make low-frequency modal decay emerge from boundaries instead of a shared
post-hoc exponential envelope.

The first implementation extends the current modal recurrence to damped modes:

```text
q_n'' + 2 * zeta_n * omega_n * q_n' + omega_n^2 * q_n = f_n
```

`zeta_n` is derived from modal boundary participation and the material loss at
the modal frequency.

Implementation steps:

1. Derive and test a discrete damped modal recurrence.
2. Compute per-mode loss from the six room surfaces.
3. Remove the global low-band RT60 envelope in the v1 path.
4. Validate modal frequency and Q against analytic cases.
5. Compare small scenes with a reference FDTD/FEM solver.
6. Decide from measured error whether a full impedance-boundary ARD backend is
   justified.

Exit gate:

- different modes can have different decay rates;
- changing one surface affects the expected modes and frequency bands;
- crossover energy remains bounded and causal;
- M0 low-band metrics improve without a downstream regression.

Current implementation status:

- [x] Derived an exact sampled damped recurrence that reduces to the previous
  pytARD recurrence when damping is zero.
- [x] Derived per-mode amplitude loss from six frequency-dependent surfaces and
  rigid-wall cosine-mode boundary participation.
- [x] Added explicit `pytard-material`, `pytard-cupy-material`, and
  `analytic-material` experimental backends.
- [x] Disabled the shared low-band RT60 envelope in material-modal mode and
  serialized modal frequency, decay rate, RT60, and Q.
- [x] Verified surface selectivity and absorbing-versus-reflective late-energy
  behavior with the exact pytARD recurrence.
- [x] Added an independent staggered pressure/velocity 3D FDTD reference with
  locally reacting impedance boundaries.
- [x] Validated the first two axial frequencies within 1% and their Q within
  20% of the impedance-boundary prediction; the estimator recovers an analytic
  damped-sinusoid Q within 2%.
- [x] Added a response-level low-frequency peak, bandwidth, and Q estimator plus
  a generated/measured bank comparison CLI.
- [x] Replaced the analytic probe's artificial rank amplitude/position phase
  with reciprocal eigenfunction source/receiver coupling and causal onset.
- [x] Replaced its fixed first-64 mode truncation with a serialized 256-mode cap,
  covering all 215 default index triplets and restoring 200–300 Hz coverage.
- [x] Ran development and zero-item-overlap measured modal comparisons; the
  corrected M1 bridge improves Q/spacing/bandwidth distributions, while the
  current material-modal loss and a scalar calibration both fail the joint
  distribution gate.
- [x] Established the phase-aware complex-impedance foundation in Pa·s/m:
  passive `Z <-> Gamma`, absorption-plus-explicit-phase conversion, schema
  interpolation/round-trip validation, and area-patch admittance mixing. No
  impedance is inferred when only absorption is known.
- [x] Added a positive-real first-order relaxation admittance, bilinear
  time-domain reflection filter, and per-wall-cell FDTD boundary state.
  Single-wall magnitude/phase, digital-pole stability, the exact static-boundary
  limit, and a nonzero-relaxation 3D case are covered by tests.
- [x] Added two provenance-bearing 100 mm glass-wool references using Tarnow's
  measured flow resistivity and the Miki phase-aware porous-layer model. Both
  pass the one-pole complex-reflection fit gate without extrapolating below
  their empirical validity range.
- [x] Demonstrated in a controlled FDTD room that a phase-aware fit moves the
  dominant peak from 85.7 Hz/Q 10.5 to 64.0 Hz/Q 15.8 versus a boundary with
  matched 80 Hz reflection magnitude but zero phase.
- [x] Added a boundary-admittance time-step gate after the first fitted prior
  exposed an edge/corner instability not covered by the interior CFL limit.
- [x] Added a strict normal-incidence complex-impedance CSV/JSON ingestion
  contract with phase, uncertainty, SI units, sample configuration, source,
  license, and passive-reflection validation.
- [x] Added a positive-real multi-pole admittance and fixed-real-pole bounded
  fit in complex pressure-reflection space. Non-negative static, low-pass, and
  high-pass parallel branches enforce passivity by construction.
- [x] Generalized the validation FDTD to one auxiliary state per wall cell and
  pole, including multi-pole metadata and the boundary time-step gate.
- [x] Connected the same rational reflection function to a minimal 1D complex
  cavity eigenproblem. The static case matches closed-form frequency/decay/Q,
  and a phase-aware boundary shifts the mode relative to a magnitude-matched
  real boundary.
- [x] Added the first licensed direct complex-impedance benchmark: CC BY 4.0
  normalized resistance/reactance for nominally identical NASA/UFSC
  perforated liners at no flow and 130 dB. It is explicitly validation-only
  because the grazing-duct, high-SPL configuration is not a room finish.
- [x] Added a passive series-RLC resonant admittance after the measured
  Helmholtz reactance crossing demonstrated that relaxation-only real poles
  fail. Alternating-frequency holdout, dense passivity, FDTD biquad state, and
  1D modal diagnostics pass.
- [x] Added a strict two-microphone impedance-tube acquisition path for the
  missing room-finish data: repeated complex H12, microphone-switch
  calibration, circular-tube and spacing validity gates, coherence,
  repeatability uncertainty, passive reduction, and uncertainty-weighted
  complex-reflection fitting.
- [ ] Obtain compatible normal-incidence room-finish measurements and map only
  matching installed configurations into the scene catalog.
- [x] Extend the accepted 1D rational-boundary formulation to a separable 3D
  nonlinear eigenproblem, connect it to an explicit experimental low-band
  renderer, and cross-check its first mode against independent FDTD.
- [ ] Validate nonlinear modal residues, ingest a compatible measured
  room-finish boundary, and only then rerun the corrected room-disjoint modal
  comparison as a production-candidate backend.
- [ ] Run the fixed downstream synthetic-to-real experiment before making the
  material-modal backend a default.

The implementation is usable for experiments but M2 has not passed its exit
gate. The default material-modal loss law is still the rejected first-order
Sabine surface-participation model. Complex phase now drives FDTD, 1D and
separable 3D nonlinear eigenvalues, and an explicit experimental renderer. The
first direct dataset validates this pipeline but is not a compatible room
material; normal-incidence room-finish data, nonlinear modal-residue
validation, angle dependence, and non-shoebox mode coupling remain required.

M2.5 direct-measurement gate:

- source: Zenodo `10.5281/zenodo.15195587`, CC BY 4.0;
- converted data: NASA and UFSC no-flow, 130 dB, KT normalized impedance,
  500–2500 Hz, with HDF5 dataset paths and checksum in each sidecar;
- report:
  `egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/impedance_zenodo_15195587_validation.json`;
- fitted NASA resonance: 1646.66 Hz, branch Q 11.69;
- held-out complex Cayley error: RMS 0.0385, maximum 0.0590;
- 4096-point passivity sweep: maximum magnitude 0.9178;
- NASA–UFSC cross-rig RMS difference: 0.1182;
- 1D phase-aware versus magnitude-only Q ratio: 2.94;
- decision: pass the pipeline-validation gate, reject automatic room-material
  catalog mapping.

M2.6 room-finish acquisition gate:

- second public-source audit found no machine-readable room-finish dataset
  satisfying phase, mounting, atmosphere/SPL, and license requirements;
- implemented
  `puresound.impedance_tube_transfer_measurement.v1` for repeated complex H12;
- microphone-switch calibration removes complex channel mismatch with a
  continuous square-root branch;
- selected bins must remain below the circular-tube transverse-mode cutoff,
  away from microphone-spacing singularities, and above the coherence gate;
- repeated installations produce real/imaginary impedance standard deviations,
  which are propagated to inverse-uncertainty reflection-domain fit weights;
- exact synthetic H12/channel-mismatch/impedance round-trip and CLI output
  contract are tested;
- decision: pass the acquisition-software gate; physical room-finish specimens
  and material-disjoint validation are still required before scene mapping.

M2.7 separable 3D impedance-mode gate:

- each wall receives an explicit passive rational admittance; no phase is
  inferred from the scene absorption catalog;
- one temporal pole jointly satisfies three complex Robin boundary
  characteristics and the 3D wave-equation dispersion relation;
- continuation from weak to full boundary strength tracks each requested
  rigid-wall mode branch;
- static x-only loss matches the existing exact 1D frequency, decay, and Q;
- a uniform cube preserves the three axial permutation degeneracies;
- the phase-aware 2.0 x 1.2 x 1.0 m glass-wool reference predicts
  63.8846 Hz / Q 17.32 versus independent FDTD 64.0049 Hz / Q 15.78;
- `analytic-impedance` is wired through the dataset generator with a strict
  six-wall JSON and validity-band check; a one-item full CLI smoke passes;
- eigenvalue and separable eigenfunction calculations pass, but the current
  RIR modal residue uses a documented engineering scale and is not yet
  production-validated;
- decision: pass the 3D eigenvalue/integration gate, keep production mapping
  and room-disjoint bank comparison blocked on real material and residue data.

M2.8 fixed-pole modal-residue gate:

- keep every M2.7 complex pole and eigenfunction fixed; fit only one global
  complex scale and one smooth frequency exponent;
- convolve the positive-pole quadrature bases with the exact FDTD Ricker
  source, use snapped cell centers and the pressure-cell volume convention;
- compare target and basis in the same 60–240 Hz validity band and only after
  source excitation, rather than fitting out-of-band FDTD energy;
- use two shoebox geometries with four training positions and two untouched
  position holdouts; each axial fundamental remains inside boundary validity;
- fitted `C = 0.444175 + j0.129146`, exponent `0.35`;
- train mean correlation / NRMSE / energy ratio:
  `0.9916 / 0.1299 / 0.9831`;
- position-holdout mean correlation / NRMSE / energy ratio:
  `0.9734 / 0.2304 / 1.0039`;
- the fitted-gain legacy `1/f` sine baseline reaches only
  `0.2467 / 0.9703 / 0.0913` on the same holdout;
- the versioned calibration is wired into `analytic-impedance`; a complete
  one-item generator smoke records `modal_residue_fdtd_validated: true`;
- decision: pass the controlled single-boundary numerical residue gate; keep
  `modal_residue_production_validated: false` until multiple installed
  boundaries and measured-room transfer functions pass.

M2.9 multi-boundary / room / grid residue gate:

- add a 50 mm rigid-backed installation-thickness variant using the same
  measured Tarnow flow resistivity and Miki model as the 100 mm reference;
  its one-pole fit has RMS/max complex-reflection error `0.00313 / 0.00506`;
- retain four training positions in rooms A/B, separate two position
  holdouts, and add an entirely unseen room C with two room holdouts;
- rerun one room-C geometry at 0.08 m after all training uses 0.12 m, creating
  an explicit grid holdout rather than silently changing numerical resolution;
- every position/room/grid split must reach correlation >=0.90, NRMSE <=0.35,
  and improve over the fitted-gain legacy `1/f` sine baseline;
- 50 mm position/room/grid correlation:
  `0.9904 / 0.9929 / 0.9961`; NRMSE: `0.1366 / 0.1187 / 0.0919`;
- 100 mm position/room/grid correlation:
  `0.9734 / 0.9933 / 0.9929`; NRMSE: `0.2304 / 0.1177 / 0.1193`;
- separate fits give `C50=0.639407+j0.060622, eta50=0.25` and
  `C100=0.444175+j0.129146, eta100=0.35`;
- a diagnostic shared fit across both variants also passes all three holdouts:
  position/room/grid correlation `0.9766 / 0.9882 / 0.9924`, NRMSE
  `0.2561 / 0.2161 / 0.1838`;
- the shared result is scoped only to two model-derived thickness variants of
  one measured flow resistivity; general boundary invariance remains false;
- a complete 50 mm generator smoke renders 123 calibrated modes and preserves
  `modal_residue_production_validated: false`;
- decision: pass numerical thickness, unseen-room and grid-transfer gates;
  next require a different directly measured room-finish material and a
  measured-room transfer function.

M2.10 source-convention and sample-rate gate:

- identify the FDTD calibration input as a pressure increment added to one
  finite-volume cell, rather than a digital source whose free-field RIR is
  `delta[n-D]/r`;
- derive and implement the free-field mapping
  `x = cell_volume/(dt*4*pi*c^2) * dq/dt`;
- convert each fitted pressure-state modal residue with
  `4*pi*c^2/(fs*pole)` before rendering it beside the `1/r` direct tap;
- version the fitted source, rendered RIR, and residue transform conventions;
  bump calibration output to v2 while retaining v1 read compatibility;
- matched-boundary direct-path validation passes axis-aligned 50/40 mm grids
  and a diagonal 30 mm grid: minimum correlation `0.9685`, maximum theoretical
  amplitude error `10.8%`, and zero best-fit sample offset;
- modal convolution identity passes at 4/8/32 kHz with maximum NRMSE
  `0.00591`; 60–240 Hz complex response across 4/8/16 kHz differs from the
  32 kHz reference by at most `4.07%`;
- decision: pass the FDTD-to-digital-RIR source-convention gate. The `1/r`
  direct tap and calibrated modal tail now share one input definition; absolute
  transducer pressure and low/high crossover level remain uncalibrated.

M2.11 complex crossover and gain-policy gate:

- find and remove the fixed 700–1300 Hz assumption: automatic RMS audit bands
  now track `[0.7*crossover, 1.3*crossover]`, while explicit bands must contain
  the actual crossover;
- demonstrate that branch-RMS equality is not a physical calibration rule:
  identical low/high inputs still produce a low gain near `0.9505` and up to
  `0.44 dB` flat-sum error;
- when the impedance backend reports a validated `1/r` source convention,
  bypass per-channel low-band energy rescaling and serialize the requested,
  applied, policy, effective band, and channel gains in item metadata;
- retain bounded RMS matching only as an explicitly reported bridge for
  uncalibrated legacy backends;
- validate fourth-order causal Linkwitz–Riley complex complementarity at
  8/16/48 kHz: maximum magnitude error `3.21e-12 dB` and branch phase
  difference `1.42e-12 degrees`;
- combine a low `1/r` direct tap with an anechoic Pyroomacoustics direct path
  at five distances. Across 60–1000 Hz, maximum magnitude/phase errors are
  `1.049 dB / 5.39 degrees`, passing `1.1 dB / 6 degrees`;
- decision: pass the digital-filter and anechoic direct crossover gate.
  Full-room modal/geometric phase continuity, boundary-reflection phase, and
  measured-room crossover remain open.

M2.12 full-room complex-crossover failure baseline:

- compare one train position, one position holdout, and one unseen-room
  holdout against the same pressure-cell FDTD transfer, without fitting gain;
- keep the 100 mm M2.10 boundary/residue pair and snapped FDTD source/receiver
  cell centers; source magnitude remains well conditioned across 168–300 Hz;
- enumerate exact shoebox image geometry through order 12 (2625 images) and
  test magnitude-only normal, complex normal, and locally reacting
  angle-aware complex reflection hypotheses;
- modal-only 60–240 Hz still passes the planned complex gate:
  correlation `0.934`, NRMSE `0.363`, energy ratio `0.919`;
- the 240 Hz magnitude-only full-room proxy fails all three cases:
  correlation `0.420`, NRMSE `4.285`, energy ratio `3.952`;
- the modal transfer itself becomes over-energetic near the upper calibration
  edge: 168–300 Hz raw/low-pass energy ratios are `5.84 / 3.67`;
- angle-aware complex reflection materially changes high-branch energy but
  does not make the hybrid pass;
- a 120/150/180/210/240 Hz scan selects 120 Hz plus angle-aware reflection as
  the least-bad diagnostic (`0.866 / 1.628 / 0.710` correlation/NRMSE/energy),
  still outside the gate;
- an 8 ms raised-cosine modal onset is the best smoothing value that preserves
  the original low-band gate, but crossover NRMSE remains `3.967`;
- decision: record the failure baseline. Do not fit a scalar, silently lower
  the crossover, refit residues, or ship onset smoothing. Move to shared
  fractional-delay `PathEvent` and coherent angle-aware early reflections,
  then rerun this exact protocol.

M3.1 versioned PathEvent and exact shoebox reference (completed 2026-07-31):

- add lossless `puresound.path_event.v1` and `path_event_set.v1` JSON contracts;
- keep physical delay separate from a complex pressure-gain spectrum so
  propagation phase cannot be counted twice;
- generate one direct and six exact first-order shoebox image paths with
  distance, delay, departure/arrival direction, surface, reflection point,
  incidence cosine, visibility, interaction policy, and directivity identity;
- preserve the project-wide free-field `1/r` RIR convention;
- store angle-aware locally reacting
  `Gamma(theta,f)=(cos(theta)-y(f))/(cos(theta)+y(f))` without pretending that
  arbitrary complex frequency samples are already a causal time-domain filter;
- add a causal one-sided third-order Lagrange renderer for real,
  frequency-independent path gains. Samples before the discrete arrival bin
  `floor(delay*fs)` are exactly zero;
- validate three geometries, including a near-grazing case: maximum geometry
  and reciprocity error `8.88e-16`, stable paths under a 1 mm source movement,
  and lossless schema round-trip;
- validate 8/16/48 kHz over 60–1000 Hz: worst fractional-delay magnitude/phase
  error `0.1001 dB / 0.5554 degrees`, zero samples before the arrival bin, and
  DC gain preserved to floating-point precision;
- decision: accept the M3.1 geometry and scalar fractional-delay gate. Do not
  render sampled complex spectra by raw IFFT. Next realize each boundary as a
  passive causal digital filter, integrate it into event rendering, and rerun
  the frozen M2.12 protocol.

M3.2 passive causal angle-filter realization (completed 2026-07-31):

- represent each digital normalized admittance as `Y(z)=B(z)/A(z)` and form
  the exact angle-aware digital Cayley transform
  `Gamma_theta(z)=(cos(theta)A(z)-B(z))/(cos(theta)A(z)+B(z))`;
- realize first-order, positive-real multi-pole, and passive resonant RLC
  admittances without raw complex-spectrum IFFT;
- serialize the stable causal filter as
  `puresound.digital_boundary_reflection_filter.v1`;
- require the supplied boundary model to reproduce every stored PathEvent
  complex-gain sample before rendering; reject mismatched models;
- cascade each causal boundary filter after the shared fractional propagation
  delay, retaining the physical arrival bin and the `1/r` convention;
- M2 reference filter at 8/16/48 kHz and five incidence cosines passes:
  worst digital-versus-analog error is `0.00279`, `0.0218 dB`, and
  `0.2397 degrees`; all tested rational models are stable and bounded-real;
- in all three frozen rooms, the rendered direct-plus-six-first-order response
  matches the identical analytic complex-angle path set with NRMSE at most
  `0.00385`, correlation at least `0.999998`, and energy ratio
  `0.99883–0.99953`;
- the same seven-path branch does not pass the full-room FDTD crossover gate:
  hybrid correlation/NRMSE/energy are `0.589 / 4.578 / 4.086`. This is not a
  filter-realization error; the early path set does not yet reproduce the
  full multiple-reflection/wave response;
- decision: accept the M3.2 causal filter gate while keeping the full-room
  gate open. Do not promote the seven-path diagnostic to the production high
  backend. Extend the coherent path set or connect a mesh tracer, then audit
  direct/early/late energy separately.

M3.3 ordered higher-order shoebox PathEvents (completed 2026-07-31):

- enumerate every integer image-source lattice point through an arbitrary
  Manhattan reflection order (currently bounded to 20), giving 7, 25, 129,
  833, and 2625 paths at orders 1, 2, 4, 8, and 12;
- fold the straight unfolded-room ray back into the physical shoebox and store
  its chronologically ordered surfaces, reflection points, incidence cosines,
  and interaction groups;
- preserve reciprocity and reproduce the existing same-order analytic complex
  image-source transfer to floating-point precision through order 12;
- cascade one passive causal angle filter for every ordered surface hit.
  Across the three frozen rooms, the order-12 time renderer matches the
  identical analytic path set with maximum NRMSE `0.01752`, minimum
  correlation `0.99991`, and energy ratio `0.99245–0.99900`;
- separate simultaneous edge/corner hits from ordinary face hits. Production
  geometry excludes them pending a physical diffraction model; an explicitly
  named diagnostic policy groups the coincident hits and multiplies legacy
  face filters only when comparison with the old analytic image product is
  required;
- order convergence is not monotonic in a coherent field: relative to order
  12, mean complex NRMSE is `9.551`, `8.093`, `3.300`, and `1.718` at orders
  1, 2, 4, and 8. More paths change both reinforcement and cancellation;
- order 12 still fails the full-room FDTD crossover gate. Its hybrid aggregate
  correlation/NRMSE/energy is `0.760 / 4.332 / 3.787`, effectively reproducing
  the existing analytic complex-angle failure rather than fixing it;
- decision: accept the M3.3 higher-order representation and renderer, reject
  “increase shoebox image order” as a standalone full-room fix, and do not
  replace the production high backend. Next audit direct/early/late complex
  energy against FDTD and evaluate a mesh engine with explicit visibility.

M3.4 direct/early/later coherent-error attribution (completed 2026-07-31):

- use the already validated free-field `1/r` PathEvent direct response as an
  explicit anchor. Do not claim that a time window can isolate direct sound:
  the 180 Hz validation pulse is wider than the 0.7–1.1 ms separation between
  direct and first-reflection arrivals in the frozen rooms;
- subtract the direct anchor from each FDTD and PathEvent output, then split
  the residual with exactly complementary 8 ms raised-cosine masks centered
  at direct plus 50 ms. Direct + early + later reconstructs the original
  response to numerical precision;
- retain a second, geometry-defined PathEvent partition by arrival time so
  boundary-filter tails remain attributable to the path that generated them;
- in 168–300 Hz, the early-reflection component alone has mean correlation
  `0.948`, NRMSE `0.331`, and transfer-norm ratio `0.999`, but its remaining
  complex error is `1.045` times the full FDTD transfer norm. Strong
  direct/early cancellation makes small phase errors consequential;
- the later component has a much larger component-relative NRMSE (`2.537`)
  but only `0.086` times the FDTD full-transfer norm. Its mean error is `0.224`
  times the full norm, far below the early-reflection contribution;
- coherent error accounting confirms that early and later error energies are
  not additive. Their squared self terms plus the explicit cross term equal
  the total squared error in every case;
- decision: accept the attribution protocol and localize the next model work
  to early-reflection phase and crossover branch interaction. Before adding
  more late rays or selecting a mesh engine, validate oblique single-wall FDTD
  reflection magnitude/phase against the same locally reacting model and
  audit the modal low-pass branch.

M3.5 staggered-grid oblique-boundary phase audit (completed 2026-07-31):

- derive the harmonic reflection coefficient of the implemented FDTD update,
  including the 3D discrete dispersion relation, pressure/velocity half-time
  staggering, and the half-cell distance between the wall face and first
  pressure cell;
- compare that coefficient with both the continuous locally reacting
  `Gamma(theta,f)` and the M3.2 PathEvent digital Cayley filter over
  168–300 Hz, five canonical incidence cosines, three tangential azimuths,
  all three wall axes, and representative directions from the actual
  order-12 early paths;
- the PathEvent filter remains close to the continuous model: actual-path
  worst complex/magnitude/phase errors are `0.00288`, `0.0218 dB`, and
  `0.243 degrees`;
- the current approximately 6 cm FDTD grid does not meet the continuous
  boundary parity gate. Canonical worst errors are `0.1238`, `0.475 dB`, and
  `7.68 degrees`; representative actual early paths reach `0.1665`,
  `0.656 dB`, and `9.75 degrees`;
- the discrete FDTD reflection remains passive, and its error decreases
  monotonically under linear refinement. At grid scales 1, 1/2, 1/4, and 1/8,
  worst phase error falls from `7.68` to `2.44`, `0.815`, and `0.303`
  degrees;
- decision: reject the coarse FDTD boundary phase as continuous ground truth.
  Do not fit the physical PathEvent reflection phase to this artifact. Next
  validate a face-pressure or phase-compensated boundary update in a
  time-domain plane-wave case, then audit the modal/geometric crossover.

M3.6 face-pressure and half-time correction experiment (completed 2026-07-31):

- add two opt-in FDTD boundary-pressure schemes while retaining
  `cell_center` as the default:
  `face_extrapolated` uses `1.5*p0-0.5*p1`, and
  `face_time_extrapolated` additionally predicts the next half step with
  `1.5*p[n]-0.5*p[n-1]`;
- extend the harmonic equation with the exact spatial extrapolation and causal
  time-predictor transfer functions;
- validate that equation independently with a 20 m 1D time-domain
  normal-incidence cross-ratio probe. Across all three schemes, worst
  measured-versus-predicted error is about `1.1e-4` complex and
  `0.0051 degrees`;
- both experimental schemes remain passive and materially reduce average
  actual-early-path error. Face plus time extrapolation reduces mean complex,
  magnitude, and phase errors from `0.0574 / 0.146 dB / 4.26 degrees` to
  `0.0224 / 0.0610 dB / 1.54 degrees`;
- neither scheme passes the worst-case continuous-reference gate. Actual
  grazing early paths still reach about `0.139` complex, `0.582 dB`, and
  `7.99 degrees`; even against a dispersion-matched target, the balanced
  face-time scheme reaches `0.0237` complex and `2.10 degrees`;
- decision: reject simple local extrapolation as a complete correction and do
  not change the production default. The remaining dominant error is the
  interior grid's angle-dependent characteristic admittance, which a local
  wall pressure predictor cannot remove. Evaluate a higher-order
  characteristic/dispersion-aware reference before rerunning full-room
  crossover.

M3.7 fourth-order staggered-grid harmonic candidate (completed 2026-07-31):

- derive the fourth-order staggered spatial symbol
  `D4/2 = sin(k*dx/2)/dx * (1 + sin(k*dx/2)^2/6)` and use it in both the
  three-dimensional dispersion relation and the normal characteristic
  admittance;
- evaluate a quadratic wall-face extrapolation
  `15/8*p0 - 5/4*p1 + 3/8*p2` and the same three-tap causal half-time
  predictor. These remove the quadratic interpolation error left by M3.6's
  two-tap linear predictors;
- compare four equations on the frozen approximately 6 cm grids: second-order
  cell center, second-order linear face/time, fourth-order linear face/time,
  and fourth-order quadratic face/time;
- fourth-order dispersion alone is insufficient: with the M3.6 linear
  predictor, actual-path worst error is `0.02372 / 0.0349 dB / 2.10 degrees`;
- the combined fourth-order/quadratic equation passes both canonical and
  actual-early continuous gates. Actual-path worst complex/magnitude/phase
  errors are `0.00750`, `0.0458 dB`, and `0.595 degrees`; corresponding means
  are `0.00229`, `0.0104 dB`, and `0.165 degrees`;
- the candidate remains passive over every scanned canonical and actual
  direction. Its fourth-order CFL number is at most `0.3927` on the frozen
  grids, below the unit stability bound;
- under joint grid/time refinement by factors 1, 1/2, 1/4, and 1/8, canonical
  worst complex error decreases `0.00750 -> 0.00106 -> 0.000267 ->
  0.0000669`, and worst phase error decreases `0.595 -> 0.0640 -> 0.0158 ->
  0.00396 degrees`;
- decision: accept exactly one *harmonic candidate*,
  `fourth_order_quadratic_face_time`, for time-domain prototyping. It is not
  yet an accepted FDTD reference: no fourth-order near-wall closure or
  time-domain cross-ratio has been implemented. Keep production/default
  `cell_center` unchanged and next validate a 1D time-domain prototype before
  modifying the 3D solver or rerunning the full-room crossover.

M3.8 fourth-order 1D near-wall time-domain closure (completed 2026-07-31):

- implement the M3.7 candidate in an independent 1D staggered time-domain
  prototype while leaving the production 3D solver unchanged;
- use the fourth-order centered interior derivative
  `9/8*(f[i]-f[i-1])-1/24*(f[i+1]-f[i-2])`;
- close the first pressure cell and first interior velocity face with the
  cubic-exact one-sided derivative weights
  `[-23/24, 7/8, 1/8, -1/24]`; mirror the weights at the rigid far wall;
- implement two samples of wall-face history for the quadratic spatial and
  causal half-time predictor selected in M3.7;
- validate the update with the same 20 m target/rigid/reference cross-ratio
  protocol. On the frozen base grid, time-domain versus harmonic worst
  complex/magnitude/phase errors are `0.000276`, `0.00184 dB`, and
  `0.00471 degrees`;
- jointly refine space/time to 1/2 and 1/4 scale. Worst errors remain
  `0.0000156 / 0.000108 dB / 0.000345 degrees` and
  `0.0000184 / 0.0000912 dB / 0.000716 degrees`; mean complex error decreases
  monotonically from `0.000104` to `0.00000616` and `0.000000963`;
- run a 1 s passive stability smoke: all samples are finite, the maximum
  absolute pressure is `2.575`, the one-dimensional fourth-order CFL number
  is `0.2267`, and tail/early RMS is `0.2367`;
- decision: accept the one-dimensional time-domain prototype and promote the
  candidate to an opt-in 3D implementation task. This is not yet an accepted
  3D boundary reference or an energy-stability proof; keep the production
  default unchanged and do not rerun full-room crossover until oblique 3D
  time-domain validation passes.

M3.9 opt-in fourth-order 3D FDTD reference (completed 2026-07-31):

- extend `FDTDReferenceConfig` with opt-in
  `spatial_derivative_order=4`,
  `near_wall_closure="third_order_one_sided"`, and
  `boundary_pressure_scheme="face_quadratic_time_quadratic"` while preserving
  the second-order `cell_center` defaults;
- implement fourth-order centered pressure-gradient and velocity-divergence
  updates on all three axes, mirrored one-sided closures at all six faces,
  and additive x/y/z divergence at edges and corners;
- retain two wall-face pressure histories per boundary and apply the M3.7
  quadratic spatial/half-time predictor independently at every face cell;
- discover that the bulk fourth-order CFL bound alone is insufficient for the
  tensor one-sided/quadratic closure: a long 3D run at effective CFL `0.39`
  excites a transverse numerical mode. Enforce and serialize the conservative
  opt-in closure cap `0.25`; the production second-order CFL path is
  unchanged;
- add validation-only spatial source/receiver weights, controlled source
  signals, and optional DC-removal bypass. A uniform 3D plane mode reduces to
  the independently implemented M3.8 1D update to `1e-12` sample tolerance;
- validate normal incidence with the 20 m broadband target/rigid/reference
  cross-ratio. Worst 168–300 Hz complex/magnitude/phase errors are
  `0.000294`, `0.00195 dB`, and `0.00520 degrees`;
- validate oblique time-domain reflection without an arbitrary wave-packet
  window. Solve the rigid tangential closure eigenproblem, drive steady
  harmonics at 270/285/300 Hz, and decompose incident/reflected amplitudes
  from two source-free pressure probes. Incidence cosine spans `0.250–0.490`;
  worst complex/magnitude/phase errors are `0.000608`, `0.00592 dB`, and
  `0.0253 degrees`;
- decision: accept the fourth-order/quadratic implementation as an opt-in 3D
  reference for the validated plane-mode scope. Do not change the production
  default or rerun the full-room crossover yet. Next require other normal
  axes, tangential azimuths/mode pairs, and edge/corner holdouts before using
  the reference to reopen crossover attribution.

M3.10 expanded fourth-order 3D holdouts (completed 2026-07-31):

- validate x/y/z normal axes, single tangential modes, and one non-equal
  two-tangential azimuth holdout at 285 Hz;
- worst plane-mode complex/phase errors are `0.000442 / 0.0268 degrees`,
  passing the frozen `0.02 / 1 degree` gates;
- raw face/edge/corner point-source reciprocity NRMSE reaches `0.637`, so the
  reference API reports it and returns the explicit bidirectional Green
  average `(Gsr + Grs) / 2`; reciprocalized NRMSE is exactly zero;
- retain the production second-order default. A stable centered/mimetic
  experimental closure was not promoted because its physical reflection
  phase error remained about 12 degrees;
- decision: accept the expanded opt-in 3D reference.

M3.11 expanded-reference crossover audit (completed 2026-07-31):

- rerun the three frozen train/position-holdout/room-holdout cases with the
  accepted fourth-order reciprocal reference and causal order-12 PathEvents;
- the 240 Hz full-room complex gate still fails; the magnitude-only production
  proxy has mean NRMSE/correlation/energy `4.567 / 0.399 / 4.217`;
- the best diagnostic is 120 Hz complex-angle with
  `1.622 / 0.869 / 0.702`, still outside the frozen gate;
- direct/early/later attribution passes its reconstructive protocol. Direct
  is exact; early-reflection NRMSE/correlation/energy is
  `0.315 / 0.953 / 1.004`, while later error has a much smaller norm relative
  to the full FDTD transfer;
- decision: complete the qualified audit and retain the explicit full-room
  complex limitation; do not fit a scalar or promote onset smoothing.

M3.12 existing mesh-engine evaluation (completed 2026-07-31):

- Pyroomacoustics 0.10.1 passes a non-convex 3D visibility/RIR smoke and
  remains useful as an independent cross-check and production default;
- its public contract does not expose ordered complex causal PathEvents,
  interior-solid furniture transmission, edge diffraction, or lossless event
  serialization;
- decision: keep PureSound PathEvents authoritative and implement only the
  scoped vertical-prism visibility required by M3, rather than creating a new
  general mesh tracer.

M3.13 furniture visibility geometry (completed 2026-07-31):

- validate serialized SceneObjects as closed vertical prisms with nonzero
  footprints, valid height/material coefficients, in-room placement, and
  transducers outside solids;
- test every source/interactions/receiver polyline segment against the prism
  in full 3D, preserving height-aware visibility and source-receiver
  reciprocity;
- serialize blocked-event to occluder IDs and keep invisible events
  inspectable rather than silently deleting their geometry;
- decision: accept height-aware furniture visibility.

M3.14 transmission, diffraction, and controlled scattering
(completed 2026-07-31):

- straight-through object transmission uses pressure gain
  `sqrt(product energy transmission)`;
- blocked direct paths receive up to two shortest visible vertical-edge
  detours using a bounded reference-frequency Fresnel-like coefficient;
- visible first-order reflections partition energy exactly into `1-s`
  specular plus four deterministic `s/N` scattering branches;
- add per-event energy partition, interaction model provenance, serialization,
  rendering, and per-path source-cardioid gain;
- the formal controlled scene passes visibility, transmission, diffraction,
  scattering-energy, reciprocity, serialization, and rendering gates;
- decision: accept the interactions and an opt-in
  `PathEventHighFrequencyBackend`; leave Pyroomacoustics as the default.

#### M3 — Coherent early-path representation and mesh backend
(completed 2026-07-31)

Goal: represent the direct path and early reflections as physically inspectable
events before rendering samples.

A `PathEvent` should carry:

- total distance and fractional delay;
- departure and arrival direction;
- ordered reflection/transmission surfaces;
- complex gain or filter per band;
- visibility, diffraction, and scattering information;
- source and receiver directivity gains.

Implementation steps:

1. [x] Define and serialize `PathEvent`.
2. [x] Generate exact shoebox direct/first-order paths as a reference.
3. [x] Render frequency-independent real gains with a shared causal
   fractional-delay filter.
4. [x] Realize angle-aware complex boundary gain as a passive causal digital
   filter and rerun the frozen M2.12 rooms as a first-order diagnostic.
5. [x] Extend coherent shoebox events through order 12, including ordered
   surfaces and identifiable direct/early/late path buckets.
6. [x] Attribute the frozen full-room complex error to direct-anchored early
   and later components with an exactly reconstructive decomposition.
7. [x] Derive and audit the implemented staggered-grid oblique boundary phase
   against continuous and PathEvent reflection models.
8. [x] Evaluate face-pressure and half-time corrections with a time-domain
   plane-wave probe; reject them as incomplete grazing-angle fixes.
9. [x] Select a fourth-order dispersion-aware harmonic boundary candidate on
   frozen canonical and actual-path directions.
10. [x] Implement and validate the fourth-order one-dimensional time-domain
   near-wall closure.
11. [x] Implement the opt-in 3D fourth-order reference and validate one
   normal-axis/oblique-tangential plane mode.
12. [x] Validate multi-axis, multi-azimuth, and edge/corner 3D holdouts.
13. [x] Audit low/high crossover branch interaction using a boundary reference
   that has passed the expanded oblique phase gate.
14. [x] Evaluate integration of an existing mesh acoustic engine before
   implementing a new tracer.
15. [x] Move furniture from post-processing into actual visibility geometry.
16. [x] Add edge diffraction, transmission, and controlled diffuse scattering.

Exit gate:

- path delays match geometry;
- nearby positions produce continuous paths;
- source-receiver swaps obey reciprocity where expected;
- early-reflection timing and C50 improve against measured rooms.

Exit result:

- maximum exact PathEvent distance/delay errors:
  `8.88e-16 m / 3.47e-18 s`;
- 1 mm position perturbations preserve event identities and change path
  distance by at most `0.9515` times the displacement;
- exact PathEvents, scoped scene interactions, and reciprocalized FDTD all
  pass their expected reciprocity gates;
- against 100 held-out measured channels, the paired complete-hybrid M3 probe
  reduces distance-bucket mean absolute median C50 gap by `38.8%`
  (`5.289 -> 3.239 dB`) and direct-excluded early-energy-centroid gap by
  `26.4%` (`3.046 -> 2.243 ms`);
- diagnostic dominant-peak timing regresses (`1.104 -> 2.740 ms`) and remains
  recorded as an M4 warning because coherent peak identity is discontinuous;
- decision: M3 exit accepted. Advance to M4 late-field density, multiband
  decay, and spatial output while retaining the failed 240 Hz full-room
  complex-crossover audit as an explicit limitation.

#### M4 — Late field, directivity, and spatial output (2–4 weeks)

Goal: produce a dense, frequency-dependent, spatially coherent late response.

Implementation decision:

- preserve causal PathEvents for the direct and early coherent response;
- render the dense tail with a multiband feedback delay network (FDN);
- parameterize FDN decay and spatial injection/output from material-derived
  octave decay plus path-traced directional-energy histograms.

The FDN is the primary dense-tail renderer because enumerating enough coherent
high-order paths to reach a diffuse tail is too expensive and retains
comb-like path regularity. Path tracing remains the parameter estimator, not a
second unbounded waveform renderer.

Implementation sequence:

1. **M4.1 measurement foundation — complete.** Add Abel-Huang normalized
   echo-density profiles, direct-relative mixing-time estimation, deterministic
   bank sampling, distance buckets, and measured-reference envelopes.
2. **M4.2 multiband/spatial target contract — complete.** Add octave-filtered
   echo density/decay, separate physical decay from measured noise floors,
   define target distributions by room class and distance, and freeze IACC plus
   diffuse-field coherence APIs without misusing source channels as receivers.
3. **M4.3 deterministic multiband FDN — complete.** Add room-scaled mutually
   incommensurate delays, an energy-preserving feedback matrix, passive
   frequency-dependent loop filters, deterministic seeding, and unit-energy
   diagnostics.
4. **M4.4 early/late coupling — complete.** Inject causal PathEvent energy into the FDN,
   crossfade around the estimated mixing time, preserve direct arrival and
   early phase, and calibrate octave decay without changing legacy mono output
   unless explicitly enabled.
5. **M4.5 spatial late field — complete.** Project one shared deterministic
   isotropic plane-wave late field to synchronized receiver arrays and
   ACN/SN3D first-order Ambisonics; validate coherence and IACC.
6. **M4.6 transducers — implementation complete.** Add real first-order source
   and receiver directivity plus an optional provenance-retaining
   HRTF-derived FIR decoder contract. The bundled audition decoder is
   explicitly analytic and is not mislabeled as measured HRTF data.

M4.1 baseline (seed `20260731`):

- 100 measured channels, 100 M1 channels, and 10 opt-in M3 channels;
- 20 ms Abel-Huang window, 1 ms hop, threshold `0.9`, and a documented 10 ms
  sustained-crossing robustness rule;
- measured/M1/M3 median mixing times: `16.0 / 21.0 / 20.5 ms`;
- all six M1 and all six M3 broadband monophonic median checks lie inside the
  measured central-80% envelopes.

That last result is a negative diagnostic, not an M4 pass: broadband
monophonic echo density alone does not distinguish the renderers. M4.2
therefore freezes octave-band and spatial/coherence gates before FDN tuning.
The report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_late_field_baseline.json`.

M4.2 baseline (seed `20260731`):

- 50 measured channels, 50 M1 channels, and the 10-channel opt-in M3 probe;
- nominal octave bands `125, 250, 500, 1000, 2000, 4000 Hz`;
- echo-density windows use at least four cycles at each lower octave edge;
- measured echo density is truncated at a reliable Lundeby intersection;
- M1 passes `31/48` and M3 passes `32/48` diagnostic median-envelope checks;
- M3 median mixing time at 2/4 kHz is `152/149 ms`, versus measured medians
  `24/24 ms` and measured p90 limits `70.2/88.2 ms`;
- M1 high-band density falls below the measured envelope later in the tail;
- measured channels require noise truncation in `16–38%` of bands, versus 0%
  for these deterministic synthetic banks;
- IACC/coherence is intentionally not computed because current bank WAV
  channels represent different sources at one receiver, not simultaneous
  receivers.

This is the first discriminative M4 target and directly motivates high-band
density injection in M4.3. The report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json`.

M4.3 isolated-core baseline (seed `20260731`):

- 16 distinct prime delay lengths from `53` to `283` samples
  (`3.31–17.69 ms` at 16 kHz), selected deterministically from the measured
  high-band mixing-time target;
- signed/permuted normalized Hadamard feedback with zero measured
  orthogonality error, unit-energy band weights, and strictly contractive
  per-band feedback operators;
- delay-proportional pressure loop gain
  `g[b,i] = 10^(-3 d[i] / (fs T60[b]))` for every measured octave target;
- all 8 structural checks pass: prime delays, orthogonality, contraction,
  unit band-weight energy, determinism, finite output, causal onset, and decay
  before the render end;
- qualified 500/1000/2000/4000 Hz T20 relative errors are
  `1.15 / 4.85 / 3.03 / 0.02%`; mixing times are
  `12 / 26 / 10 / 24 ms`; all qualified decay, mixing-time, and late-NED
  checks lie inside the frozen M4.2 requirements;
- 125/250 Hz remain diagnostic: their T20 errors are `21.72 / 15.57%` because
  the production hybrid low/modal branch and the parallel-octave crossover
  have not yet been coupled to this isolated core;
- spectral-flatness/ripple output is diagnostic only; metallic coloration
  still requires the M4 listening exit.

The M4.3 report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_fdn_report.json`; its audition impulse
is `egs/rir_generation/exp/rir_realism/m4/rir_m4_fdn_core/rir_m4_fdn_core.wav`. This does not change the
production hybrid generator or the default Pyroomacoustics backend. M4.4 adds
that coupling behind a separate explicit opt-in backend.

M4.4 coupled baseline (scene/FDN seed `20260731`):

- add the explicit `path-events-m4` high backend; `pyroomacoustics` remains the
  default and `path-events-m3` remains the unchanged coherent-path diagnostic;
- preserve PathEvent samples exactly through the transition start, use a
  complementary cosine/sine equal-power transition centered 24 ms after each
  physical direct arrival, and stop coherent-path injection after the 16 ms
  transition;
- drive each source channel's FDN with its own early PathEvent response and a
  stable room/source-derived seed; octave RT60 comes from the serialized
  material-first scene rather than the fixed M4.2 median;
- solve the positive quadratic gain root so finite post-transition energy is
  equal to the original PathEvent response, including the coherent/FDN cross
  term rather than matching RMS independently;
- pass all 8 structural gates: shape, same-seed determinism, finite output,
  causality, exact early preservation, post-transition energy preservation,
  distinct channel seeds, and unchanged production default;
- material-target T20 relative errors at 500/1000/2000/4000 Hz are
  `2.83 / 5.32 / 0.69 / 3.47%`; M4 median mixing times are
  `22 / 26 / 30 / 34 ms`, all inside the frozen M4.2 measured envelopes;
- late normalized echo density changes from M3
  `0.560 / 0.000 / 0.000 / 0.000` to M4
  `1.041 / 1.109 / 0.992 / 1.001`, improving the distance to each measured
  median;
- the complete modal/high crossover is sample-exact before the transition;
  median/max absolute C50 changes are at most `1/3 dB`, and early-energy
  centroid changes are at most `1/2.5 ms` under the formal gates.

The report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_path_event_fdn_coupling_report.json`; paired M3
and M4 high/full-hybrid WAVs are under `egs/rir_generation/exp/rir_realism/m4/rir_m4_coupling/`. This is one
deterministic material-first fixture, not a measured-room or coloration exit.
M4.5 owns multi-receiver output matrices, coherence, and IACC.

M4.5 spatial baseline (scene/spatial seed `20260731`):

- one source is rendered to a synchronized 17 cm two-receiver array and to
  ACN/SN3D first-order Ambisonics (`W/Y/Z/X`) from the same plane-wave field;
- 256 seeded, rotated Fibonacci-sphere directions carry independent
  octave-band noise shaped by causal RMS envelopes from one passive multiband
  FDN; receiver delays are physical fractional propagation delays;
- all 12 structural gates pass, including exact determinism, causal onset,
  exact early preservation, per-channel post-transition energy, and unchanged
  production default;
- all 5 spatial gates pass. The bin-weighted qualified 500 Hz–4 kHz
  complex-coherence RMSE is `0.187197`, late IACC_L4 is `0.323144`, and every
  reported octave-coherence error remains inside the frozen model-derived
  tolerance. Full-spectrum RMSE `0.216982`, which includes frequencies outside
  the qualified M4 FDN bands, remains a diagnostic rather than an exit gate.

The report is `egs/rir_generation/phases/m4_spatial_late_field/reports/m4_spatial_rir_report.json`; array and
Ambisonic artifacts are under `egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/`.

M4.6 decoder baseline:

- source and receiver PathEvents support omni, cardioid, hypercardioid, and
  figure-eight real first-order pressure patterns;
- `AmbisonicBinauralDecoder` accepts causal FIRs with shape `[2, 4, taps]` and
  retains sample rate, reference identity, decoder kind, and provenance;
- all 11 decoder contract, causality, determinism, distinct-ear, and IACC
  analysis checks pass;
- the audition BRIR uses an analytic headless decoder and is explicitly not a
  measured HRTF. A multi-tap fixture proves external FIR injection but is also
  not labeled as measurement evidence.

The report is `egs/rir_generation/phases/m4_spatial_late_field/reports/m4_binaural_brir_report.json`; the
analytic audition artifact is `egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_analytic_brir_2ch.wav`.

M4 completion is deliberately reported at two evidence levels:

- **implementation exit: passed.** M4.1 through M4.6 and their computational
  artifacts are complete;
- **empirical/production exit: open.** Measured synchronized multi-receiver
  validation, a licensed calibrated HRTF decoder, and controlled listening for
  metallic coloration/spatial plausibility are still required.

This split is serialized by
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_exit_report.json`. Pyroomacoustics remains the
default; the M4 spatial renderer is an explicit opt-in API/CLI.

Add:

- room-dependent mixing time;
- band-dependent echo density and decay;
- talker/loudspeaker spherical-harmonic directivity;
- omni, cardioid, and device-mounted microphone models;
- microphone arrays and Ambisonic output;
- optional HRTF rendering to BRIR.

Exit gate:

- late tails do not exhibit obvious metallic coloration;
- octave decay and echo density match the target distributions;
- array coherence and IACC are plausible;
- mono output remains backward compatible.

#### M5 — Measured-room inverse calibration and learned residual (implementation complete; empirical exit open)

Goal: close the remaining measured/synthetic gap using sparse, controlled real
measurements.

Measurement protocol:

- repeated exponential sine sweeps;
- calibrated loudspeaker and microphone response;
- source/receiver position and orientation;
- room geometry or coarse mesh;
- temperature and humidity;
- retained raw sweeps, deconvolution configuration, and noise estimates.

For each representative room, begin with roughly 12–30 source-receiver
measurements. Fit material, scattering, directivity, and late-field parameters
with a differentiable approximate renderer. Only then train a residual model.

Residual losses should include:

- multiresolution STFT distance;
- energy-decay curve distance;
- direct/early arrival timing;
- octave-band acoustic metrics;
- spatial coherence;
- causality and decay regularization.

Exit gate:

- held-out positions in calibrated rooms improve;
- held-out rooms improve rather than merely being memorized;
- the physical and learned contributions can be ablated independently;
- generated output remains physically valid under interpolation.

Implementation sequence:

1. **M5.1 — measurement and loss contract.** Freeze the controlled-room
   campaign schema, room-disjoint split semantics, retained acquisition assets,
   and an independently auditable reference loss.
2. **M5.2 — synthetic recovery.** Hide parameters from synthetic scenes, fit
   them back from their RIRs, and reject parameters that are not identifiable
   even before real measurement uncertainty is introduced.
3. **M5.3 — measured-room fit.** Fit train positions in controlled rooms and
   report held-out positions separately from held-out physical rooms.
4. **M5.4 — joint spatial calibration.** Add scattering, directivity, and
   multiband late-field parameters only where synchronized receivers provide
   enough evidence.
5. **M5.5 — constrained learned residual.** Learn only the remaining error,
   with explicit causality/decay constraints and physical-only,
   residual-only, and combined ablations.
6. **M5.6 — exit decision.** Run held-out-room acoustic, listening, and
   downstream gates before enabling the calibrated renderer for production.

M5.1 implementation result (2026-08-01):

- [x] Added strict `puresound.rir_measurement_campaign.v1` types for room
  geometry, calibrated transducers, pose uncertainty, environment, repeated
  ESS captures, raw/noise/inverse/deconvolved assets, SHA-256 provenance, and
  room-disjoint train/validation/test assignment.
- [x] Added a campaign audit that verifies retained assets and hashes, 12 or
  more measurements per room, repeated sweeps, and at least one synchronized
  spatial capture.
- [x] Added `puresound.rir_calibration_loss.v1`, combining multiresolution
  STFT, direct-relative energy decay, arrival timing, octave energy/T20,
  complex spatial coherence, causality, and late-decay regularization. The
  NumPy/SciPy implementation is a reference/evaluation contract, not an
  autograd optimizer.
- [x] Added a structurally valid template and deterministic contract validator.
  All M5.1 implementation probes pass.
- [x] Audited 64 metadata items from each existing measured train/held-out
  bank. They remain useful acoustic references, but contain none of the nine
  controlled acquisition evidence groups required by M5.1; measured inverse
  fitting therefore remains open rather than being falsely claimed.

Evidence:

- report: `egs/rir_generation/phases/m5_calibration/reports/m5_measurement_contract_report.json`;
- non-evidence template:
  `egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json`;
- protocol and field semantics:
  `docs/audio/rir_measurement_campaign_zh-TW.md`.

M5.1 implementation exit is **PASS**. Controlled-measurement readiness is
**OPEN**. M5.2 may proceed with synthetic recovery without claiming a
measured-room fit; M5.3 cannot close until a real controlled campaign satisfies
the contract.

M5.2 synthetic-recovery baseline (2026-08-01):

- [x] Added a causal approximate renderer that keeps geometry/direct response
  fixed and exposes shared mixing time, coherent early-reflection gain, four
  octave RT60 values, and four octave late gains under explicit box bounds.
- [x] Fit the ten hidden parameters jointly from three synthetic source/receiver
  positions with bounded nonlinear least squares and three deliberately distant
  initializations.
- [x] All starts converged to the same noise-free ground truth; maximum parameter
  spread was `2.35e-12`. Scaled-Jacobian condition number was `17.22`, with full
  local column rank.
- [x] The independent M5.1 oracle mean loss on two unseen positions fell from
  `2.26933` to `1.69e-14` (greater than 99% reduction), while every output
  remained causal.
- [x] Added target/initial/recovered holdout WAVs and a strict report at
  `egs/rir_generation/phases/m5_calibration/reports/m5_synthetic_recovery_report.json`.

This is intentionally an **inverse-crime baseline**: the same noise-free
approximate renderer family creates and fits the targets. It proves parameter
ordering, bounds, multi-position fitting, multi-start convergence, local
sensitivity, holdout evaluation, and independent loss plumbing. It does not
prove robustness to measurement noise/model mismatch, invert the complete M4
renderer, or fit a real room. Those limitations are serialized in the report.

M5.2b robust synthetic recovery (2026-08-01):

- [x] Added deterministic measurement perturbations spanning 32–38 dB SNR,
  per-position calibrated gain error up to 1.2 dB, latency offsets from -6 to
  +11 samples, unmodelled early paths, and an independent 1.25x-decay late
  component.
- [x] Retained raw perturbed RIRs and applied only the known gain/latency
  correction before fitting. Noise and renderer mismatch remain in the target.
- [x] Added a noise-aware smooth M4-proxy objective with waveform, coherent
  early, broadband decay, and octave decay residuals. It is optimized with
  SciPy finite differences and is explicitly not an autograd or complete-M4
  inverse renderer.
- [x] Recovered mixing time within `0.0235 ms`, early gain within `0.0214 dB`,
  all octave RT60 values within `1.43%`, and all late gains within `0.106 dB`.
- [x] Two distant starts converged within `1.03e-6`; the scaled Jacobian
  condition number is `8.12` and remains full rank locally.
- [x] On two separately perturbed held-out positions, the independent M5.1
  total fell `61.4%` from the initial model. Octave error was `0.00846` versus
  `0.03341` for the waveform-only ablation; maximum RT60 and late-gain errors
  also improved.
- [x] Recorded that the global absolute peak was not the direct arrival in two
  of five noisy/model-mismatched cases. The controlled protocol must use the
  geometry-bounded arrival window or a validated onset detector.

The 15/15 M5.2b gates pass. Evidence and six holdout RIR artifacts are in
`egs/rir_generation/phases/m5_calibration/reports/m5_robust_recovery_report.json` and
`egs/rir_generation/exp/rir_realism/m5/rir_m5_robust_recovery/`. M5.3 remains blocked on a real controlled
campaign.

M5.2c actual M4 parameter-profile mapping (2026-08-01):

- [x] Connected the inverse fit to the real M4
  `PathEvent -> equal-power transition -> multiband FDN` renderer rather than
  the smooth proxy. Pyroomacoustics remains the unchanged production default.
- [x] Treated mixing time as a discrete outer profile because it changes prime
  FDN delay lengths discontinuously; fitted coherent-reflection gain and three
  octave RT60 targets in the bounded continuous inner solve.
- [x] Used two order-4 PathEvent training positions and two unseen positions.
  Targets retain 8% alternate-FDN-topology mismatch and 42 dB SNR noise.
- [x] Selected the hidden `24 ms` mixing profile over `20/28 ms`; the best to
  second-best cost ratio is `0.0820`. Every inner solve converged with locally
  full-rank Jacobian; best scaled condition number is `3.14`.
- [x] Recovered coherent-reflection gain within `0.661 dB`; octave RT60 errors
  are `0.392%`, `1.86%`, and `4.23%`. The independent held-out M5.1 total fell
  `71.7%`, while physical-arrival causality and exact pre-transition
  preservation remained intact.
- [x] Added a strict 14-gate report and target/initial/recovered holdout WAVs at
  `egs/rir_generation/phases/m5_calibration/reports/m5_m4_parameter_mapping_report.json` and
  `egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping/`.

M5.2c is the first mapping to the actual M4 renderer, but its
`coherent_reflection_gain_db` is still an aggregate proxy. It does not identify
individual wall absorption/scattering, prove global identifiability outside the
supplied mixing grid, or complete measured-room fitting. The next parallel
model task is a grouped material/path identifiability ablation; M5.3 remains
blocked on a qualifying controlled campaign.

M5.2d grouped material/path identifiability (2026-08-01):

- [x] Reparameterized ordered PathEvents by six effective boundary-reflection
  adjustments, with one dB pressure adjustment applied per boundary hit before
  the actual M4 early/FDN coupling.
- [x] All six boundary groups are locally full rank on three order-4 training
  positions: normalized condition number `2.74`, maximum column correlation
  `0.540`, maximum recovery error `0.00123 dB`, and two-start spread
  `4.21e-10 dB` at 48 dB SNR.
- [x] The held-out M5.1 total fell `64.8%`.
- [x] Explicitly duplicated each sensitivity into absorption-loss and
  specular-scattering-loss columns. Rank stayed `6/12`; all six scattering
  duplicates were rejected. Mono coherent RIRs identify only effective
  reflection loss, so scattering is deferred to synchronized M5.4 evidence.

M5.3 measured runner implementation (2026-08-01):

- [x] Added retained-asset/hash readiness, deterministic SHA-256 position
  fit/holdout assignment, per-train-room M4/profile fitting, held-out-position
  reports, and separate validation/test physical-room reports.
- [x] Ran the complete runner on an explicitly non-evidence synthetic campaign
  with train/validation/test rooms and synchronized receivers; all eight runner
  implementation gates pass.
- [x] Ran the same CLI against the current campaign template. It exits blocked
  before fitting because assets/hashes and acquisition evidence are absent.
- [x] Defined what `all_train_room_m4_profiles_converged` asserts
  (2026-08-03, `puresound.m4_profile_convergence.stable_minimum.v1`). The M6
  excitation fix moved the optimization landscape: the fit now reaches a
  *lower* cost (0.0333 against the frozen 0.0404) while no longer tripping
  `ftol`. Measured on the fixture, the point is stationary in every direction
  that matters — no coordinate step of 1e-3, 1e-2 or 1e-1 of the bound span
  lowers the cost — while the reported first-order optimality reads 6.2e-2
  because the objective is locally rough. `success` and `optimality` are
  therefore both wrong criteria. Convergence is now decided by restarting the
  solve from its own answer, which resets the collapsed trust region: the
  point counts as a minimum only if the restart cannot lower the cost by more
  than 1e-3 relative. A deliberately truncated fit is still rejected, and both
  solves are recorded in the report rather than a single boolean.
- [ ] Execute the runner on a real controlled repeated-ESS campaign. Current
  legacy banks remain ineligible (`0/64` for all nine required evidence groups).

M5.4 synchronized spatial calibration implementation (2026-08-01):

- [x] Added a fail-closed candidate profiler that requires at least two
  synchronized receivers and evaluates scattering/directivity/late-field
  candidates with the M5.1 spatial-aware loss.
- [x] The actual M4 four-candidate fixture selects the hidden
  scattering-plus-cardioid candidate; 11/11 gates pass. Mono input is rejected.
- [x] Recorded that current scattering changes coherent early PathEvents, while
  the post-80-ms FDN field is scattering-independent. Scattering is therefore
  selected by synchronized early/spectral/octave evidence, not falsely by an
  identical late-coherence term.
- [ ] Repeat on measured synchronized arrays.

M5.5 constrained residual implementation (2026-08-01):

- [x] Added a direct-relative shared residual fitted only from physical-model
  error. It has zero pre-direct support, a 50-ms-onward block decay ceiling, and
  an explicit residual/physical energy budget.
- [x] Added physical-only, residual-only, and combined ablations plus an
  interpolated-output causality check.
- [x] On a third, room-disjoint synthetic room, combined M5.1 total fell
  `67.9%` from physical-only; all 12 structural/ablation gates pass.
- [ ] Train/evaluate the residual on real controlled train/held-out rooms and
  run the fixed downstream speech task.

M5.6 aggregate exit (2026-08-01):

- [x] Added `puresound.m5_exit.v1`, nine implementation-stage gates, seven
  required artifact checks, and invariants preventing false measured claims.
- [x] **M5 implementation exit: PASS.** M5.1–M5.6 code, validators, synthetic
  fixtures, fail-closed measured runner, and ablations are complete.
- [ ] **M5 empirical/production exit: OPEN.** A qualifying controlled campaign,
  measured held-out positions/rooms, synchronized spatial calibration,
  measured residual training, controlled listening, and room-disjoint
  downstream evidence are still missing. Production remains disabled.

Evidence is aggregated in
`egs/rir_generation/phases/m5_calibration/reports/m5_exit_report.json`. This is the terminal M5
implementation result; the remaining work is external empirical evidence, not
an unimplemented optimizer path.

#### M6 — Production RIR bank v2 (M6.1–M6.6 implementation complete; promotion blocked)

Goal: package the accepted simulator into reproducible training banks.

Required outputs:

- room-disjoint train, validation, and test manifests;
- scene and renderer version hashes;
- material and room-type distributions;
- calibrated and normalized RIR variants where required;
- quality-control metrics per item;
- generation throughput and failure statistics;
- a final synthetic-only, mixed, and real-RIR downstream comparison.

Implementation sequence:

1. **M6.1 — bank contract.** Freeze deterministic acoustic-space splits,
   content hashes, generator/renderer provenance, signal variants, QC state,
   and fail-closed release semantics.
2. **M6.2 — reproducible generator integration.** Materialize the task plan,
   generate/resume by split, and emit the M6 manifest from actual outputs.
3. **M6.3 — per-item QC and quarantine.** Measure causality, clipping, DRR,
   clarity, decay, spectral, density, and spatial contracts before admission.
4. **M6.4 — distribution and variant release.** Publish available synthetic,
   mixed, and real variants with frozen distributions; unavailable measured or
   mixed recipes remain explicitly blocked rather than being fabricated.
5. **M6.5 — bank-level and downstream evaluation.** Compare measured acoustic
   distributions, throughput/failures, listening, and room-disjoint tasks.
6. **M6.6 — production decision.** Promote only content-addressed evidence-
   backed profiles/items; otherwise remain candidate or draft.

M6.1 bank-contract result (2026-08-01):

- [x] Added `puresound.rir_bank.v2` with canonical manifest SHA-256, per-asset
  WAV/metadata/scene hashes, audio shapes, signal/level variants, and QC state.
- [x] Added SHA-256 acoustic-space assignment and independent room/acoustic-
  space leakage checks across train/validation/test.
- [x] Added generator/config/code and renderer/backend/scene/evidence
  provenance. A development profile cannot claim a production release.
- [x] The deterministic non-evidence fixture passes 12/12 contract gates,
  including asset/manifest tamper, unsafe-path, leakage, false-production, and
  legacy-reader controls.
- [x] Integrate the contract into the actual parallel/resumable generator
  (M6.2); current fixture is not a production or acoustic-quality bank.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_bank_contract_report.json`;
- fixture manifest: `egs/rir_generation/exp/rir_realism/m6/rir_m6_bank_contract/rir_bank_manifest.json`;
- Chinese contract: `docs/audio/rir_bank_v2_zh-TW.md`.

M6.2 reproducible-generator result (hardened 2026-08-02):

- [x] Added opt-in `--emit-m6-manifest` integration to the real
  `generate_hybrid_rir.py`; the legacy path and Pyroomacoustics default remain
  unchanged.
- [x] Materialize and hash the complete task plan before rendering. Each item
  has a stable task seed, acoustic-space split, scene hash, renderer profile,
  expected audio shape, and generation-config hash.
- [x] Emit content-addressed train/validation/test JSONL indexes, the v2
  manifest, and a strict post-generation audit from actual WAV/JSON outputs.
- [x] Require an explicit split when `PreGeneratedRoomBank` reads an M6 root;
  unsplit access fails closed while legacy banks retain their old behavior.
- [x] Resume now verifies task/config/code-revision identity, scene hash, WAV
  content hash, audio header, and records critical runtime package versions. A
  corrupted item is regenerated without touching valid items; changing the
  generation config or code revision invalidates incompatible outputs.
- [x] Pyroomacoustics generation seeds both NumPy and libroom RNGs. The 17-gate
  fixture uses the actual default Pyroomacoustics high backend and produces
  identical serial, two-worker, and independent fresh-run manifest/item/task-
  plan hashes. After one WAV is tampered, resume regenerates exactly `1/6` and
  restores the original manifest; revision/config changes are fail-closed.
- [x] Add M6.3 per-item acoustic QC and quarantine before any candidate bank is
  admitted or promoted.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_reproducible_generation_report.json`;
- hardened validation outputs: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/reproducible/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_reproducible_generation.py`.

M6.3 per-item QC/quarantine result (hardened 2026-08-02):

- [x] Added a content-addressed `puresound.rir_bank_qc.physical.v1` policy and
  deterministic reports for structural integrity, causality/direct timing,
  tail energy, DRR, C50/C80, noise-aware decay, spectral tilt, octave bands,
  and Abel echo density.
- [x] Separate `fail`, `not_evaluable`, and `not_applicable`. Source-indexed
  5-channel RIRs cannot falsely claim synchronized-array IACC/coherence.
- [x] Publish pass-only train/validation/test candidate indexes and a quarantine
  index with per-item report path/hash and explicit reasons. Original WAV and
  metadata assets remain unchanged.
- [x] `PreGeneratedRoomBank` excludes failed items by default; candidate and
  production manifests admit only QC PASS items. Debug inclusion is explicit.
- [x] If quarantine empties any split, release remains `draft`. A complete
  three-split QC bank becomes `candidate`, never `production` from QC alone.
- [x] The actual M6.2 generator fixture passes `6/6`; silent, pre-arrival,
  sparse-late-field, and deliberately late-direct-arrival negative controls
  are quarantined. The formal validator passes 13/13 gates, including active
  octave-decay coverage, deterministic hashes, and QC-report tamper.
- [x] Start M6.4 distribution and variant release; publish the two valid
  synthetic recipes and fail closed for missing measured/mixed variants.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_item_qc_report.json`;
- hardened outputs and negative controls: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/item_qc/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_item_qc.py`.

M6.4 distribution/variant-release result (2026-08-01):

- [x] Added `puresound.rir_bank_release.v1`, content-addressed acoustic
  distribution snapshots, variant lineage, and train/validation/test recipe
  indexes.
- [x] Published `synthetic_calibrated` and a deterministic
  `synthetic_peak_normalized` variant. The latter applies one common gain per
  item to peak `0.98`; DRR, C50/C80, T20, acoustic-space identity, and split
  remain invariant within numerical tolerance.
- [x] Added `PreGeneratedReleaseBank(recipe_id=..., split=...)`. It audits the
  release, enforces recipe origin weights, and propagates release/recipe/
  variant identity into sample metadata.
- [x] Canonicalized the non-acoustic libsndfile float-WAV `PEAK` timestamp so
  repeated builds are byte-identical instead of changing once per wall-clock
  second.
- [x] The formal release validator passes 12/12 gates, including two-build
  determinism, recipe consumption, sample-exact parent/child transform and
  parent-hash lineage, and index-tamper rejection.
- [ ] `real_native` and `mixed_calibrated_real` remain blocked because no
  QC-passed measured M6 variant was supplied. This is a truthful incomplete
  empirical release, not an M6.4 implementation failure.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_variant_release_report.json`;
- hardened candidate release fixture: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/variant_release/release_a/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_variant_release.py`.

M6.5 bank-evaluation result (hardened 2026-08-02):

- [x] Added content-addressed bank-level distribution comparison, actual
  generation-throughput/failure validation, controlled-listening contract,
  and room-disjoint multi-seed downstream contract.
- [x] Distribution evaluation confirms calibrated/normalized scale-invariant
  metrics and records synthetic-to-measured as `not_evaluable` when a measured
  reference is absent; it never substitutes synthetic fixtures for real data.
- [x] Listening evidence requires randomized double blind assignment, common
  loudness gain, hidden reference/degraded anchor, room-disjoint stimuli,
  content-addressed responses/analysis, and at least 20 people for empirical
  status.
- [x] Downstream evidence recomputes recipe train/test acoustic-space hashes,
  requires at least three unique seeds and frozen model/training recipes, and
  requires every primary confidence lower bound to improve.
- [x] Downstream confidence intervals are recomputed from paired per-seed
  improvements. Empirical listening evidence must include content-addressed
  assignment, response, and analysis records whose participant result and
  confidence interval can be recomputed.
- [x] The formal validator passes 14/14 implementation gates and rejects
  unblinded listening, split-identity tamper, a single-seed claim, a forged
  positive confidence interval, and non-human responses relabelled empirical.
- [ ] **M6.5 empirical exit remains OPEN.** Current listening/downstream inputs
  are clearly labelled contract fixtures, not human responses or trained-model
  results; measured-distribution evidence is also absent. Production remains
  disabled.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_bank_evaluation_report.json`;
- hardened evaluation artifacts: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/evaluation/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_evaluation.py`.

M6.6 production-decision result (hardened 2026-08-02):

- [x] Added `puresound.m6_production_decision.v1`: an immutable candidate
  release is promoted by a content-addressed certificate instead of rewriting
  every bank asset, variant manifest, distribution, and recipe hash.
- [x] The decision binds the M6.4 release SHA, M6.5 evaluation SHA, ready
  synthetic/real/mixed recipes, QC state, pinned generator revisions,
  production-approved renderer profiles, evidence files, and three role
  sign-offs (acoustics, ML, release owner).
- [x] Evidence files must exist under safe relative paths and match their
  declared SHA-256. Listening/downstream artifact hashes must equal the hashes
  evaluated by M6.5; renderer approval files must match every profile.
- [x] Certificate validation re-runs the bound release/evidence audits and
  recomputes the canonical decision checks; it does not trust self-declared
  booleans merely because an attacker also recomputed the outer hash.
- [x] `PreGeneratedReleaseBank(require_production=True)` accepts only a valid
  approved certificate. Candidate usage remains available without that flag.
- [x] The formal validator passes 15/15 implementation gates, rejecting unsafe
  evidence paths, evaluation flag/hash tamper, forged approval flags, a
  competent all-true/rehashed certificate forgery, direct editing of candidate
  status, and production-reader access to a blocked bank.
- [ ] **Production promotion remains BLOCKED.** The current certificate lists
  missing real/mixed recipes, production renderer approval, measured/listening/
  downstream empirical evidence, evidence files, and three sign-offs.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_production_decision_report.json`;
- hardened decision artifacts: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/production/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_production_decision.py`.

Post-M6 training-pilot readiness (updated 2026-08-02):

- [x] Wire `PreGeneratedReleaseBank` into the existing
  `AudioEffectAugmentor`/dynamic-dataset YAML path with `bank_type: release`.
- [x] Require an explicit recipe and split; reject release-only options in
  legacy room-bank mode. Dataset role must equal the release split, and
  release/variant/split provenance reaches the collated training batch.
- [x] Bound the augmentor's simulated-RIR cache with LRU eviction; reject
  manifestless M6 layouts, invalid manifest items, out-of-range channels, and
  pending items in candidate/production readers.
- [x] Add a ready-to-copy training YAML and a non-overwriting pilot script for
  matched 1,000-room × 4-RIR Pyroomacoustics and PathEvents-M4 candidates.

Hardened matched preflight result (2026-08-02):

- [x] Completed a new 30-room × 2-item preflight for both high backends with
  fixed `v1/mixed`, seed `1337`, calibrated `16 kHz / 1.6 s`, and shared
  `pytard-cupy-material` low band. Each backend produced 60 items / 300
  channels; all 60 matched scene/room/acoustic-space/split/seed/shape identities
  agree, both banks are 60/60 QC PASS with zero quarantine, and both releases
  pass audit. The paired report is
  `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/preflight_validation_summary.json`.
- [x] Confirmed the intended acoustic difference: M4 has median C50/C80 of
  10.28/15.01 dB versus Pyroomacoustics 6.10/8.03 dB, median T20 0.472 s versus
  0.990 s, and absolute T20-to-scene-RT60 error 0.080 s versus 0.327 s. M4 is
  therefore drier and more early-energy dominant in this sample, and tracks the
  material RT60 more closely; this is not a measured-room realism verdict.
- [ ] The earlier A/B run was stopped and predates the hardening, so it remains
  diagnostic only. Re-run both full 4,000-item pilots in clean, pinned output
  directories, compare throughput/QC yield/distributions, obtain measured,
  listening, and downstream evidence, then train the first room-disjoint model
  A/B before scaling to a 50k–200k item bank. Pyroomacoustics remains the default
  while M4 is opt-in.

The preflight is explicitly `pass_with_provenance_warnings`: it used a dirty code
revision, has only two validation and two test items per backend, and does not
contain measured RIRs, human listening, or downstream-model results. Pyroom's
configured air absorption is applied internally but is not serialized in the
high-band metadata as completely as M4's policy, so provenance work remains.

### 5. Experiment discipline

Each experiment changes one major factor at a time:

```text
v0  current hybrid baseline
v1a + multiband materials, legacy peak normalization
v1b + calibrated gain using the same material scenes
v2  + source/receiver directivity rendering
v3  + lossy low-frequency modes
v4  + coherent mesh early paths
v5  + spatial multiband late field
v6  + inverse calibration / residual
```

Every row records:

- code and configuration revision;
- bank seed and manifest;
- generation time;
- M0 acoustic summary;
- downstream result with confidence interval;
- known limitations.

### 6. Risks and controls

| Risk | Control |
|------|---------|
| Full wave simulation becomes computationally prohibitive | Start with per-mode boundary loss and maintain a small reference solver only for validation |
| Metrics are optimized without audible or downstream benefit | Require real-room downstream and, for rendering, listening-test gates |
| Real measurements entangle room and transducer response | Preserve calibration data and model transducers separately |
| A learned model memorizes rooms | Split by room and report held-out-position and held-out-room results separately |
| Multiple RIR metric implementations disagree | Centralize them in one tested library module |
| Backend integration creates licensing or reproducibility problems | Complete a make-or-integrate review before adopting it |
| New metadata breaks existing recipes | Version the schema and retain legacy readers |

### 7. Immediate development queue

The active queue starts with M0:

- [x] Centralize DRR, clarity, decay, and spectral metrics.
- [x] Add analytic unit tests for decay estimates and failure cases.
- [x] Refactor `compare_bank_acoustics.py` to use the shared metrics.
- [x] Add deterministic JSON output suitable for checked experiment reports.
- [x] Add octave-band analysis.
- [x] Freeze and document the v0 generator configuration.
- [x] Run the first generated-versus-measured bank comparison.
- [x] Select the first M1 change from measured evidence.
- [x] Complete the M1 material-first schema, renderer, calibrated mode, and
  100-item acoustic probe.
- [x] Build a causal 50-item M1 audition bank with full WAV/metadata quality
  gates and deterministic dry/wet near/far previews.
- [x] Add measured-RIR noise-floor detection/truncation to decay fits.
- [x] Begin M2 per-mode low-frequency boundary damping.
- [x] Add a small independent 3D FDTD case for modal frequency/Q validation.
- [x] Benchmark the first M2 material-modal low band against measured modal
  distributions; it improves modal spacing but fails the Q gate due to excess
  damping, so it remains disabled for training banks.
- [x] Define and test the phase-aware complex-impedance/reflection contract
  without inventing phase from diffuse absorption.
- [x] Integrate a passive causal frequency-dependent admittance into the
  validation FDTD and verify its single-wall complex reflection response.
- [x] Add provenance-bearing glass-wool/Miki references, a complex-reflection
  fitting gate, and a phase-versus-magnitude FDTD modal diagnostic.
- [x] Add the direct complex-measurement contract, passivity-by-construction
  multi-pole fit, multi-state FDTD boundary, and 1D complex modal eigenvalue
  reference.
- [x] Audit public sources, ingest CC BY 4.0 NASA/UFSC normalized complex liner
  measurements, fit a passive series-RLC branch with alternating-frequency
  holdout, and validate the same branch in FDTD and the 1D eigenproblem.
- [x] Implement the calibrated repeated-H12 impedance-tube reduction and
  Traditional Chinese physical measurement protocol needed to acquire a
  compatible room-finish prior without inventing reflection phase.
- [x] Record the M2.12 full-room complex-crossover failure baseline and reject
  scalar gain, crossover relocation, and onset smoothing as standalone fixes.
- [x] Define and serialize shared direct/first-order `PathEvent` geometry,
  angle-aware complex gain spectra, and a causal scalar fractional-delay
  renderer; pass the M3.1 geometry/reciprocity/continuity gate.
- [x] Realize stored angle-aware complex boundary spectra as passive causal
  digital filters, verify the renderer against the identical analytic paths,
  and rerun the frozen M2.12 rooms as a seven-path diagnostic.
- [x] Extend the coherent PathEvent set through order 12, preserve ordered
  surface interactions, and verify spectral plus causal time rendering against
  the same-order analytic image set.
- [x] Audit direct-anchored early/later complex response against FDTD with
  complementary windows and coherent cross-term accounting; localize the
  dominant full-transfer error to early-reflection phase/interference.
- [x] Derive the exact staggered-grid oblique reflection and show that the
  approximately 6 cm FDTD boundary phase is not a continuous-reference match,
  while remaining passive and converging under grid refinement.
- [x] Validate face-pressure and half-time boundary predictors against the
  harmonic equation in a time-domain probe; reject both as incomplete
  continuous-reference fixes and retain the legacy default.
- [x] Evaluate a higher-order characteristic/dispersion-aware boundary, pass
  expanded multi-axis/azimuth holdouts, audit modal/geometric crossover
  branches, and evaluate an existing mesh engine; the order-12 full-room
  crossover remains an explicit failed diagnostic.
- [ ] Replace diffuse absorption-as-modal-loss with a low-frequency impedance
  prior from a compatible room-finish measurement in the production 3D
  backend, then rerun the room-disjoint M2 probe.
- [x] Audit direct and early reflection energy, integrate opt-in furniture
  interactions, and pass the paired measured C50/continuous early-timing M3
  exit before changing the late-field model.
- [x] Begin M4 with Abel-Huang normalized echo density, a room-response mixing
  time estimator, and a deterministic measured/M1/M3 bank baseline.
- [x] Add noise-aware octave-band late-field echo-density/decay targets and
  explicit binaural-IACC/diffuse-field-coherence contracts before FDN tuning;
  record that existing source-channel banks cannot evaluate spatial output.
- [x] Implement the isolated deterministic internally contractive multiband
  FDN core against the M4.2 targets; pass structural and 500 Hz–4 kHz measured
  decay/density/mixing gates while retaining 125/250 Hz as diagnostics.
- [x] Couple causal PathEvent direct/early energy to the FDN at the estimated
  mixing region with exact early preservation and post-transition energy
  matching; expose it only through `path-events-m4` and keep all earlier/default
  backends unchanged.
- [x] Add synchronized multi-receiver/ACN-SN3D Ambisonic output from one shared
  plane-wave field; pass deterministic model-derived diffuse-coherence/IACC
  gates, first-order receiver directivity, and the optional HRTF FIR decoder
  contract without changing the default backend.
- [ ] Close the separate M4 empirical/production exit with measured
  synchronized multi-receiver RIRs, a licensed calibrated HRTF decoder, and
  controlled listening for metallic coloration and spatial plausibility.
- [x] Complete M5.1 controlled-room campaign schema, room-disjoint split,
  multi-objective calibration-loss contract, reusable template, and legacy-bank
  readiness audit.
- [x] Complete the M5.2 noise-free synthetic-recovery inverse calibration and
  local-identifiability baseline with multi-start and held-out-position gates.
- [x] Extend M5.2 evidence with controlled noise/model-mismatch perturbations,
  calibrated nuisance correction, a smooth M4-consistent multi-term proxy, and
  waveform-only ablation.
- [x] Map mixing time, aggregate coherent-path gain, and octave RT60 through the
  actual M4 PathEvent/multiband-FDN coupling with discrete topology profiling.
- [x] Split the coherent proxy into six effective boundary/path groups, accept
  the full-rank groups, and reject mono absorption/scattering separation.
- [x] Complete the readiness-gated M5.3 runner, synchronized M5.4 candidate
  profile, constrained M5.5 residual/ablations, and M5.6 aggregate exit.
- [ ] Acquire or ingest a controlled M5 campaign containing retained repeated
  raw ESS recordings, transducer calibrations, geometry/poses/environment,
  deconvolution provenance, and synchronized receiver channels.
- [ ] Close M5 empirical/production exit with measured position/room holdouts,
  controlled listening, and room-disjoint downstream evidence.
- [x] Complete M6.1 versioned bank contract with deterministic acoustic-space
  splits, content hashes, provenance, negative controls, and fail-closed
  production claims.
- [x] Complete M6.2 actual generator integration with exact
  serial/parallel/resume reproducibility and config isolation.
- [x] Complete M6.3 per-item QC/quarantine before building M6 release
  candidates.
- [x] Complete M6.4 content-addressed distributions, signal variants, release
  recipes, and fail-closed measured/mixed recipe handling.
- [x] Complete M6.5 bank/listening/downstream evaluation contracts and
  implementation gates without manufacturing empirical evidence.
- [x] Complete M6.6 immutable promotion-certificate implementation and
  fail-closed production reader.
- [x] Integrate release recipes into training augmentation and freeze a matched
  Pyroomacoustics/PathEvents-M4 pilot workflow.
- [ ] Supply measured, human-listening, trained downstream, renderer-approval,
  and sign-off evidence so the M6.6 decision can change from blocked to
  approved.

### 8. First M0 baseline observation

Date: 2026-07-30  
Report: `egs/rir_generation/exp/rir_realism/m0/rir_benchmark_v0/acoustics.json`  
Frozen configuration: `egs/rir_generation/phases/m0_baseline/config/baseline_v0.json`  
Sampling: 300 channels per bank, seed 0, broadband plus valid octave bands

Compared banks:

- synthetic: `hybrid_rir_16k_levels/wide`;
- measured: `real_rir_16k_train_view`.

Selected distance-bucket medians:

| Bank | Distance | DRR (dB) | C50 (dB) | qualified T30 (s) | spectral tilt (dB/oct) |
|------|----------|----------|----------|-------------------|------------------------|
| synthetic | 0–1 m | 1.61 | 10.48 | 0.55 | 1.12 |
| measured | 0–1 m | -2.11 | 15.02 | 0.32 | -2.29 |
| synthetic | 2–3.5 m | -8.51 | 4.98 | 0.56 | 2.50 |
| measured | 2–3.5 m | -4.25 | 12.91 | 0.33 | -2.60 |
| synthetic | 3.5–6 m | -9.62 | 4.10 | 0.67 | 3.31 |
| measured | 3.5–6 m | -6.50 | 6.02 | 0.34 | -2.63 |

Decay medians above require fit R² >= 0.9. Synthetic T30 coverage was 100%;
measured coverage was only 62–79% below 6 m and 0% above 6 m. Before decay
parameters are used for material fitting, M0 therefore needs measured-RIR
noise-floor detection or truncation. The raw low-quality 6 m+ fits produced
spurious values near 100 s and are now excluded from summaries rather than
reported as room decay.

The strongest stable gap is spectral: synthetic responses tilt upward while
measured responses tilt downward by roughly 2–3 dB/octave. The 2–3.5 m C50 gap
also shows that synthetic energy is distributed too heavily into the late field
relative to measured early reflections. The first M1 implementation target is
therefore **frequency-dependent surface materials**, followed by a focused
early/late energy audit. Broadband RT60 tuning alone cannot fix either gap.

### 9. Noise-corrected M0 decay baseline

Date: 2026-07-30  
Report: `egs/rir_generation/exp/rir_realism/m0/rir_benchmark_v0/acoustics_noise_corrected.json`  
Sampling: the same 300 channels per bank and seed 0

The shared analyzer now estimates 10 ms block energy, a stationary tail floor,
the Lundeby-style decay/noise intersection, available dynamic range, and
decay-line R². Reliable responses are truncated at the intersection and have
expected integrated noise energy removed. Clean synthetic decays retain their
full uncorrected curve.

| Bank | Distance | T30 (s) | qualified T30 | noise-corrected | median dynamic range |
|------|----------|---------|---------------|-----------------|----------------------|
| synthetic | 0–1 m | 0.55 | 100.0% | 0.0% | n/a (no stationary floor) |
| measured | 0–1 m | 0.33 | 65.9% | 39.0% | 61.8 dB |
| measured | 1–2 m | 0.29 | 67.9% | 34.0% | 63.9 dB |
| measured | 2–3.5 m | 0.32 | 82.4% | 42.6% | 60.8 dB |
| measured | 3.5–6 m | 0.35 | 65.7% | 22.9% | 56.5 dB |
| measured | 6 m+ | 2.82 | 9.5% | 9.5% | 47.5 dB |

The correction removes the earlier approximately 100 s noise-tail artifacts.
The 6 m+ result remains low-confidence because it represents only two qualified
channels; it must not be treated as the typical measured decay. Below 6 m, the
stable conclusion remains a measured T30 near 0.3–0.35 s, substantially shorter
than the v0 synthetic 0.55–0.67 s.

### 10. M2 modal validation and corrected probe

Date: 2026-07-30  
Development report:
`egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics_dense_dev.json`  
Item-heldout report:
`egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics_dense_item_heldout.json`

All probe variants contain the same 20 rooms × 5 positions: their 100
serialized scenes and generator configurations compare equal. The analyzer
uses 35–300 Hz, a fixed 0.8 s gate beginning 20 ms after the direct peak, and
the same peak/Q settings for synthetic and measured responses.

The first report at `modal_acoustics.json` exposed two analytic-probe defects
and is now superseded for model selection:

1. mode amplitude and phase were rank/position heuristics instead of
   `phi(source) * phi(receiver)` coupling;
2. sorting and retaining only the first 64 modes removed almost all synthetic
   200–300 Hz peaks.

The corrected probe uses reciprocal rectangular eigenfunction coupling,
`1/omega` impulse-response weighting, causal modal onset, and a 256-mode cap
that covers all 215 possible non-DC triplets at the default index limit.
Node, reciprocity, and causality tests pass.

Development medians (measured sampling seed 0):

| Variant | peaks/channel | median Q | bandwidth | peak spacing | prominence |
|---------|---------------|----------|-----------|--------------|------------|
| corrected M1 bridge | 11 | 39.17 | 4.05 Hz | 14.04 Hz | 13.90 dB |
| M2 material loss, scale 1.0 | 9 | 22.97 | 5.86 Hz | 22.58 Hz | 13.73 dB |
| M2 material loss, scale 0.58 | 14 | 39.27 | 3.39 Hz | 15.01 Hz | 14.42 dB |
| measured | 6 | 35.76 | 3.97 Hz | 14.89 Hz | 13.56 dB |

Item-heldout medians (measured sampling seed 1, zero measured-item overlap):

| Variant | peaks/channel | median Q | bandwidth | peak spacing | prominence |
|---------|---------------|----------|-----------|--------------|------------|
| corrected M1 bridge | 11 | 38.58 | 4.21 Hz | 14.77 Hz | 14.01 dB |
| M2 material loss, scale 1.0 | 8 | 22.24 | 6.06 Hz | 23.56 Hz | 12.89 dB |
| M2 material loss, scale 0.58 | 13 | 38.96 | 3.50 Hz | 15.50 Hz | 13.64 dB |
| measured | 6 | 42.58 | 3.48 Hz | 12.57 Hz | 14.07 dB |

For the item-heldout set, Wasserstein distances normalized by measured IQR are:

| Variant | Q | spacing | bandwidth | peak count |
|---------|---|---------|-----------|------------|
| corrected M1 bridge | 0.087 | 0.080 | 0.183 | 0.280 |
| M2 material loss, scale 1.0 | 0.505 | 0.614 | 0.364 | 0.325 |
| M2 material loss, scale 0.58 | 0.304 | 0.183 | 0.270 | 0.387 |

The scalar 0.58 was estimated from the development aggregate. It restores the
Q median but creates too many narrow visible modes and does not beat the
corrected M1 bridge jointly on development or heldout items. It is retained
only as a serialized diagnostic knob, not an accepted constant.

Decision:

- accept eigenfunction endpoint coupling, causal onset, and complete modal
  coverage as analytic-probe corrections;
- reject the current diffuse-absorption-to-modal-loss mapping and its scalar
  calibration as an M2 production model;
- retain the exact damped recurrence and boundary-participation code as tested
  infrastructure;
- next model low-frequency complex impedance, validate frequency-dependent
  phase/loss against FDTD, and require a true room-disjoint measured split.

The heldout sample is item-disjoint, not proven room-disjoint, so even the
corrected M1 bridge result is diagnostic rather than a training-bank acceptance.

---

## PureSound RIR 模組化重構計畫

狀態：R0–R7 全部完成；shim 已退場，呼叫端全數改用正規路徑  
版本：v1.1  
日期：2026-08-02  
範圍：`puresound/audio` 底下與 RIR 生成、物理模型、空間渲染、校準、bank 管理相關的程式

本文件規劃並記錄模組化重構。R0–R7 已全部執行完畢，各階段結果附在對應小節；重構未改變任何 M 系列算法的物理假設。既有的整體算法與 milestone 狀態，仍以 [`RIR_REALISM_PLAN.md`](#physically-grounded-rir-realism-plan) 為準；新舊 import 路徑對照見 [`docs/audio/rir_package_migration.md`](docs/audio/rir_package_migration.md)。

### 1. 重構目的

目前 RIR 功能已經從單純的 `rir_generator` 擴展到：

```text
場景與材料
  → 阻抗／模態／FDTD 物理模型
  → 低頻與高頻 renderer
  → path event 與 late field
  → crossover／空間／雙耳組裝
  → metrics／calibration
  → M6 bank、QC、release、production decision
```

功能已能運作，但目前大多數模組仍直接放在 `puresound/audio` 根目錄，造成：

1. `hybrid_rir.py` 同時負責資料模型、場景取樣、backend、交叉濾波、障礙物幾何、metadata 與檔案輸出。
2. `rir_path_events.py` 同時負責 schema、幾何 visibility、image-source 路徑生成與波形 renderer。
3. `rir_metrics.py` 同時包含時間、頻譜、衰減、雙耳與陣列空間分析。
4. bank 的 manifest、item QC、release、evaluation 與 production decision 有清楚的概念差異，但目前以互相 import 的平面檔案表示。
5. 核心物理演算法被 `torch`、`torchaudio`、filesystem 與 optional backend 細節牽連，難以單獨測試或重用。

重構的目標不是一次性重寫算法，而是建立清楚的分層，使每個模組只有一種主要責任，並保留現有 import 路徑與 M6 bank contract 的相容性。

### 2. 現況盤點

#### 2.1 主要大型模組

| 現有檔案 | 約略行數 | 主要問題 |
|---|---:|---|
| `puresound/audio/hybrid_rir.py` | 3,240 | 低頻、高頻、場景、幾何、crossover、輸出全部混在一起 |
| `puresound/audio/rir_path_events.py` | 2,396 | schema、幾何、路徑生成、directivity、fractional delay、render 混在一起 |
| `puresound/audio/fdtd_reference.py` | 1,870 | 解析邊界參考、FDTD solver、source convention、測試診斷集中 |
| `puresound/audio/rir_metrics.py` | 1,516 | temporal、spectral、decay、echo density、IACC、spatial coherence 混在一起 |
| `puresound/audio/rir_bank_qc.py` | 1,251 | item 分析、multiprocessing、index 寫入、release audit 混在一起 |
| `puresound/audio/rir_bank_release.py` | 1,031 | variant materialization、distribution、recipe、release audit 混在一起 |
| `puresound/audio/rir_bank_manifest.py` | 880 | schema、hash、split policy、filesystem 操作與 audit 混在一起 |

#### 2.2 依賴集中點

目前最常被其他 RIR 模組使用的核心為：

- `rir_metrics`：9 個下游模組使用。
- `acoustic_impedance`：7 個物理與 path-event 模組使用。
- `rir_bank_manifest`：5 個 bank 模組使用。
- `rir_path_events`、`rir_scene`：各被 5 個 rendering/calibration 模組使用。

直接 import cycle 目前不明顯，但尚未有架構規則阻止未來形成 cycle。語意上的高耦合主要在：

```text
hybrid_rir
  ├─ scene sampling
  ├─ low-frequency backends
  ├─ high-frequency backends
  ├─ crossover
  ├─ obstacle geometry
  ├─ metadata
  └─ dataset I/O

bank loader
  ├─ manifest
  ├─ release
  └─ production certificate

release
  └─ QC

production decision
  ├─ evaluation
  ├─ release
  └─ manifest
```

#### 2.3 必須保留的現有 API

下列 API 已被生成腳本、phase validator 與測試使用，不能在第一階段直接刪除：

- `puresound.audio.hybrid_rir.HybridRIRConfig`
- `HybridRIRScene`、`PolygonObstacle`
- `AnalyticModalLowFrequencyBackend`
- `ImpedanceModalLowFrequencyBackend`
- `GpuARDPytARDBackend`、`GpuARDPytARDCuPyBackend`
- `PyroomacousticsHighFrequencyBackend`
- `PathEventHighFrequencyBackend`、`PathEventFDNHighFrequencyBackend`
- `generate_hybrid_rir`
- `sample_hybrid_rir_scene`、`sample_material_first_rir_scene`
- `upgrade_hybrid_scene_to_v2`
- `hybrid_crossover`
- `write_hybrid_rir_dataset_item`

目前也有少數測試與 CLI 直接使用 `hybrid_rir` 的私有 helper，例如 `_solve_modal_ard`、`_calibrate_pytard_signal`、`_apply_rt60_decay_envelope`、`_align_high_band_direct`。遷移前必須把這些 helper 分類成：正式 API、測試專用 API，或只保留在 compatibility shim 的 legacy symbol。

### 3. 目標架構

目標是在 `puresound/audio/rir/` 建立真正的 domain package。初期不移除舊的平面檔案；舊檔案先變成 re-export compatibility shim。

```text
puresound/audio/rir/
├── __init__.py
├── api.py                 # 對外穩定入口
├── contracts.py           # RIR tensor、metadata、backend protocol
├── scene/
│   ├── __init__.py
│   ├── schema.py          # RoomSceneV2、Pose、SurfaceMaterial 等
│   ├── sampling.py        # room/source/obstacle sampling
│   ├── materials.py       # material catalog 與 realization
│   └── geometry.py        # polygon、room、visibility 基礎幾何
├── physics/
│   ├── __init__.py
│   ├── propagation.py     # air absorption、sound speed、travel time
│   ├── impedance/
│   │   ├── admittance.py
│   │   ├── priors.py
│   │   ├── measurements.py
│   │   ├── tube.py
│   │   ├── fitting.py
│   │   ├── modes.py
│   │   └── residues.py
│   └── wave/
│       ├── fdtd.py
│       ├── low_frequency.py
│       └── source_convention.py
├── path_events/
│   ├── __init__.py
│   ├── schema.py          # ComplexPathGainSpectrum、PathEvent、PathEventSet
│   ├── geometry.py        # visibility、scene interaction、image source
│   ├── generator.py       # shoebox/scene path event generation
│   ├── directivity.py     # orientation、source/receiver directivity
│   └── renderer.py        # fractional delay、path-event waveform
├── render/
│   ├── __init__.py
│   ├── backend.py         # backend protocol、registry、capability metadata
│   ├── low_frequency/
│   │   ├── pytard.py
│   │   ├── analytic_modal.py
│   │   └── impedance_modal.py
│   ├── high_frequency/
│   │   ├── pyroomacoustics.py
│   │   ├── path_event.py
│   │   └── fdn.py
│   ├── crossover.py       # causal LR crossover、energy matching
│   ├── coupling.py        # path-event 與 FDN coupling
│   ├── spatial.py         # receiver array、Ambisonics
│   ├── binaural.py        # BRIR decoder/renderer
│   └── hybrid.py          # orchestration，不放底層幾何或 solver
├── metrics/
│   ├── __init__.py
│   ├── temporal.py        # direct、DRR、clarity、arrival、decay
│   ├── spectral.py        # octave band、tilt、頻譜 response
│   ├── density.py         # echo density、mixing time、noise floor
│   ├── spatial.py         # IACC、array coherence、diffuse coherence
│   └── report.py          # analyze_rir 與統一輸出 schema
├── calibration/
│   ├── __init__.py
│   ├── loss.py
│   ├── inverse_m4.py
│   ├── inverse_m5.py
│   ├── synthetic_recovery.py
│   ├── measured_campaign.py
│   └── residual.py
└── bank/
    ├── __init__.py
    ├── schema.py          # manifest、item、split、release schema
    ├── storage.py         # WAV/metadata/index filesystem adapter
    ├── loader.py          # training-time PreGeneratedRoomBank
    ├── qc.py
    ├── release.py
    ├── evaluation.py
    └── production.py
```

#### 3.1 分層規則

依賴只能由上往下：

```text
api / CLI adapter
        ↓
render、calibration、bank
        ↓
path_events、scene、metrics
        ↓
physics、contracts
        ↓
numpy/scipy 基礎運算
```

具體規則：

1. `contracts` 不得 import renderer、bank 或 filesystem。
2. `scene` 不得 import `torch`、`torchaudio` 或 Pyroomacoustics。
3. `physics` 只負責物理量與 solver，不寫 WAV、manifest 或 training metadata。
4. `path_events` 可以使用 scene 與 physics，但不能依賴 bank。
5. `render` 可以組裝 scene/path/physics，但不能直接修改 bank manifest。
6. `metrics` 接受 array-like 與明確的分析設定，不能依賴特定 renderer。
7. `bank` 可以使用 metrics，但 bank schema 不得反向依賴 hybrid renderer。
8. optional backend（PyTARD、CuPy、Pyroomacoustics）必須 lazy import。
9. 所有 filesystem 與 CLI 行為放在 adapter 層；核心函數優先回傳 immutable dataclass 或 NumPy array。
10. 舊的 `puresound.audio.*` import 路徑在遷移完成前必須維持可用。

### 4. 分階段遷移計畫

#### R0：API 與 contract freeze

目的：在移動任何程式前，固定行為與輸出格式。

工作項目：

- 列出 `hybrid_rir` 的正式 public symbols 與 legacy/private symbols。
- 固定 `[channels, samples]`、sample rate、dtype、causality 與 metadata contract。
- 為 `RoomSceneV2`、`PathEventSet`、M6 manifest/release 建立 contract fixtures。
- 建立 import boundary 測試，禁止新模組跨層 import。
- 確認 optional dependency 缺失時，scene/metrics/schema 仍可 import。

完成條件：現有測試全數通過，且每個保留 API 都有相容性測試。

##### R0 實作結果（2026-08-02）

- [x] 建立 `puresound/audio/rir/` package 與 `contracts.py`。`contracts` 只 import
  stdlib 與 NumPy，不碰 torch、torchaudio、renderer、bank 或 filesystem。
- [x] 定義 `RIRArray`（`[channels, samples]` 佈局驗證、dtype 轉換、因果邊界檢查）、
  `RenderContext`（backend 真正需要的 render 參數子集，與 scene sampling knobs 分離）、
  `BackendCapabilities`（含顯式 `deterministic_for_fixed_seed`）與
  `validate_rir_metadata`。
- [x] 凍結 API inventory：`hybrid_rir` 22 個 public symbols 分成
  egs 使用（15）、僅測試使用（5）、無使用者（2）三類；16 個被外部引用的 private
  helper 分成「應提升為 public」（6，egs CLI 直接 import）與「僅測試」（8）。
- [x] 建立 `RoomSceneV2`、`PathEventSet`、M6 manifest 的 golden fixtures，
  同時以檔案與原始碼中的 SHA-256 常數雙重釘住。
- [x] 建立 layer-direction lint 與 optional-dependency 守門測試。

新增檔案：

- `puresound/audio/rir/{__init__,contracts}.py`；
- `test/test_rir_r0_{api_inventory,contracts,golden_fixtures,import_boundaries}.py`；
- `test/fixtures/rir_r0/{room_scene_v2,path_event_set,bank_manifest}.json`。

R0 完全是新增，沒有修改任何既有檔案。115 個 R0 測試通過；M6 validator 套件
30/30、RIR 相關回歸 218/218 維持通過。

R0 過程中確認並記錄的兩件事：

1. **NumPy/Torch 邊界是單一一行**——`hybrid_rir.py:2162` 的
   `torch.as_tensor(rir, dtype=torch.float32)`。其餘全程 NumPy。契約已據此凍結，
   R2 拆 renderer 時不得增加第二個轉換點。
2. **`RoomSceneV2` 的物件往返不是 byte-stable**——`from_dict(to_dict())` 會把
   整數座標 `[0, 0, 0]` 加寬成 `[0.0, 0.0, 0.0]`，canonical JSON hash 因此改變。
   數值相等，且 M6 resume 不受影響（它 hash 的是從 JSON 載入的原始 dict，而 JSON
   文字往返會保留 int/float 區別）。但任何未來「重建 scene 物件再重算
   `scene_sha256`」的程式都會拿到不同的 hash。R0 以測試釘住現況；R1 若要修正，
   必須是明示決定並確認沒有既有 digest 依賴它。

#### R1：抽出 contracts 與 scene

來源：`rir_scene.py`、`rir_materials.py`、`hybrid_rir.py` 的 scene sampling/obstacle data。

拆分：

- `RoomSceneV2`、`SurfaceMaterial`、`Pose` 等移到 `scene/schema.py`。
- room/source/obstacle sampling 移到 `scene/sampling.py`。
- polygon 與 room geometry 基礎工具移到 `scene/geometry.py`。
- material catalog/realization 移到 `scene/materials.py`。
- `HybridRIRConfig` 與 backend protocol 移到 `contracts.py` 或 `render/backend.py`。

完成條件：scene/schema 不需載入 torch 或 optional renderer；舊 import 仍可用。

##### R1 實作結果（2026-08-02）

分五個可獨立驗證的小步執行，每步都以 R0 golden fixture 的 SHA-256 當關卡：

| 步驟 | 內容 | 結果 |
|---|---|---|
| R1a | `rir_scene.py` → `rir/scene/schema.py` | 836 行，`git mv` 保留歷史 |
| R1b | `rir_materials.py` → `rir/scene/materials.py` | 381 行 |
| R1c | `HybridRIRConfig` → `contracts.py` | 純資料，落 layer 0 |
| R1d | 純幾何 → `rir/scene/geometry.py` | 10 個函式，177 行 |
| R1e | scene sampling → `rir/scene/sampling.py` | 18 個定義，619 行 |

`hybrid_rir.py` 從 3,240 行降到 **2,585 行**（−20%）。舊的 `rir_scene` /
`rir_materials` 保留為 re-export shim；`hybrid_rir` 以別名保留全部既有名稱。

**`HybridRIRConfig` 為何落在 layer 0**：`scene/sampling` 需要它，而 layer 2 不得
import layer 3 的 `render`，所以計畫提供的兩個位置只有 `contracts.py` 可行。
`RIRBackend` protocol 則刻意留在 `hybrid_rir` 等 R2——它的簽名要引用 scene 型別，
放 layer 0 只能弱化成 `Any`，等 `render/backend.py` 出現才有正確的家。

**私有 helper 就地正名**：R0 分類為「應提升為 public」的 6 個，以及測試用的
`_obstacle_floor_coverage`，在新模組裡都取得正式名稱（`sample_point`、
`min_feasible_rt60`、`polygon_distance` 等），`hybrid_rir` 則以
`x as _x` 別名維持舊呼叫端。新 package 沒有底線命名的公開 API，舊 recipe 也不必改。

**R0 測試在此發揮作用，並據此修正判準**：inventory 測試原本檢查「定義於此模組」，
搬移後兩次正確地擋下改動。契約其實是「可從舊路徑 import」，因此改為
(a) `hybrid_rir.__all__` 必須等於凍結清單、(b) 定義於此的 public symbol 不得超出
清單、(c) 被外部引用的 private helper 必須仍可 import。同時替 `hybrid_rir` 補上
先前缺少的 `__all__`（§2 現況盤點列為問題之一）。

**過程中被測試攔下的兩個真實錯誤**（皆已修正）：

1. 以 `str.replace` 改名時沒有詞界保護，`max_obstacle_floor_coverage` 被誤傷成
   `maxobstacle_floor_coverage`；
2. `ast` 節點的 `lineno` 指向 `def`/`class` 那行而不含裝飾器，導致 `@dataclass`
   留在原檔成為孤兒，並疊加到後面的 `PytARDWaveBackend`。修正為取
   `min(node.lineno, decorator_list[*].lineno)`。

第二點值得記住：任何以 AST 行號搬移程式碼的工具都必須處理裝飾器，否則會產生
「能 import、但行為錯誤」的靜默損壞。

驗證：R0 三份 golden digest 完全未變（scene `d951…`、path events `772a…`）；
RIR 相關 187 項、M6 整合 30/30、全套 552 passed。全套仍有 6 個**既有**失敗，
與 R1 無關——它們引用 `phases/` 重整前的扁平腳本路徑，並缺少
`egs/rir_generation/measurements/` 資料，建議單獨修。

#### R2：拆解 hybrid renderer

來源：`hybrid_rir.py`。

拆分責任：

- PyTARD/GPU adapter → `render/low_frequency/pytard.py`。
- analytic/impedance modal backend → `render/low_frequency/`。
- Pyroomacoustics backend → `render/high_frequency/pyroomacoustics.py`。
- path-event high backend → `render/high_frequency/path_event.py`。
- FDN backend → `render/high_frequency/fdn.py`。
- crossover、alignment、causal clipping → `render/crossover.py`。
- `generate_hybrid_rir` 保留為高階 orchestration。
- dataset WAV/JSON 寫入移到 CLI/storage adapter。

完成條件：`hybrid.py` 只負責 pipeline composition；任何 backend 可以以 protocol 注入並獨立測試。

##### R2 實作結果（2026-08-02）

`hybrid_rir.py` 從 2,585 行降到 **497 行**，只剩三個定義：`generate_hybrid_rir`
（241 行 orchestration）、`_realized_acoustics_metadata`、`_config_metadata`。
其餘全部成為 re-export，舊 import 路徑不變。

新增 14 個模組：

```text
rir/render/arrays.py              37   共用 [channels, samples] 整形
rir/render/backend.py             34   RIRBackend protocol
rir/render/crossover.py          262   LR crossover、energy match、對齊、causal clip
rir/render/low_frequency/
    modal_damping.py             167   材料 → per-mode 損耗（pytard 與 analytic 共用）
    pytard.py                    701   pytARD CPU/CuPy backend
    analytic_modal.py            158
    impedance_modal.py           293
rir/render/high_frequency/
    obstacles.py                 224   post-hoc 遮蔽與散射
    pyroomacoustics.py           212   production default
    path_event.py                175   M3 coherent
    fdn.py                       168   M4 early+FDN late
rir/bank/storage.py                    dataset WAV/JSON 寫檔 adapter
```

**兩處為了達成分層而做的最小行為保持改動**：

1. `PathEventHighFrequencyBackend` 原本以
   `isinstance(self, PathEventFDNHighFrequencyBackend)` 選 metadata 字串，
   使 base class 依賴自己的 subclass，兩者無法分檔。改為未加型別註解的 class
   attribute `_LATE_PATH_AIR_ABSORPTION_POLICY`，由 subclass 覆寫。未加註解是關鍵：
   `dataclass` 只收 `__annotations__`，所以它不會變成 field。行為完全等價。
2. `RIRBackend` protocol 落在 `render/backend.py` 而非 `contracts.py`。放 layer 0
   的話簽名只能寫成 `Any`（contracts 不得 import scene）；放 layer 3 才能正確
   引用 `HybridRIRScene | RoomSceneV2`。

**被測試抓到的一個搬移專屬破壞**：`_default_pytard_root()` 以
`Path(__file__).parents[1]` 定位 vendored pytARD。在舊位置那是 `puresound/`，
搬到 `rir/render/low_frequency/` 後變成 `rir/render/`，pytARD 直接找不到。已改為
錨定套件本身（`Path(puresound.__file__).parent`），對未來搬移免疫。

這一類 `__file__` 相對路徑是 AST 搬移工具**看不到**的破壞，golden fixture 也蓋
不到（fixture 走 analytic backend）。搬移含檔案系統路徑的模組時要主動 grep
`__file__`。

**工具化**：R1e 的兩個錯誤（`str.replace` 無詞界、`ast.lineno` 不含裝飾器）已寫成
scratchpad 的可重用抽取工具，R2 五個步驟共用，未再發生同類錯誤。工具另含
`check_module()` 偵測疊加裝飾器、`audit_no_mangled_identifiers()` 比對搬移前後的
識別字集合。

驗證：R0 三份 golden digest 未變；RIR 相關 151 項、M6 整合 30/30、全套 552 passed，
6 個既有失敗不變。

#### R3：拆解 path-event pipeline

來源：`rir_path_events.py`。

- dataclass/schema 與 JSON round-trip → `path_events/schema.py`。
- polygon visibility、segment intersection、scene interaction → `path_events/geometry.py`。
- shoebox image-source 與 scene path generation → `path_events/generator.py`。
- directivity → `path_events/directivity.py`。
- fractional-delay kernel、render、arrival partition → `path_events/renderer.py`。

完成條件：可以只建立 PathEventSet、只做 geometry audit，或只 render 已存在的 PathEventSet。

##### R3 實作結果（2026-08-02）

`rir_path_events.py`（2,245 行）拆成六個模組而非計畫的五個：
`schema` 663、`geometry` 479、`interactions` 636、`generator` 398、
`directivity` 60、`renderer` 256。多出的 `interactions.py` 是因為
`augment_scene_path_events_with_interactions` 單一函式就有 536 行，硬併入
`geometry.py` 會讓該檔逼近 1,100 行，違反 §5 的規模準則。

搬移時漏掉 `_BOUNDARY_GEOMETRY`——它是 `AnnAssign`（帶型別註解的賦值），而我的
常數蒐集只掃 `ast.Assign`。已補；此後的模組級常數改為同時掃兩種節點。

#### R4：拆解 metrics 與 spatial renderer

來源：`rir_metrics.py`、`multiband_fdn.py`、`spatial_late_field.py`、`spatial_rir.py`、`binaural_renderer.py`。

- metrics 按 temporal/spectral/density/spatial 分開。
- `analyze_rir` 保留為 report facade。
- FDN design/render 與 spatial late-field assembly 分開。
- Ambisonics decoder 與 BRIR renderer 不依賴 bank 或 calibration。

完成條件：mono、array、Ambisonics、binaural 路徑共用同一套低層 metrics 與 contracts。

##### R4 實作結果（2026-08-02）

`rir_metrics.py`（1,516 行）拆成
`core`/`temporal`/`spectral`/`density`/`spatial`/`report` 六個模組。
`analyze_rir` 留在 `report.py` 作為 facade，維持所有 bank 工具的單一入口。

FDN、coupling、spatial、binaural 四個模組本來就是單一職責，這一階段對它們是
**重新定位**而非拆分，且四個都直接落在 §3 目標樹的
`render/{coupling,spatial,binaural}.py` 與 `render/multiband_fdn.py`。
`rir_attribution.py` 不屬 calibration，歸入 `metrics/attribution.py`。

#### R5：拆解 calibration 與 measured pipeline

來源：`rir_calibration.py`、`rir_inverse_calibration.py`、`rir_m4_inverse_calibration.py`、`rir_m5_pipeline.py`、`rir_measured_calibration.py`、`rir_measurement_campaign.py`、`rir_constrained_residual.py`。

- measurement schema 與 filesystem audit 分開。
- loss/metrics 只處理數值輸入。
- M4/M5 inverse fit 各自成為 calibration strategy。
- measured runner 只負責載入資料、呼叫策略、輸出 fit report。
- residual model 維持 causal/decay contract，不直接依賴 CLI。

完成條件：synthetic recovery、measured fit、M4/M5 validator 可以使用相同的 renderer/metrics contract。

##### R5 實作結果（2026-08-02）

七個 calibration 模組整檔搬入 `rir/calibration/`：`loss`（M5.1 loss 契約）、
`synthetic_recovery`（M5.2）、`inverse_m4`、`inverse_m5`、`measured_runner`
（M5.3 fail-closed runner）、`measured_campaign`（M5.1 acquisition 契約）、
`residual`（M5.5）。全部原本就有 `__all__` 且職責單一，不需再拆。

#### R6：拆解 M6 bank pipeline

來源：`rir_bank.py`、`rir_bank_manifest.py`、`rir_bank_qc.py`、`rir_bank_release.py`、`rir_bank_evaluation.py`、`rir_bank_production.py`。

分層：

1. `bank/schema.py`：純 dataclass、hash、split policy、serialization。
2. `bank/storage.py`：WAV/metadata/index 的 filesystem 操作。
3. `bank/loader.py`：training-time loader 與 cache。
4. `bank/qc.py`：item QC 與 quarantine，不建立 production decision。
5. `bank/release.py`：variant/recipe materialization。
6. `bank/evaluation.py`：distribution、throughput、listening/downstream evidence。
7. `bank/production.py`：只讀取各項 evidence 並產生 fail-closed decision。

完成條件：loader 可以只依賴 schema/storage；M6 的 manifest hash、split disjointness、QC hash、release lineage 與 production certificate 行為完全不變。

##### R6 實作結果（2026-08-02）

六個 bank 模組搬入 `rir/bank/`：`schema`（原 manifest）、`loader`、`qc`、
`release`、`evaluation`、`production`，與 R2 已建立的 `storage` 併齊。

**`bank/__init__.py` 刻意不做任何 import。** 第一版讓它 re-export `loader`，
結果把 `torch` 拉進每一個 manifest 讀取者，`rir_bank_manifest` 從 torch-free
變成 torch-dependent——R0 的 boundary 測試立刻擋下。`schema` 必須能在沒有
torch/torchaudio 的環境載入，這個性質比 `__init__` 的便利重要。

M6 全套 30/30 通過，manifest hash、split、QC、release lineage 與 certificate
行為皆不變。

#### R7：compatibility shim 與舊模組退場

- `puresound/audio/hybrid_rir.py` 改為 re-export facade。
- `puresound/audio/rir_scene.py`、`rir_path_events.py`、bank 相關舊檔案保留相容入口。
- 更新 egs 與 tests 到新路徑。
- 對私有 helper 設定 deprecation 或移為測試 fixture。
- 加入 migration guide，最後才評估刪除舊實作。

##### R7 實作結果（2026-08-02）

- **`physics/` 層補齊**：目標樹 §3 列了 `physics/`，但 R1–R6 沒有任何階段指派它。
  11 個模組搬入 `rir/physics/{impedance,wave}/` 與 `physics/propagation.py`。
- **`hybrid_rir.py` 搬成 `render/hybrid.py`**，扁平路徑改為手寫 shim。
- **33 個 compatibility shim** 覆蓋所有舊路徑。自動產生的 shim 只轉 `__all__`，
  漏了兩處外部實際依賴的私有名稱（`hybrid_rir` 的 14 個相容別名、
  `rir_bank_evaluation._paired_t_confidence_interval`），由一支「掃描全 repo
  對扁平模組的 import，逐一驗證 shim 是否具備該屬性」的檢查抓出並補齊。
- **`rir/api.py`**：37 個名稱的穩定門面。docstring 明講它會載入 renderer stack
  （含 torch），要省 import 成本就直接取用分層模組。
- **migration guide**：`docs/audio/rir_package_migration.md`，含完整新舊對照表、
  私有 helper 更名表、分層規則，以及「shim 何時該退場」的前提條件。

R0 inventory 測試在 R1–R7 期間共擋下 4 次改動，每次都不是搬錯，而是測試的判準
需要隨遷移精確化。四次修正都收斂到同一個原則：**契約是「可從舊路徑 import」，
不是「定義於該檔」**：

1. 公開介面改以 `hybrid_rir.__all__` 為準（並補上它原本缺少的 `__all__`）；
2. 私有 helper 改檢查 `hasattr` 而非定義位置；
3. 同一 package 內姊妹模組共用私有 helper 屬正常，跨 package 才算違規；
4. 自述為 `Compatibility shim` 的模組整體豁免——它存在的目的就是 re-export。

另外把「不得拉入 torch/torchaudio/pyroomacoustics/cupy」的守門從扁平 shim 擴充到
12 個正規模組（`PURE_PACKAGE_MODULES`）。這個性質原本只在會消失的 shim 上被驗證，
現在釘在遷移後大家真正會 import 的位置。

**R7 驗收**：

- 全套 564 passed / 6 failed，6 個失敗與遷移前完全相同（既有的缺檔與
  `phases/` 重整遺留的扁平腳本路徑）；
- M6 整合 30/30；R0 三份 golden digest 未變；
- 端到端實跑 `generate_m6_bank.py`（6 房）：生成 → QC 6/6 → release audit PASS
  → `status=candidate`，證明搬完全部模組後 CLI 仍可用；
- `contracts`、`scene.*`、`metrics`、`path_events`、`physics.*`、`bank.schema`
  全部可在不載入 torch 的情況下 import。

**遷移後規模**：`puresound/audio/rir/` 共 73 個模組、28,701 行。

##### R7 後續：shim 退場（同日）

計畫原本讓 shim 無限期留存，理由是「刪除的收益小於改寫呼叫端的擾動」。這個判斷
在盤點後不成立：shim 最正當的用途是**改不到的外部消費者**，但這裡沒有——沒有
entry point、沒有設定檔以字串引用模組路徑、沒有 pickle 依賴，216 處全在 repo 內。
留著的代價則是同一個東西有兩種 import 方式，而新程式碼會照抄看到的那種。

- AST 改寫 214 個 import 敘述、92 個檔案，全部指向正規模組；私有別名一併解析到
  它們的公開名稱（`_solve_modal_ard` → `pytard.solve_modal_ard` 等）。
- 刪除 35 個 shim。外部引用的私有名稱從 15 個降到 1 個
  （M6.5 validator 的 `_paired_t_confidence_interval`，已登記在 inventory 測試）。
- `render/hybrid.py` 的 `__all__` 從 22 項收斂到 1 項（`generate_hybrid_rir`）。
  其餘 21 項是為 shim 服務的 re-export，沒有任何呼叫端使用；同時移除 24 個
  底線別名與 11 個未用 import，模組回歸純 orchestration。
- inventory 測試改寫成遷移後的守門：`__all__` 完整性、跨 package 私有 import
  登記、shim 不得重生、以及兩個 dead-code 候選（`PytARDWaveBackend`、
  `_sample_source_in_shell`）的「確實無人使用」宣稱。

改寫器刻意跳過 `puresound/audio/*.py` 以免動到 shim 本身，但那裡也住著兩個**真實
模組**——`impulse_response.py` 與 `augmentation.py`——它們因此指向已刪除的模組。
由 repo 全域掃描抓出並修正。教訓：以「目錄」界定豁免範圍不安全，該以「檔案性質」
界定。

驗證：573 passed，6 個既有失敗不變；golden digest 未變。

### 5. 測試與驗收策略

每個階段都必須同時通過以下四類測試：

#### 行為相容

- 現有 RIR 單元測試不退化。
- 同 seed 的 scene、path event、RIR metadata 保持 deterministic。
- 舊 import path 與新 import path 產生相同結果。

#### 數值相容

- crossover 前後的 causality、direct arrival、energy matching 不變。
- 低頻 modal/impedance/FDTD 的 reference fixture 不變。
- path-event delay、gain、visibility 與 reconstruction error 不變。

#### bank contract

- M6 manifest/release/QC/evaluation/production validator 全數通過。
- train/validation/test split 不混用。
- hash、release lineage、QC report identity 不變。

#### 架構品質

- `contracts`、`scene`、`physics` 可在沒有 optional renderer 的環境 import。
- 新 package 不允許反向依賴 CLI 或 bank storage。
- 每個 production module 的 public API 有 docstring 與 type hints。
- 單一模組建議不超過約 500–700 行；超過時必須有明確理由。

### 6. 非目標

本次重構不包含：

- 改變 M4/M5/M6 的物理模型或參數。
- 把 Pyroomacoustics 替換成新的預設 backend。
- 在沒有 benchmark 的情況下調整音響效果。
- 改變 M6 release schema 或 training data split 規則。
- 將所有舊 API 一次刪除。
- 在本階段加入 neural RIR generator。

### 7. 主要風險與處理方式

| 風險 | 影響 | 處理方式 |
|---|---|---|
| 大量既有程式直接 import `hybrid_rir` | 高 | 先保留 facade 與 re-export |
| 測試依賴私有 helper | 中高 | 先分類 helper，再建立明確測試 API |
| optional backend 在 import 時被載入 | 中 | lazy import 與 capability check |
| metadata/hash 行為細微改變 | 高 | 先建立 golden fixtures 與 canonical JSON tests |
| 大檔案移動造成 git review 困難 | 中 | 每個階段只移動一個 bounded domain |
| bank schema 與 loader 同時改動 | 高 | schema/storage/loader 分三階段遷移 |
| 物理核心與 torch dtype 混用 | 中 | contracts 明確規定 NumPy/Torch 邊界 |

### 8. 第一個實作批次

第一批只做 R0，不拆任何演算法：

1. 建立 API inventory 與 import compatibility tests。
2. 建立 `RoomSceneV2`、`PathEventSet`、M6 manifest 的 golden fixtures。
3. 建立 package boundary lint/test，禁止新的跨層依賴。
4. 定義 `RIRArray`、`RenderContext`、`BackendCapabilities` 與 metadata contract。
5. 針對 `hybrid_rir.py` 的 private helper 做 public/legacy/test-only 分類。

R0 完成並通過後，才開始 R1 的實際檔案拆分。這樣可以先確定「搬家不改行為」，再逐步改善模組邊界。

### 9. 當前決策

- 採用新 `puresound/audio/rir/` domain package。
- 舊 flat modules 暫時保留為 compatibility shim。
- `hybrid_rir` 只保留高階 orchestration facade。
- core contracts 與 scene/physics 不得依賴 torch、torchaudio、CLI 或 filesystem。
- M6 bank schema 與現有 release/QC 行為視為 frozen contract。
- R0（contract freeze）、R1（contracts 與 scene）、R2（hybrid renderer）已完成。
- 正規位置為 `puresound/audio/rir/{contracts.py,scene/,render/,bank/storage.py}`；
  `rir_scene`、`rir_materials` 與 `hybrid_rir` 的對應名稱都是 re-export。
- `hybrid_rir.py` 已收斂為 497 行的 orchestration facade。
- R3（path-event pipeline）尚未開始。`rir_path_events.py`、`rir_metrics.py` 與
  bank 相關模組仍在舊位置。

---

## 附錄：自產品文件遷出的結果紀錄

2026-08-04 文件重整時，自 `docs/audio/rir_bank_v2_zh-TW.md` 與
`docs/audio/rir_measurement_campaign_zh-TW.md` 遷出的結果段落，原文照錄（heading 降兩級）。

#### 7. M6.1 正式結果

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

##### 12.2 強化後 matched backend preflight（2026-08-02）

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

#### 6. M5.1 已完成的實作

主要入口：

- `puresound/audio/rir_measurement_campaign.py`：schema、strict JSON、hash
  與 campaign audit；
- `puresound/audio/rir_calibration.py`：七項 reference loss 與 term-level
  diagnostics；
- `egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py`：deterministic
  loss probes、template 與 legacy bank readiness audit；
- `egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json`：只有結構
  範例，不是實測證據；
- `egs/rir_generation/phases/m5_calibration/reports/m5_measurement_contract_report.json`：正式
  M5.1 report。

驗證 probe 確認：identity fidelity terms 為零、8-sample delay 可被偵測、
pre-arrival energy 會受罰、spatial perturbation 可被偵測，而且 mono spatial
term 明確不可評估。針對 schema、hash corruption、同步語意與 loss 的 10 個
測試全數通過。

#### 7. 現有 measured bank 的真實狀態

Validator 分別抽查既有 train 與 held-out view 各 64 個 metadata。兩者目前
只保存 `channel_map`、distance、`origin=real` 與 RT60 類資訊；以下九類 M5
證據在抽樣中都是 0/64：

1. 穩定實體 room identity；
2. geometry／mesh；
3. source position 加 orientation；
4. receiver position 加 orientation；
5. source／receiver response calibration；
6. temperature／humidity／pressure；
7. repeated raw ESS；
8. deconvolution inverse／config／noise；
9. synchronized receiver channel semantics。

結論不是「舊資料沒用」。它仍可做 DRR、C50、decay、頻譜與 echo-density
distribution reference；但缺失的 acquisition evidence 無法從最後 RIR WAV
逆推出來，所以不能被升格成 M5 controlled inverse-calibration dataset。

#### 8. M5.2 synthetic recovery 已完成什麼

在等待／規劃受控量測時，M5.2 先完成第一個 synthetic recovery baseline：

1. 從已知 scene 生成 target RIR；
2. 隱藏 material、mixing-time、late-decay 等一小組參數；
3. 從錯誤初值開始最小化同一 loss；
4. 在未參與 fitting 的 source／receiver positions 比較 recovery；
5. 用 multi-start 與 scaled Jacobian singular values 檢查局部不可識別方向；
6. 只有能在 synthetic holdout 穩定找回的參數，才進 M5.3 measured fit。

實作 `puresound.rir_synthetic_recovery.v1` 是一個刻意簡化、保持因果的
approximate renderer。Geometry 與 direct response 視為已知，只開放十個
shared-room parameters：

- 一個 mixing time；
- 一個 coherent early-reflection gain；
- 500／1000／2000／4000 Hz 四個 RT60；
- 同四個 octave bands 的 late gain。

Direct component 永遠不被 crossfade；early 與 late 使用連續 equal-power
transition。每個 late band 的 pressure envelope 為

\[
a_b(t)=10^{-3t/T_{60,b}},
\]

所有 mixing time、gain 與 RT60 都受明確 box bounds 限制。三個 train
positions 共用同一組房間參數，但各有不同 distance、early path fixture 與
deterministic band-limited late excitation，因此 optimizer 必須找出可跨位置
解釋資料的參數。

正式 validator 從三組差異很大的初值做 bounded nonlinear least squares。
三次都回到同一組隱藏 ground truth，最大 parameter spread 為
`2.35e-12`；scaled Jacobian condition number 是 `17.22`，且具有完整 local
column rank。獨立 M5.1 oracle 在兩個 unseen positions 的 mean loss 由
`2.26933` 降到 `1.69e-14`，所有輸出保持 direct arrival 前嚴格為零。

這個結果是必要但很弱的第一關，屬於 **inverse-crime baseline**：target 與
fitter 使用同一個 noise-free model family，所以精確 recovery 是預期結果。
它證明的是 parameter serialization／bounds、multi-position objective、
multi-start、local sensitivity、holdout 與 M5.1 oracle 接線都正確；它沒有
證明：

- measurement noise 或 clock drift 下仍穩定；
- approximate renderer 能吸收完整 M4 renderer 的 model mismatch；
- 真實材料、scattering 或 directivity 已被找回；
- global identifiability；
- measured-room fit 已完成。

報告為 `egs/rir_generation/phases/m5_calibration/reports/m5_synthetic_recovery_report.json`；同一
holdout position 的 target／錯誤初值／recovered WAV 位於
`egs/rir_generation/exp/rir_realism/m5/rir_m5_synthetic_recovery/`。下一個可平行開發切片是加入 controlled
noise/model mismatch；下一節已完成這個 robust synthetic gate。真正的
M5.3 仍必須等待符合前述契約的 controlled campaign。

##### 8.1 M5.2b：噪聲、已知 nuisance 與未知 model mismatch

M5.2b 不再讓 fitter 看到乾淨 target。每個 train／holdout position 都加入：

- 32–38 dB SNR 的 broadband acquisition noise；
- -0.8 至 +1.2 dB 的 per-position gain calibration error；
- -6 至 +11 samples 的 deconvolution latency offset；
- nominal model 沒有的 early taps；
- 一個獨立 stochastic late component，其 RT60 為 nominal band 的 1.25 倍。

系統同時保存 `raw_rir` 和 `corrected_rir`。只有 campaign metadata 已知的 gain
與 latency 會被移除；noise、extra paths 和 secondary decay 刻意留在 fitting
target。這是在測 robust estimation，不是用 ground truth 把所有誤差清乾淨。

新的 `puresound.rir_robust_recovery_objective.v1` 使用四組 smooth residual：

\[
r(\theta)=
\left[
\sqrt{w_w}r_{\mathrm{wave}},
\sqrt{w_e}r_{\mathrm{early}},
\sqrt{w_b}r_{\mathrm{broadband\ decay}},
\sqrt{w_o}r_{\mathrm{octave\ decay}}
\right].
\]

Decay residual 以 8 ms window 的 log energy 計算；只有高於估計 noise energy
20 dB 的 windows 參與 fitting。這個門檻非常重要：若把已進入 noise floor 的
高頻尾端也當成房間衰減，optimizer 會把 noise plateau 解讀成較長 RT60。

它稱為 M4-consistent proxy，是因為 direct／coherent early／broadband late／
octave late 的分工與 M4 相同；但目前仍以 SciPy finite-difference least
squares 最佳化 surrogate，不是 autograd，也尚未對完整 PathEvent＋FDN
renderer 求導。

正式結果：

- mixing time absolute error：`0.0235 ms`；
- early gain error：`0.0214 dB`；
- 最大 octave RT60 relative error：`1.43%`；
- 最大 late gain error：`0.106 dB`；
- 兩個遠距初值的最大 parameter spread：`1.03e-6`；
- scaled-Jacobian condition number：`8.12`，local full rank；
- held-out M5.1 total：相對初始值下降 `61.4%`；
- held-out octave error：robust objective `0.00846`，waveform-only ablation
  `0.03341`。

另外，五個擾動案例中有兩個的 global absolute peak 並不是 direct arrival。
這不是小細節：高 DRR 以外的 RIR、未建模反射或 noise spike 都可能比 direct
大。正式量測應以 geometry (d/c) 建立 arrival search window，或使用另行
驗證的 onset detector；不能直接 `argmax(abs(rir))`。

M5.2b 的 15/15 gates 全數通過，報告位於
`egs/rir_generation/phases/m5_calibration/reports/m5_robust_recovery_report.json`，六個 holdout
RIR artifacts 位於 `egs/rir_generation/exp/rir_realism/m5/rir_m5_robust_recovery/`。它仍不是 measured-room fit；
下一節已把第一組參數映射到 actual M4 renderer。

##### 8.2 M5.2c：actual PathEvent＋multiband FDN parameter profile

M5.2c 不再用 smooth surrogate 產生 candidate，而是直接通過 M4 的
`PathEvent -> equal-power transition -> multiband FDN`。由於 mixing time 會
離散改變 prime delay topology，演算法以 `20/24/28 ms` 做 outer profile；每個
profile 內再以 bounded least squares 估 coherent-reflection aggregate gain 和
500／1000／2000 Hz RT60。

Target 保留 8% alternate-FDN-seed mismatch 與 42 dB SNR noise。兩個 order-4
PathEvent positions 用於 fitting，另兩個 positions 只做 holdout。14/14 gates
通過：正確選到 hidden `24 ms`、best／second cost ratio `0.0820`、best
Jacobian condition number `3.14`、coherent gain error `0.661 dB`、最大 RT60
error `4.23%`，held-out M5.1 total 下降 `71.7%`。所有 candidate 維持
physical-arrival causality，M4 transition 前樣本完全不變，Pyroomacoustics
production default 也沒有改動。

報告位於
`egs/rir_generation/phases/m5_calibration/reports/m5_m4_parameter_mapping_report.json`，三個 holdout
RIR artifacts 位於 `egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping/`。這只識別 aggregate
coherent gain，不代表已從 RIR 分離出單一牆面的 absorption／scattering；
也不是 measured-room fit。下一個 M5.2d 是逐 material／path group 的
identifiability ablation。

##### 8.3 M5.2d：哪些 material/path groups 真的可辨識

每條 PathEvent 都保留撞到的 surface sequence。M5.2d 對每次 boundary hit
施加一個 effective pressure adjustment，因此同一參數會一致影響所有包含該
牆面的高階路徑。三個 order-4 train positions 可找回 west／east／south／
north／floor／ceiling 六組：condition number `2.74`、最大 gain error
`0.00123 dB`、held-out M5.1 total 下降 `64.8%`。

但若把每面牆同時開放 `absorption loss` 和 `specular scattering loss`，兩者在
mono coherent amplitude 上的 Jacobian columns 完全相同；rank 由應有的 12
只有 6。演算法因此保留六個 `effective_reflection`，拒絕六個 scattering
duplicates。這不是最佳化失敗，而是資料本身沒有足夠觀測；scattering 必須等
M5.4 的 synchronized receiver evidence。

##### 8.4 M5.3：runner 已完成，但不合格資料不會開始 fitting

`fit_m5_measured_campaign.py` 的執行順序是：

1. 驗全部 retained assets 與 SHA-256；
2. 確認每個 room 的 repeated ESS、noise、inverse、calibration、geometry、
   environment 與 synchronized channel semantics；
3. 以 `campaign_id + room_id + measurement_id` 的 SHA-256 固定選出
   train-room position holdout；
4. 只在 `position_fit` 估每個 train room 的 M4 topology、RT60 與六面
   effective reflection；
5. 分開回報 train-position、position-holdout、validation-room 與 test-room；
6. 未見房間只用 train-room population median parameters，不偷 fit test room。

完整 synthetic campaign fixture 已走通 runner 的所有階段與 8/8 gates，而且
metadata 明確標成 `qualifies_as_measured_evidence=false`。現有 template／legacy
bank 則在 readiness 階段退出，M4 optimizer 完全不會被呼叫。正式狀態報告為
`m5_measured_fit_status_report.json`：M5.3 runner implementation PASS，真實
measured fit 仍 BLOCKED。

目前 reference runner 支援有 `dimensions_m` 的 shoebox campaign。只有 mesh 的
campaign 必須先註冊能產生 PathEvent 的 mesh backend，不能把 mesh 悄悄縮成
shoebox。

##### 8.5 M5.4：同步 receiver 才能校正 spatial groups

`select_spatial_calibration_candidate` 至少要求兩個同步 channel。它用同一份
M5.1 loss 比較 actual M4 的 scattering、receiver directivity 與 late-field
candidates；mono 直接拋錯，不會得到假的 spatial zero loss。四候選 fixture
正確選回 hidden scattering＋opposed-cardioid 組合，11/11 gates 通過。

證據邊界也要保留：現行 M4 scattering 只分配 first-order coherent PathEvent
energy，80 ms 後的 FDN field 尚未依 scattering 改變。因此 scattering 是由
synchronized early／spectral／octave total 選中；late spatial coherence 主要
驗 shared field 與 directivity，不能把相同的 late term 說成 scattering 證據。

##### 8.6 M5.5：learn residual，不重學 basic physics

新的 residual model 先算 `target - physical`，再把每條 residual 對齊自己的
direct arrival，除以 physical tail norm，從多個 train rooms 取 robust median。
它受到三個硬限制：

- direct arrival 前永遠為零；
- 50 ms 後每 10 ms block 不得比最大 `RT60=0.8 s` 的 pressure decay 更慢；
- normalized residual energy 不得超過 physical energy 的 `0.15`。

M5.5 同時輸出 physical-only、residual-only、combined，避免只報最好的一條。
在完全未參與 fitting 的第三個 synthetic room，combined total 相對
physical-only 下降 `67.9%`，residual-only 明顯較差；causality、decay、energy
budget 與 parameter interpolation 共 12/12 gates 通過。這證明 residual
contract／ablation plumbing 可用，尚不代表已在真實房間訓練 neural model。

##### 8.7 M5.6：完成的是 implementation，不是捏造 empirical PASS

`validate_m5_exit.py` 彙整 M5.1、M5.2／2b／2c／2d、M5.3 runner、M5.4、
M5.5、必要 WAV/campaign artifacts 與禁止 false claim 的 invariants。結果為：

- **M5 implementation exit：PASS**；
- **M5 empirical／production exit：OPEN**；
- production enablement：`ready=false`，Pyroomacoustics default 不變。

empirical exit 還缺：合格 repeated-ESS campaign、measured position holdout、
measured physical-room holdout、measured synchronized spatial calibration、
measured residual training、controlled listening 與 room-disjoint downstream
task。這些是必須真的取得／執行的外部證據，不能由 synthetic fixture 生成。

近期 inverse-acoustic rendering 研究也採用 differentiable rendering 與稀疏
觀測來估 room parameters，例如
[AV-DAR](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Differentiable_Room_Acoustic_Rendering_with_Multi-View_Vision_Priors_ICCV_2025_paper.html)
與 [DiffRIR / Hearing Anything Anywhere](https://masonlwang.com/hearinganythinganywhere/)。
PureSound 的策略更保守：先用現有可稽核物理 renderer 做 recovery baseline，
確認 identifiability，最後才加入 learned residual。

