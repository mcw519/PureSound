# puresound/* Refactor Plan

Status: **COMPLETED**（2026-07-29 盤點、決策、執行同日完成）

| 階段 | commit | 結果 |
|---|---|---|
| P1 backbone 一等公民化＋改名 | `a5812f3` | 全 backbone 匯出＋可達性測試；4 個 typo 識別字改正、GE2E/Triplet/F1 歸位 |
| P2 死碼移除 | `ddf2b52` | base_nn.py(992 行) 刪除；test/ 根腳本歸位 |
| P3 schema 拼字硬換 | `7cf7e02` | lightning_*/normalized 全面換新，40 檔 |
| P4a 通用/專用分層 | `7805e50` | ns.py 骨架＋hooks，VI 機制歸 VoiceIsolationDataset；雙指紋 bit-identical |
| P4b heads 搬家＋siso 分區 | `1f83685` | heads → lobe/heads.py（ckpt key 0 缺）；siso 分區 |
| P5 docs 同步 | `fd2662f` | 6 新頁、3 重寫、狀態欄、0 斷鏈 |
| P6 測試理性化 | 本 commit | ruff 政策入 pyproject（F/E9 全 repo 綠）、白名單移除、+5 tests（171 綠） |

最終閘（全過）：171 tests、ruff 全 repo、22 config parse+build、v7 bit-identical、
資料集雙指紋 bit-identical、streaming parity 63dB。
Scope: `puresound/`（25,251 行，其中 vendored `third_party/pytARD` ~3,600 行不動）、
`docs/`、`test/`。`egs/` 只做被動同步（import 路徑、config key 拼字），不重構。

---

## 0. 原則與已定案決策

1. **puresound 是模型庫**：所有 backbone（ConvTasNet / SkiM / TFGridNet / UnetTcn /
   DPRNN / DPARN / DPCRN / EcapaTdnn）**全部保留**，未來可能使用（user 定案）。
   推論：它們必須是「一等公民」——從 config 可達（`getattr(nnet, type)`）、有
   forward 煙測、docs 有頁。目前 ConvTasNet / DPRNN / UnetTcn / Unet **不在
   `nnet/__init__` 匯出清單內，config 寫這些 type 會直接 AttributeError**——要修。
2. **通用 vs 專用分開**：通用合成/訓練機制屬 `dynamic_base` / `ns` / `siso`；
   voice-isolation 專用機制（real pool、turn-taking、mix_mode、target_absent、
   overlap gating）下沉到 `voice_isolation`。
3. **TSE / SV 條線凍結為 legacy**：`task/tse.py`、`task/sv.py`、`system/miso.py`、
   `EncPredClassBase`、`dataset/kaldi_base.py`、`streaming/dparn.py` 保持可用、
   不重寫、docs 標註狀態。它們的既有行為由現有測試守著的部分照舊。
4. **VAD gate 基礎設施全數保留**（user 既有指示）：`VADHead`、gate 訓練路徑、
   gate loss。允許搬檔（ckpt key 綁 attribute 名，不綁定義檔路徑），不允許刪。
5. **每一步都有硬閘**（見 §7），任何一步失敗即回退該 commit。

---

## 1. 現況盤點（證據摘要）

### 1.1 使用圖（誰真的被用）

| 層 | 活躍 | Legacy（凍結） | 無引用 |
|---|---|---|---|
| task | `ns.py`、`voice_isolation.py`、`sampler.py` | `tse.py`、`sv.py` | — |
| system | `base.py`、`siso.py`（`EncDecMaskBase`）、`optim.py`、`logger.py` | `miso.py`、`EncPredClassBase` | — |
| nnet | `dpcrn`、`dparn`、`ecapa_tdnn`、`features`、`unet(Unet)`、`masker`（streaming 用） | — | **`base_nn.py`（992 行，前 Lightning 訓練包裝，被 `system/` 全面取代，全 repo 零 import）** |
| nnet（模型庫資產，僅測試引用） | `conv_tasnet`、`skim`、`tfgridnet`、`unet(UnetTcn)`、`dprnn` — **保留** | — | — |
| nnet/lobe | activation、attention、cnn、dsp、encoder、multiframe、norm、rnn、stft、trivial | — | `group_op.py`、`pooling.py` — **視同模型庫資產保留**（未來模型的積木），補進 docs 狀態標註 |
| nnet/loss | sdr、stft_loss、asr_feature、residual、dist、vad、spk | `metrics.py`（內容是 **GE2ELoss**，放錯地方） | — |
| audio | 全部活躍（io/dsp/noise/volume/vad/spectrum/rir_bank/room_simulator/hybrid_rir/augmentaion/impluse_response） | — | — |
| dataset | `dynamic_base.py`、`parser.py` | `kaldi_base.py` | — |
| streaming | `dpcrn.py` | `dparn.py` | — |

### 1.2 命名/拼字債

| 現況 | 應為 | 影響面 |
|---|---|---|
| `audio/augmentaion.py` | `augmentation.py` | 7 個 import 點 |
| `audio/impluse_response.py` | `impulse_response.py` | 4 個 import 點；**docs 那頁已拼對**（`docs/audio/impulse_response.md`），目前文件與程式碼名字不一致 |
| class `EcapaTdnnExtracotr` | `EcapaTdnnExtractor` | 3 個 SV/TSE config 以字串引用 |
| class `FrequecyEQLayer` | `FrequencyEQLayer` | 2 個 config 以字串引用 |
| config key `lighting_module` / `lighting_trainer_args` | `lightning_*` | 45+ yaml、4 個 egs main.py |
| config key `gain_nomalized_to` / `audio_gain_nomalized_to` | `*_normalized_*` | 全部訓練 yaml + dataset 建構參數 |

### 1.3 結構問題

- **繼承邊界反轉**：`NoiseSuppressionDataset`（1,330 行）持有全部 voice-isolation
  專用機制；掛名的 `VoiceIsolationDataset` 只剩 metadata 薄殼（205 行）。
- **兩個 metrics 命名空間**：`puresound/metrics.py`（評測，活躍）與
  `nnet/loss/metrics.py`（實為 GE2ELoss）。
- `system/siso.py` 669 行：通用 Lightning 管線與可選特性（channel_consistency、
  gate freeze、dry_blend/spec_floor）交織，無清楚分區。
- `VADHead` / `DistHead` 定義在 `nnet/dpcrn.py`——通用 head 積木住在單一 backbone
  檔內。
- `test/` 根目錄混入兩支非測試腳本：`simulate_room_scene.py`、
  `generate_simulated_training_data.py`。
- `test/run_repo_checks.py` 內建手寫 lint 白名單（`CHANGED_LINT_TARGETS`），已過時。

### 1.4 文件債

- `docs/nnet|system|task|dataset` 停在 **2026-05-15**；`docs/audio` 07-09、
  `docs/streaming` 07-03。
- 缺頁（活躍程式碼無文件）：`streaming/dpcrn`（只有 `dparn_onnx.md`）、
  `loss/asr_feature`、`loss/residual`、`loss/dist`、DPCRN 的 heads、
  `audio/rir_bank`（確認）、`recipes` 現行 tuple 契約。
- 記載已死程式碼：`nnet/base_nn.md`。
- 三個套件內 README（`nnet/`、`system/`、`dataset/`）語境過舊。
- 無 mkdocs.yml——docs 為手維護樹，唯一標準訂為「與程式碼一致」。

### 1.5 測試現況

- 155 passed。核心路徑（DPCRN 訓練/streaming/gate/real-pool/一致性/dist head/
  hybrid RIR/rir bank）覆蓋良好。
- backbone shape 煙測（`test_backbone.py` 7 條）——**保留**，它們是模型庫資產的
  唯一回歸防護。
- 真缺口（少量、有必要才補）：`system/optim.py` param-group 邏輯、
  `task/voice_isolation.py` metadata 發射的直接測試、P3 之後的新舊 config key
  等價測試。
- Backup config 引用三個已刪 loss（`ScalarAuxiliaryLoss` 等）——僅出現於
  `config/exp/backup/`，屬歷史文件，不處理。

---

## 2. P1 — 模型庫一等公民化＋機械改名（低風險）

1. `nnet/__init__.py` 匯出全部 backbone：補 `ConvTasNet`、`DPRNN`、`UnetTcn`、
   `Unet`（修正「config 可達性」缺口）。
2. `git mv audio/augmentaion.py audio/augmentation.py`、
   `git mv audio/impluse_response.py audio/impulse_response.py`；全 repo import 掃改
   （puresound/ egs/ test/ sdk/）。**不留 shim**（D2 定案），一次換乾淨。
3. 類名改正、**不留 alias**（D2 精神延伸）：
   - `EcapaTdnnExtracotr` → `EcapaTdnnExtractor`
   - `FrequecyEQLayer` → `FrequencyEQLayer`
   - repo 內引用這兩個名字的 yaml 同步換新名（legacy config 屬被動同步範圍）。
4. `nnet/loss/metrics.py` 的 `GE2ELoss` 併入 `nnet/loss/spk.py`（speaker loss 同居），
   刪 `loss/metrics.py`；`loss/__init__` 匯出不變。
5. 測試：`test_backbone.py` 加 DPARN 一條煙測補齊「每個匯出 backbone 都有 forward
   煙測」的不變量（DPARN 現只被 streaming 測試間接蓋到）；加一條
   「`nnet/__init__` 匯出的每個 backbone type 都能 `getattr` 到」的 parametrized 測試。

驗證：155+ 綠、全 config parse、`grep -r augmentaion|impluse|Extracotr|Frequecy` 僅剩 alias/shim 定義處。

## 3. P2 — 死碼移除（低風險、範圍已縮）

1. 刪 `nnet/base_nn.py`（992 行）＋ `docs/nnet/base_nn.md`（D1 已核准，
   動手前再全掃一次引用確認為零）。
   依據：前 Lightning 訓練包裝（`SoTaskWrapModule`/`EncDecMaskerBaseModel`），
   被 `system/` 全面取代，全 repo（含 egs、test、sdk）零 import。
2. `test/simulate_room_scene.py`、`test/generate_simulated_training_data.py`
   移出 test/（D1b 已核准）：前者移 `egs/rir_generation/`（房間場景試聽工具）
   或若已被涵蓋則刪；後者功能已被
   `egs/voice_isolate/scripts/check_training_data.py --dump` 覆蓋 → 刪。

驗證：155+ 綠（收集階段無 import error）、ruff 全綠。

## 4. P3 — Config schema 拼字相容（中風險、面廣但機械）

1. **直接硬換**（D3 定案），不留舊 key：`lighting_module`→`lightning_module`、
   `lighting_trainer_args`→`lightning_trainer_args`、`gain_nomalized_to`→
   `gain_normalized_to`；Python 端 kwarg/attribute `audio_gain_nomalized_to`→
   `audio_gain_normalized_to` 一併改名。讀取點全掃：`recipes.py`、四個 egs
   `main.py`、`streaming/` 的 config 驗證、tests 內嵌 config dict、
   voice_isolate scripts。
2. Repo 內全部 runnable yaml（含 `config/exp/`，**不含 `config/exp/backup/`**——
   歷史文件，本就標註不可跑）換成正確拼字。
3. 測試由既有全套把守（改壞任何讀取點 155 個測試會抓到），不另寫新舊 key 等價
   測試（舊 key 已不存在）。

驗證：155+ 綠、全 config parse、v7 經 default infer config 輸出 bit-identical。

## 5. P4 — 通用/專用分層（高風險、單獨 commit、可獨立 revert）

1. **ns → voice_isolation 下沉**：`_load_real_pool`、`_sample_realfar_interferers`、
   realnear 分支、turn-taking（`_sample_turn_script`/`_apply_overlap_gating` 的
   override 部分）、`mix_mode`、`augmentation_realfar/realnear` 參數，從
   `NoiseSuppressionDataset` 移入 `VoiceIsolationDataset`。
   - **RNG 保序是硬約束**：搬移不得改變 `torch.rand`/`random` 呼叫順序；
     以既有「seeded item bit-identical」測試把守，voice_isolation 與 ns 兩個
     task type 各驗一次。
   - `NoiseSuppressionDataset` 回到通用語義：前景 + interferer + noise +
     device chain。`egs/noise_suppression` 的 config（task 未寫 voice_isolation 者）
     行為不變。
2. **heads 搬家**：`VADHead`、`DistHead` → `nnet/lobe/heads.py`；`dpcrn.py`
   `from .lobe.heads import ...`。ckpt key（`backbone.vad_head.*`）不受影響——
   以「v6_gate ckpt strict load 缺 0 鍵」測試驗證。
3. **siso.py 分區**：不改行為，把可選特性整併為清楚的區塊（channel-consistency
   一組、gate-freeze 一組、inference knobs 一組），`__init__` 參數與文件對齊。

驗證（本階段全套）：155+ 綠、seeded bit-identical ×2、v7 bit-identical、
streaming parity 63dB、`overfit_check.py --gate` 煙測可跑。

## 6. P5 — 文件同步（低風險）

1. 刪：`docs/nnet/base_nn.md`（隨 P2）。
2. 補：`docs/streaming/dpcrn_onnx.md`、`docs/nnet/loss/{asr_feature,residual,dist}.md`、
   `docs/nnet/lobe/heads.md`、`docs/audio/rir_bank.md`（若缺）、
   `docs/recipes.md` 更新為現行 20-tuple 契約與新舊 key 相容表。
3. 改：`docs/index.md` 與各 index 的模組表對齊現況；backbone 頁加「狀態」欄
   （active / library asset / legacy）；`docs/system/miso.md`、`docs/task/{tse,sv}.md`
   標 legacy；三個套件內 README 重寫為指向 docs 的短版。
4. 拼字：`impulse_response.md` 在 P1 改名後與程式碼一致，檢查頁內 import 範例。

驗證：docs 內所有 `puresound.*` 引用經腳本比對存在（一次性檢查腳本，不入庫）。

## 7. P6 — 測試理性化（低風險）

1. `test/run_repo_checks.py`：刪 `CHANGED_LINT_TARGETS` 白名單，改全 repo
   `ruff check .`＋快測集。
2. 補三個小測試（僅真缺口）：`system/optim.py` param-group／lr-factor 邏輯、
   `voice_isolation` metadata 發射（NaN 語義）、P3 的 key 等價（已列 P3）。
3. 不新增任何「為覆蓋率而寫」的測試；backbone 煙測保留現狀＋P1 的兩條。

## 8. 全程硬閘（每個 commit 後必跑）

```bash
uv run pytest test -q                      # 155+ 全綠
# 全部 config parse + build（20 個，不含 backup）
# v7 ckpt 經 config/infer_dpcrn.yaml 輸出 bit-identical（diff 0.0）
# streaming parity：scripts/streaming_onnx.py verify → 63.0 dB PASS
# seeded dataset item bit-identical（test_realfar_interferer 既有 + P4 加強版）
ruff check puresound/ egs/ test/           # F 級 0（E402 慣例除外）
```

## 9. 開放決策點

| # | 問題 | 決議（2026-07-29） |
|---|---|---|
| D1 | `nnet/base_nn.py` 刪除？ | **刪**（確認零引用後移除；git 史保留） |
| D1b | test/ 根下兩支非測試腳本 | **照建議**：`simulate_room_scene.py` 移 rir_generation、`generate_simulated_training_data.py` 刪 |
| D2 | 檔名改名留相容 shim？ | **不留**，一次換乾淨。連帶：類名改正也不留 alias，config 直接換新名 |
| D3 | config key 拼字過渡策略 | **直接硬換**，不留舊 key；repo 內 runnable yaml 全換（backup 除外） |
| D4 | commit 粒度 | **P1–P6 各一 commit，P4 拆 a/b 兩個** |

## 10. 明確不做

- 不動 `third_party/pytARD`（vendored）。
- 不重寫 TSE / SV / miso / kaldi_base / streaming.dparn（凍結標 legacy）。
- 不刪任何 backbone、lobe 積木、其測試（模型庫原則）。
- 不動 `config/exp/backup/`（歷史文件）。
- 不引入 mkdocs／CI 等新基礎設施（本輪只求文件與程式碼一致）。
