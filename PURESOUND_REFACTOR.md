# puresound 模組重構評估與計劃

> 本檔是 `puresound/` 函式庫的**架構審查結論與重構工作計劃**。它不是產品文件：各模組的
> 使用方式見 [docs/index.md](docs/index.md)，RIR 子系統的實驗歷史見
> [RIR_EXP_LOG.md](RIR_EXP_LOG.md)。
>
> 審查日期：2026-08-17。審查範圍：`puresound/` 143 個檔案 / 49,155 行（不含
> `third_party/` 46 檔 5,482 行）。`egs/` 只在它反映 `puresound/` 抽象缺口時才列入。
>
> 做完的項目在「執行紀錄」打勾並註明 commit；**發現本身不刪**，保留完整歷史。

## 0. 總評

這份 codebase 的知識密度很高，註解寫的是「為什麼」而不是「做什麼」（DDP deadlock 的
成因、`sync_dist=False` 為何必要、peak normalization 抹掉距離位準線索的副作用、
`arange` 不含上界的 bug）。這是**最大資產，重構時逐字搬運，不要順手精簡**。

問題不在寫得糟，而在於三條軸線是一路長出來的、沒有回頭收斂：config 傳遞、loss 分派、
`__getitem__`。

### 現況量測

| 區塊 | 檔數 | 行數 | 佔比 |
|---|---:|---:|---:|
| `audio/rir/` | 76 | 31,114 | 63% |
| 訓練核心（`task`+`system`+`nnet`+`dataset`+`streaming`） | — | 15,306 | 31% |
| `audio/`（非 rir） | — | ~2,700 | 5% |
| **合計（不含 third_party）** | **143** | **49,155** | |
| `third_party/pytARD`（vendored，ruff 已排除） | 46 | 5,482 | — |

`ruff --select F,E9`（repo 現行政策）全綠。以下發現全部來自人工審查加上
`--select ALL` 的顧問級訊號。

---

## 1. 架構層面

### A1. Config 參數隧道 ⭐⭐⭐

同一包設定值以「攤平的 20 個獨立參數」穿過 5 層：

| 層 | 位置 | 形態 |
|---|---|---|
| 1 | `recipes.py:17` `load_siso_recipe_config` | 回傳 **20-tuple**（位置相依） |
| 2 | `system/runner.py:64` `RecipeConfig` | **20 欄位** dataclass（為了讓 tuple 可讀而生） |
| 3 | `egs/*/main.py init_dataloader` | 16 個位置參數，docstring 明言「positional signature is part of the recipe's surface」 |
| 4 | `system/runner.py:102` `build_dataloaders` | 17 個 keyword 參數 |
| 5 | `task/ns.py:57` / `dataset/dynamic_base.py:17` | 20 / 17 個參數，落地成 15 個 `self.augmentation_*_args` |

代價可實測：新增一個 augmentation block（`realfar`/`realnear`）要同步改 6 個檔案。
`RecipeConfig` 的 docstring 已自承 "a recipe reading twenty positional fields cannot be
checked by eye"——那是徵狀，不是解法。

中途**沒有 schema 驗證**。`_enabled_config` 只檢查 `used`，其餘全靠各消費點自己
`cfg.get("prob", 0.0)`。config 打錯字（`porb`）不會報錯，只會安靜地變成機率 0——做消融
實驗時這是最貴的一種 bug。

### A6. augmentation 軸的複雜度已經在腐蝕（2026-08-17 量測）⭐⭐⭐

A1 原本被歸在 P3「等你要加第 3 個 augmentation block 再說」。量完之後這個判斷是錯的：
腐蝕不是未來式，已經發生了。

**config 側**

| 量測 | 數字 |
|---|---|
| 現役 `train_dpcrn.yaml` 的 augmentation 頂層區塊 | 14 |
| 其中的旋鈕（葉節點） | 71（整份 config 175 個、469 行） |
| 巢狀最深 | 3 層 |
| pipeline 讀得到的 config key | 62 個，散在 **128 個讀取點** |
| 讀法 | **57 處硬取 `cfg["k"]`（缺就 KeyError）/ 71 處軟取 `cfg.get("k", 預設)`（缺就靜默用預設）** |
| schema 驗證 | 無 |

「硬取 vs 軟取」沒有規則可循——同一個 block 內兩種混用，所以「漏寫一個 key 會怎樣」
無法從 config 本身判斷，得去翻程式碼。

**已經累積的死旋鈕**：`augmentation_query_distance` 整叢 **10 個旋鈕**（`derive_from_bank`、
`bank_margin`、`range`、`peak_distance`、`peak_prob`、`peak_half_width`、`near_floor`、
`far_ceiling`、`gate_silence.prob`、`gate_silence.margin`）在 commit `380da2e` 移除
distance-query 軸之後**沒有任何程式碼讀取**，卻仍存在於 **33 份 config**。同類的還有
`prob_by_origin` / `turn_taking_prob_by_origin` / `real`。現役 default config 已清乾淨，
但每一份 exp config 都還帶著——下次有人照抄 exp config 開新實驗，就會以為自己在調一個
早就不存在的機制。

沒有 schema 的直接後果就是這個：**刪掉機制不會讓它的 config 報錯**。

**pipeline 側**

| 量測 | 數字 |
|---|---|
| `ns.__getitem__` 的 `if` 分支 | 42 |
| 其中以 `torch.rand` 抽機率的增強區塊 | 17 |
| 一路帶下去、必須保持互相一致的訊號 | 8 |
| `noisy_speech` 被重新指派 | 8 次 |
| `target_speech` / `added_noise` | 各 6 次 |

最脆弱的是 **flag + 延後重播** 這個樣式。SRC / IIR / HPF / volume 四段各自：先設
`flag_*` 並保存抽到的參數，然後在**一兩百行之後**把同一組參數重播到 `added_noise` 上：

| 耦合 | 設定於 | 重播於 | 跨距 |
|---|---:|---:|---:|
| `flag_src` | L649 | L855 | **212 行** |
| `flag_iir` | L704 | L885 | 187 行 |
| `flag_hpf` | L719 | L890 | 177 行 |
| `flag_volume` | L745 | L898 | 159 行 |

新增任何一段會動到 `noisy_speech` 的增強，都必須記得在 200 行外補一段對應的重播，否則
`added_noise` 會與 `noisy_speech` 靜默不同步。沒有任何機制在檢查這件事。

**而 `added_noise` 沒有人用。** 它進了 `sample` 字典（`ns.py:946`），但
`NoiseSuppressionCollateFunc` **不會把它收進 batch**——它只出現在該 class 的一段
docstring 註解裡。全 repo（`puresound/`、`egs/`、`test/`）沒有任何地方讀
`batch["added_noise"]`。也就是說：整條 pipeline 裡最纏繞的一塊機制（4 組跨百行耦合、
約 90 行程式碼），產出的東西在 batch 邊界就被丟掉了，而且沒人發現。

同一類但程度較輕的還有 `far_target`：有被 collate，但唯一的消費者是一支 eval 腳本；
它的註解說是給 `FarReconstructionLoss` 用的，而**那個 loss 不存在**。
（對照組：`consistency_noise` 是活的——`ResidualReferenceLoss` 預設就讀它。）

**這改變了什麼**

A1（config 參數隧道）與 A6 是同一個問題的兩面：**一包沒有型別、沒有 schema、沒有
擁有者的 dict，穿過 5 層再散進 128 個讀取點**。它的成本已經不是「未來加東西會麻煩」，
而是三筆已發生的事實：33 份 config 帶著 10 個死旋鈕、四組跨百行的隱式耦合、以及一塊
沒人要卻仍在維護的 90 行機制。

計劃相應調整：把 config schema 驗證（原 P3-2）與死旋鈕/死 payload 清理拉到 **P2**，
見下方 2-5 / 2-6。

### A2. 側通道狀態（temporal coupling）⭐⭐

模組間靠「上一次呼叫留下的 attribute」溝通，共 4 組：

- `system/siso.py:164` `self.last_mask` — `forward()` 寫、`training_step` 讀，註解自承
  "overwritten by every forward, so read it right after the call"
- `nnet/dpcrn.py:235` `backbone.last_vad_logits` / `last_dist_preds` → `system/siso.py:295`
  用 `getattr` 撈
- `audio/augmentation.py:560` `augmentor._last_rir_meta` → `task/ns.py:216`
  `getattr(self.augmentor, "_last_rir_meta")`（跨模組讀私有屬性，而
  `apply_rir` 的**回傳值裡本來就有** `metadata`）
- `task/ns.py:379` `self._last_overlap_fraction` / `_last_turn_taking`

單執行緒訓練下正確，但契約完全隱式：任何人在 forward 與讀取之間插一次 forward 就靜默
壞掉，且沒有 test 抓得到。

### A3. Loss 分派的 if/elif 鏈 + 一條懸空契約 ⭐⭐

`system/siso.py:307-341` 是 6 段 `elif getattr(loss_func, "uses_XXX", False)`。每加一種
loss 簽章就要改核心迴圈一次（OCP 違反）。且**順序有語意**：`uses_vad_logits` 排第一，
所以 `BackgroundVADHeadBCELoss` 必須把 `uses_vad_logits = False` 明寫出來
（`nnet/loss/vad.py:208`）才不會被上一個分支吃掉。這種「靠宣告順序才正確」的設計不會
有人第二次讀對。

**其中一條分支的對端不存在**：`last_background_vad_logits` 全 repo 只有兩處出現——
`siso.py:298` 讀它，以及 `test_siso_compute_loss.py` 的 test double 設它。沒有任何
backbone 產生它（commit `380da2e` 移除 conformer 軸時，head 走了、loss 與分派分支留下）。

> 這與「有些 model type 沒被用到不代表不該存在」是不同的東西：backbone 沒被用到是
> **閒置的庫存資產**（該留）；這條是**核心迴圈裡指向不存在對端的分支**。
> 處置見 P0-5——依 2026-07-16 的「gate 機制全套不得移除」決定，保留 hook，只修錯誤訊息。

同一個 loss reduce 迴圈（`overall_loss = []` → 迴圈裡改賦成 tensor）存在 **4 份**：
`siso.py:305`、`siso.py:568`、`miso.py:202`、`miso.py:222`。

### A4. `runner.py` 只服務 SISO ⭐

`runner.py` 的抽象方向正確（recipe 只擁有 task，不擁有 argparse），但只覆蓋 SISO。結果
`egs/target_speaker_extraction/main.py` 497 行、`egs/speaker_embedding/main.py` 369 行
各自帶著一份 argparse + trainer 組裝 + scoring/inference——正是 runner 存在要消滅的
東西。`init_siso_model` 在 `recipes.py:44` 與 tse main:174 也是兩份。

（TSE/SV 目前凍結 legacy，此項不排入近期計劃，見 P3-4。）

### A5. `audio/rir` 的邊界

**做得好，不要動**：`rir/__init__.py` 明確說出「訓練路徑只碰 `bank.loader` 一個模組」，
並由 `test/test_rir_r0_import_boundaries.py` 用 LAYER_RANK 表**機械化強制**分層，同時
保證 `contracts`/`scene`/`metrics` 不需要 torch 就能 import。這是整份 codebase 架構紀律
最高的地方。

**空頭的第二套契約（已於 P1-5 處置）**：`rir/api.py` 自稱 "Stable public entry
point"，但**零個生產呼叫端**——`egs/rir_generation` 全部直接深入
`physics.impedance.modes`（13 次）、`physics.wave.fdtd`（7 次）等內層模組。而且它也服務
不了：頂層 7 支腳本需要的 40 個符號裡 **25 個不在 api 的出口**，`phases/` 是 191 個裡缺
173 個——那些正是 api 自己說「internal helper 或 stage-specific tool」的東西，代表它劃的
線與呼叫端需要的線不是同一條。沒有呼叫者就沒有東西會在它壞掉時報錯，那個穩定性宣稱是
空頭的。處置見 P1-5：保留 re-export，撤掉宣稱。

---

## 2. 複用性

### B1. `ns.__getitem__` 與 `tse.__getitem__` 有 332 行逐字重複 ⭐⭐⭐

difflib 行級實測：677 行 vs 513 行，**332 行完全相同**，最大連續區塊 **108 行**
（`# SRC` 起的整段 SRC/IIR/HPF/volume 裝置鏈）與 **78 行**（`added_noise` 重新套用同一條
鏈）。

這是模組裡最大的一塊複製貼上，而且是會漂移的那種：`ns.py:508` 有個修過的 bug
（`arange` 不含上界，害速度擾動只剩減速一半），註解寫得很清楚；`tse.py` 那份是不是也修
了，得逐行比對才知道。

> TSE/SV 為凍結 legacy，因此計劃是**只從 `ns.py` 抽出可重用元件**，`tse/sv` 維持不動，
> 不追求三邊統一（見 P2-1）。

### B2. `if_none_else` 定義 3 份、呼叫 84 次 ⭐⭐

`task/ns.py:28`、`task/tse.py:15`、`task/sv.py:12` 三份一模一樣的實作（就是
`a if a is not None else b`）。84 次呼叫裡絕大多數是同一個運算式
`if_none_else(self.target_sr, self.ori_audio_sr)`——光 `ns.__getitem__` 一個函式就出現
20 次以上。這應該是 `DynamicBaseDataset` 上的一個 `@property audio_sr`。

### B3. `streaming/dpcrn.py` 與 `streaming/dparn.py` 重複 308 行 ⭐⭐

567 行 vs 512 行，**308 行相同**（ratio 0.57），最大連續區塊 43 + 22 行（都在 ONNX 匯出
段）。`_require`、`_as_list`、`_tensor_shape`、`_down_step`、`_up_step`、
`export_*_onnx`、`Streaming*Ort` 幾乎逐字兩份。`streaming/dpcrn.py` 檔頭已自承
"Mirrors puresound/streaming/dparn.py"——知道問題，但沒抽。真正的差異只有 state 欄位與
block step（`dpcrn` 多一組 lookahead delay line）。

### B4. 輔助 head 只綁在 DPCRN 上

`nnet/lobe/heads.py` 寫得很好——docstring 明確說「任何 bottleneck 是 `[N,C,F,T]` 的
backbone 都能掛」，且說明 checkpoint key 由 attribute 名決定。但實際上**只有 `dpcrn.py`
掛了**，掛法（`if head_cfg is not None and head_cfg.get("enabled")` + `self.last_xxx =`）
是 30 行手寫樣板。換成 DPARN 或 TFGridNet 要再抄一次。缺一層共用掛載契約。

### B5. `nnet.__init__.__all__` 是手寫同步清單

`nnet/__init__.py:14` 的註解要求「Keep this in sync with the imports above」——人工同步
契約。`test_backbone.py:236` 只驗證「config 裡出現的 type 都取得到」，不驗證反向。實務
效果：`UnifiedConvEncDec`（在 `encoder.py`、被 test 用）不在 `__all__` 裡，config 寫不
出來。這不是「該刪」的問題（庫存資產應保留），是「庫存清單與貨架不一致」的問題。

---

## 3. Clean code 細節

### C1. God method

| 函式 | 行數 | 圈複雜度 |
|---|---:|---:|
| `task/ns.py:326` `__getitem__` | 677 | **44** |
| `task/tse.py:220` `__getitem__` | 512 | 36 |
| `task/sv.py:60` `__getitem__` | 274 | 19 |
| `dataset/dynamic_base.py:123` `gen_meta` | 140 | 26 |

`ns.__getitem__` 已抽出 5 個 hook（`_plan_row` / `_prepare_foreground` /
`_sample_interferers` / `_turn_taking_override` / `_mix_foreground_with_interferers`）——
**方向完全正確**，`voice_isolation` 靠這 5 個 hook 就把整個 near/far 軸乾淨地疊上去，
是這份 code 裡最漂亮的一次設計。但抽的只有前半段的「來源決策」，後半段 400 行的裝置鏈
（SRC→IIR→HPF→volume→codec→packet loss→peak guard→VAD 標記→metadata）完全沒動。

參數量前三：`nnet/unet.py:274` `UnetTcn.__init__` **30 個參數**、`unet.py:523` 25 個、
`nnet/dpcrn.py:121` 21 個。

### C2. 型別註記與實作不符

- `task/ns.py:65`、`task/tse.py:32`、`task/sv.py:28`：`augmentation_speech_args:
  Optional[int]`，實際傳的是 `Dict`。三份都錯，複製貼上傳染。
- 宣告 `-> torch.Tensor` 但回傳 tuple：`audio/augmentation.py:466` `apply_rir`、
  `audio/augmentation.py:563` `apply_2nd_iir_response`、`nnet/masker.py:130`
  `apply_wiener`、`nnet/masker.py:162` `apply_mvdr`。
- `AudioEffectAugmentor` 整套 `apply_*` 回傳 `(wav, (a, b, c))` 這種裸 tuple-of-tuple，
  呼叫端靠位置解包（`noisy, (added_noise, _, _) = ...`）。改 NamedTuple/dataclass 就能
  自我說明。

### C3. 函式庫層直接 `print`

54 處，集中在 `dataset/dynamic_base.py`（26）、`system/base.py`（5）、
`dataset/kaldi_base.py`（4）、`audio/io.py`（4）。`gen_meta` 一次列印約 15 行語料統計——
DDP 下每個 rank 每次建 dataset 都印一遍。函式庫不該決定輸出去哪裡；這也讓「靜默載入」
在被當 SDK 用時不可能。

### C4. 死碼與空實作

- `dataset/dynamic_base.py:209` 與 `:216`：`if len(gender_meta["other"][cid]) < 0:` ——
  長度不可能 < 0，**這個過濾器從來沒生效過**。print 訊息寫「remove corpus id because
  speaker numbers less than 4」，可見原意是 `< 4`。意圖與實作不符，且沒有 test 抓得到
  （兩條分支只差一行 print）。
- `dataset/dynamic_base.py:117` `def __len__(self): pass` —— 回傳 `None`。目前靠
  `batch_sampler` 繞過，但任何走 `len(dataset)` 的路徑會拿到 `TypeError`。
- 裸 `raise NameError`（無訊息）4 處：`system/siso.py:202`、`system/miso.py:177`、
  `nnet/unet.py:355`、`nnet/dpcrn.py:42`。錯誤型別本身也不對（該是 `ValueError`）。
- `system/siso.py:152` `if wav.dim() != 2 and wav.shape[0] == 1:` —— 意圖顯然是
  `dim() == 3`；現在 4 維且 `shape[0]==1` 也會 squeeze 一層。三個模組各一份相同寫法。

### C6. 多取樣率路徑上的 per-item seed 不能跨 process 重現（2026-08-17 於 P1 驗證途中發現，已修）⭐

> **範圍更正**：初次回報時我寫成「直接推翻 `runner.py:166` 的可重現性宣稱」，那是**錯的**。
> 實測後確認現役 recipe 走的是另一條分支，宣稱對它們成立。以下是修正後的版本。

`dataset/dynamic_base.py` `choose_an_utterance_by_speaker_name` 有兩條選池分支：

| 條件 | 池子來源 | 是否決定性 |
|---|---|---|
| `select_with_sr_as_key is None` | `meta[spk]["utts"]` 這個 dict 的 key 順序（= metafile 順序） | ✅ |
| 給了 `select_with_sr_as_key` | `sr_meta[sr][spk]` 這個 list，**但被 `set` 繞了一圈** | ❌（修正前） |

走哪條由 `runner.py:147` 的 `select_by_sr_first = False if target_sample_rate else True`
決定：設了具體 `target_sample_rate` → sampler 送出 `sr=None` → 走第一條。**所有出貨的
config 都設了具體值**，所以現役 recipe 一直走在決定性那條。實測 `batch_sr=None` 下
**0 / 1187** 個雜湊隨 `PYTHONHASHSEED` 改變。

只有 `target_sample_rate: null`（多取樣率語料）會走第二條，而那條在修正前實測
**99 / 1187** 個雜湊隨 `PYTHONHASHSEED` 改變。根因：

```python
check_key_list = set(target_speech_pool)
...
target_speech_pool = list(check_key_list)   # <- str hash 順序
tgt_key = random.sample(target_speech_pool, k=1)[0]
```

`random.sample` 抽的是索引，索引落在一個順序隨 `PYTHONHASHSEED` 改變的 list 上，所以
同一個 seed 在不同 process 會選到不同 utterance。示範（seed 都是 1234）：

| PYTHONHASHSEED | `list(set(...))` 選到 | `sorted(set(...))` 選到 |
|---|---|---|
| 0 | `spk_utt0` | `spk_utt3` |
| 1 | `spk_utt3` | `spk_utt3` |
| 2 | `spk_utt1` | `spk_utt3` |

`task/ns.py` 已經在別處避開同一個陷阱（`random.sample(sorted(spk_pool), ...)`、
`random.choice(sorted(echo_pool))`）——有人知道這件事，只是 `dynamic_base` 這條漏了。

修法見 P1-6：不繞 `set`，直接保留 list 的 metafile 順序，與上面那條分支對齊。因為現役
recipe 不走這條，修正對它們是零影響。

### C5. 推理策略混進訓練模組

`dry_blend` / `spec_floor`（`system/siso.py:126-243`）是部署期的過度抑制補償參數，住在
LightningModule 的 `forward()` 裡。docstring 誠實標了 "inference-only knobs"，但它們仍
增加訓練路徑的分支數，且部署端要複製這段邏輯才能對齊。應該是獨立的 `Postprocessor`，
訓練與部署共用同一份。

---

## 4. 明確「不要動」的部分

1. **註解品質** —— 見總評。重構時逐字搬運。
2. **`rir` 分層 + import boundary test** —— 見 A5，架構紀律的示範。
3. **RNG 順序不變性紀律** —— `ns.py` 每個 knob 都標
   「absent/disabled block never touches the RNG stream」，讓舊 recipe 可 bit-identical
   重現。**這是後面所有重構的硬約束。**
4. **backbone / lobe 作為模型庫資產** —— 零引用不等於該刪（2026 定案，見記憶
   `backbones-are-library-assets`）。本檔沒有任何一條建議刪 model type。
5. **VAD gate 基礎設施** —— 2026-07-16 決定「gate 機制全套留在現役程式，清理時不得
   移除」。P0-5 因此採保留路線。
6. **`test/run_repo_checks.py` 三層 suite** —— quick/standard/full 分層與 xdist
   單執行緒 pin 都有實測數據支撐。

---

## 5. 重構計劃

依「風險/收益比」排序，不依重要性。每階段都給可機械驗證的驗收條件。

### P0 — 零行為風險（已完成，見執行紀錄）

| # | 事項 | 驗收 |
|---|---|---|
| 0-1 | 移除 `dynamic_base.py:209,216` 從未生效的 `< 0` 過濾器 | 行為逐位元不變；若真要 `< 4` 過濾，那是**刻意的行為變更**，另案處理 |
| 0-2 | 修 3 處 `Optional[int]` → `Optional[Dict]`；修 4 處 `-> torch.Tensor` 實回 tuple | 純註記，無執行期變化 |
| 0-3 | 4 處裸 `raise NameError` → 帶訊息的 `ValueError` | 既有 test 全綠 |
| 0-4 | `DynamicBaseDataset.__len__` 從 `pass` 改為 `raise NotImplementedError` | Lightning `sized_len` 同時捕捉 `TypeError`/`NotImplementedError`，且 runner 用 `use_distributed_sampler=False`，無呼叫端 |
| 0-5 | `BackgroundVADHeadBCELoss` 的懸空 hook：**保留**（依「gate 不得移除」），只把誤導的錯誤訊息改成子類感知 | 設定該 loss 而 backbone 無對應 head 時，錯誤訊息指向正確的屬性名 |

### P1 — 低風險，純結構（約 1–2 天）

| # | 事項 | 做法 | 驗收 |
|---|---|---|---|
| 1-1 | 消除 `if_none_else` 三份定義 | 移到 `DynamicBaseDataset` 的 `@property audio_sr`；`tse/sv` 保留自己那份（凍結不動），只改 `ns.py`/`voice_isolation.py` | `ns.py` 的 36 處呼叫歸零；固定 seed 下前 32 個 item 逐位元相同 |
| 1-2 | 統一 4 份 loss reduce 迴圈（已完成） | `BaseLightningModule.reduce_losses(...)` | 見執行紀錄 |
| 1-3 | 函式庫 `print` → `logging`（已完成） | module logger；rank 過濾放在 handler 上而非 `gen_meta` 裡——library 一律送出，由應用端的 handler 決定丟棄，`dataset/` 不必知道 rank 這件事 | 見執行紀錄 |
| 1-4 | `AudioEffectAugmentor` 回傳值改 NamedTuple | `apply_rir` → `RirResult(wav, rir_id, mode, metadata)`；同時讓 `ns.py:216` **改讀回傳值**而非 `_last_rir_meta` | `_last_rir_meta` 的外部讀取歸零（grep 可驗證） |
| 1-5 | 決斷 `rir/api.py`（已完成，選 B：撤掉宣稱） | 見執行紀錄 | — |
| 1-6 | 修 C6：sr-keyed 選池不再繞 `set`（已完成） | `dataset/dynamic_base.py` `choose_an_utterance_by_speaker_name` | 見執行紀錄 |

### P2 — 中風險，需 bit-identical 驗證（約 3–5 天）

**硬前提**：先寫一支「RNG 指紋」測試——固定 seed 跑 `VoiceIsolationDataset` 前 32 個
item，把 `noisy_speech`/`clean_speech`/全部 metadata scalar 的 sha256 存成 golden
fixture。**沒有這支 test 就不要開始 P2**：`ns.py` 的正確性有一半在 RNG 呼叫順序上，那是
任何 code review 都看不出來的。

P1 期間已經做出一支可用的原型（見執行紀錄；1187 個雜湊、約 6 秒）。C6 / P1-6 修完後
它兩條路徑都跨 `PYTHONHASHSEED` 穩定，所以 promote 成 repo 內的 test 只差把語料
fixture 從 scratchpad 移進 `test/`。

| # | 事項 | 做法 |
|---|---|---|
| 2-1 | 拆 `ns.__getitem__` 的裝置鏈 | 抽 `task/device_chain.py`：`DeviceChain.apply(...) -> ChainResult`，內含 SRC/IIR/HPF/volume/codec/packet-loss 與 `added_noise` 重放。**只從 `ns.py` 抽，`tse/sv` 維持凍結** |
| 2-2 | 拆 `_apply_overlap_gating`（複雜度 14、142 行） | 拆成 `_turn_taking_envelope` / `_bernoulli_envelope`，共用 `_gate`/`_smooth_env` |
| 2-3 | `streaming` 抽共用基底 | `StreamingFrameModelBase` 承載 `_require`/`_as_list`/`_down_step`/`_up_step`/ONNX 匯出/`*Ort`；`dpcrn`/`dparn` 只留 state 定義與 block step |
| 2-4 | 輔助 head 掛載樣板化 | `AuxHeadMixin.attach_heads(cfg, enc_channels)` + `collect_side_outputs() -> dict`。**checkpoint key 必須維持 `backbone.vad_head.*`** |
| 2-5 | **config schema 驗證**（原 P3-2，因 A6 提前） | 對 14 個 augmentation 區塊寫一份 schema，`load_hparam` 之後檢查：未知 key 直接報錯、缺 key 依 schema 決定「必填」或「預設值」。這同時消滅「硬取 vs 軟取」的隨機性——規則寫在 schema 裡，不在 128 個讀取點裡 |
| 2-6 | 清死旋鈕與死 payload（依賴 2-5） | schema 一上線，33 份 config 的 `augmentation_query_distance` 等 10 個旋鈕會立刻報錯，順勢清掉；同時決斷 `added_noise`（無人讀 → 連同 4 組 flag 重播一起刪，約 −90 行）與 `far_target`（`FarReconstructionLoss` 不存在 → 補 loss 或降級為 eval-only 並標註） |

**2-5 / 2-6 的驗收**：schema 上線後，現役 `train_dpcrn.yaml` 必須零錯誤通過；33 份 exp
config 的死旋鈕全部被報出來；刪掉 `added_noise` 後 RNG 指紋逐位元不變（它不影響任何抽樣
——但**必須實測**，因為它經過 `apply_clipping_distortion`，要確認那條路徑真的不耗 RNG）。

**驗收**：2-1/2-2 後 RNG 指紋 test 逐位元通過；2-3 後 ONNX 匯出對同一 ckpt 產生數值相同
的輸出（tol 1e-6）；2-4 後 `dpcrn_v8/v9/v10` 三個 ckpt 都能 `strict=True` 載入。

### P3 — 大型，有動機時再做

| # | 事項 | 觸發條件 |
|---|---|---|
| 3-1 | **Config 物件化**：20-tuple → 巢狀 dataclass（`AugmentationConfig` 聚合全部 `augmentation_*`），一路傳到 dataset 只剩 3–4 個參數 | 2-5 的 schema 落地之後——schema 本身就是 dataclass 的形狀，屆時物件化幾乎是把 schema 換個寫法 |
| ~~3-2~~ | **Config schema 驗證** → 因 A6 提前為 **2-5** | — |
| 3-3 | **Loss 分派改註冊制**：`loss.required_inputs = ("vad_logits",)` + `{name: provider}` 表取代 if/elif 鏈 | 下次要新增第 7 種 loss 簽章時 |
| 3-4 | `runner.py` 泛化到 MISO/SV，收掉 tse/sv main 的 866 行重複 | TSE/SV 解凍時。**目前凍結中，不要碰** |
| 3-5 | `dry_blend`/`spec_floor` 抽成獨立 `Postprocessor` | 要做 SDK 部署對齊時 |

---

## 6. 執行紀錄

### 2026-08-17 — P0 完成

驗收：`ruff check`（F,E9）全綠；`test/run_repo_checks.py --suite standard`
**615 → 616 passed**（+1 為 P0-5 新增的回歸測試），無既有測試變動。

| # | 改動 | 檔案 |
|---|---|---|
| 0-1 | 移除從未生效的 corpus 過濾器（條件是 `len(...) < 0`），保留統計 print，改成明寫 `self.all_corpus_id = set(all_corpus_id)`。**沒有**把邊界「修正」成訊息宣稱的 `< 4`——那會改變訓練分佈，屬於刻意的實驗變更，不是清理 | `dataset/dynamic_base.py` |
| 0-2 | 3 處 `augmentation_speech_args: Optional[int]` → `Optional[Dict]`；4 處 `-> torch.Tensor` 實回 tuple 的註記改對（`apply_rir`、`apply_2nd_iir_response`、`apply_wiener`、`apply_mvdr`） | `task/{ns,tse,sv}.py`、`audio/augmentation.py`、`nnet/masker.py` |
| 0-3 | 4 處裸 `raise NameError` → 帶訊息並列出合法值的 `ValueError` | `system/siso.py`、`system/miso.py`、`nnet/unet.py`、`nnet/dpcrn.py` |
| 0-4 | `DynamicBaseDataset.__len__` 從 `pass`（回傳 `None`）改為 `raise NotImplementedError`，docstring 說明為何動態合成沒有固定 epoch 大小 | `dataset/dynamic_base.py` |
| 0-5 | `BackgroundVADHeadBCELoss` 的懸空 hook **保留**；`VADHeadBCELoss` 的錯誤訊息改為子類感知（`_logits_attr` / `_head_config_key` / `_target_key`），並在 `siso.py` 的分派點註明「沒有現役 backbone 產生它、分支刻意留著」 | `nnet/loss/vad.py`、`system/siso.py` |

同步更新的文件：`docs/dataset/dynamic_base.{md,zh-TW.md}`（corpus 過濾器與
`__len__` 的敘述已過時）、`docs/nnet/loss/vad.{md,zh-TW.md}`（class body 引用）。

#### 兩項判斷的理由

**0-1 為何不改成 `< 4`**：那不是修 bug，是加一個從未存在過的過濾器。目前所有已發布
的 ckpt（`dpcrn_v8/v9/v10`）都是在「無 corpus 過濾」的分佈上訓練的；把邊界接上會靜默
改變語料組成，讓新舊實驗不可比。要加就該當成一次有對照的實驗。

**0-5 為何保留而非移除**：2026-07-16 已有決定「VAD gate 機制全套留在現役程式（真實
資料軸重用），清理時不得移除」。背景 head 屬於同一套機制。但保留不等於維持誤導——原
訊息會叫使用者去開 `vad_head`（他可能已經開了），現在會正確指向
`background_vad_head`。回歸測試：
`test/test_losses/test_vad_loss.py::test_missing_head_errors_name_the_head_the_loss_actually_wants`。

#### 未被測試覆蓋、改以實測驗證的一項

0-4 的 `__len__` 沒有任何現存測試觸及（全 repo 只有 `KaldiFormBaseDataset` 被
`len()` 呼叫）。改動前實測確認：`DataLoader(dataset, batch_sampler=...)` 在
`num_workers=0` 與 `num_workers=2` 下都能正常建構與迭代，`len(dataloader)` 取自
batch_sampler，Lightning 的 `sized_len()` 對這種 dataset 回傳 `None`（它同時捕捉
`TypeError` 與 `NotImplementedError`），`runner.py` 又設了
`use_distributed_sampler=False`，故無任何呼叫端受影響。

### 2026-08-17 — P1-1、P1-4 完成

驗收：`ruff check` 全綠；`--suite standard` **616 passed**（與 P0 後相同，無測試變動）；
RNG 指紋 **1187 個雜湊逐位元相同**（`PYTHONHASHSEED=0`，理由見 C6）。

**1-1 — `if_none_else` 收斂成 property**

- `DynamicBaseDataset` 新增 `audio_sr` 與 `sample_length` 兩個 property。兩者都在
  `target_sr` 為 None 時回退到 `ori_audio_sr`，而 `training_sample_length is None`
  ⟺ `target_sr is None`，所以兩者對「何時回退」的判斷一致。
- `task/ns.py`：`audio_sr` ×29、`sample_length` ×6 取代，`if_none_else` 定義刪除。
- `task/voice_isolation.py`：`audio_sr` ×2、`sample_length` ×2 取代，import 移除。
- `task/{tse,sv}.py` 的本地 `if_none_else` **維持不動**（凍結 legacy）。定義數 3 → 2，
  現役路徑的呼叫點 35 → 0。
- 順手記錄一處**沒有**修的不對稱：`ns.py` 的 `added_noise` 裁切用的是原始屬性
  `self.training_sample_length`（`target_sr: null` 時為 None，等於不裁），與上面兩行
  改用 `sample_length` 的裁切不一致。對齊它會改變那類 recipe 的產物，已加註解說明。

**1-4 — 殺掉 `_last_rir_meta` 側通道**

- `audio/augmentation.py` 新增 `RirDetail(rir_id, info)`（帶 `.metadata` property）與
  `RirApplied(wav, detail)`，`apply_rir` 改回傳後者。兩者都是 plain 2-tuple，所以
  既有的 `wav, (rir_id, info) = ...` 解包與 `info["metadata"]` 全部照舊——**tse/sv、
  7 個 test 呼叫點、3 個 `egs/rir_generation` 呼叫點都不用改**。
- `dataset/dynamic_base.py`：`apply_source_level_target_reverb` 回傳
  `ForegroundReverb(noisy, clean, metadata)`（仍是 3-tuple，呼叫端不變）；
  `apply_source_level_interferer_reverb` 改回傳 `ReverbedSource(wav, metadata)`——
  它本來就收到 metadata 卻丟掉，逼呼叫端去讀私有屬性。
- `task/ns.py` 的 `_build_synthetic_interferers` 改讀回傳的 `reverbed.metadata`，
  `getattr(self.augmentor, "_last_rir_meta", None)` 刪除。
- 呼叫端配合：`ns.py` echo playback 與 `tse.py:330` 各加 `.wav`（tse 是凍結 legacy，
  但這是共用 API 變更必須跟上的一行機械修改，不是重構它）。
- `_last_rir_meta` 屬性**保留**（REPL / eval 腳本的便利），但註解改成明說「library 內
  已無讀取者」。驗收 grep：外部讀取 0 處。

**副產物：RNG 指紋 harness**

`scratchpad/rng_fingerprint.py`（session scratchpad，未入庫）。建 4 speaker × 4 utt 的
合成語料，開滿所有 augmentation block，對 32 筆 seeded item + 1 個 collated batch 取
1187 個 sha256。P1 的兩項改動都用它驗過逐位元相同。要 promote 成正式 test 見 P2 硬前提。

### 2026-08-17 — P1-6 完成（C6）

驗收：`ruff check` 全綠；`--suite standard` **616 → 617 passed**（+1 為新回歸測試）。
兩段式指紋驗證：

| 判準 | 結果 |
|---|---|
| A. 現役路徑（`batch_sr=None`）必須完全不變 | **1187 雜湊逐位元不變** ✓ |
| B. 多取樣率路徑（`batch_sr=16000`）跨 `PYTHONHASHSEED` 0/1/2 | 不穩定數 **99 → 0** ✓ |
| （B 相對修正前的 hashseed=0 實現值有 102 個變動——預期如此，抽到的 utterance 換了） | — |

改動：`select_with_sr_as_key` 分支不再 `deepcopy → set → list`，改成保留
`sr_meta[sr][spk]` 的 list 順序，`ignoring_utt_list` 用 list comprehension 過濾。與
另一條分支（dict key 順序）對齊，兩條都是 metafile 順序。

回歸測試 `test/test_utils/test_dynamic_and_ns_dataset.py::
test_sr_keyed_utterance_pool_keeps_metafile_order`：用 8 筆 utterance 的池子（短池子
可能碰巧與 set 順序相同），monkeypatch `random.sample` 攔下它實際收到的池子，斷言等於
`sr_meta[sr][spk]`，含 `ignoring_utt_list` 過濾後的情況。**已反向驗證**：把實作改回
`list(set(...))` 這支測試會 fail。

> 為什麼不用 subprocess 跨 `PYTHONHASHSEED` 測真正的性質？那才是直接測法，但兩次
> python + torch 啟動要 ~5 秒，對一個 25 秒的 suite 是 20% 成本。改為釘住可在
> process 內觀察的不變量（池子順序），代價是理論上有人能寫出「順序對、但仍不決定性」
> 的實作騙過它——實務上不會發生。

### 2026-08-17 — P1-2、P1-3 完成

驗收：`ruff check` 全綠；`--suite standard` **617 → 628 passed**（+11 為新的 logging
契約測試）；RNG 指紋在兩條路徑上都與 P1-6 後的基準逐位元相同。

**1-2 — 四份 loss reduce 迴圈收成一份**

`BaseLightningModule.reduce_losses(invoke, loss_funcs=None, weights=None)`：呼叫端只
提供 `invoke(loss_func) -> tensor`，因為那是四處**唯一**不同的部分（SISO 依 dispatch
flag 路由 backbone 側輸出、MISO 不路由、分類頭吃 `(pred, target)`）。

- `siso.EncDecMaskBase.compute_loss`：6 段 dispatch 原封搬進一個 closure（順序語意的
  註解一併搬過去），reduce 的部分消失。
- `siso.EncPredClassBase.compute_loss`：13 行 → 1 行。
- `miso.compute_loss` / `compute_loss2`：同上，後者用 `loss_funcs=` / `weights=` 指向
  conditional 分支的那組。

一處刻意的行為差異：累加改成 out-of-place（`total = total + w`），舊迴圈是 `+=` 直接
在第一個 weighted tensor 上做 in-place。已驗證數值、總和與**梯度**三者都完全相同
（3-loss 隨機輸入對照舊迴圈逐位元比對）。空 loss list 從回傳 `[]` 變成回傳 `None`
——兩者都是壞掉的 recipe，`None` 至少誠實。

**1-3 — `print` → `logging`**

52 處 `print` 轉成 module logger（`logging.getLogger(__name__)`），跨 13 個檔案。
level 依語意分派：語料統計 / augmentor 初始化 / ckpt 載入摘要 = INFO；設定被忽略、
取樣率不符、空片段重試、參數對不上 = WARNING；`create_folder` 的 benign race = DEBUG。

**保留唯一一個 `print`**：`base.on_test_epoch_end` 印的 metric 分數是 `--scoring`
的產出，不是關於過程的訊息。走 logging 會讓「把吵雜訊息關掉」連帶把結果也關掉。

新模組 `puresound/logging_setup.py`（**純 stdlib**——`puresound/__init__.py` 會匯入
它，而 `import puresound` 不能拉進 torch，`test_rir_r0_import_boundaries.py` 在管這
件事，已驗證仍為 torch-free）。設計上的兩個取捨：

1. *為什麼 library 仍預設掛 handler*（一般而言是反模式）：`egs/` 與 `tools/` 底下有
   84 個檔案 import 這個套件，其中 20 個會建 dataset / 載 ckpt / 開音檔，沒有任何一個
   設定 logging。只掛 NullHandler 會讓它們靜默失去平常在讀的輸出。改成預設掛上、但
   給三種接手方式（`PURESOUND_LOG_AUTOCONFIG=0` / `setLevel` /
   `configure_library_logging(force=True)`），既不回歸也真的可控。
2. *為什麼 rank 從環境變數讀而不是 `torch.distributed`*：最吵的輸出發生在建 dataset
   時，那時 Lightning 還沒初始化 process group，`dist.is_initialized()` 在每個 rank
   都是 False，問 torch 會得到「全部都是 rank 0」而印 N 次。

輸出格式維持 `%(message)s` 並寫到 **stdout**，所以畫面與 shell 重導向的行為跟原本的
`print` 完全一樣（已用語料統計逐行比對）。

新測試 `test/test_utils/test_library_logging.py`（11 項）釘住三個性質：預設有輸出、
可完全靜音、只有 rank 0 輸出（外加 `all_ranks` 豁免、rank 逐筆讀取、
`configure_library_logging` 的冪等性與不碰他人 handler）。**已反向驗證**：拿掉 rank
filter → 1 fail；拿掉 autoconfig 開關 → 1 fail；改寫到 stderr → 3 fail。

文件：`docs/index.{md,zh-TW.md}` 新增「函式庫輸出」一節與 module overview 條目。

### 2026-08-17 — P1-5 完成（選 B）

兩個選項的實測差異：讓呼叫端改走 api（選項 A）需要 api 從 37 個出口長到 62 個，等於把
`BankSplitPolicy`、`canonical_json_sha256` 這類內部細節升格成公開承諾——那是削弱分層而
不是加強；或者維持 37 個讓腳本混用兩種 import 風格，比現在全部深入還難讀。選 B。

改動：

- `rir/api.py` 的 docstring 改成誠實描述——「常用符號的方便集合，**不是**穩定性邊界」，
  並寫明真正被強制執行的契約在下一層（`contracts.py` + `test_rir_r0_import_boundaries.py`
  的 LAYER_RANK 表，後者會讓跨層 import 直接 fail build）。
- `test_rir_r0_api_inventory.py`：拿掉那組「essential 名單」斷言（在守一條沒有消費者的
  線），保留「`__all__` 的名字都要解析得到」——那條仍有價值，是防 re-export 因為底層改名
  而靜默腐爛。
- 同步 4 份 docs（`rir_package_migration.{md,zh-TW}`、`rir_realism_algorithm.{md,zh-TW}`）
  裡沿用「stable façade / 穩定表面」的說法。

驗收：`--suite standard` 628 passed（不變）；`test_rir_r0_api_inventory.py` 與
`test_rir_r0_import_boundaries.py` 82 passed。

### 下一步

1. **P2-5 / P2-6（config schema + 清死旋鈕）建議優先**——見 A6，這不是預防性重構，是
   已經發生的腐蝕：33 份 config 帶著 10 個死旋鈕、`added_noise` 那塊 90 行機制產出無人
   消費。
2. P2-1（拆裝置鏈）動之前先把 RNG 指紋 harness 入庫成 test——C6 修完後已無 flaky 障礙，
   只差搬 fixture。
3. P1 已全部完成。
