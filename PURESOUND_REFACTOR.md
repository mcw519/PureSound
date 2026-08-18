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

**硬前提（已完成，2026-08-17）**：`ns.py` 的正確性有一半在 RNG 呼叫順序與「缺 key 時
哪個預設值開火」上，兩者讀 code 都看不出來。現在有兩件工具：

- `test/test_utils/test_synthesis_fingerprint.py`（2 項、約 6 秒）——**自我驗證、不需
  golden 檔**。斷言「把每個旋鈕寫滿」與「只寫必填」合成出相同音訊，因為 model 預設值
  就該等於讀取點原本內聯的值；以及同一 seed 兩次結果相同。
- `tools/rng_fingerprint.py`——refactor 用的 before/after 比對。測試抓不到「兩份 config
  一起移動」的變更（刪掉一個 stage、換兩個 RNG 抽取順序），那要靠基準比對。

為什麼不放 golden 雜湊進測試：任何**刻意**的合成變更都得重生 golden 檔，而 reviewer
無法分辨「合理重生」與「改壞了才重生」。before/after 本質上是 refactor 當下的工具，
不是常駐斷言。

| # | 事項 | 做法 |
|---|---|---|
| 2-1 | 拆 `ns.__getitem__` 的裝置鏈 | 抽 `task/device_chain.py`：`DeviceChain.apply(...) -> ChainResult`，內含 SRC/IIR/HPF/volume/codec/packet-loss 與 `added_noise` 重放。**只從 `ns.py` 抽，`tse/sv` 維持凍結** |
| 2-2 | 拆 `_apply_overlap_gating`（複雜度 14、142 行） | 拆成 `_turn_taking_envelope` / `_bernoulli_envelope`，共用 `_gate`/`_smooth_env` |
| 2-3 | `streaming` 抽共用基底 | `StreamingFrameModelBase` 承載 `_require`/`_as_list`/`_down_step`/`_up_step`/ONNX 匯出/`*Ort`；`dpcrn`/`dparn` 只留 state 定義與 block step |
| 2-4 | 輔助 head 掛載樣板化 | `AuxHeadMixin.attach_heads(cfg, enc_channels)` + `collect_side_outputs() -> dict`。**checkpoint key 必須維持 `backbone.vad_head.*`** |
| 2-5 | **config schema 驗證**（已完成） | 見執行紀錄 |
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

### 2026-08-17 — P2-5 完成（config schema 驗證）

驗收：`ruff check` 全綠；`--suite standard` **628 → 651 passed**（+23 新測試）；
**7 份出貨 config 全部零錯誤通過**；33 份帶死旋鈕的 exp config 全部被報出來。

新模組 `puresound/config_schema.py`，掛在 `recipes.load_siso_recipe_config`——那是唯一
的匯流點（兩個訓練進入點、streaming 匯出器、12 支 eval 腳本都走它），所以檢查發生在
任何音訊被合成之前。

**規則**（都不是憑品味，是從程式碼實際讀法推導的）

- 未知 key → 錯誤。未知的 `augmentation_*` 頂層區塊 → 錯誤。
- `REQUIRED` = 程式碼硬取 `cfg["k"]`；`OPTIONAL` = 走 `cfg.get("k", 預設)`。這一條直接
  消滅了 A6 記的「57 處硬取 vs 71 處軟取、沒有規則」——規則現在寫在 schema 一個地方，
  不散在 128 個讀取點。
- 必填只在區塊**啟用時**檢查（停用的區塊被 `_enabled_config` 擋在 dataset 之前，根本
  不會被讀）。例外是 `used` 本身：`augmentation_hpf` / `augmentation_volume` 走
  `config.get(...)` 而非 `_enabled_config`，即使關閉也會進到 dataset 並硬取 `used`。
- `Delegated` 標記那些整包被 spread 進建構子的 sub-block（bank loader、room simulator、
  VAD labeler）——它們自己就會拒絕未知 keyword argument，在 schema 再抄一份只會腐爛。
  先例是 `init_drr_contrast`，它本來就對自己那塊做這件事；2-5 等於把它推廣到全部。
- 所有問題一次報完，不是遇到第一個就停——帶 10 個死旋鈕的 config 應該一次講完。

**dry run 抓到一個我原本寫錯的 schema**：`augmentation_speed` 有**兩種方言**——
NS / voice_isolation 讀 `speed_range`（[lo, hi] 區間），speaker_embedding 讀
`speed_change`（離散清單）+ `treat_as_new_speaker`。兩者都不能單獨標 REQUIRED，否則會
打死另一個 task。改用 `ANY_OF` 表達「兩者擇一」。**這本身是 A6 的又一個切面**：同一個
區塊名底下長出兩套互不相容的方言，因為那個名字沒有擁有者。統一它是對凍結 task 的行為
變更，記為發現、不在此處理。

當時掃出的 33 份失敗 exp config 後續已完成清理：現役 8 份移除死旋鈕，25 份無法執行的
backup config 從 live config tree 刪除，歷史內容交由版本控制保存。

**刻意不做型別檢查**：`prob: "0.5"` 這種錯誤目前仍會通過 schema（但會在 `float()` 轉換
時炸）。加型別檢查會擴大誤判面，而靜默失效的那一類是 key 名，不是型別。列為可選延伸。

### 2026-08-17 — Pydantic canonical schema v2（取代 2-5，並吃掉 A1 / A4 / 3-1 / 3-2）

config 機制整個換成 Pydantic v2 的 typed recipe（`puresound/config/`），舊時代路徑全部
移除。經過兩輪：先做出可運作的 typed 遷移，再收斂成 canonical v2。

驗收：`ruff` 全綠；`--suite standard` **698 passed**；35 份現役 recipe 全數 strict 驗證
通過並納入回歸測試；**RNG 指紋 1187 雜湊 × 2 份 fixture 對照「遷移前 dict 時代」的基準
逐位元相同**。

那 2 份 fixture 是本案的安全網：一份「全寫滿」、一份「只寫必填」。後者讓 49 個原本寫在
讀取點的預設值全部開火——搬錯任何一個，minimal 會偏離而 full 不動。這是風險最集中處，
因為 model 原本是 `fade_samples: int | None = None` 而讀取點是 `.get("fade_samples", 400)`。

**review 過程中找到並修掉的實質 bug**：`to_legacy_dict` 的 `exclude_none=True` 會刪掉
明確寫成 `null` 的 key，而執行期硬取——`egs/noise_suppression/config/dpcrn.yaml --training`
會在 `KeyError: 'gain_normalized_to'` 死在啟動。測試抓不到，因為沒有測試建 dataloader。

**最終形態**

- **canonical v2**：`schema_version` / `purpose` / `task` 全部必填。不推斷 legacy 格式、
  不接受 task alias、沒有 normalization 層——loader 只剩 76 行。
- **strict mode**：禁止 numeric string 等隱式轉型。現役 YAML 同步修正（只有型別寫法，
  例如 `filter_min_utterance_per_speaker: 1.` → `1`，無任何數值變更）。
- **A1 參數隧道消失**：20-tuple loader、20 欄位 `RecipeConfig`、17 參數的
  `build_dataloaders` 全部刪除；`BaseRecipe.augmentation_kwargs()` 由 model 欄位自動
  推導轉發清單。
- **A4 重複消失**：`speaker_embedding` 與 `target_speaker_extraction` 兩支 main 自帶的
  dataloader builder 改為委派 `runner.build_dataloaders`。
- **`dataset_role` 與 `pipeline_role` 分離**：前者是這個 split 的身分，後者是它從哪個
  RIR bank split 取樣。舊行為是兩者都吃 `dataset_role` 的預設值 `"train"`——也就是
  validation 其實在用 train 的 bank split。現在改成明示：4 份 release-bank config 都寫上
  `validation_pipeline_role: train` 並附理由（房間泛化另跑一輪對 `split: test`），
  行為不變但決定變成可見的。
- **speed 方言由 task class 在任何資料 I/O 之前選定**（`speed_augmentation_model`
  class attribute），錯誤的 Pydantic model 會被重新驗證。
- 停用的 augmentation 一律傳 `None`（已實測與傳 `used: False` 行為等價）。
- 移除：TSE `add_ir_response` 死旋鈕、inference 的 optimizer/scheduler/loss placeholder、
  重複的 `runner.load_config`、TSE 自己的 loss builder、公開 parsing helper。
- `init_siso_model` 不再就地改 `recipe.model`（改成複製 `features` 再注入 peq）。

**三個設計決定**

1. **dataset 邊界接受 model 或 mapping**（`as_block`）：直接建構 dataset 的測試與腳本
   也會過 schema——比遷移前強，遷移前那條路完全沒有驗證。
2. **`with_overrides`**：model 是 frozen，eval 腳本不能再就地改 dict；覆寫走同一套驗證，
   `training_length_seconds=-1` 在覆寫當下就被拒。
3. **`PURESOUND_CONFIG_SCHEMA=warn` 逃生門移除**：typed 之後沒有降級態可回傳。

**仍是 dict 的地方（刻意）**：`model` 區塊、`optimizer.args` / `scheduler.args` /
`loss_func[].args` / `vad_label.args`。它們是 spread 進建構子的，建構子自己會拒絕未知
keyword argument。

**已刪除且部分不可復原**：`config/exp/backup/` 的 25 份設定與該空目錄。其中 1 份是
git tracked（可由版本歷史復原），其餘 24 份原本受 `.gitignore` 排除，**無法由 git 復原**。
它們引用的機制早已不存在，本來就無法重現當初的實驗。

### 2026-08-17 — P2-6 完成（死 payload：`added_noise`）

驗收：`ruff` 全綠；`--suite standard` **700 passed**；`tools/rng_fingerprint.py` 對兩條
路徑做 before/after 比對——**改動前後只有 `added_noise` 這個 key 消失，其餘 hash 全部
逐位元相同**（ns 路徑 864 → 840，差 24 = 每個 item 一個；TSE 路徑 144 → 128，差 16）。

**刪了什麼**：`added_noise` 進 sample 字典但 collate 從不收它，全 repo 沒有任何地方讀
`batch["added_noise"]`。維持它同步的是 4 組 flag + 延後重播（SRC / IIR / HPF / volume，
各跨 159–212 行），也就是 A6 記的「pipeline 最脆弱的樣式」。

| 檔案 | 淨變化 |
|---|---:|
| `task/ns.py` | −89 |
| `task/tse.py` | −83（同一份逐字複製，B1 那 332 行重複的其中 78 行） |
| `task/sv.py` | −6（純死區域變數；`added_noise += tensor` 那行本來就是壞的 list 運算，因為從不被讀所以沒人發現） |

`AudioEffectAugmentor.add_bg_noise` 仍回傳它——那是 augmentor 的 API，不是這條 payload。

**為什麼那 4 組重播不耗 RNG**（指紋證實，但值得寫下理由）：重播用的都是「已經抽好的
參數」版本——`apply_2nd_iir_response(a_coeffs=, b_coeffs=)`、`apply_hpf(cutoff, q)`、
`sox_volume_perturbed(vol_ratio=)`、`apply_clipping_distortion(min_quantile=,
max_quantile=)`。隨機分支只在參數為 None 時才走。

**`far_target` 不刪，改註解**：它的註解說是給 `FarReconstructionLoss` 用的，而那個 loss
不存在。但它**有被讀**——`scripts/eval_indomain.py` 用它量遠場洩漏，那是近/遠場軸的主要
診斷。改成如實描述「這是 eval 產出，不是訓練目標」。

**順手修好的工具缺陷**：`tools/rng_fingerprint.py` 原本只支援 3-tuple 的 item key，對
speaker_embedding 與 TSE（兩者的 `__getitem__` 只吃 2-tuple、沒有 per-item seed）會直接
壞掉。現在對那兩個 task 改為在每個 item 前手動 seed 全部 RNG。

**順手發現、未修**：`tse.py` 的 `add_n_cases` 只支援純量，不像 `ns.py` 會處理 `[lo, hi]`
範圍——TSE recipe 寫成範圍會 crash。schema 允許兩種形態，所以這是 TSE 側的缺口。
TSE 為凍結 legacy，記錄不修。

### 2026-08-17 — P2-1 完成（抽出裝置鏈）

驗收：`ruff` 全綠；`--suite standard` **709 passed**（+9 新契約測試）；
`tools/rng_fingerprint.py` 對照抽取前：**840 個 hash 逐位元相同**（TSE 路徑未動，
128 個 hash 亦相同）。

`ns.__getitem__` **588 → 400 行**，圈複雜度 **44 → 25**（P0 時量到的 44 是起點）。
新模組 `puresound/task/device_chain.py`（267 行）承載 SRC / IIR / HPF / volume /
codec / packet-loss 與收尾的 overload guard。

**為什麼是這條切線**：混音完成之後剩下的全部是「擷取與傳輸」——重取樣、麥克風傾斜、
高通、增益、VoIP codec、掉封包。它們對「說話者」與「房間」一無所知，所以不該長在
`__getitem__` 中段。切在這裡，`__getitem__` 剩下的就只有「這一列是什麼」的決策。

**模組明寫的三條契約**（各有測試，且都反向驗證過會 fail）

1. **stage 順序是契約不是細節**——每個 stage 都從共用 RNG stream 抽值，換兩個順序就會
   改變 seeded recipe 的產物，即使每個 stage 本身沒變。
2. **停用的 stage 不得碰 RNG stream**——機率抽取放在短路**內**。把它移到短路外，測試會
   抓到。這正是「加新旋鈕後舊 recipe 仍能 bit-identical 重現」的來源。
3. **哪些訊號能被碰**：SRC / IIR / HPF / volume 是線性通道，用同一組參數同時作用在混音
   與乾淨目標上（目標是模型要「穿過該通道」還原的東西）；codec 與 packet loss 只打混音
   ——它們是傳輸損傷，目標要維持未受損的評分基準。第 3 條在結構上也成立：
   `_codec` / `_packet_loss` 的簽章只收發 `noisy`，碰不到 target。

**P2-6 讓這件事變簡單**：`added_noise` 的 4 組跨百行重播先被刪掉，這條鏈只剩 noisy /
target 兩條訊號要保持一致，工作量比原估少一半。

### 2026-08-17 — 裝置鏈 per-row provenance

`DeviceChain` 現在回報它對每一列實際做了什麼（`ChainResult.applied`，14 個數值
scalar），`ns.py` emit、`NoiseSuppressionCollateFunc` collate（voice_isolation 走
`super()` 免費繼承），`eval_indomain.py --by-bucket` 依它分組。

驗收：`--suite standard` **714 passed**；指紋 **既有 840 個 hash 完全不變**，只新增
24×14 = 336 個新 key。

**為什麼做**：`eval_indomain --by-bucket` 已經在依 `realized_speech_sir` / `noise_snr` /
`overlap_fraction` / `drr_gap` 分組看 SI-SDRi——「依這一列實際經歷了什麼切開看」是這個
repo 已經在用的方法。而裝置鏈是整條 pipeline 唯一完全不 emit 的一段，所以
「過度抑制是不是集中在被 SRC 降頻或 HPF 削過的列」這類問題**問不出來**。現役 recipe 的
src(0.5) / ir(0.3) / hpf(0.25) / volume(0.5) 都開著且只打部分列，今天就分得出組。

**為什麼同時接消費端**：`rir_provenance` 那 9 個 key 是反面教材——為 traceability 而加，
`rir_release_sha256` 與 `rir_renderer_profile_id` 的消費端至今是 **0**。加上剛刪掉的
`added_noise`，這會是第三次。所以規則是：**emit 與讀它的分析放在同一次改動裡**。

**設計約束**：只放數值 scalar，且**每一列都帶齊每一個 key**（`*_applied` 用 0.0/1.0，
未觸發的參數用 NaN）。這樣才能搭現有的 scalar collate（每個 key 一次 `torch.cat`）；
字串要走 `RIR_PROVENANCE_KEYS` 那條路徑，也就是沒人用的那條。

**一支刻意刪掉的測試**：「記錄本身不得多抽亂數」在這棵樹上寫不出誠實的測試——任何寫法
兩邊都會跑到記錄程式碼，必定通過。那是 before/after 性質，屬於
`tools/rng_fingerprint.py`。不可能失敗的測試比沒有更糟，它宣告了一個它不提供的保證。

### 2026-08-17 — TSE 併入 DeviceChain（B1 重複再減 139 行）

驗收：`ruff` 全綠；`--suite standard` **714 passed**；指紋 TSE 既有 128 個 hash 不變
（只多 16×14 個 provenance key），NS 路徑 1176 個完全不變。

`tse.py` **−139 行**：inline 裝置鏈換成 `DeviceChain`。三個 task 的 stage 順序本來就
完全相同（src → ir → hpf → volume），TSE 沒有 codec / packet_loss 而缺席的 stage
不耗 RNG，所以是天然相容。TSE 的 collate 不繼承 NS 的，provenance 的 `torch.cat`
要自己加一份。

**SV 沒有併進來**，這是刻意的：`SpeakerEmbeddingDataset.__getitem__` 只回傳
`noisy_speech` + `speaker_id`，**沒有 target**。`DeviceChain` 整個「線性通道要同步作用
在一對訊號上」的契約對它沒有意義，硬套就得餵假 target、白跑一遍所有濾波器。要收斂
應該是給 `DeviceChain` 一個單訊號入口，而不是為了消重複扭曲抽象。

**overload guard 從無到有（行為變更）**：`overload_guard` 成為建構參數，TSE 打開。
理由與 ns 相同——`EncDecCondMaskBase` 一樣把輸出 clamp 到 [−1, 1]，超過滿刻度的 target
是模型構不到的，那一列的 loss 有個永遠跨不過的底。實測 8/8 觸發、且**保持 target/
mixture 的位準比**。出貨的 TSE recipe 整條鏈都停用，所以對它是 no-op。

**量測途中發現的既有問題**：`apply_2nd_iir_response` 會**靜默硬夾 ±1**——
`torchaudio.functional.lfilter` 的 `clamp=True` 是預設值。實測 pair 以峰值 3.64
進鏈時，IIR 把 mixture 夾到 1.000 而 target（0.477）不受影響，**mixture 與 target
的位準關係就此被破壞**。已於 2026-08-18 修正，見下節。

### 2026-08-18 — 裝置鏈的聲學物理修正：線性級就要是線性的

驗收：`ruff` 全綠；`--suite standard` **727 passed**（+13）；8 個反向驗證（把每條
性質各自打破一次）全部被抓到。

#### 這不是一個 clamp，是六個

從 IIR 的 `clamp=True` 往下挖，發現**整條類比路徑的後端全部在滿刻度飽和**，而且
沒有一個說出口：`lfilter` 預設 clamp、每個 `biquad` 都建在它上面、每個 sox effect
都經過定點格式來回。在現役 `train_dpcrn.yaml` 上實測（300 列）：

| 級 | 呼叫數 | 非線性 | 最大 rel-err |
|---|---:|---:|---:|
| 2nd-order IIR（換能器響應） | 158 | **28.5%** | 0.72 |
| volume（sox `vol`） | 328 | **19.2%** | 0.63 |
| SRC 來回（sox 那一支） | 148 | **16.2%** | 0.25 |
| HPF（rumble filter） | 144 | 6.2% | 0.42 |
| speed perturb（sox） | 336 | 5.4% | 0.015 |
| media coloring | 70 | 該配方 0% | — |

而現役配方寫的是 `clipping_prob: 0.`——**它一個非線性都沒要**。全部是函式庫預設。

media coloring 是最惡劣的一個：兩顆 biquad 夾完之後，末尾的 RMS 還原又把被破壞的
訊號放大回原位準，**把削波藏起來**——一個披著濾波器名字的 waveshaper。

#### 為什麼位準關係會壞

夾在**固定絕對位準**上，而 mixture 是兩者中比較大的那個，所以被壓扁的永遠是
mixture，比較安靜的 target 原封不動通過。superposition 就此斷掉，mixture 不再是
它的來源在配方要求的 SIR 下的和。實測（400 列，同 seed，扣掉純位準差）：

| | mixture（模型輸入） | target（評分參考） |
|---|---:|---:|
| 波形 SNR < 60 dB 的列 | **9.0%** | 3.3% |
| 波形 SNR < 20 dB 的列 | 0.8% | 0.3% |
| 逐點誤差 > 峰值 1% | **9.0%** | 2.8% |

三倍的不對稱正是預測的破壞模式。

#### 修法：把物理講清楚

音壓域沒有滿刻度。`±1` 在這條鏈裡只在**一個**地方有意義——類比轉數位的那一刻。

1. **`puresound/audio/dsp.py: apply_linear(fn, wav)`**——線性後端就要在它的隱含
   削波器構不到的位準上跑。線性算子 `H` 與任意 `a > 0` 滿足 `H(x) == H(a·x)/a`，
   所以縮進後端的合法範圍、套用、再縮回來：結果是該算子真正的輸出，飽和從不觸發。
   後端增益超過預留 headroom 就退一階重試，而不是回傳一個悄悄錯掉的數字。
   `fn` 不得消耗亂數（重試會再呼叫一次）——現有呼叫端全是純濾波/增益。
2. **每個模型線性算子的地方都用它**：HPF、media coloring 的兩顆 biquad、sox 的
   vol / speed / pitch / rate。`rand_add_2nd_filter_response` 直接 `clamp=False`
   ——API 有這個旗標時就直說，精確且零成本。
3. **`DeviceChain._analogue_to_digital`**：原本收尾的 overload guard 改名並移到
   類比↔數位的交界（volume 之後、codec 之前）。它做的事本來就是**增益配置**而非
   削波——同一個純量除在兩條訊號上，錄音師把前級調到不撞軌的那個動作。放在
   codec 前面才對：codec 是數位 sink，餵它超過滿刻度的訊號等於讓它在內部夾，
   那是只打在 mixture 上、而且哪裡都沒記錄的非線性。

配方要求的過載仍然只有一處，`_volume` 的 quantile 削波：**用 mixture 實際抽到的
門檻同時夾兩條訊號**。它本身是尺度不變的，所以這次修正沒有動到它。

#### 分佈搬動了多少

`converter` 觸發率 6.4% → 28.8%（真實線性峰值超過滿刻度的列變多了，它本來就該
接手），但**輸出 RMS 中位數只從 −20.5 移到 −20.8 dBFS**，p5/p95 不動。也就是說
訓練資料的位準分佈實質沒變，變的是那 9% 原本被 waveshape 掉的列。

抽樣序列**完全沒動**：500 列的 provenance 逐鍵比對，除了 `overload_rescaled`
旗標本身以外 500/500 相同。所有變更都是值的變更，不是流的變更。

#### 順帶洗掉的一個混淆

SRC 那一級丟硬幣在 sox 與 torchaudio 兩個 resampler 之間，註解說是為了「拓寬
artifact 分佈而不是綁死一家的濾波器」。實際上 sox 那支會夾、torchaudio 那支不會
——這枚硬幣一半的效果跟抗混疊濾波器無關，是在決定要不要削波。現在兩支都是線性的，
它才真的只在測那件事。

**留待決定（未做）**：`absolute_floor` 掛在 chain 之前、標稱絕對 dBFS，但
converter 之後整列會被重新配置增益，所以那個「絕對」是相對於轉換器前的參考點。
以電容式麥克風的自身雜訊/房間底噪而言（兩者都隨前級走）現在的位置是對的，
不精確的只是文件用語。真要模型化轉換器自身的電子雜訊，那一份要加在 converter
之後、且不隨增益走。

### 下一步

1. **P2-6（清死旋鈕與死 payload）**——schema 已經把 33 份 config 的問題全部列出來了；
   `added_noise` 那塊（4 組 flag 重播、約 90 行、無人消費）也可以一起處理。
   此項已完成，包含現役設定與無法執行的 backup 設定。
2. P2-2（拆 `_apply_overlap_gating`，複雜度 14）、P2-3（streaming 抽共用基底，
   308 行重複）、P2-4（輔助 head 掛載樣板化）。
3. `__getitem__` 仍是複雜度 25 的長函式——剩下的是「這一列是什麼」的決策
   （row plan、interferer、mix、target-absent、echo、noise、VAD 標記、metadata）。
   要再降就是把 noise 那段也抽出去，但它與 room_scene 耦合，切線不像裝置鏈那麼乾淨。
4. P1、P2-1、P2-5、P2-6 已完成。
