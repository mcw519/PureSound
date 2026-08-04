# RIR 生成工具

這個目錄放置近距／遠距語音增強所使用的房間脈衝響應（Room Impulse
Response, RIR）工具。推薦的訓練資料流程是 M6 入口：它會把
**確定性生成、逐項 QC 與 release 封裝**串成一個可重現的指令。

英文版本：[`README.md`](README.md)

## 一個模擬單元是什麼

每個單元由一個五聲道 RIR WAV 和同 stem 的 JSON sidecar 組成：

| 聲道 | 聲源 | 預期距離 |
|---:|---|---|
| 0 | `near_0` | 近距，通常 `< 1 m` |
| 1 | `near_1` | 近距，通常 `< 1 m` |
| 2 | `far_0` | 遠距，通常 `> 2 m` |
| 3 | `far_1` | 遠距，通常 `> 2 m` |
| 4 | `far_2` | 遠距，通常 `> 2 m` |

接收端是一支麥克風。每個房間／acoustic space sample 會記錄房間尺寸、
表面與材料成因、家具／障礙物、麥克風位置、聲源位置、renderer 設定及實際
量出的聲學指標。同一個 acoustic space 可以產生多個不同麥克風／聲源配置的
單元；M6 會讓 train、validation、test 在房間與 acoustic space 上互斥。

一般 hybrid renderer 以低頻波／模態成分和高頻幾何成分合成 RIR，並透過因果
crossover 接合。JSON sidecar 是資料契約的一部分；不可只複製 WAV 而遺漏
metadata。低頻 solver 在有限 modal／voxel 展開下可能留下極小的數值前置殘留；
合成前會依每個 source 的 `floor(distance / sound_speed * sample_rate)` 清零，
保留到達 sample 本身，確保 M6 的幾何因果條件成立。

## M 系列代表什麼

M 標籤是 renderer 與資料 bank 的里程碑，不是音質分數。M 編號較大，也不代表
所有較早的 backend 已經可以直接當成 production。

| 里程碑 | 意義 | 目前實際狀態 |
|---|---|---|
| **M0** | 凍結的 RT60 驅動 hybrid baseline（`v0`），作為回歸與比較基準。 | 只作 reference，不是首選訓練設定。 |
| **M1** | Material-first scene：房間類型、表面／材料先驗、障礙物與相關聲學成因。 | 目前最推薦的物理 scene 基礎。 |
| **M2** | 複數阻抗、頻率相關模態損耗、阻抗量測與 residue calibration。 | 實驗性 opt-in；production mapping 尚未凍結。 |
| **M3** | 用 PathEvents 表示直接聲、早期反射與相干波路徑。 | 已實作，可作高頻 opt-in backend。 |
| **M4** | 空間化 late field：PathEvents 加上 multiband late-field／FDN。 | 已實作，可 opt-in；不是預設 bank renderer。 |
| **M5** | 受控真實房間量測、反演校準、受限 residual 與空間校準。 | implementation gates 已有，但仍缺真實房間 empirical evidence。 |
| **M6** | 確定性訓練 bank 契約、QC、variants、evaluation 與不可變 production decision。 | 可生成 candidate bank；production promotion 仍受證據門檻限制。 |

### M6 的六個子階段

1. **M6.1 — contract：** versioned manifest、provenance、hash，以及固定的
   train／validation／test split。
2. **M6.2 — generation：** 先建立完整 task plan，再以逐項 seed 執行平行／
   resume 生成，最後做 audit。
3. **M6.3 — QC：** 每個 item 的物理檢查、只收 PASS 的 index，以及失敗／無法
   評估項目的 quarantine。
4. **M6.4 — release：** 凍結 calibrated 與 peak-normalized variants 和訓練
   recipes。
5. **M6.5 — evaluation：** 評估聲學分布、吞吐、聆聽與 downstream model
   證據契約。
6. **M6.6 — decision：** 以 append-only promotion certificate 綁定 release、
   證據 hash、renderer approval 與 sign-off。

M6.1–M6.6 的 implementation gates 已完成，但目前的 synthetic release 仍是
**candidate**，不是 production-approved bank。真實／混合資料、受控聆聽、
downstream model 結果及 production approvals 不會由生成器自動製造。

### Review 後的強化（2026-08-02）

M6 bank 路徑現在會同時固定 NumPy 與 libroom RNG，讓 Pyroomacoustics ray tracing
在 serial、parallel、fresh run 與 resume 下都可重現；resume identity 也綁定
code revision 與 runtime package versions。低頻建議設定改用單一 sample 的因果
delta 激勵，不再使用會在所有 item 留下固定 comb null 的舊 bipolar 激勵。

M4 路徑也不是只調 Pyroomacoustics 參數：FDN filterbank 現在從 DC 完整覆蓋到
Nyquist，晚場能量由材料 RT60 外插而不是被有限階 path tail 鎖住；空間輸出共用
同一個 field gain，並加入材料邊界相位先驗、空氣吸收、source directivity 與
逐 path 障礙物作用。

M6 admission／evidence 也已收緊：late-arrival 與 octave-decay gate 實際生效；
calibrated float RIR 不再套用錯誤的 unit-peak 上限；不安全 item path、缺 manifest
的 M6 layout 會 fail closed；variant 音訊 lineage 逐 sample 驗證；downstream CI
由 per-seed 結果重算；production certificate 會重新稽核所綁定的 release 與
evidence。修正前已生成的 bank 不應被描述成 post-hardening 結果，請使用新的
output directory 重新生成。

### 強化後 matched preflight 結果（2026-08-02）

第一個修正後的 smoke campaign 已完成，輸出在
[`exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/`](exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/)。
設定是 30 個 matched rooms × 每房 2 個單元（每個 backend 60 items／300
channels）、`v1/mixed`、seed `1337`、calibrated `16 kHz / 1.6 s`，低頻使用
GPU backend `pytard-cupy-material`。兩個 backend 的 scene hash、room／acoustic
space ID、split、seed 與 audio shape 全部 60/60 對上；兩邊都是 60/60 QC PASS、
0 quarantine、release audit PASS；實際 calibrated train reader 也各自成功讀取
56 個 PASS items。

配對後的中位數如下：

| 指標 | Pyroomacoustics | PathEvents-M4 | M4 − Pyroom |
|---|---:|---:|---:|
| DRR | -4.83 dB | -3.24 dB | +1.59 dB |
| C50 | 6.10 dB | 10.28 dB | +4.18 dB |
| C80 | 8.03 dB | 15.01 dB | +6.97 dB |
| T20 | 0.99 s | 0.47 s | -0.52 s |
| absolute T20 − scene RT60 error | 0.327 s | 0.080 s | M4 較接近 |

這表示在這批樣本裡，M4 比較乾、早期能量較多，而且 broadband T20 更貼近
scene RT60。因果到達誤差（中位數 0.033 ms）、最後 sample 精確為零、390 Hz
固定 comb 檢查與 M4 高頻尾場 coverage 都通過。這證明強化後資料鏈與 M4 演算法
確實有執行預期的機制，但不能因此宣稱 M4 比真實房間更合理：目前 validation、
test 各只有 2 個 items、code revision 是 dirty，也沒有 measured RIR、真人聆聽或
downstream model 結果。因此兩個 release 都仍是 **candidate**，Pyroomacoustics
仍是預設 backend，完整 4,000-item pilot 仍是下一個必要步驟。

這次觀察到的 aggregate 生成成本約為：Pyroomacoustics 使用 2 workers 時
6.5 秒／item，M4 在 resume 後使用 8 workers 時 8.2 秒／item。M4 目前主要受
CPU PathEvent／material boundary 限制；GPU 主要加速共同的低頻 solve，增加
worker 不會讓 M4 高頻變成 GPU-bound。

完整的 manifest／QC／release hash 與 provenance 警告在
[`preflight_validation_summary.json`](exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/preflight_validation_summary.json)。

## 最接近可用作訓練資料的模擬設定

目前最接近可直接拿來做訓練資料的路徑如下：

| 設定 | 建議值 | 原因 |
|---|---|---|
| Bank pipeline | `generate_m6_bank.py` | 一次產生 manifest、QC index 與 release recipe。 |
| Scene | `--scene-version v1 --room-type mixed` | 使用 material-first 且相關的房間／裝修／障礙物成因。 |
| 輸出 | `--output-mode calibrated --record-realized-metrics` | level 語意穩定，並保留可稽核聲學指標。 |
| 低頻 | `--low-backend pytard-material` | CPU 波／模態 solver；使用材料頻率相關模態阻尼與因果 delta 激勵，不再帶有舊版固定 comb 特徵。 |
| 高頻 | `--backend path-events-m4` | 預設：八度衰減形狀比 pyroomacoustics 貼近實測 8 倍（pyro 高頻殘響約 2.2 倍長）。要速度或 A/B 對照時再指定 `--backend pyroomacoustics`。 |
| 取樣率／長度 | `16 kHz / 1.6 s` | 符合目前語音增強 bank 的建議設定。 |
| Bank 大小 | `1000 rooms × 4 items = 4000 items` | 適合作為第一個訓練 pilot，且保留 room-disjoint split。 |
| Seed／workers | `1337 / 8` | 內容可重現，worker 數不會改變 item identity。 |
| 訓練 recipe | `synthetic_calibrated`、`split: train` | 只取通過 QC 的 train items。 |

這是目前推薦的**合成 candidate**，不是「等同真實房間」的宣稱。M4 高頻
backend（`path-events-m4`）可以用來做 matched ablation，但第一輪訓練不應
在沒有比較的情況下取代 Pyroomacoustics baseline。

### 生成推薦的 M6 candidate

請從 repository root 執行：

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot \
  --backend path-events-m4 \
  --n-rooms 1000 \
  --rir-per-room 4 \
  --num-workers 8 \
  --seed 1337 \
  --sample-rate 16000 \
  --duration 1.6 \
  --scene-version v1 \
  --room-type mixed \
  --output-mode calibrated \
  --low-backend pytard-material \
  --record-realized-metrics
```

這一條指令會依序執行 M6.2 generation、M6.3 QC 與 M6.4 release，產生：

```text
egs/rir_generation/exp/rir_realism/m6/training_pilot/
├── pyroomacoustics_bank/       # WAV/JSON、manifest、split index、QC
└── pyroomacoustics_release/    # 已 audit 的 training variants 與 recipes
```

如果中途停止，可以用完全相同的參數重跑；M6 resume 會在跳過 item 前檢查
task、config、scene 與 audio identity。既有 release 不會被覆寫；要產生新的
bank 請換一個 `--output-dir`。

若要建立 matched M4 高頻 candidate，其他設定與 seed 都保持不變，只更換輸出
目錄與 renderer：

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot_m4 \
  --backend path-events-m4 \
  --n-rooms 1000 --rir-per-room 4 --num-workers 8 --seed 1337 \
  --sample-rate 16000 --duration 1.6 \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --low-backend pytard-material \
  --record-realized-metrics
```

這個 M4 bank 應先作為 matched realism ablation；在 M4／M5 empirical evidence
通過前，不要把它和 baseline bank 靜默混合，也不要標成 production-approved。

### 同一個 scene 的 backend 比較示意圖

下面的 overview 已直接取代原本的單一 backend 示意圖。左右兩側使用完全相同
的 v1 scene、聲源／接收端幾何、`16 kHz / 1.6 s` 設定、calibrated output 與
相同低頻 backend，只有高頻 renderer 不同：

- 左側：Pyroomacoustics high-frequency renderer；
- 右側：M4 PathEvent early response 加上 FDN late field。

兩個版本的 scene metadata hash 都是
`6bb48bb7b7f5bca0c61a7765544d8568ac946faf955ca87a358d43dd9a69d1df`，因此波形
與 Schroeder decay 顯示的是 matched backend ablation，而不是兩個不同房間的
獨立抽樣結果。

重新產生這些圖：

```bash
for backend in pyroomacoustics path-events-m4; do
  PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_hybrid_rir.py \
    --output-dir /tmp/readme_assets/$backend --n-rooms 1 --rir-per-room 1 \
    --sample-rate 16000 --duration 1.6 --scene-version v1 --room-type mixed \
    --output-mode calibrated --low-backend pytard-material \
    --high-backend $backend --seed 1423 --record-realized-metrics
done
```

seed 用 1423 而非 pipeline 預設，是為了讓圖中的房間落在 bank 的殘響中位數而不
是長尾：100 房間 pilot 的 scalar scene RT60 中位數是 0.56 s，但 95 百分位是
2.34 s。這個 scene 是 0.55 s，且 RT60 隨頻率下降（與中位數一致）。seed 1337 的
第一個房間恰好是 2.82 s 的全硬表面離群值，當示意圖會誤導讀者。

![Pyroomacoustics 與 M4 的同 scene 比較](assets/overview.png)

path 圖與壓力場動畫也使用同一個 scene。path 只由幾何決定；動畫則是實際低頻
modal pressure slice，所以兩個高頻 variant 共用同一份動畫。

![同 scene 的反射路徑](assets/paths_ch2.png)

![低頻 modal 壓力場動畫](assets/field_ch2.gif)

### 聽一個 M4 item

RIR 本身聽起來只是一個 click。以下是同一段 LibriSpeech 乾淨語音，分別與上圖那
個 M4 item 的近場與遠場 channel 卷積，所以這一對是**同一個房間內的距離比較**：

| 檔案 | 內容 |
|---|---|
| [`m4_sample_dry.wav`](assets/m4_sample_dry.wav) | 乾淨源，無房間 |
| [`m4_sample_near_near_0.wav`](assets/m4_sample_near_near_0.wav) | 經 `near_0`，0.45 m |
| [`m4_sample_far_far_2.wav`](assets/m4_sample_far_far_2.wav) | 經 `far_2`，3.89 m |
| [`m4_sample_rir.wav`](assets/m4_sample_rir.wav) | 五通道 RIR 本身，float32，calibrated 位準 |

兩個卷積檔共用同一個增益，以保留彼此的位準關係；乾淨參考另外縮放，因為
calibrated RIR 的振幅編碼的是參考 SPL 而不是聆聽音量。可聽出的距離線索主要來自
direct-to-reverberant ratio 而不是音量——在這麼殘響的房間裡擴散場幾乎與距離無
關，所以兩者 RMS 只差約 1 dB，聽感卻差很多。

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/build_readme_sample.py \
  --rir /tmp/readme_assets/path-events-m4/room_000000/room_000000_000000.wav
```

### 使用 GPU 生成低頻部分

先安裝與 CUDA 版本相符的 CuPy，再選擇 `pytard-cupy-material` backend：

```bash
uv pip install cupy-cuda12x

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot_gpu \
  --backend path-events-m4 \
  --n-rooms 1000 --rir-per-room 4 \
  --num-workers 1 --gpu-devices 0 \
  --low-backend pytard-cupy-material \
  --seed 1337 --sample-rate 16000 --duration 1.6 \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --record-realized-metrics
```

只有低頻模態 solve 使用 GPU；高頻的 Pyroomacoustics 或 PathEvents renderer
仍然在 CPU 執行。建議每張 GPU 配一個 worker（兩張 GPU 使用
`--gpu-devices 0,1 --num-workers 2`），因為每個 CuPy worker 都會建立獨立的
CUDA context 與 memory pool。

### 小型 smoke test

M6 必須同時具備三個 split，因此至少準備幾個 room。下面只用來驗證流程，
不適合作為訓練設定：

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/smoke \
  --backend path-events-m4 \
  --n-rooms 6 \
  --rir-per-room 1 \
  --num-workers 1 \
  --low-backend analytic \
  --duration 0.4
```

### 在 augmentation 中使用 release

必須明確指定 release recipe 與 split：

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

`usage_role` 必須和 dataset role／split 一致，training dataset 因此不能靜默讀取
validation 或 test。在 M6.5 empirical evidence 和 M6.6 promotion certificate
尚未完成前，`require_production: false` 是有意為之。遇到 M6 manifest 時，不要
把未分割的 bank root 直接交給訓練程式。

## 公開指令

以下指令都從 repository root 執行，完整參數請加 `--help`：

| 指令 | 用途 |
|---|---|
| `generate_hybrid_rir.py` | 生成單批 simulated hybrid RIR。 |
| `generate_m6_bank.py` | 生成、QC 並封裝 M6 synthetic candidate。 |
| `render_spatial_rir.py` | 生成 receiver-array、FOA 或選用的 BRIR。 |
| `plot_rir.py` | 繪製波形、EDC、image-source path，以及實際低頻壓力場動畫。 |
| `inspect_bank.py` | 統計單一 RIR folder 的 metadata 分布。 |
| `compare_bank_acoustics.py` | 比較多個 bank 的 DRR、C50、decay 與頻譜統計。 |
| `compare_modal_acoustics.py` | 比較低頻模態 peak、spacing、bandwidth、Q。 |

### 實際低頻聲場壓力動畫

`field` 子指令是獨立的 2-D FDTD 示意模型。若要觀看低頻 modal recurrence
實際重建出的壓力場，使用 `low-field`：

```bash
PYTHONPATH=. python egs/rir_generation/plot_rir.py low-field \
  --rir egs/rir_generation/exp/rir_realism/m6/training_pilot_gpu/pyroomacoustics_bank/room_000349/room_000349_000000.wav \
  --backend pytard --channel 0 --t-ms 80 --gif
```

這個指令會重新跑單筆 sample，從 modal state 重建固定高度的
`p(x, y, z_slice, t)` 切片，並輸出 MP4、選用的 GIF、contact sheet，以及可重用
的 `.npz` 診斷資料。一般 M6 生成流程不會保存壓力場 snapshot。動畫使用 solver
的相對壓力單位，不包含最後的 output calibration，也不包含高頻 band。
`--slice-z` 可以指定切片高度；預設使用 receiver 高度。如果 sidecar metadata
沒有記錄生成 recipe，請用 `--low-sample-rate` 與
`--spatial-samples-per-wavelength` 對齊原本設定。

## 目錄結構

```text
egs/rir_generation/
├── generate_hybrid_rir.py       # 單批 simulated RIR
├── generate_m6_bank.py          # 公開 M6 一鍵流程
├── render_spatial_rir.py        # 空間／陣列 rendering
├── plot_rir.py                  # 繪圖
├── inspect_bank.py              # folder 統計
├── compare_*_acoustics.py       # bank 比較
├── examples/                    # 可重現 recipes
├── tools/                       # audition、measured-RIR、bank helpers
├── exp/rir_realism/             # 保留的 M0–M6 參考 artifacts
└── phases/
    ├── m0_baseline/
    ├── m1_material/
    ├── m2_impedance/
    ├── m3_wave_path/
    ├── m4_spatial_late_field/
    ├── m5_calibration/
    └── m6_bank/
```

各 phase 的 script、config、report 和 fixture 都留在對應 milestone 目錄；
保留的生成證據集中在 [`exp/rir_realism/`](exp/rir_realism/)；根目錄只保留
穩定、可給其他人直接使用的工具。

## 依賴與延伸文件

CPU M6 路徑使用 repository 內的 `pytard` 實作，以及高頻的 Pyroomacoustics：

```bash
pip install pyroomacoustics
```

若安裝相容的 CuPy，也可以使用 `pytard-cupy` 做 GPU 低頻加速；推薦的 CPU
pilot 不需要 GPU。

延伸閱讀：

- [`docs/audio/rir_realism_algorithm_zh-TW.md`](../../docs/audio/rir_realism_algorithm_zh-TW.md)：完整物理與算法說明。
- [`docs/audio/rir_bank_v2_zh-TW.md`](../../docs/audio/rir_bank_v2_zh-TW.md)：M6 bank 契約與證據規則。
- [`docs/audio/hybrid_rir.md`](../../docs/audio/hybrid_rir.md)：hybrid renderer 細節。
- [`docs/audio/rir_scene_v2.md`](../../docs/audio/rir_scene_v2.md)：M1 scene/material schema。
- [`RIR_REALISM_PLAN.md`](../../RIR_REALISM_PLAN.md)：整體 roadmap。
