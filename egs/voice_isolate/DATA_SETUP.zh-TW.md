# 資料準備 —— 從零到可訓練

English version: [`DATA_SETUP.md`](DATA_SETUP.md)

`config/train_dpcrn.yaml` 可以從零一次訓練完成，但它不附帶資料。本頁是從公開
語料到那份 recipe 所讀路徑之間的完整鏈路。recipe 裡所有以 `/path/to/` 開頭的
路徑都是佔位符，必須換成你自己的。

## 預設 recipe 需要什麼

| recipe 欄位 | 需要什麼 | 由誰產生 |
|---|---|---|
| `dataset.train_metafile` / `valid_metafile` | 語者不重疊的語音 metafile | §1 |
| `augmentation_noise.noise_folder` | 噪音語料 | §1 |
| `augmentation_real_far.pool_manifest` | 真實遠距離錄音 | §2 |
| `augmentation_real_near.pool_manifest` | 真實近場錄音 | §2 |
| `rir_bank.banks[core/expand/wide].folder` | 合成 RIR bank，依難度分層 | §3 |
| `rir_bank.banks[real].folder` | 實測 RIR 轉成的 bank | §4 |

以下指令都從 repository root 執行。

## 1. 語音與噪音

DNS-5 兩者都有。先取得並重取樣：

```bash
uv run python egs/voice_isolate/prepare_dns_challenge.py --help
```

接著建 metafile、切成連續 chapter（curriculum 會訓練到 30 秒的列，chapter 語料
是長列得以成立的前提），再依語者切分，確保沒有語者橫跨 train/dev：

```bash
uv run python egs/voice_isolate/prepare_metafile.py ...        # -> data/dns5-read.list

uv run python egs/voice_isolate/scripts/split_by_speaker.py data/dns5-read.list \
    --train-out data/dns5-read.train.list --dev-out data/dns5-read.dev.list \
    --dev-speaker-frac 0.05 --seed 0

uv run python egs/voice_isolate/scripts/build_chapter_corpus.py \
    --metafile data/dns5-read.train.list \
    --out-dir /your/scratch/dns5_read_16k_chapters \
    --out-metafile data/dns5-read.chapters.train.list
```

`augmentation_noise.noise_folder` 指向 DNS-5 的噪音目錄。

## 2. 真實錄音池

同一套語料、依距離切成兩個池。VOiCES 是公開語料（CC-BY）。遠場池提供遠距離
干擾者；近場池提供真實近講列，是模型不會把使用者刪掉的關鍵。

```bash
uv run python egs/voice_isolate/scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 1.0 \
    --out data/realfar_pool/voices.train.jsonl

uv run python egs/voice_isolate/scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 0 --max-distance 1.0 \
    --out data/realfar_pool/voices.near.train.jsonl
```

它產出的是完成的波形而非 RIR，並依語者切分 train/held-out。

## 3. 合成 RIR bank，切成難度層

先產生一個 bank，再切出 curriculum 會依序走過的 `core` / `expand` / `wide`
三層。這些 view 全部是 symlink，留著幾乎不佔空間。

```bash
# 產生（完整參數見 egs/rir_generation/README.md）
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
    --output-dir /your/scratch/hybrid_rir_16k \
    --backend path-events-m4 --n-rooms 1000 --rir-per-room 4 \
    --num-workers 8 --seed 1337 --sample-rate 16000 --duration 1.6 \
    --scene-version v1 --room-type mixed --output-mode calibrated \
    --record-realized-metrics

# 切層：core / expand / wide / stress / all
uv run python egs/rir_generation/tools/bank/build_bank_view.py levels \
    /your/scratch/hybrid_rir_16k \
    /your/scratch/hybrid_rir_16k_levels
```

`levels` 依 RT60 與最壞情況的近／遠場 DRR 差切分，而且各層是**累積**的，所以
curriculum 可以依序走：

| 層級 | RT60 | 最壞情況近／遠 DRR 差 |
|---|---|---|
| `core` | 0.20–0.45 s | ≥ 6 dB |
| `expand` | 0.20–0.65 s | ≥ 3 dB |
| `wide` | 0.20–0.85 s | ≥ 3 dB |
| `stress` | 不屬於以上任何一層的項目 | — |

把 `rir_bank.banks[core/expand/wide].folder` 指向
`/your/scratch/hybrid_rir_16k_levels/{core,expand,wide}`。

## 4. 實測 RIR

`real` 這個 bank 要的是實測脈衝響應，不是合成的。多個公開語料可用（ACE、
dEchorate、BUT ReverbDB、BRUDEX、AIR）；使用前請自行確認各語料的授權條款。

```bash
uv run python egs/rir_generation/tools/measured/real_rir_to_bank.py --help
uv run python egs/rir_generation/tools/bank/build_bank_view.py levels \
    /your/scratch/real_rir_16k /your/scratch/real_rir_16k_levels
```

把 `rir_bank.banks[real].folder` 指向產生出來的 view。

## 5. 投入 GPU 之前先檢查

```bash
uv run python egs/voice_isolate/scripts/check_training_data.py \
    egs/voice_isolate/config/train_dpcrn.yaml --n 64 --dump 8
```

它會回報 manifest 的語者不重疊性、路徑是否存在、在實際取樣項目上量到的近／遠
DRR 差與距離／RT60 分佈，並 dump 幾個混音讓你聽。通過這關，代表路徑正確、而且
取樣到的資料裡真的存在近／遠對比。

接著在投入長時間訓練前，確認 pipeline 真的學得動：

```bash
uv run python egs/voice_isolate/scripts/overfit_check.py \
    egs/voice_isolate/config/train_dpcrn.yaml --steps 800 --device cuda
```

更多工具與各自用途：[`scripts/README.zh-TW.md`](scripts/README.zh-TW.md)。
