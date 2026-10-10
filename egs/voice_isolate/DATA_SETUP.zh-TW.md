# 資料準備 —— 從零到可訓練

English: [`DATA_SETUP.md`](DATA_SETUP.md)

`config/train_dpcrn.yaml` 一次 run 從零訓練，但它不附資料。本頁是從公開語料到那份 recipe
所讀路徑的整條鏈。recipe 裡每個以 `/path/to/` 開頭的路徑都是要替換的佔位符；`data/...`
路徑則相對於 `egs/voice_isolate/`。

## 預設 recipe 需要什麼

| Recipe 鍵 | 要什麼 | 在哪一節建 |
|---|---|---|
| `dataset.train_metafile` / `valid_metafile` | 語者互斥、章節長度的語音 metafile | §1 |
| `augmentation_noise.noise_folder` | 16 kHz 的噪音語料 | §1 |
| `augmentation_realfar.pool_manifest` | 真實的遠場錄音 | §2 |
| `augmentation_realnear.pool_manifest` | 真實的近場錄音 | §2 |
| `augmentation_reverb.simulator.pregenerated.banks` 的 `core` / `expand` / `wide` | 依難度切分的合成 RIR bank | §3 |
| `augmentation_reverb.simulator.pregenerated.banks` 的 `real` | 做成 bank 的實測 RIR | §4 |

除非該段另有說明，指令都從 repo 根目錄執行。

## 1. 語音與噪音

DNS-5 兩者都有。語料準備是與其他 recipe 共用的函式庫程式碼——見
[`docs/usage/data_preparation.zh-TW.md`](../../docs/usage/data_preparation.zh-TW.md)。一個指令
掃描朗讀語音 subset、一次轉成 16 kHz，並寫出語者互斥的 train/dev 一對：

```bash
uv run python -m puresound.dataset.corpus.dns_challenge speech /path/to/audio/dns-5 \
    --output-dir egs/voice_isolate/data --id-prefix dns-read --subset read_speech \
    --utt-id-style stem \
    --train-metafile egs/voice_isolate/data/dns5-read.train.list \
    --valid-metafile egs/voice_isolate/data/dns5-read.dev.list \
    --valid-ratio 0.05 --seed 0 \
    --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/clean_fullband
```

utterance id 保留語料自己的檔名（`--utt-id-style stem`），因為下一步要從 id 讀出 segment
編號。recipe 的訓練列最長 30 秒，所以把 10 秒的片段接回整個章節，兩個切分都要做：

```bash
for split in train dev; do
  uv run python egs/voice_isolate/scripts/build_chapter_corpus.py \
      --metafile egs/voice_isolate/data/dns5-read.$split.list \
      --out-dir /your/scratch/dns5_read_16k_chapters/$split \
      --out-metafile egs/voice_isolate/data/dns5-read.chapters.$split.list
done
```

噪音則把 DNS-5 的噪音資料夾轉檔一次，再把 `augmentation_noise.noise_folder` 指向轉好的樹：

```bash
uv run python -m puresound.dataset.corpus.dns_challenge noise /path/to/audio/dns-5 \
    --output-dir egs/voice_isolate/data --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband
```

## 2. 真實錄音池

兩個池都來自 VOiCES（公開，CC BY），依喇叭到麥克風的距離切分。遠場池提供遠處的干擾者；
近場池提供真實的近講列，讓模型不會把使用者刪掉。

```bash
cd egs/voice_isolate
uv run python scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 1.0 \
    --out data/realfar_pool/voices.train.jsonl

uv run python scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 0 --max-distance 1.0 \
    --out data/realfar_pool/voices.near.train.jsonl
```

這些是完成的錄音，不是 RIR，並依語者切成 train/held-out。`--distractor none`（預設）保留
乾淨的單獨遠場講話者，那是近場隔離該用的干擾者；`--stats-only` 只印出池的統計、不寫檔。

## 3. 合成 RIR bank，切成難度等級

先產生一個 bank，再切出 curriculum 要走過的 `core` / `expand` / `wide` view。view 只是
symlink，保留它們不花任何成本。完整參數與直譯器要求見
[`egs/rir_generation/README.zh-TW.md`](../rir_generation/README.zh-TW.md)。

```bash
# 產生：寫出 <out>/path-events-m4_bank/（item）與 <out>/path-events-m4_release/
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
    --output-dir /your/scratch/hybrid_rir_16k \
    --backend path-events-m4 --n-rooms 1000 --rir-per-room 4 \
    --num-workers 8 --seed 1337 --sample-rate 16000 --duration 1.6 \
    --scene-version v1 --room-type mixed --output-mode calibrated \
    --record-realized-metrics

# 把 bank 切成等級：core / expand / wide / stress / all
PYTHONPATH=. .venv/bin/python egs/rir_generation/tools/bank/build_bank_view.py levels \
    /your/scratch/hybrid_rir_16k/path-events-m4_bank \
    /your/scratch/hybrid_rir_16k_levels
```

產生器預設用 `--low-backend pytard-material` 算低頻段，這需要你自行安裝的
[pytARD](https://github.com/gpuard/pytARD) checkout（AGPL-3.0，不在 PyPI 上；把
`PURESOUND_PYTARD_ROOT` 指向它——見[相依套件](../rir_generation/README.zh-TW.md#相依套件)）。
若要避開這個相依，加上 `--low-backend analytic-material`：它是不需安裝任何東西的矩形房間模態
模型，因此做出的 bank 與預設產生的不會完全相同。

`levels` 量測每個 item 的 RT60 與最壞情況的近/遠 DRR 差距（`min(DRR_near) - max(DRR_far)`），
而且等級是累積的，curriculum 可以依序走過：

| 等級 | RT60 | 最壞情況的近/遠 DRR 差距 |
|---|---|---|
| `core` | 0.20–0.45 s | ≥ 6 dB |
| `expand` | 0.20–0.65 s | ≥ 3 dB |
| `wide` | 0.20–0.85 s | ≥ 3 dB |
| `stress` | 不屬於以上任何一級的有效 item | -- |
| `all` | 每個有效 item，不過濾 | -- |

`--drr-window-ms` 必須與 recipe 的 `drr_window_ms`（預設 2.5）一致。把 `core`、`expand`、
`wide` 三個 bank 成員指向 `/your/scratch/hybrid_rir_16k_levels/{core,expand,wide}`。

## 4. 實測 RIR

`real` 這個 bank 成員要的是實測的脈衝響應，其近場聲道是真正一公尺以內的響應。
`real_rir_to_bank.py` 分兩步轉換公開語料——先用該語料專屬的掃描器產生 manifest，再交給
共用的 bank 寫出器：

```bash
# 掃描一個語料：ace、brudex、dechorate、diffrir 或 slr28
PYTHONPATH=. .venv/bin/python egs/rir_generation/tools/measured/real_rir_to_bank.py scan dechorate \
    --input /path/to/dEchorate --staging /your/scratch/dechorate_wav \
    --manifest /your/scratch/dechorate.manifest.jsonl

# 寫出 16 kHz 的 bank，依距離切分近/遠
PYTHONPATH=. .venv/bin/python egs/rir_generation/tools/measured/real_rir_to_bank.py from-manifest \
    --manifest /your/scratch/dechorate.manifest.jsonl \
    --output /your/scratch/real_rir_16k/dechorate --target-sr 16000
```

每個語料各做一次，再用 `build_bank_view.py merge --source TAG=PATH ... --output DIR` 把各
bank 合成一個訓練 view，並把 `real` 成員指向它。腳本的 `--help` 記載了每個語料的幾何與
距離處理方式。BUT ReverbDB 不要放進訓練 bank：benchmark 的第 7b 與第 8 關是用它建的。
請自行確認每個語料的授權是否適用你的用途。

## 5. 投入 GPU 之前先檢查

```bash
cd egs/voice_isolate
uv run python scripts/check_training_data.py config/train_dpcrn.yaml --n 64 --dump 8
```

它會報告 manifest 的語者互斥性、路徑是否存在、在實際抽樣 item 上量到的近/遠 DRR 差距與
距離、RT60 分佈，並輸出幾個混音讓你聽。通過它的 recipe，路徑是對的，近/遠對比也確實存在於
抽樣資料中。

接著在長時間訓練前，先確認這條管線學得起來：

```bash
uv run python scripts/overfit_check.py config/train_dpcrn.yaml --steps 800 --device cuda
```

更多工具，以及每個工具的用途：[`scripts/README.zh-TW.md`](scripts/README.zh-TW.md)。
