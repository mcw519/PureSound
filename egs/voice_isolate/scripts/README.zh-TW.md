# scripts — voice_isolate 工具

English version: [`README.md`](README.md)

檔名對應腳本的功能：

| 前綴 | 意義 |
|---|---|
| `build_*` | 建立一次資料集、測試集或 pool，之後重複使用 |
| `check_*` | 在信任一個輸入或一次執行結果之前先做健檢 |
| `eval_*`  | 在某一個軸線上為 checkpoint 評分 |
| `dump_*`  | 印出／匯出原始資訊 |

除非另有註明，所有腳本都從 repo root 執行。Eval 腳本預設 `--device cpu`，
這樣才不會跟訓練用的 GPU 搶資源，而且所有腳本在適用時都接受已發布的推論旋鈕：
`--dry-blend a`（`out = a*enh + (1-a)*mix`，`1.0` = 關閉）與 `--spec-floor f`
（`|enh| >= f*|mix|` 逐 bin 計算、保留相位，`0.0` = 關閉；僅限 complex-mask
模型）。兩者都只影響推論、實作在 `siso.forward` 裡，對訓練沒有任何影響。已
發布的 `dpcrn_v8` checkpoint 是**用 `--dry-blend 0.9`** 評分的——見
`../pretrained_ckpt/README.md`。

## 1. 資料準備

| script | 用途 |
|---|---|
| `split_by_speaker.py` | 對一份 metafile 做語者互斥的 train/dev 切分（依 `spkid` 重新分組，讓 dev 語者絕不會洩漏進 train）。 |
| `build_real_recording_pool.py` | 把真實遠距錄音（VOiCES）索引進訓練 pipeline 抽取用的 pool manifest：遠場干擾者（`--min-distance 1.0`）與真實近場 keep 列（`--min-distance 0 --max-distance 1.0`）。輸出的是完成品波形而非 RIR，並依語者切分 train/held-out。 |
| `check_training_data.py` | 檢視一份 recipe 實際會訓練到什麼：manifest 的語者互斥性＋路徑存在性、在真實抽樣 item 上量到的近／遠 DRR-gap 與距離／RT60 分布、實際算出的 target-to-residual ratio，以及一份供試聽的 wav dump。 |

```bash
uv run python egs/voice_isolate/scripts/split_by_speaker.py egs/voice_isolate/data/dns5-read.list \
    --train-out egs/voice_isolate/data/dns5-read.train.list --dev-out egs/voice_isolate/data/dns5-read.dev.list \
    --dev-speaker-frac 0.05 --seed 0

uv run python egs/voice_isolate/scripts/build_real_recording_pool.py \
    --voices-root /path/to/VOiCES --split train --min-distance 1.0 \
    --out egs/voice_isolate/data/realfar_pool/voices.train.jsonl

uv run python egs/voice_isolate/scripts/check_training_data.py \
    egs/voice_isolate/config/train_dpcrn.yaml --n 64 --dump 8
```

## 2. 訓練期檢查

| script | 用途 |
|---|---|
| `check_training_run.sh` | 一鍵完成：找出這次執行最新的 checkpoint、印出 loss curve、跑 in-domain 指標，印出 PASS / MARGINAL / FAIL。 |
| `eval_indomain.py` | **主要判準指標。** 重用 training/valid 的合成 pipeline，對 enhanced 與 early-reverb target 算 SI-SDRi；`--by-bucket` 會拆出 counter_level / 1N+0F / overlap / distance 各 bucket（要看困難的 bucket，不要只看 aggregate）。 |
| `overfit_check.py` | 這條 pipeline 到底學不學得起來？用真正的訓練 loss 對一個固定 batch 做 overfit（`--pick-hardest-of N` 可避開 passthrough 已經贏的 batch）。`--gate` 則改為凍結分離器、只 fit 逐幀 gate head。這是管線通不通的測試，不是泛化測試。 |
| `dump_loss_curves.py` | 把 tensorboard 的 scalar curve 印成精簡的 ASCII。 |

```bash
bash egs/voice_isolate/scripts/check_training_run.sh config/train_dpcrn.yaml [device] [n_batches] [ckpt]
uv run python egs/voice_isolate/scripts/eval_indomain.py <config> --ckpt <ckpt> \
    --device cpu --n-batches 40 --by-bucket --dump-distribution
uv run python egs/voice_isolate/scripts/overfit_check.py \
    egs/voice_isolate/config/train_dpcrn.yaml --steps 800 --device cuda
```

## 3. 真實聲學 WER（建一次，評任何 checkpoint）

在實測 RIR（BUT ReverbDB）上做的任務對齊近／遠 benchmark，搭配 LibriTTS 的
ground-truth 逐字稿，所以 WER 不是 whisper 對 whisper 的自我比較。

```bash
# 3a. 建立凍結集合（一次性）
uv run python egs/voice_isolate/scripts/build_wer_set.py --n-items 200 \
    --out data_report/but_wer_set --seed 1234

# 3b. 為一個 checkpoint 評分（每個 checkpoint 重複一次）
uv run python egs/voice_isolate/scripts/eval_wer.py config/exp/eval_but_real.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --set-dir data_report/but_wer_set --device cuda --asr faster-whisper --asr-model large-v3
# Azure 雲端 STT（uv pip install azure-cognitiveservices-speech；SPEECH_KEY/SPEECH_REGION 環境變數）：
SPEECH_KEY=... SPEECH_REGION=eastus ... --asr azure
```

`--asr` 切換 backend（`auto`/`faster-whisper`/`openai-whisper`/`azure`）；快取
與輸出檔名都以 backend 為 key，所以切換 backend 不會沿用舊快取——**只能在同一個
backend 內比較 WER**（enh 對 mix，不能跨辨識器比較）。太弱的辨識器會把過度抑制
藏起來，所以判斷 deletion 要用強的辨識器。

## 4. Dawn Chorus（參考指標）+ ASR 輔助工具

`eval_dawn_chorus.py`——Dawn Chorus 真實錄音 WER/SI-SDRi。**注意：它的 FG/BG
切分不是以距離定義的，所以線索與這個近／遠任務不匹配**——保留作 do-no-harm
的參考，不是主要判準。也匯出了 `init_asr`/`wer_breakdown`，供 `eval_wer.py`
重用。

## 5. 真實錄音行為（部署 gate）

模擬的 far-only 探針會把遠場壓制高估超過 20 dB，而 in-domain 指標完全看不到
keep 側的失敗。以下兩個腳本在真實錄音上同時量測兩側。

| script | 用途 |
|---|---|
| `eval_far_suppression.py` | 壓制側：真實遠場語音依距離被壓制了多少。`--corpus voices` 依 (room, mic) 分桶，span 取自錄音自身的能量；`--corpus realman` 依標註距離分桶，span 取自逐樣本對齊的直達路徑參考。希望超過 1 m 時非常負，1 m 以內約 0 dB。 |
| `eval_keep_robustness.py` | 保留側：近場語音在其擷取鏈逐漸偏離訓練用的擷取鏈時（dry → 模擬 RIR → 實測 RIR → 真實錄音），有噪音／無噪音下各存活多少。若階梯很陡，代表 keep 決策是錨定在擷取鏈上、而非距離上。 |
| `eval_realcase.py` | 在真實片段（`windows.json`）上手動標註 keep/suppress 區段，把兩種失敗方向分開評分。若 case 目錄附有第二套系統的輸出，會一併評分。 |
| `eval_turntaking.py` | 相同的雙側計分卡，區段由一組凍結 turn-taking 集合的 target 能量自動推得。 |
| `build_turntaking_set.py` | 建立上述凍結的 (mix, target) turn-taking 集合；`--rir-folder` 可換成實測 RIR bank 來產生真實房間的輪替。 |
| `eval_gate.py` | 在留存的模擬集合上，逐幀 gate-head 計分卡（recall / specificity / balanced accuracy / BCE）——因為 gate 訓練不動 mask 路徑，這是唯一能看出 gate 進展的視角。 |
| `eval_domain_gap.py` | Domain-gap 分解，不是 checkpoint 評分器：在同時擁有真實錄音（R）與實測脈衝響應（M）的匹配 VOiCES (room, mic) 三元組上，建出同源的合成 bank 版本（S），回報 R-vs-M 的 `fit_db`（LTI-convolution 天花板——非線性＋時變＋噪音底）與 M-vs-S 的 `fit_db`（RIR-bank 保真度），外加噪音底；`--ckpt` 會加上每一腿各自的 behavioral suppression。產出 `data_report/domain_gap_v7.jsonl`，正是第 8 階段真實化修正（`dpcrn_v8`）背後的證據。 |

`eval_realcase.py` 與 `eval_turntaking.py` 接受 `--gate`，把一個訓練好的 gate
head 當成逐幀增益套用到輸出上，在只有 mask 的那一列旁邊加上 `gate_soft` /
`gate_hard` 列。

```bash
uv run python egs/voice_isolate/scripts/eval_far_suppression.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --corpus voices --voices-root /path/to/VOiCES --device cuda --per-bucket 40

uv run python egs/voice_isolate/scripts/eval_keep_robustness.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt v6=egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt \
    --ckpt v7=egs/voice_isolate/pretrained_ckpt/dpcrn_v7.ckpt \
    --ckpt v8=egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --device cuda

uv run python egs/voice_isolate/scripts/eval_realcase.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --cases-dir egs/voice_isolate/data_report/qvf22_real_cases --device cpu
```

以上所有項目（加上 in-domain 與 WER 階段）都能透過一個指令為一個 checkpoint
一次跑完：

```bash
cd egs/voice_isolate && bash run_full_benchmark.sh <ckpt> <tag> [device] [dry_blend]
```

## 6. 推論／部署

| script | 用途 |
|---|---|
| `demo.py` | Gradio demo：離線增強、checkpoint 下拉選單（掃描 `config/infer_dpcrn.yaml` 的 `trainer.work_folder` → `../pretrained_ckpt/`，所以每個版本與匯出的 `streaming/*.onnx` 都會出現）、旋鈕滑桿、頻譜圖。 |
| `streaming_onnx.py` | 逐幀 streaming ONNX 的 `export` / `infer` / `benchmark` / `verify`。`verify` 會先依回報的演算法延遲對齊，才進行評分——省略這一步的比較會把 30 ms 延遲讀成誤差。 |

```bash
uv run python egs/voice_isolate/scripts/demo.py --config_path egs/voice_isolate/config/infer_dpcrn.yaml

uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.json --provider cpu
```

## 改名／合併（較舊的 log 使用舊名稱）

| 舊名稱 | 現名稱 |
|---|---|
| `indomain_sisdri.py` | `eval_indomain.py` |
| `eval_but_wer.py` / `build_but_wer_set.py` | `eval_wer.py` / `build_wer_set.py` |
| `eval_gate_vad.py` | `eval_gate.py` |
| `validate_data.py` | `check_training_data.py`（改寫：改成量測抽樣到的 item，而非重新推導模擬器） |
| `run_valid.sh` | `check_training_run.sh` |
| `dump_tb.py` | `dump_loss_curves.py` |
| `dump_turntaking_samples.py` | `build_turntaking_set.py` |
| `build_realfar_pool.py` | `build_real_recording_pool.py` |
| `probe_channel_keep.py` | `eval_keep_robustness.py` |
| `probe_retransmitted_farfield.py` + `probe_realman_farfield.py` | `eval_far_suppression.py --corpus {voices,realman}` |
| `eval_realcase_faronly.py` + `eval_realcase_gated.py` | `eval_realcase.py [--gate]` |
| `eval_turntaking_set.py` + `eval_turntaking_gated.py` | `eval_turntaking.py [--gate]` |
| `overfit_sanity.py` + `overfit_gate_sanity.py` | `overfit_check.py [--gate]` |
| `bench_qvf22_realcases.py` | 已移除——對同一批片段做無參考的 DNSMOS/RMS 比較，`eval_realcase.py` 已直接評分這些片段，而且 DNSMOS 對遠場洩漏沒有反應 |
