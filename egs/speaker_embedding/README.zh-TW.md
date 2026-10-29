# 語者 Embedding

English version: [`README.md`](README.md)

## 訓練自己的語者 embedding 模型

    # 產生 manifest
    python -m puresound.dataset.corpus.kaldi --utt2gender_path=utt2gender.txt --insert_root_path=/corpus/path manifest wav2scp utt2spk
    # 訓練
    python main.py --training work.yaml

## 萃取語者 embedding

    python main.py --inference --ckpt_path ckpt_path work.yaml

## 計算 EER 分數

    # 為每個 utterance 萃取 embedding
    python main.py --inference --ckpt_path ckpt_path work.yaml
    # 計分
    python local/compute_eer.py eer_trail_file embeddings_folder score_file

## 預訓練模型

| MODEL | Files | Feature | Backbone | Loss func | Vox-O EER |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PS-spk-v1 | [onnx](pretrained/PS-spk-v1.onnx)<br>[pytorch](pretrained/PS-spk-v1.ckpt)<br>[config](conf/PS-spk-v1.yaml) | Fbank80 | ECAPA-TDNN | AAMSoftmax<br>(m=0.3) | 1.10 |
| PS-spk-v1-1 | [onnx](pretrained/PS-spk-v1-1.onnx)<br>[pytorch](pretrained/PS-spk-v1-1.ckpt)<br>[config](conf/PS-spk-v1-1.yaml) | Fbank80 | ECAPA-TDNN | AAMSoftmax<br>(m=0.3, c=3, k=5) | 0.99 |

| PARAMETER | DESCRIPTION |
|:---:|:---:|
| m | margin |
| c | sub-center |
| k | top-K class |

## 判定門檻

`puresound infer`、網頁介面與 `demo.py` 在兩段 embedding 的餘弦相似度高於模型門檻時判為同一人，
門檻寫在 [`model_zoo/catalog.yaml`](../../model_zoo/catalog.yaml)：PS-spk-v1-1 為 **0.37**，
PS-spk-v1 為 **0.39**。兩者都在 LibriSpeech test-clean（40 位朗讀者、2,620 段語句，任兩段語句
即一組試驗）上校正，取「同性別的不同朗讀者有 1% 被誤判為同一人」的那一點，四捨五入到 0.01。
完整紀錄見
[`speaker_threshold_librispeech_test_clean_20261023.json`](../../model_zoo/benchmarks/speaker_threshold_librispeech_test_clean_20261023.json)。

| LibriSpeech test-clean | PS-spk-v1-1 | PS-spk-v1 |
|---|---|---|
| EER，全部試驗 | 1.41%（門檻 0.33） | 1.87%（門檻 0.32） |
| EER，同一朗讀者的不同章節 | 1.54% | 2.03% |
| EER，每段只取前 3 秒 | 1.64% | 2.11% |
| 同性別冒充者誤收 1% | 門檻 0.372，誤拒 2.5% | 門檻 0.392，誤拒 3.9% |
| 先前的預設值 0.8 | 拒掉 83% 的同一人試驗 | 拒掉 83% |

分數是原始的餘弦相似度。`local/compute_eer.py` 計算 EER 時會換算成 (score + 1) / 2，
EER 不變，但門檻代表的意義會變。LibriSpeech 是乾淨的朗讀語音；嘈雜、遠場或電話錄音會讓同一人的
分數降低，在那類資料上使用前請先重新校正：

    uv run python local/calibrate_threshold.py LibriSpeech/test-clean \
        --speakers LibriSpeech/SPEAKERS.TXT --out threshold.json

## 匯出模型為 ONNX 格式

    # 產生 ONNX 模型，輸出為 ckpt_path.onnx
    python main.py --export_onnx --pretrained_ckpt_path ckpt_path work.yaml

## 用 `Gradio` 執行你的 demo

    python demo.py --address=yours_ip_address --port=yours_port onnx_model_path
