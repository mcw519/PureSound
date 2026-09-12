# 語者 Embedding

English version: [`README.md`](README.md)

## 訓練自己的語者 embedding 模型

    # 產生 manifest
    python prepare_metafile.py --utt2gender_path=utt2gender.txt --insert_root_path=/corpus/path manifest wav2scp utt2spk
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

## 匯出模型為 ONNX 格式

    # 產生 ONNX 模型，輸出為 ckpt_path.onnx
    python main.py --export_onnx --pretrained_ckpt_path ckpt_path work.yaml

## 用 `Gradio` 執行你的 demo

    python demo.py --address=yours_ip_address --port=yours_port onnx_model_path
