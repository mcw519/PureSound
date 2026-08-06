# Speaker Embedding

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

## Train your own speaker embedding model

    # Generate manifest
    python prepare_metafile.py --utt2gender_path=utt2gender.txt --insert_root_path=/corpus/path manifest wav2scp utt2spk
    # Training
    python main.py --training=True work.yaml

## Extract speaker embedding

    python main.py --inference=True --ckpt_path=ckpt_path work.yaml

## Caculate the EER scores

    # Extract embedding for each utterances
    python main.py --inference=True --ckpt_path=ckpt_path work.yaml
    # Scoring
    python local/compute_eer.py eer_trail_file embeddings_folder score_file

## Pretrained models

| MODEL | Files | Feature | Backbone | Loss func | Vox-O EER |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PS-spk-v1 | [onnx](pretrained/PS-spk-v1.onnx)<br>[pytorch](pretrained/PS-spk-v1.ckpt)<br>[config](conf/PS-spk-v1.yaml) | Fbank80 | ECAPA-TDNN | AAMSoftmax<br>(m=0.3) | 1.10 |
| PS-spk-v1-1 | [onnx](pretrained/PS-spk-v1-1.onnx)<br>[pytorch](pretrained/PS-spk-v1-1.ckpt)<br>[config](conf/PS-spk-v1-1.yaml) | Fbank80 | ECAPA-TDNN | AAMSoftmax<br>(m=0.3, c=3, k=5) | 0.99 |

| PARAMETER | DESCRIPTION |
|:---:|:---:|
| m | margin |
| c | sub-center |
| k | top-K class |

## Export the model to ONNX format

    # Generate the ONNX model as ckpt_path.onnx
    python main.py --export_onnx=True --pretrained_ckpt_path=ckpt_path work.yaml

## Running your demo by using `Gradio`

    python demo.py --address=yours_ip_address --port=yours_port onnx_model_path