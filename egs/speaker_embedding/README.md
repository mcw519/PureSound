# Speaker Embedding

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
    python local/compute_eer.py eer_trail_file embeddings_folder

## Pretrained models

| MODEL | Feature | Backbone | Loss func | Vox-O EER |
|:---:|:---:|:---:|:---:|:---:|
| [PS-spk-v1](pretrained/PS-spk-v1.onnx) | Fbank80 | ECAPA-TDNN | AAMSoftmax (m=0.3) | 1.48 |
| [PS-spk-v2](pretrained/PS-spk-v2.onnx) | Fbank80 | ECAPA-TDNN | SphereFace2 (m=0.2) | 1.44 |

## Export the model to ONNX format

    # Generate the ONNX model as ckpt_path.onnx
    python main.py --export_onnx=True --pretrained_ckpt=ckpt_path work.yaml

## Running your demo by using `Gradio`

    python demo.py --address=yours_ip_address --port=yours_port onnx_model_path