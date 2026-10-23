# Speaker Embedding

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

## Train your own speaker embedding model

    # Generate manifest
    python -m puresound.dataset.corpus.kaldi --utt2gender_path=utt2gender.txt --insert_root_path=/corpus/path manifest wav2scp utt2spk
    # Training
    python main.py --training work.yaml

## Extract speaker embedding

    python main.py --inference --ckpt_path ckpt_path work.yaml

## Caculate the EER scores

    # Extract embedding for each utterances
    python main.py --inference --ckpt_path ckpt_path work.yaml
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

## Decision threshold

`puresound infer`, the web app and `demo.py` accept a trial when the cosine
similarity of the two embeddings is above the model's threshold in
[`model_zoo/catalog.yaml`](../../model_zoo/catalog.yaml): **0.37** for
PS-spk-v1-1 and **0.39** for PS-spk-v1. Both were set on LibriSpeech test-clean
(40 readers, 2,620 utterances, every pair of utterances a trial) at the point
where 1% of trials between two readers of the same sex are accepted, rounded to
0.01. The full record is
[`speaker_threshold_librispeech_test_clean_20261023.json`](../../model_zoo/benchmarks/speaker_threshold_librispeech_test_clean_20261023.json).

| LibriSpeech test-clean | PS-spk-v1-1 | PS-spk-v1 |
|---|---|---|
| EER, every trial | 1.41% (at 0.33) | 1.87% (at 0.32) |
| EER, same reader in two chapters | 1.54% | 2.03% |
| EER, first 3 s of each utterance | 1.64% | 2.11% |
| 1% false acceptance, same-sex impostors | at 0.372, 2.5% false rejection | at 0.392, 3.9% false rejection |
| The former default, 0.8 | rejects 83% of same-reader trials | rejects 83% |

Scores are raw cosine similarities. `local/compute_eer.py` maps them to
(score + 1) / 2 for its EER, which leaves the EER unchanged but not what a
threshold means. LibriSpeech is clean read speech: noisy, far-field or telephone
recordings lower same-speaker scores, so calibrate on such data before relying on
these points there:

    uv run python local/calibrate_threshold.py LibriSpeech/test-clean \
        --speakers LibriSpeech/SPEAKERS.TXT --out threshold.json

## Export the model to ONNX format

    # Generate the ONNX model as ckpt_path.onnx
    python main.py --export_onnx --pretrained_ckpt_path ckpt_path work.yaml

## Running your demo by using `Gradio`

    python demo.py --address=yours_ip_address --port=yours_port onnx_model_path