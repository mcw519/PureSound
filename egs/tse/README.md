# TSE
Target Speech Extraction.

## Data structure definition
Majorly we followed the Kaldi's data format.

### Basic data
All of below files has same format as "\<uttid> \<path>".

    wav2scp.txt: input mixture path
    wav2ref.txt: reference target speech path
    wav2spk.txt: speaker list in mixture, connected by "-"
    ref2spk.txt: target speaker id

### Enrollment file
Different like above, behind first columns would be treated as the path of enrollment speech.

    ref2list: enrolment audios path

For example, this case uttid's (1272-128104-0000_2035-147961-0014_1) enrollment speech could be choiced from "dev-clean/1272/141231/1272-141231-0008.flac" or "dev-clean/1272/141231/1272-141231-0019.flac".
    
    1272-128104-0000_2035-147961-0014_1 dev-clean/1272/141231/1272-141231-0008.flac dev-clean/1272/141231/1272-141231-0019.flac

## Generate [LibriMix](https://github.com/JorisCos/LibriMix) dataset & metadata

    git clone https://github.com/JorisCos/LibriMix
    cd LibriMix 
    ./generate_librimix.sh storage_dir

    # generate TSE metadata
    cd local
    sh create_metadata.sh storage_dir output_dir librispeech_dir

## Records

1. Libri2Mix-clean-16k-max TD-ConvTasNet(gLN, non causal)
    - conf: libri2mix_max_2spk_clean_16k_1a.yaml
    - training data: train-100, max, clean, 3 seconds

        | Metric | Mode | Dev | Test |
        |:---:|:---:|:---:|:---:|
        | Si-SNRi | max | 12.727 | 12.699 |

2. Libri2Mix-clean-16k-max TD-ConvTasNet(bN1d, causal)
    - conf: libri2mix_max_2spk_clean_16k_1b.yaml
    - training data: train-100, max, clean, 3 seconds

        | Metric | Mode | Dev | Test |
        |:---:|:---:|:---:|:---:|
        | Si-SNRi | max | 8.160 | 8.085 |

3. Libri2Mix-clean-16k-max TD-DP-SkiM(causal)
    - conf: libri2mix_max_2spk_clean_16k_1c.yaml
    - training data: train-100, max, clean, 3 seconds

        | Metric | Mode | Dev | Test |
        |:---:|:---:|:---:|:---:|
        | Si-SNRi | max | 9.954 | 10.043 |


## Demo App

There is a simple demo app in the [./demo](./demo) which based on streaming recording and real-time decoding.

### Install extra dependencies

we need at least 1.12.x torch and 0.12.x torchaudio version.

    pip install sv-ttk
    pip install playsound
    pip install "torch>=1.12.0"
    pip install "torchaudio>=0.12.0"

and then,

    cd demo && python demo_app.py

### Usage:

Enroll:     
Click the `Enroll` button to start the enrollment stage, and click angain to finish the recording. During recording, you have to say some enrollment sentence for geneating speaker embedding.

Clear:  
Click the `Clear` button will delete all status include enrolled speaker embedding.

On/Off switch:  
Click the `On/Off` switch button for starting or stopping your target speaker speech separation function.

Show:  
You can `Show` the record and processed spectrogram and also playback the processed result.