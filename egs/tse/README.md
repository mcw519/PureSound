# TSE
Target Speech Extraction.

## Data structure definition
Majorly we followed the Kaldi's data format.

### Basic data
All of below files has same format as "\<uttid> \<path>".

    wav2scp.txt: input mixture
    wav2ref.txt: reference target speech
    wav2spk.txt: speaker list in mixture, connected by "-"
    ref2spk.txt: target speaker id

### Enrollment file
Different like above, behind first columns would be treated as the path of enrollment speech.

    ref2list: enrolment audios path

For example, this case uttid's (1272-128104-0000_2035-147961-0014_1) enrollment speech could be choiced from "dev-clean/1272/141231/1272-141231-0008.flac" or "dev-clean/1272/141231/1272-141231-0019.flac".
    
    1272-128104-0000_2035-147961-0014_1 dev-clean/1272/141231/1272-141231-0008.flac dev-clean/1272/141231/1272-141231-0019.flac

## Records

1. Libri2Mix-clean-16k-max TD-ConvTasNet(gLN)
    - conf: libri2mix_max_2spk_clean_16k_1a.yaml
    - training data: train-100, max, clean, 3 seconds

    | Metric | Mode | Dev | Test |
    |:---:|:---:|:---:|:---:|
    | Si-SNR | max | 12.727 | 12.700 |
    | Si-SNRi | max | 12.727 | 12.699 |

2. Libri2Mix-clean-16k-max TD-ConvTasNet(bN1d)
    - conf: libri2mix_max_2spk_clean_16k_1b.yaml
    - training data: train-100, max, clean, 3 seconds

    | Metric | Mode | Dev | Test |
    |:---:|:---:|:---:|:---:|
    | Si-SNR | max | - | 8.086 |
    | Si-SNRi | max | - | 8.085 |