# NS
Noise Suppression speech enhancement.

## Data structure definition
Majorly we followed the Kaldi's data format.

### Basic data
All of below files has same format as "\<uttid> \<path>".

    wav2scp.txt: input mixture path
    wav2ref.txt: reference target speech path

## Records of VCTK+DEMAND

| Model | Params | Lookahead | PSEQ | STOI |
|:-----:|:------:|:---------:|:----:|:----:|
| DPCRN-1a | 1,380,043 |  0  | 2.496 | 0.938 |
| DPCRN-1b | 1,380,043 | 640 | 2.814 | 0.944 |
| DPARN-1a | 1,215,179 |  0  | 2.568 | 0.940 |
| DPARN-1b | 1,215,179 | 640 | 2.924 | 0.947 |
