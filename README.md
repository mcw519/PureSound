# PureSound
We hope this repo can help you to listen pure clean voice/sound.

## Install & Test
    git clone <project-url>
    cd puresound && sh build_puresound.sh

## Repo struct & Recipes

    PureSound
    ├─puresound
    │  ├─nnet: DNN realted models and loss function.
    │  │  ├─lobe
    │  │  └─loss
    │  ├─src: Audio I/O and others processing.
    │  ├─streaming: DNN models streaming structure.
    │  └─task: PyTorch trainer and Dataset.
    ├─egs
    │  ├─ns: Example of Noise Supression
    │  └─tse: Example of Target Speech Extraction
    │      ├─demo: Simple demo App for real-time recording and extract your voice