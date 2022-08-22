# PureSound
We hope this repo can help you to listen pure clean voice/sound.

## Install & Test
    git clone <project-url>
    cd puresound && sh build_puresound.sh

## Repo struct & Recipes

    PureSound
        |-- puresound
            | -- src: Audio I/O and others processing.
            | -- nnet: DNN realted models and loss function.
            | -- streaming: DNN models streaming structure.
            | -- task: PyTorch trainer and Dataset.
        | -- egs
            | -- tse: Sample of target speech extraction or target speaker VAD task.
            | -- ns: Smaple of noise supression task.