# PureSound
We hope this repo can help you to listen pure clean voice/sound.

## Install & Test
    git clone <project-url>
    cd puresound && sh build_puresound.sh

## Repo struct & Recipes
After merged the v2 branch, all of previous codes used to train a model has been adapted to utilize [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
Belows only list the major parts  

    puresound
    ├── audio: <related to audio processing>
    |   ├── augmentaion.py
    │   ├── dsp.py
    │   ├── impluse_response.py
    │   ├── io.py
    │   ├── noise.py
    │   ├── spectrum.py
    │   └── volume.py
    ├── dataset: <data manifest format and its parser>
    │   ├── base.py
    │   ├── dynamic_base.py
    │   ├── kaldi_base.py
    │   └── parser.py
    ├── nnet: <neural network modules>
    │   ├── dparn.py
    │   ├── dpcrn.py
    │   ├── ecapa_tdnn.py
    │   ├── features.py
    │   ├── masker.py
    │   ├── skim.py
    │   │── unet.py
    │   ├── lobe
    │   │   ├── encoder.py
    │   │   ├── stft.py
    │   │   └── trivial.py
    │   ├── loss
    │   │   ├── sdr.py
    │   │   ├── spk.py
    │   │   └── stft_loss.py
    ├── system: <lightning based modules>
    │   ├── base.py
    │   ├── logger.py
    │   ├── optim.py
    │   └── siso.py
    ├── task: <extending dataset modules for each speech task>
    │   ├── ns.py
    │   └── sv.py
    ├── metrics.py
    └── utils.py

Some samples:

    egs
    ├── default_config.yaml
    ├── noise_suppression
    │   ├── config
    │   │   ├── dparn.yaml
    │   │   └── dpcrn.yaml
    │   ├── main.py
    │   └── prepare_metafile.py
    └── speaker_embedding
        ├── conf
        │   └── ecapa_tdnn.yaml
        ├── local
        │   ├── compute_eer.py
        │   └── voxceleb-O-trail-file.txt
        │── main.py
        └── prepare_metafile.py
