# puresound.nnet

Model library for speech tasks. Every backbone stays available and reachable
from a recipe config (`getattr(nnet, backbone["type"])`), whether or not a
current recipe uses it; `nnet/__init__.py` is the authoritative export list.

    puresound/nnet/
    ├── dpcrn.py / dparn.py / dprnn.py / skim.py / conv_tasnet.py /
    │   tfgridnet.py / unet.py / ecapa_tdnn.py     # backbones
    ├── features.py                                # encoder-to-backbone features
    ├── masker.py                                  # mask application utilities
    ├── lobe/                                      # building blocks (rnn/cnn/attention/
    │                                              #  norm/heads/...)
    └── loss/                                      # loss library (same config-reachable rule)

Full API reference: `docs/nnet/`.
