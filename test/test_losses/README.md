# Loss Tests

繁體中文版本：`README.zh-TW.md`

Training loss tests live here. Today that means activity/VAD losses
(`VADActivityLoss`, `VADHeadBCELoss`) — recipe loading, external VAD
targets, and imbalanced-frame handling. `puresound/nnet/loss/` also
defines waveform (SDR), spectral (STFT/multi-resolution), and speaker
(GE2E/triplet) losses, but those currently have no dedicated unit tests
in this directory.
