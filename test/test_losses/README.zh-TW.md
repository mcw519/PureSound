# 損失函式測試

English version: `README.md`

這裡收錄的是訓練用 loss 的測試。目前實際涵蓋的是 activity/VAD
losses（`VADActivityLoss`、`VADHeadBCELoss`）——包括從 recipe 載入、
外部 VAD targets，以及 imbalanced-frame（樣本不平衡）處理。
`puresound/nnet/loss/` 底下其實還定義了 waveform（SDR）、spectral
（STFT/multi-resolution）與 speaker（GE2E/triplet）等 loss，但這些
目前在本目錄下都還沒有專屬的 unit test。
