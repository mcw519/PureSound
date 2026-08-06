# puresound.nnet

English version: [`README.md`](README.md)

語音任務的模型庫（model library）。每個 backbone 都保持可用，並可從 recipe
config 存取到（`getattr(nnet, backbone["type"])`），不論目前是否有任何 recipe
在使用它；`nnet/__init__.py` 是唯一權威的 export 清單。

    puresound/nnet/
    ├── dpcrn.py / dparn.py / dprnn.py / skim.py / conv_tasnet.py /
    │   tfgridnet.py / unet.py / ecapa_tdnn.py     # backbones
    ├── features.py                                # encoder 到 backbone 的特徵轉換
    ├── masker.py                                  # mask 套用工具
    ├── lobe/                                      # 組成積木（rnn/cnn/attention/
    │                                              #  norm/heads/...）
    └── loss/                                      # loss 函式庫（同樣遵循可從 config 存取的規則）

完整 API 參考文件：`docs/nnet/`。
