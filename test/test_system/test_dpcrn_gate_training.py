import torch

from puresound.nnet import DPCRN, FeatureEncoder
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.system.siso import EncDecMaskBase


def _make_gate_model() -> EncDecMaskBase:
    encoder = ConvEncDec(
        fft_length=64,
        win_length=64,
        hop_length=32,
        fmin=0,
        fmax=8000,
        sr=16000,
        trainable=False,
    )
    features = FeatureEncoder(
        feats_type="complex",
        drop_stft_first_bin=True,
        trainable=False,
    )
    backbone = DPCRN(
        input_dim=32,
        channels=(2, 4, 8),
        kernel_t=(2, 2),
        stride_t=(1, 1),
        dilation_t=(1, 1),
        kernel_f=(5, 3),
        stride_f=(2, 2),
        dilation_f=(1, 1),
        delay=(0, 0),
        rnn_hidden=4,
        vad_head={"enabled": True, "hidden": 8, "kernel_t": 5},
    )
    return EncDecMaskBase(
        encoder,
        features,
        backbone,
        train_vad_head_only=True,
        gate_head_lr_factor=1.0,
    )


def test_dpcrn_gate_head_has_frame_logits_and_preserves_mask_shape():
    model = _make_gate_model().eval()
    wav = torch.randn(2, 1024)

    enhanced = model(wav)
    logits = model.backbone.last_vad_logits

    assert enhanced.shape == wav.shape
    assert logits is not None
    assert logits.shape[0] == wav.shape[0]
    assert logits.shape[-1] > 0


def test_dpcrn_without_gate_head_remains_backward_compatible():
    backbone = DPCRN(
        input_dim=32,
        channels=(2, 4, 8),
        kernel_t=(2, 2),
        stride_t=(1, 1),
        dilation_t=(1, 1),
        kernel_f=(5, 3),
        stride_f=(2, 2),
        dilation_f=(1, 1),
        delay=(0, 0),
        rnn_hidden=4,
    )
    features = torch.randn(2, 2, 32, 11)

    mask = backbone(features)

    assert mask.shape == features.shape
    assert backbone.vad_head is None
    assert backbone.last_vad_logits is None


