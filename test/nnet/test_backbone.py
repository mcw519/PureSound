"""Every backbone in the library builds from its constructor and maps its input
to an output of the same shape. The inputs are short: what is under test is the
shape contract and that each configuration's code path runs, not the numbers."""

import pytest
import torch

import puresound.nnet as nnet
from puresound.nnet.conv_tasnet import ConvTasNet, GatedTCN
from puresound.nnet.dparn import DPARN
from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.dprnn import DPRNN
from puresound.nnet.skim import SkiM
from puresound.nnet.tfgridnet import TFGridNet
from puresound.nnet.unet import UnetTcn

EMBED = 192


def _conv_tasnet(embed):
    return ConvTasNet(
        512, EMBED if embed else 0, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3,
        tcn_dilated_basic=2, per_tcn_stack=8,
        tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0] if embed else [0] * 8,
        tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
    )


def _unet_tcn():
    return UnetTcn(
        embed_dim=EMBED, embed_norm=True, input_dim=256, activation_type="PReLU",
        norm_type="gLN", channels=(2, 32, 64, 128, 128, 128, 128), transpose_t_size=2,
        transpose_delay=True, skip_conv=False, kernel_t=(2,) * 6, kernel_f=(5,) * 6,
        stride_t=(1,) * 6, stride_f=(2,) * 6, dilation_t=(1,) * 6, dilation_f=(1,) * 6,
        delay=(0,) * 6, tcn_layer="gated", tcn_kernel=3, tcn_dim=256,
        tcn_dilated_basic=2, per_tcn_stack=5, repeat_tcn=3,
        tcn_with_embed=[1, 0, 0, 0, 0], tcn_norm="gLN", dconv_norm=None, causal=False,
    )


def _dpcrn():
    return DPCRN(
        input_dim=256, activation_type="PReLU", norm_type="bN2d",
        channels=(2, 32, 32, 32, 64, 128), transpose_t_size=2, transpose_delay=False,
        skip_conv=False, kernel_t=(2,) * 5, kernel_f=(5, 3, 3, 3, 3), stride_t=(1,) * 5,
        stride_f=(2, 2, 1, 1, 1), dilation_t=(1,) * 5, dilation_f=(1,) * 5,
        delay=(0,) * 5, rnn_hidden=128,
    )


def _tfgridnet():
    return TFGridNet(
        inp_channel_dim=2, input_dim=256, channel_dim=32, lstm_dim=128, n_block=2,
        block_delay_frames=0, kernel_f_size=5, kernel_t_size=5, f_stride=4, n_head=4,
        channel_qk=4, attent_range=100,
    )


BACKBONES = [
    pytest.param(lambda: _conv_tasnet(False), (1, 512, 50), False, id="conv_tasnet"),
    pytest.param(lambda: _conv_tasnet(True), (1, 512, 50), True, id="conv_tasnet-embed"),
    pytest.param(_unet_tcn, (1, 2, 256, 50), True, id="unet_tcn-embed"),
    pytest.param(_dpcrn, (1, 2, 256, 100), False, id="dpcrn"),
    pytest.param(
        lambda: DPARN(input_dim=256, norm_type="cLN", channels=(2, 32, 32, 32, 64, 128)),
        (1, 2, 256, 50), False, id="dparn",
    ),
    pytest.param(
        lambda: SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=True),
        (1, 512, 200), False, id="skim-overlap",
    ),
    pytest.param(
        lambda: SkiM(512, 256, 512, 4, 32, causal=False, seg_overlap=False, dropout=0.1),
        (1, 512, 200), False, id="skim-noncausal",
    ),
    pytest.param(
        lambda: SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=False, embed_dim=EMBED,
                     embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion="Gate"),
        (1, 512, 200), True, id="skim-embed",
    ),
    pytest.param(
        lambda: DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=False),
        (1, 512, 200), False, id="dprnn",
    ),
    pytest.param(
        lambda: DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True, embed_dim=EMBED,
                      embed_norm=True, block_with_embed=[0, 1, 1, 0]),
        (1, 512, 200), True, id="dprnn-overlap-embed",
    ),
    pytest.param(_tfgridnet, (1, 2, 256, 120), False, id="tfgridnet"),
]


@pytest.mark.backbone
@pytest.mark.parametrize("build, shape, conditioned", BACKBONES)
def test_backbone_forward_keeps_the_input_shape(build, shape, conditioned):
    model = build()
    x = torch.rand(*shape)
    with torch.no_grad():
        y = model(x, torch.rand(1, EMBED)) if conditioned else model(x)
    assert y.shape == x.shape


@pytest.mark.backbone
@pytest.mark.parametrize(
    "type_name",
    ["ConvTasNet", "DPARN", "DPCRN", "DPRNN", "SkiM", "TFGridNet", "Unet", "UnetFsmn",
     "UnetTcn", "EcapaTdnnExtractor"],
)
def test_backbone_reachable_from_config(type_name):
    """Every model in the library must be resolvable the way recipe configs do it:
    ``getattr(nnet, backbone["type"])``. A model missing from ``nnet/__init__``
    exists on disk but cannot be named by any config."""
    assert callable(getattr(nnet, type_name))


@pytest.mark.backbone
@pytest.mark.parametrize("kernel", [1, 3])
def test_causal_gated_tcn_keeps_every_frame_for_any_kernel(kernel):
    block = GatedTCN(8, 16, kernel=kernel, dilation=2, causal=True, tcn_norm="cLN").eval()
    x = torch.randn(2, 8, 30)
    with torch.no_grad():
        assert block(x).shape == x.shape


@pytest.mark.backbone
@pytest.mark.parametrize("overlap", [False, True])
def test_causal_skim_rows_do_not_see_each_other(overlap):
    """Segments of different rows share one flattened axis inside the memory
    LSTM; the one-segment shift must stop at each row's first segment."""
    torch.manual_seed(0)
    model = SkiM(16, 8, 16, 3, 8, causal=True, seg_overlap=overlap).eval()
    x = torch.randn(2, 16, 40)
    other = x.clone()
    other[0] += torch.randn_like(other[0])
    with torch.no_grad():
        alone, perturbed = model(x), model(other)
        row_only = model(x[1:])
    assert torch.equal(alone[1], perturbed[1])
    assert torch.allclose(alone[1:], row_only, atol=1e-6)


@pytest.mark.backbone
@pytest.mark.parametrize("overlap", [False, True])
def test_embedding_free_dprnn_is_seeded_by_the_enrollment_not_by_film(overlap):
    """With no FiLM blocks configured the enrollment sequence is the only
    conditioning, so changing it must change the output."""
    torch.manual_seed(0)
    model = DPRNN(16, 8, 16, 2, 8, causal=True, seg_overlap=overlap, embedding_free_tse=True).eval()
    x, enroll, other = torch.randn(2, 16, 40), torch.randn(2, 16, 40), torch.randn(2, 16, 40)
    with torch.no_grad():
        seeded, reseeded = model(x, enroll), model(x, other)
    assert seeded.shape == x.shape
    assert not torch.allclose(seeded, reseeded)
