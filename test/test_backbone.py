import sys

import pytest
import torch
from puresound.nnet.conv_tasnet import ConvTasNet
from puresound.nnet.dprnn import DPRNN
from puresound.nnet.skim import SkiM
from puresound.nnet.unet import UnetTcn

sys.path.insert(0, './')


@pytest.mark.backbone
def test_conv_tasnet_backbone():
    model = ConvTasNet(512, 0, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
        tcn_with_embed=[0, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal')
    input_x = torch.rand(1, 512, 100)
    y = model(input_x)
    assert input_x.shape == y.shape


@pytest.mark.backbone
def test_conv_tasnet_dvec_backbone():
    model = ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
        tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal')
    input_x = torch.rand(1, 512, 100)
    input_dvec = torch.rand(1, 192)
    y = model(input_x, input_dvec)
    assert input_x.shape == y.shape


@pytest.mark.backbone
def test_unet_tcn_backbone():
    model = UnetTcn(embed_dim=192, embed_norm=True, input_type='RI', input_dim=512, activation_type='PReLU', norm_type='gLN',
        channels=(1, 32, 64, 128, 128, 128, 128), transpose_t_size=2, transpose_delay=True, skip_conv=False,
        kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
        stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2),
        dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1), delay=(0, 0, 0, 0, 0, 0),
        tcn_layer='gated', tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2, per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
        tcn_norm='gLN', dconv_norm=None, causal=False)
    input_x = torch.rand(1, 512, 100)
    input_dvec = torch.rand(1, 192)
    y = model(input_x, input_dvec)
    assert input_x.shape == y.shape


@pytest.mark.backbone
def test_skim_backbone():
    model = SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
    input_x = torch.rand(1, 512, 1000)
    y = model(input_x)
    assert input_x.shape == y.shape

    model = SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=False)
    input_x = torch.rand(1, 512, 1000)
    y = model(input_x)
    assert input_x.shape == y.shape

    model = SkiM(512, 256, 512, 4, 32, causal=False, seg_overlap=False, dropout=0.1)
    input_x = torch.rand(1, 512, 1000)
    y = model(input_x)
    assert input_x.shape == y.shape

    model = SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=False, embed_dim=192, embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion='Gate')
    input_x = torch.rand(1, 512, 1000)
    embed = torch.rand(1, 192)
    y = model(input_x, embed)
    assert input_x.shape == y.shape


@pytest.mark.backbone
def test_dprnn_backbone():
    model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
    input_x = torch.rand(1, 512, 1000)
    y = model(input_x)
    assert input_x.shape == y.shape

    model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=False)
    input_x = torch.rand(1, 512, 1000)
    y = model(input_x)
    assert input_x.shape == y.shape

    model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True, embed_dim=192, embed_norm=True, block_with_embed=[0, 1, 1, 0])
    input_x = torch.rand(1, 512, 1000)
    input_d = torch.rand(1, 192)
    y = model(input_x, input_d)
    assert input_x.shape == y.shape
