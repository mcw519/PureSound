import sys

import pytest
import torch
import torch.nn as nn
from puresound.nnet.base_nn import SoTaskWrapModule
from puresound.nnet.conv_tasnet import TCN, ConvTasNet
from puresound.nnet.lobe.encoder import FreeEncDec
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling
from puresound.nnet.loss.aamsoftmax import AAMsoftmax
from puresound.nnet.loss.sdr import SDRLoss

sys.path.insert(0, './')


@pytest.mark.nnet
def test_conv_tasnet():
    model = ConvTasNet(512, 0, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
        tcn_with_embed=[0, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal')
    input_x = torch.rand(1, 512, 100)
    with torch.no_grad():
        y = model(input_x)

    assert input_x.shape == y.shape


@pytest.mark.nnet
def test_conv_tasnet_dvec():
    model = ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
        tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal')
    input_x = torch.rand(1, 512, 100)
    input_dvec = torch.rand(1, 192)
    with torch.no_grad():
        y = model(input_x, input_dvec)

    assert input_x.shape == y.shape


@pytest.mark.nnet
def test_so_task_warpper_2loss():
    sdr_loss_func = SDRLoss.init_mode('sisnr', threshold=-30)
    aam_loss_func = AAMsoftmax(input_dim=192, n_class=666, margin=0.2, scale=30)

    model = SoTaskWrapModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
        speaker_net=nn.ModuleList(
                [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
        loss_func_wav=sdr_loss_func,
        loss_func_spk=aam_loss_func,
        mask_constraint='ReLU',)
    
    input_x = torch.rand(2, 16000*10)
    ref_x = torch.rand(2, 16000*10)
    input_enroll = torch.rand(2, 16000*5)
    ref_cls = torch.LongTensor([66, 67])
    loss = model(noisy=input_x, enroll=input_enroll, ref_clean=ref_x, spk_class=ref_cls, alpha=10, return_loss_detail=False)
    try:
        loss.backward()
    except:
        raise RuntimeError


def test_so_task_warpper_1loss():
    sdr_loss_func = SDRLoss.init_mode('sisnr', threshold=-30)
    
    model = SoTaskWrapModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
        speaker_net=nn.ModuleList(
                [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
        loss_func_wav=sdr_loss_func,
        loss_func_spk=None,
        mask_constraint='ReLU',)
    
    input_x = torch.rand(2, 16000*10)
    ref_x = torch.rand(2, 16000*10)
    input_enroll = torch.rand(2, 16000*5)
    loss = model(noisy=input_x, enroll=input_enroll, ref_clean=ref_x, spk_class=None, alpha=10, return_loss_detail=False)
    try:
        loss.backward()
    except:
        raise RuntimeError


@pytest.mark.nnet
def test_tse_td_tse_conv_tasnet_v0():
    model = SoTaskWrapModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
        speaker_net=nn.ModuleList(
                [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
        loss_func_wav=None,
        loss_func_spk=None,
        mask_constraint='ReLU',)

    input_x = torch.rand(1, 16000*10)
    input_enroll = torch.rand(1, 16000*5)
    y = model.inference(input_x, input_enroll)
    assert input_x.shape == y.shape
