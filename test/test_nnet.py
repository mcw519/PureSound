import sys

import pytest
import torch
import torch.nn as nn
from puresound.nnet.base_nn import SoTaskWrapModule
from puresound.nnet.conv_tasnet import TCN, ConvTasNet, GatedTCN
from puresound.nnet.lobe.encoder import ConvEncDec, FreeEncDec
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling
from puresound.nnet.lobe.trivial import Magnitude
from puresound.nnet.loss.aamsoftmax import AAMsoftmax
from puresound.nnet.loss.metrics import GE2ELoss, TripletLoss
from puresound.nnet.loss.sdr import SDRLoss
from puresound.nnet.skim import SkiM
from puresound.nnet.unet import UnetTcn

sys.path.insert(0, './')


@pytest.mark.nnet
def test_so_task_warpper_modulist():
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


def test_so_task_warpper_sequential():
    spk_net = [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
            [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]
    model = SoTaskWrapModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
        speaker_net=nn.Sequential(*spk_net),
        loss_func_wav=None,
        loss_func_spk=None,
        mask_constraint='ReLU',)
    
    input_x = torch.rand(1, 16000*10)
    input_enroll = torch.rand(1, 16000*5)
    y = model.inference(input_x, input_enroll)
    assert input_x.shape == y.shape


@pytest.mark.nnet
def test_so_task_warpper_3loss():
    sdr_loss_func = SDRLoss.init_mode('sisnr', threshold=-30)
    aam_loss_func = AAMsoftmax(input_dim=192, n_class=666, margin=0.2, scale=30)
    triplet_loss_func = TripletLoss(margin=1, add_norm=True, distance='Euclidean')

    model = SoTaskWrapModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
        speaker_net=nn.ModuleList(
                [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
        loss_func_wav=sdr_loss_func,
        loss_func_spk=aam_loss_func,
        loss_func_others=triplet_loss_func,
        mask_constraint='ReLU',)
    
    input_x = torch.rand(2, 16000*3)
    ref_x = torch.rand(2, 16000*3)
    input_enroll = torch.rand(2, 16000*5)
    ref_cls = torch.LongTensor([66, 67])
    loss = model(noisy=input_x, enroll=input_enroll, ref_clean=ref_x, spk_class=ref_cls, alpha=10, return_loss_detail=False)
    try:
        loss.backward()
    except:
        raise RuntimeError


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
    
    input_x = torch.rand(2, 16000*3)
    ref_x = torch.rand(2, 16000*3)
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
    
    input_x = torch.rand(2, 16000*3)
    ref_x = torch.rand(2, 16000*3)
    input_enroll = torch.rand(2, 16000*5)
    loss = model(noisy=input_x, enroll=input_enroll, ref_clean=ref_x, spk_class=None, alpha=10, return_loss_detail=False)
    try:
        loss.backward()
    except:
        raise RuntimeError


def test_so_task_warpper_contrastive():
    ge2e_loss = GE2ELoss(nspks=2, putts=2*3)
    
    model = SoTaskWrapModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
        speaker_net=nn.ModuleList(
                [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
        loss_func_wav=None,
        loss_func_spk=ge2e_loss,
        mask_constraint='ReLU',)
    
    input_x = torch.rand(6, 16000*3)
    input_enroll = torch.rand(6, 16000*5)
    spk_class = torch.Tensor([[1, 1, 1], [2, 2, 2]])
    loss = model(noisy=input_x, enroll=input_enroll, spk_class=spk_class)
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
    dvec = model.inference_tse_embedding(input_enroll)
    assert input_x.shape == y.shape
    assert dvec.shape[1] == 192


@pytest.mark.nnet
def test_tse_unet_tcn_v0():
    model = SoTaskWrapModule(
        encoder=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=128, trainable=True, output_format='Complex'),
        masker=UnetTcn(embed_dim=192, embed_norm=True, input_type='RI', input_dim=512, activation_type='PReLU', norm_type='gLN',
                        channels=(1, 32, 64, 128, 128, 128, 128), transpose_t_size=2, transpose_delay=True, skip_conv=False,
                        kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
                        stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2),
                        dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1), delay=(0, 0, 0, 0, 0, 0),
                        tcn_layer='gated', tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2, per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
                        tcn_norm='gLN', dconv_norm=None, causal=False),
        speaker_net=nn.ModuleList(
            [Magnitude(drop_first=False)] + \
            [GatedTCN(256, 128, 3, dilation=2**i, causal=False, tcn_norm='gLN') for i in range(5)] + \
            [AttentiveStatisticsPooling(256, 128), nn.Conv1d(256*2, 192, 1, bias=False)]),
        loss_func_wav=None,
        loss_func_spk=None,
        mask_constraint='linear',
        drop_first_bin=True,)

    input_x = torch.rand(1, 16000*10)
    input_enroll = torch.rand(1, 16000*5)
    y = model.inference(input_x, input_enroll)
    dvec = model.inference_tse_embedding(input_enroll)
    assert dvec.shape[1] == 192


@pytest.mark.nnet
def test_tse_skim_v0():
    model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True),
            masker=SkiM(input_size=128, hidden_size=256, output_size=128, n_blocks=4, seg_size=150, seg_overlap=True, causal=True,
                embed_dim=192, embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion='FiLM'),
            speaker_net=nn.ModuleList(
                [TCN(128, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(128, 128), nn.Conv1d(128*2, 192, 1, bias=False)]),
        loss_func_wav=None,
        loss_func_spk=None,
        mask_constraint='ReLU',)

    input_x = torch.rand(1, 16000*10)
    input_enroll = torch.rand(1, 16000*5)
    y = model.inference(input_x, input_enroll)
    dvec = model.inference_tse_embedding(input_enroll)
    assert input_x.shape == y.shape
    assert dvec.shape[1] == 192
