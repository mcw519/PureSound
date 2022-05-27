from typing import Optional

import torch
import torch.nn as nn

from nnet.base_nn import TaskWarpModule
from nnet.conv_tasnet import TCN, ConvTasNet, GatedTCN
from nnet.lobe.encoder import ConvEncDec, FreeEncDec
from nnet.lobe.pooling import AttentiveStatisticsPooling
from nnet.loss.aamsoftmax import AAMsoftmax
from nnet.loss.sdr import SDRLoss

# Modules
free_encoder = FreeEncDec(win_length=400, hop_length=200, laten_length=512)
stft_encoder = ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=160, output_format='Complex', trainable=True, sr=16000)
conv_tasnet_v0 = ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=4, tcn_dilated_basic=2, per_tcn_stack=8,
                            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], norm_type='gLN', causal=False, tcn_layer='normal')
conv_tasnet = ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=4, tcn_dilated_basic=2, per_tcn_stack=8,
                            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], norm_type='gLN', causal=False, tcn_layer='gated')
tse_speaker_net = nn.ModuleList(
    [GatedTCN(512, 128, 3, dilation=2**i, causal=False, norm_type='gLN') for i in range(5)] + [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]
    )

sisnr_loss = SDRLoss.init_mode('sisnr')
ce_loss = nn.CrossEntropyLoss()
aam_loss = AAMsoftmax(192, 251, margin=0.2, scale=30)


# Models
def get_model(name: str):
    if name == 'td_tse_conv_tasnet_v0':
        model = TaskWarpModule(
            encoder=free_encoder,
            masker=conv_tasnet_v0,
            speaker_net=nn.ModuleList(
                    [TCN(512, 256, 3, dilation=2**i, causal=False, norm_type='gLN') for i in range(5)] + \
                    [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
            loss_func_wav=sisnr_loss,
            loss_func_spk=aam_loss,
            mask_constraint='ReLU')
    
    elif name == 'td_tse_conv_tasnet_v1':
        model = TaskWarpModule(
            encoder=FreeEncDec(win_length=400, hop_length=200, laten_length=512),
            encoder_spk=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=160, output_format='Magnitude', trainable=True, sr=16000),
            masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=4, tcn_dilated_basic=2, per_tcn_stack=8,
                tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], norm_type='gLN', causal=False, tcn_layer='normal'),
            speaker_net=nn.ModuleList(
                [TCN(257, 256, 3, dilation=2**i, causal=False, norm_type='gLN') for i in range(5)] + \
                [AttentiveStatisticsPooling(257, 128), nn.Conv1d(257*2, 192, 1, bias=False)]),
            loss_func_wav=SDRLoss.init_mode('tsdr'),
            loss_func_spk=AAMsoftmax(192, 251, margin=0.2, scale=30),
            mask_constraint='ReLU')

    else:
        raise NameError

    return model
