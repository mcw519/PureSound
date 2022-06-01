from typing import Optional

import torch
import torch.nn as nn

from puresound.nnet.base_nn import SoTaskWrapModule
from puresound.nnet.conv_tasnet import TCN, ConvTasNet, GatedTCN
from puresound.nnet.lobe.encoder import ConvEncDec, FreeEncDec
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling
from puresound.nnet.loss.aamsoftmax import AAMsoftmax
from puresound.nnet.loss.sdr import SDRLoss


# Loss
def init_loss(hparam):
    sig_loss = hparam['LOSS']['sig_loss']
    cls_loss = hparam['LOSS']['cls_loss']
    sig_threshold = hparam['LOSS']['sig_threshold']

    # Init SDR-based loss function
    if sig_loss.lower() in ['sisnr', 'sdsdr', 'sdr', 'tsdr']:
        sig_loss = SDRLoss.init_mode(sig_loss.lower(), threshold=sig_threshold)
    else:
        sig_loss = None
    
    # Init Class-fication loss function
    if cls_loss.lower() == 'cross_entropy':
        cls_loss == nn.CrossEntropyLoss()

    elif cls_loss.lower() == 'aamsoftmax':
        cls_loss = AAMsoftmax(input_dim=hparam['LOSS']['embed_dim'], n_class=hparam['LOSS']['n_class'], margin=hparam['LOSS']['margin'], scale=hparam['LOSS']['scale'])
    
    else:
        cls_loss = None

    return sig_loss, cls_loss


# Models
def init_model(name: str, sig_loss: Optional[nn.Module] = None, cls_loss: Optional[nn.Module] = None, **kwargs):
    if name == 'td_tse_conv_tasnet_v0':
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
            masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
                tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], norm_type='gLN', causal=False, tcn_layer='normal'),
            speaker_net=nn.ModuleList(
                    [TCN(512, 256, 3, dilation=2**i, causal=False, norm_type='gLN') for i in range(5)] + \
                    [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)

    elif name == 'td_tse_conv_tasnet_v0_causal':
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
            masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
                tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], norm_type='cLN', causal=True, tcn_layer='normal'),
            speaker_net=nn.ModuleList(
                    [TCN(512, 256, 3, dilation=2**i, causal=False, norm_type='gLN') for i in range(5)] + \
                    [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)

    else:
        raise NameError

    return model
