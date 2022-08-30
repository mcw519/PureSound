from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from puresound.nnet.base_nn import SoTaskWrapModule
from puresound.nnet.dparn import DPARN
from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.lobe.encoder import ConvEncDec, FreeEncDec
from puresound.nnet.loss.sdr import SDRLoss
from puresound.nnet.loss.stft_loss import (MultiResolutionSTFTLoss,
                                           over_suppression_loss)
from puresound.nnet.skim import SkiM


# Loss
def init_loss(hparam):
    sig_loss = hparam['LOSS']['sig_loss']
    sig_threshold = hparam['LOSS']['sig_threshold']

    # Init SDR-based loss function
    if sig_loss.lower() in ['sisnr', 'sdsdr', 'sdr', 'tsdr']:
        sig_loss = SDRLoss.init_mode(sig_loss.lower(), threshold=sig_threshold)
    
    elif sig_loss.lower() == 'l1':
        sig_loss = lambda enh, ref, dummy: F.l1_loss(enh, ref)
    
    elif sig_loss.lower() == 'stft':
        loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, dummy: loss(enh, ref)
    
    elif sig_loss.lower() == 'sdr_stft':
        sdr_loss = SDRLoss.init_mode('sdr', threshold=sig_threshold)
        loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, dummy: torch.log(loss(enh, ref)) + sdr_loss(enh, ref, dummy)
    
    elif sig_loss.lower() == 'sdr_ov':
        sdr_loss = SDRLoss.init_mode('sdr', threshold=sig_threshold)
        sig_loss = lambda enh, ref, dummy: sdr_loss(enh, ref, dummy) + over_suppression_loss(enh, ref)
    
    elif sig_loss.lower() == 'l1_stft':
        loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, dummy: loss(enh, ref) + F.l1_loss(enh, ref)

    elif sig_loss.lower() == 'stft_ov':
        loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, dummy: loss(enh, ref) + over_suppression_loss(enh, ref)

    else:
        sig_loss = None
    
    return sig_loss


# Models
def init_model(name: str, sig_loss: Optional[nn.Module] = None, **kwargs):
    if name == 'ns_dpcrn_v0_causal':
        """
        Total params: 1,380,043
        Lookahead(samples): 384
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=128, trainable=True, output_format='Complex'),
            masker=DPCRN(input_type='RI', input_dim=512, activation_type='PReLU', norm_type='bN2d', dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128), transpose_t_size=2, transpose_delay=False, skip_conv=False, kernel_t=(2, 2, 2, 2, 2), kernel_f=(5, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1), stride_f=(2, 2, 1, 1, 1), dilation_t=(1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1), delay=(0, 0, 0, 0, 0), rnn_hidden=128),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            drop_first_bin=True,
            mask_constraint='linear',
            f_type='Complex',
            mask_type='Complex',
            **kwargs)
    
    elif name == 'ns_dpcrn_v1_causal':
        """
        Total params: 1,116,655
        Lookahead(samples): 600
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=200, hop_length=100, laten_length=128, output_active=True),
            masker=DPCRN(input_type='Real', input_dim=128, activation_type='PReLU', norm_type='cLN', dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128), transpose_t_size=2, transpose_delay=False, skip_conv=True, kernel_t=(2, 2, 2, 2, 2), kernel_f=(3, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1), stride_f=(1, 1, 1, 1, 1), dilation_t=(1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1), delay=(1, 1, 1, 1, 1), rnn_hidden=128),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            mask_constraint='ReLU',
            **kwargs)

    elif name == 'ns_skim_v0_causal':
        """
        Total params: 5294209
        Lookahead(samples): 16
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True),
            masker=SkiM(input_size=128, hidden_size=256, output_size=128, n_blocks=4, seg_size=150, seg_overlap=False, causal=True),
        speaker_net=None,
        loss_func_wav=sig_loss,
        loss_func_spk=None,
        mask_constraint='ReLU',
        **kwargs)
    
    elif name == 'ns_dparn_v0_causal':
        """
        Total params: 1,215,179
        Lookahead(samples): 384 or 512
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=128, trainable=True, output_format='Complex'),
            masker=DPARN(input_type='RI', input_dim=512, activation_type='PReLU', norm_type='bN2d', dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128), transpose_t_size=2, transpose_delay=False, skip_conv=False, kernel_t=(2, 2, 2, 2, 2), kernel_f=(5, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1), stride_f=(2, 2, 1, 1, 1), dilation_t=(1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1), delay=(1, 0, 0, 0, 0), rnn_hidden=128, nhead=8),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            drop_first_bin=True,
            mask_constraint='linear',
            f_type='Complex',
            mask_type='Complex',
            **kwargs)

    else:
        raise NameError

    return model
