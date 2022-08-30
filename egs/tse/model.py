from typing import Optional

import torch
import torch.nn as nn
from puresound.nnet.base_nn import SoTaskWrapModule
from puresound.nnet.conv_tasnet import TCN, ConvTasNet, GatedTCN
from puresound.nnet.lobe.encoder import ConvEncDec, FreeEncDec
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling
from puresound.nnet.lobe.rnn import SingleRNN
from puresound.nnet.lobe.trivial import Magnitude
from puresound.nnet.loss.aamsoftmax import AAMsoftmax
from puresound.nnet.loss.metrics import F1_loss, GE2ELoss, TripletLoss
from puresound.nnet.loss.sdr import SDRLoss
from puresound.nnet.loss.stft_loss import (MultiResolutionSTFTLoss,
                                           over_suppression_loss)
from puresound.nnet.skim import SkiM
from puresound.nnet.unet import UnetTcn


# Loss
def init_loss(hparam):
    sig_loss = hparam['LOSS']['sig_loss']
    cls_loss = hparam['LOSS']['cls_loss']
    sig_threshold = hparam['LOSS']['sig_threshold']

    # Init SDR-based loss function
    if sig_loss.lower() in ['sisnr', 'sdsdr', 'sdr', 'tsdr']:
        sig_loss = SDRLoss.init_mode(sig_loss.lower(), threshold=sig_threshold)
    
    elif sig_loss.lower() == 'sisnr_stft':
        sdr_loss = SDRLoss.init_mode('sisnr', threshold=sig_threshold)
        stft_loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, others: stft_loss(enh, ref) + sdr_loss(enh, ref, others)
    
    elif sig_loss.lower() == 'sisnr_ov':
        sdr_loss = SDRLoss.init_mode('sisnr', threshold=sig_threshold)
        sig_loss = lambda enh, ref, others: sdr_loss(enh, ref, others) + over_suppression_loss(enh, ref)

    elif sig_loss.lower() == 'f1':
        f1_loss = F1_loss()
        sig_loss = lambda enh, ref, others: f1_loss(enh, ref)

    else:
        sig_loss = None
    
    # Init Class-fication loss function
    if cls_loss.lower() == 'cross_entropy':
        cls_loss == nn.CrossEntropyLoss()

    elif cls_loss.lower() == 'aamsoftmax':
        cls_loss = AAMsoftmax(input_dim=hparam['LOSS']['embed_dim'], n_class=hparam['LOSS']['n_class'], margin=hparam['LOSS']['margin'], scale=hparam['LOSS']['scale'])
    
    elif cls_loss.lower() == 'ge2e':
        assert hparam['TRAIN']['contrastive_learning']
        cls_loss = GE2ELoss(nspks=hparam['TRAIN']['p_spks'], putts=hparam['TRAIN']['p_utts'], add_norm=True)

    else:
        cls_loss = None

    if hparam['LOSS']['cls_loss_other'] is None:
        return sig_loss, cls_loss

    else:
        if hparam['LOSS']['cls_loss_other'].lower() == 'triplet':
            cls_loss_other = TripletLoss(margin=0.3, add_norm=True, distance='consine')
        
        else:
            raise NotImplementedError
        
        return sig_loss, cls_loss, cls_loss_other


# Models
def init_model(name: str, sig_loss: Optional[nn.Module] = None, cls_loss: Optional[nn.Module] = None, **kwargs):
    if name == 'td_tse_conv_tasnet_v0':
        """
        Total params: 10,156,311
        Lookahead(samples): infinite
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
            masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
                tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_layer='normal'),
            speaker_net=nn.ModuleList(
                    [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                    [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)

    elif name == 'td_tse_conv_tasnet_v0_causal':
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
            masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
                tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], tcn_norm='bN1d', dconv_norm='bN1d', causal=True, tcn_layer='normal'),
            speaker_net=nn.ModuleList(
                    [TCN(512, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                    [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)

    elif name == 'tse_unet_tcn_v0':
        """
        Total params: 13,372,725
        Lookahead(samples): infinite
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=128, trainable=True, output_format='Complex'),
            masker=UnetTcn(embed_dim=192, embed_norm=True, input_type='RI', input_dim=512, activation_type='PReLU', norm_type='gLN',
                            channels=(1, 32, 64, 128, 128, 128, 128), transpose_t_size=2, transpose_delay=True, skip_conv=False, kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
                            stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2), dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1), delay=(0, 0, 0, 0, 0, 0),
                            tcn_layer='gated', tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2, per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
                            tcn_norm='gLN', dconv_norm='gGN', causal=False),
            speaker_net=nn.ModuleList(
                [Magnitude(drop_first=False)] + \
                [GatedTCN(256, 128, 3, dilation=2**i, causal=False, tcn_norm='gLN') for i in range(5)] + \
                [AttentiveStatisticsPooling(256, 128), nn.Conv1d(256*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='linear',
            drop_first_bin=True,
            **kwargs)
    
    elif name == 'tse_unet_tcn_v0_causal':
        """
        Total params: 13,372,725
        Lookahead(samples): 1152
        Receptive Fields(samples): 24960
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=128, trainable=True, output_format='Complex'),
            masker=UnetTcn(embed_dim=192, embed_norm=True, input_type='RI', input_dim=512, activation_type='PReLU', norm_type='bN2d',
                            channels=(1, 32, 64, 128, 128, 128, 128), transpose_t_size=2, transpose_delay=True, skip_conv=False, kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
                            stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2), dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1), delay=(0, 0, 0, 0, 0, 0),
                            tcn_layer='gated', tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2, per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
                            tcn_norm='bN1d', dconv_norm='bN1d', causal=True),
            speaker_net=nn.ModuleList(
                [Magnitude(drop_first=False)] + \
                [GatedTCN(256, 128, 3, dilation=2**i, causal=False, tcn_norm='gLN') for i in range(5)] + \
                [AttentiveStatisticsPooling(256, 128), nn.Conv1d(256*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='linear',
            drop_first_bin=True,
            **kwargs)
    
    elif name == 'tse_unet_tcn_v1':
        """
        Total params: 14,404,917
        Lookahead(samples): infinite
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(fft_length=512, win_type='hann', win_length=512, hop_length=128, trainable=True, output_format='Complex'),
            masker=UnetTcn(embed_dim=192, embed_norm=True, input_type='RI', input_dim=512, activation_type='PReLU', norm_type='gLN',
                            channels=(1, 32, 64, 128, 128, 128, 128), transpose_t_size=2, transpose_delay=True, skip_conv=False, kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
                            stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2), dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1), delay=(0, 0, 0, 0, 0, 0),
                            tcn_layer='gated', tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2, per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
                            tcn_norm='gLN', dconv_norm='gGN', causal=False, tcn_use_film=True),
            speaker_net=nn.ModuleList(
                [Magnitude(drop_first=False)] + \
                [GatedTCN(256, 128, 3, dilation=2**i, causal=False, tcn_norm='gLN') for i in range(5)] + \
                [AttentiveStatisticsPooling(256, 128), nn.Conv1d(256*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='linear',
            drop_first_bin=True,
            **kwargs)
    
    elif name == 'tse_skim_v0':
        """
        Total params: 15,575,570
        Lookahead(samples): infinite
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True),
            masker=SkiM(input_size=128, hidden_size=256, output_size=128, n_blocks=4, seg_size=150, seg_overlap=False, causal=False,
                embed_dim=192, embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion='FiLM'),
            speaker_net=nn.ModuleList(
                [TCN(128, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(128, 128), nn.Conv1d(128*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)
    
    elif name == 'tse_skim_v0_causal':
        """
        Total params: 6,375,442
        Lookahead(samples): 16
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True),
            masker=SkiM(input_size=128, hidden_size=256, output_size=128, n_blocks=4, seg_size=150, seg_overlap=False, causal=True,
                embed_dim=192, embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion='FiLM'),
            speaker_net=nn.ModuleList(
                [TCN(128, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(128, 128), nn.Conv1d(128*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)
    
    elif name == 'tse_skim_v1_causal':
        """
        Total params: 6,249,219
        Lookahead(samples): 16
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True),
            masker=SkiM(input_size=128, hidden_size=256, output_size=128, n_blocks=4, seg_size=150, seg_overlap=False, causal=True,
                embed_dim=192, embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion='FiLM'),
            speaker_net=nn.ModuleList(
                [SingleRNN(rnn_type='LSTM', input_size=128, hidden_size=192, bidirectional=True, dropout=0.05)] + \
                [AttentiveStatisticsPooling(128, 128), nn.Conv1d(128*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            **kwargs)
    
    elif name == 'tse_skim_v0_causal_vad':
        """
        Total params: 1,181,392
        Lookahead(samples): 16
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True),
            masker=SkiM(input_size=128, hidden_size=64, output_size=128, n_blocks=2, seg_size=150, seg_overlap=False, causal=True,
                embed_dim=192, embed_norm=True, block_with_embed=[1, 1], embed_fusion='FiLM'),
            speaker_net=nn.ModuleList(
                [TCN(128, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
                [AttentiveStatisticsPooling(128, 128), nn.Conv1d(128*2, 192, 1, bias=False)]),
            loss_func_wav=sig_loss,
            loss_func_spk=cls_loss,
            mask_constraint='ReLU',
            output_constraint='Sigmoid',
            **kwargs)

    else:
        raise NameError

    return model
