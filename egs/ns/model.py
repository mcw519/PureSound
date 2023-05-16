from typing import Optional

import torch.nn as nn
from puresound.nnet.base_nn import SoTaskWrapModule
from puresound.nnet.dparn import DPARN
from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.nnet.loss.sdr import SDRLoss
from puresound.nnet.loss.stft_loss import MultiResolutionSTFTLoss, over_suppression_loss


# Loss
def init_loss(hparam):
    sig_loss = hparam["LOSS"]["sig_loss"]
    sig_threshold = hparam["LOSS"]["sig_threshold"]

    # Init loss function
    if sig_loss.lower() in ["sisnr", "sdsdr", "sdr", "tsdr"]:
        sig_loss = SDRLoss.init_mode(sig_loss.lower(), threshold=sig_threshold)

    elif sig_loss.lower() == "stft":
        loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, dummy: loss(enh, ref)

    elif sig_loss.lower() == "stft_ov":
        loss = MultiResolutionSTFTLoss()
        sig_loss = lambda enh, ref, dummy: loss(enh, ref) + over_suppression_loss(
            enh, ref
        )

    else:
        sig_loss = None

    return sig_loss


# Models
def init_model(name: str, sig_loss: Optional[nn.Module] = None, **kwargs):
    if name == "ns_dpcrn_v0_causal":
        """
        Total params: 1,380,043
        Lookahead(samples): 384
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(
                fft_length=512,
                win_type="hann",
                win_length=512,
                hop_length=128,
                trainable=True,
                output_format="Complex",
            ),
            masker=DPCRN(
                input_type="RI",
                input_dim=512,
                activation_type="PReLU",
                norm_type="bN2d",
                dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128),
                transpose_t_size=2,
                transpose_delay=False,
                skip_conv=False,
                kernel_t=(2, 2, 2, 2, 2),
                kernel_f=(5, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1),
                stride_f=(2, 2, 1, 1, 1),
                dilation_t=(1, 1, 1, 1, 1),
                dilation_f=(1, 1, 1, 1, 1),
                delay=(0, 0, 0, 0, 0),
                rnn_hidden=128,
            ),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            drop_first_bin=True,
            mask_constraint="linear",
            f_type="Complex",
            mask_type="Complex",
            **kwargs
        )

    elif name == "ns_dpcrn_v0":
        """
        Total params: 1,380,043
        Lookahead(samples): 1024; (384+128*(6-1)); semi-causal
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(
                fft_length=512,
                win_type="hann",
                win_length=512,
                hop_length=128,
                trainable=True,
                output_format="Complex",
            ),
            masker=DPCRN(
                input_type="RI",
                input_dim=512,
                activation_type="PReLU",
                norm_type="bN2d",
                dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128),
                transpose_t_size=2,
                transpose_delay=True,
                skip_conv=False,
                kernel_t=(2, 2, 2, 2, 2),
                kernel_f=(5, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1),
                stride_f=(2, 2, 1, 1, 1),
                dilation_t=(1, 1, 1, 1, 1),
                dilation_f=(1, 1, 1, 1, 1),
                delay=(0, 0, 0, 0, 0),
                rnn_hidden=128,
            ),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            drop_first_bin=True,
            mask_constraint="linear",
            f_type="Complex",
            mask_type="Complex",
            **kwargs
        )

    elif name == "ns_dparn_v0_causal":
        """
        Total params: 1,215,179
        Lookahead(samples): 384
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(
                fft_length=512,
                win_type="hann",
                win_length=512,
                hop_length=128,
                trainable=True,
                output_format="Complex",
            ),
            masker=DPARN(
                input_type="RI",
                input_dim=512,
                activation_type="PReLU",
                norm_type="bN2d",
                dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128),
                transpose_t_size=2,
                transpose_delay=False,
                skip_conv=False,
                kernel_t=(2, 2, 2, 2, 2),
                kernel_f=(5, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1),
                stride_f=(2, 2, 1, 1, 1),
                dilation_t=(1, 1, 1, 1, 1),
                dilation_f=(1, 1, 1, 1, 1),
                delay=(0, 0, 0, 0, 0),
                rnn_hidden=128,
                nhead=8,
            ),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            drop_first_bin=True,
            mask_constraint="linear",
            f_type="Complex",
            mask_type="Complex",
            **kwargs
        )

    elif name == "ns_dparn_v0":
        """
        Total params: 1,215,179
        Lookahead(samples): 1024; (384+128*(6-1)); semi-causal
        Receptive Fields(samples): infinite
        """
        model = SoTaskWrapModule(
            encoder=ConvEncDec(
                fft_length=512,
                win_type="hann",
                win_length=512,
                hop_length=128,
                trainable=True,
                output_format="Complex",
            ),
            masker=DPARN(
                input_type="RI",
                input_dim=512,
                activation_type="PReLU",
                norm_type="bN2d",
                dropout=0.1,
                channels=(1, 32, 32, 32, 64, 128),
                transpose_t_size=2,
                transpose_delay=True,
                skip_conv=False,
                kernel_t=(2, 2, 2, 2, 2),
                kernel_f=(5, 3, 3, 3, 3),
                stride_t=(1, 1, 1, 1, 1),
                stride_f=(2, 2, 1, 1, 1),
                dilation_t=(1, 1, 1, 1, 1),
                dilation_f=(1, 1, 1, 1, 1),
                delay=(0, 0, 0, 0, 0),
                rnn_hidden=128,
                nhead=8,
            ),
            speaker_net=None,
            loss_func_wav=sig_loss,
            loss_func_spk=None,
            drop_first_bin=True,
            mask_constraint="linear",
            f_type="Complex",
            mask_type="Complex",
            **kwargs
        )

    else:
        raise NameError

    return model
