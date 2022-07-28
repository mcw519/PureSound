from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .lobe.rnn import SingleRNN
from .lobe.trivial import spectral_compression
from .unet import Unet


class DPRNNblock2D(nn.Module):
    """
    DPRNN-2D modul which be parts of DPCRN.

    Args:
        input_size: input 4D-tensor's channel dimension
        hidden_size: RNN's hidden dimension
        dropout: dropout rate
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.) -> None:
        super().__init__()
        
        self.intra_rnn = SingleRNN('LSTM', input_size, hidden_size, bidirectional=True, dropout=dropout)
        self.intra_norm = nn.LayerNorm(input_size)

        self.inter_rnn = SingleRNN('LSTM', input_size, hidden_size, bidirectional=False, dropout=dropout)
        self.inter_norm = nn.LayerNorm(input_size)
        
    def forward(self, x: torch.Tensor, intra_skip: bool = True, inter_skip: bool = True) -> torch.Tensor:
        """
        Args:
            input tensor has shape as [N, ch, C, T]
        Inputs:
            x -- [N, ch, C, T]

        Returns:
            output -- [N, ch, C, T]
        """
        self.intra_rnn.rnn.flatten_parameters()
        self.inter_rnn.rnn.flatten_parameters()

        x_intra_skip = x.clone()
        N, CH, C, T = x.shape

        # intra-chunk, time independent and frequency dependent
        x = x.transpose(1, -1).reshape(N * T, C, CH) # [N, CH, C, T] -> [N, T, C, CH] -> [N*T, C, CH]
        x = self.intra_rnn(x.permute(0, 2, 1)) # [N*T, C, CH] -> [N*T, CH, C]
        x = x.permute(0, 2, 1) # [N*T, CH, C] -> [N*T, C, CH]
        x = self.intra_norm(x)
        x = x.reshape(N, T, C, -1)
        x = x.transpose(1, -1) # [N, CH, C, T]
        
        if intra_skip:
            x = x_intra_skip + x # [N, CH, C, T]

        x_inter_skip = x.clone()

        # inter-chunk, time dependent and frequency independent
        x = x.permute(0, 2, 3, 1).reshape(N*C, T, -1) # [N, CH, C, T] -> [N, C, T, CH] -> [N*C, T, CH]
        x = self.inter_rnn(x.permute(0, 2, 1)) # [N*C, T, CH] -> [N*C, CH, T]
        x = x.permute(0, 2, 1) # [N*C, CH, T] -> [N*C, T, CH]
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(N, C, CH, T)
        x = x.permute(0, 2, 1, 3)

        if inter_skip:
            x = x_inter_skip + x
        
        return x


class DPCRN(Unet):
    def __init__(self,
                input_type: str = 'RI',
                input_dim: int = 512,
                activation_type: str = 'PReLU',
                norm_type: str = 'bN2d',
                dropout: float = 0.05,
                channels: Tuple = (1, 32, 32, 32, 64, 128),
                transpose_t_size: int = 2,
                transpose_delay: bool = False,
                skip_conv: bool = False,
                kernel_t: Tuple = (2, 2, 2, 2, 2),
                stride_t: Tuple = (1, 1, 1, 1, 1),
                dilation_t: Tuple = (1, 1, 1, 1, 1),
                kernel_f: Tuple = (5, 3, 3, 3, 3),
                stride_f: Tuple = (2, 2, 1, 1, 1),
                dilation_f: Tuple = (1, 1, 1, 1, 1),
                delay: Tuple = (0, 0, 0, 0, 0),
                rnn_hidden: int = 128,
                spectral_compress: bool = False,
                ):
        super().__init__(input_type, input_dim, activation_type, norm_type, dropout, channels, transpose_t_size, skip_conv,
            kernel_t, stride_t, dilation_t, kernel_f, stride_f, dilation_f, delay)
        
        self.transpose_delay = transpose_delay
        self.rnn_hidden = rnn_hidden
        self.spectral_compress = spectral_compress

        # DPRNN block
        self.dprnn_block1 = DPRNNblock2D(input_size=channels[-1], hidden_size=rnn_hidden, dropout=dropout)
        self.dprnn_block2 = DPRNNblock2D(input_size=channels[-1], hidden_size=rnn_hidden, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, C, T]
            
        Returns:
            output tensor has shape [N, C, T]
        """
        if self.spectral_compress:
            x = spectral_compression(x, alpha=0.3, dim=1)

        if self.input_type.lower() == 'ri':
            _re, _im = torch.chunk(x, 2, dim=-2)
            x = torch.stack([_re, _im], dim=1) # [N, C, T] -> [N, 2, C, T]
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1) # [N, 1, C, T]
        
        skip = [x.clone()]

        # forward CNN-down layers
        for cnn_layer in self.cnn_down:
            x = cnn_layer(x) # [N, ch, C, T]
            skip.append(x)

        # forward dprnn
        x = self.dprnn_block1(x) # [N, ch, C, T]
        x = self.dprnn_block2(x) # [N, ch, C, T]

        # forward CNN-up layers
        for i, cnn_layer in enumerate(self.cnn_up):
            if self.skip_conv:
                x += self.skip_cnn[i](skip[-i-1])
            else:
                x = torch.cat([x, skip[-i-1]], dim=1)
           
            x = cnn_layer(x)
            if self.t_kernel != 1:
                if self.transpose_delay:
                    x = x[..., (self.t_kernel-1):] # transpose-conv with t-kernel size would increase (t-1) length
                else:
                    x = x[..., :-(self.t_kernel-1)] # transpose-conv with t-kernel size would increase (t-1) length
        
        if self.input_type.lower() == 'ri':
            _re = x[:, 0, :, :]
            _im = x[:, 1, :, :]
            x = torch.cat([_re, _im], dim=1)
        
        else:
            x = x.squeeze(1) # [N, 1, C, T] -> [N, C, T]

        return x

    @property
    def get_args(self) -> Dict:
        return {
            'input_type': self.input_type,
            'input_dim': self.input_dim,
            'activation_type': self.activation_type,
            'norm_type': self.norm_type,
            'dropout': self.dropout,
            'channels': self.channels,
            'transpose_t_size': self.transpose_t_size,
            'transpose_delay': self.transpose_delay,
            'skip_conv': self.skip_conv,
            'kernel_t': self.kernel_t,
            'stride_t': self.stride_t,
            'dilation_t': self.dilation_t,
            'kernel_f': self.kernel_f,
            'stride_f': self.stride_f,
            'dilation_f': self.dilation_f,
            'delay': self.delay,
            'rnn_hidden': self.rnn_hidden,
            }
