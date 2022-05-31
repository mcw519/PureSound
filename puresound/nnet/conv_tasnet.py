from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lobe.cnn import DepthwiseSeparableConv1d
from .lobe.norm import get_norm


class TCN(nn.Module):
    """
    Args:
        in_channels: input feature(channel) dimension
        hid_channels: hidden feature(channel) dimension
        kernel: kernel size
        dilation: dilation size
        dropout: if not 0. applies dropout
        emb_dim (int): if not zero, concate in right_conv's input
        causal (bool): padding by causal scenario, others padding to same length between input and output
        norm_type: the type of normalization layer
    """
    def __init__(self, in_channels: int, hid_channels: int, kernel: int, dilation: int, dropout: float = 0.,
            emb_dim: int = 0, causal: bool = False, norm_type: str = 'gLN') -> None:
        super().__init__()
        self.causal = causal
        self.norm_type = norm_type
        norm_cls = get_norm(norm_type)
        
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels+emb_dim, hid_channels, kernel_size=1, bias=False, groups=1),
            nn.PReLU(),
            norm_cls(hid_channels))
        
        self.dconv = nn.Sequential(
            DepthwiseSeparableConv1d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                hid_channels=None,
                kernel=kernel,
                dilation=dilation,
                skip=False,
                causal=causal),
            nn.PReLU(),
            norm_cls(hid_channels),
            nn.Dropout(p=dropout))
        
        self.out_conv = nn.Conv1d(hid_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor, embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, C, T]
            embed: conditional tensor shape [N, C]
        
        Returns:
            output tensor has shape [N, C, T]
        """
        res = x.clone()
        
        if embed is not None:
            embed = embed.unsqueeze(2) # [N, C, 1]
            embed = embed.repeat(1, 1, x.size(2)) # [N, C, T]
            x = torch.cat([x, embed], dim=1)
        
        x = self.in_conv(x)
        x = self.dconv(x)
        x = self.out_conv(x)
        x = x + res
        
        return x


class GatedTCN(nn.Module):
    """
    Gated-TCN block with conditional input. TCN based on Conv1D layer.
    
    Args:
        in_channels: input feature(channel) dimension
        hid_channels: hidden feature(channel) dimension
        kernel: kernel size
        dilation: dilation size
        dropout: if not 0. applies dropout
        emb_dim (int): if not zero, concate in right_conv's input
        causal (bool): padding by causal scenario, others padding to same length between input and output
        norm_type: the type of normalization layer
    """
    def __init__(self, in_channels: int, hid_channels: int, kernel: int, dilation: int, dropout: float = 0.,
            emb_dim: int = 0, causal: bool = False, norm_type: str = 'gLN') -> None:
        super().__init__()
        self.causal = causal
        self.padd = (kernel - 1) * dilation // 2 if not causal else (kernel - 1) * dilation
        self.norm_type = norm_type
        norm_cls = get_norm(norm_type)
        
        self.in_conv = nn.Conv1d(in_channels, hid_channels, kernel_size=1, bias=False, groups=1)
        
        self.left_conv = nn.Sequential(
            nn.Conv1d(hid_channels, hid_channels, kernel_size=kernel, dilation=dilation, bias=False, padding=self.padd, groups=1),
            norm_cls(hid_channels),
            nn.PReLU(),
            nn.Dropout(p=dropout)
        )

        right_in_dim = hid_channels + emb_dim
        self.right_conv = nn.Sequential(
            nn.Conv1d(right_in_dim, hid_channels, kernel_size=kernel, dilation=dilation, bias=False, padding=self.padd, groups=1),
            norm_cls(hid_channels),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv1d(hid_channels, in_channels, kernel_size=1, bias=False, groups=1)
    
    def forward(self, x: torch.Tensor, embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, C, T]
            embed: conditional tensor shape [N, C]
        
        Returns:
            output tensor has shape [N, C, T]
        """
        res = x.clone()
        x = self.in_conv(x)
        
        if embed is not None:
            embed = embed.unsqueeze(2) # [N, C, 1]
            embed = embed.repeat(1, 1, x.size(2)) # [N, C, T]
            x_r = torch.cat([x, embed], dim=1)
        
        else:
            x_r = x
        
        x = self.left_conv(x) * self.right_conv(x_r)
        x = self.out_conv(x)
        
        if self.causal:
            x = x[..., :-self.padd] + res
        
        else:
            x = x + res
        
        return x


class ConvTasNet(nn.Module):
    """
    Re-implement Conv-TasNet structure for supporting multi-input which is waveform with dvec(embedding).
    TCN layer refers from 
    
    Args:
        input_dim: Input feature dimension, in TF-based system this is FFT bin size.
        embed_dim: Embedding feature dimension.
        embed_norm: If True, applies the 2-norm on the input embedding.
        tcn_kernel: TCN's kernel size, defaule is 3.
        tcn_dim: TCN's hidden dimansion.
        tcn_dilated_basic: TCN's dilation basic.
        per_tcn_stack: Number of TCN layers in one TNC-stack.
        repeat_tcn: Repeat N TCN-stacks.
        tcn_with_embed: In each TCN-stack, where need to inject the embedding.
        norm_type: Normalization methods.

    Note:
        We ignore the encoder/decoder here for easy optimizing different front-end encoder/decoder.
    """
    def __init__(self,
                input_dim: int = 512,
                embed_dim: int = 256,
                embed_norm: bool = False,
                tcn_layer: str = 'gated',
                tcn_kernel: int = 3,
                tcn_dim: int = 256,
                tcn_dilated_basic: int = 2,
                per_tcn_stack: int = 5,
                repeat_tcn: int = 4,
                tcn_with_embed: List = [1, 0, 0, 0, 0],
                norm_type: str = 'gLN',
                causal: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed_norm = embed_norm
        self.tcn_layer = tcn_layer
        self.tcn_dim = tcn_dim
        self.tcn_kernel = tcn_kernel
        self.per_tcn_stack = per_tcn_stack
        self.repeat_tcn = repeat_tcn
        self.tcn_dilated_basic = tcn_dilated_basic
        self.tcn_with_embed = tcn_with_embed
        self.norm_type = norm_type
        self.causal = causal

        if self.tcn_layer.lower() == 'gated':
            tcn_cls = GatedTCN
        else:
            tcn_cls = TCN

        assert per_tcn_stack == len(tcn_with_embed)
        self.tcn_list = nn.ModuleList()
        for _ in range(repeat_tcn):
            _tcn = []
            
            for i in range(per_tcn_stack):
                if tcn_with_embed[i]:
                    _tcn.append(tcn_cls(input_dim, tcn_dim, kernel=tcn_kernel, dilation=tcn_dilated_basic**i, emb_dim=embed_dim,
                                        causal=causal, norm_type=norm_type))
                else:
                    _tcn.append(tcn_cls(input_dim, tcn_dim, kernel=tcn_kernel, dilation=tcn_dilated_basic**i, emb_dim=0,
                                        causal=causal, norm_type=norm_type))

            self.tcn_list.append(nn.ModuleList(_tcn))
            
    def forward(self, x: torch.Tensor, dvec: torch.Tensor):
        """
        Args:
            x: Input mixture feats [N, C, T]
            dvec: Speaker tensor with shape [N, embed_dim]
        
        Returns:
            TF-mask as same shape of input_dim
        """
        # normalize
        if self.embed_norm:
            dvec = F.normalize(dvec, p=2, dim=1)

        # forward TCN block
        for r in range(self.repeat_tcn):
            for i in range(len(self.tcn_list[r])):
                if self.tcn_with_embed[i]:
                    x = self.tcn_list[r][i](x, dvec)
                else:
                    x = self.tcn_list[r][i](x)
        
        return x
    
    @property
    def get_args(self) -> Dict:
        return {
            'input_dim': self.input_dim,
            'embed_dim': self.embed_dim,
            'embed_norm': self.embed_norm,
            'norm_type': self.norm_type,
            'tcn_layer': self.tcn_layer,
            'tcn_dim': self.tcn_dim,
            'tcn_kernel': self.tcn_kernel,
            'tcn_dilated_basic': self.tcn_dilated_basic,
            'repeat_tcn': self.repeat_tcn,
            'per_tcn_stack': self.per_tcn_stack,
            'tcn_with_embed': self.tcn_with_embed,
            'causal': self.causal,
            }
