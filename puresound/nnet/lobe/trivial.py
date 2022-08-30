from types import LambdaType
from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Variable

from .norm import ChanLN


class LambdaLayer(nn.Module):
    def __init__(self, lambda_func: LambdaType):
        super().__init__()
        self.lambd = lambda_func
    
    def forward(self, x: torch.Tensor, *kwargs) -> Any:
        return self.lambd(x, kwargs)


class Magnitude(nn.Module):
    """
    This is a layer for converting stft-complex form to magnitude form.

    Args:
        if drop_first will remove the first bin value
        if log1p, return log1p(x)
    """

    def __init__(self, drop_first: bool = True, log1p: bool = False) -> None:
        super().__init__()
        self.drop_first = drop_first
        self.log1p = log1p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor x has shape [N, C, T, 2] or [N, C, T]
        """
        if x.dim() == 4:
            _re = x[..., 0]
            _im = x[..., 1]
        
        elif x.dim() == 3:
            _re, _im = torch.chunk(x, 2, dim=1)
        
        else:
            raise TypeError
        
        if self.drop_first:
            _re = _re[:, 1:, :]
            _im = _im[:, 1:, :]
        
        mag = torch.sqrt(_re.pow(2) + _im.pow(2) + 1e-8)
        if self.log1p:
            mag = torch.log1p(mag)
        
        return mag


class Gate(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, embed_size: int, dropout: float = 0.):
        super().__init__()
        self.in_conv = nn.Conv1d(input_size, hidden_size, kernel_size=1, bias=False, groups=1)
        
        self.left_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=1, bias=False, padding=0, groups=1),
            ChanLN(hidden_size),
            nn.PReLU(),
            nn.Dropout(p=dropout)
        )

        right_in_dim = hidden_size + embed_size
        self.right_conv = nn.Sequential(
            nn.Conv1d(right_in_dim, hidden_size, kernel_size=1, dilation=1, bias=False, padding=0, groups=1),
            ChanLN(hidden_size),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv1d(hidden_size, input_size, kernel_size=1, bias=False, groups=1)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Args:
            input tensor x has shape [N, C, T]
            condition tensor has shape [N, C]
        
        Returns:
            output tensor has shape [N, C, T]
        """
        res = x.clone()
        x = self.in_conv(x)

        condition = condition.unsqueeze(-1) # [N, C, 1]
        condition = condition.repeat(1, 1, x.size(2)) # [N, C, T]
        x_r = torch.cat([x, condition], dim=1)
        
        x = self.left_conv(x) * self.right_conv(x_r)
        x = self.out_conv(x)
        x = x + res
        
        return x


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    FiLM = linear(c)*f(x) + linear'(c), where c is speaker or other vectors
    """
    def __init__(self, feats_size: int, embed_size: int, input_norm: bool = True):
        super().__init__()
        self.cond_scale = nn.Conv1d(feats_size+embed_size, feats_size, kernel_size=1, bias=False)
        self.cond_bias = nn.Conv1d(feats_size+embed_size, feats_size, kernel_size=1, bias=False)

        self.inp_norm = input_norm
        if self.inp_norm:
            self.norm = nn.LayerNorm(feats_size)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Args:
            input tensor x has shape [N, C, T]
            condition tensor has shape [N, C]
        
        Returns:
            output tensor has shape [N, C, T]
        """
        if self.inp_norm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        condition = condition.unsqueeze(-1) # [N, C, 1]
        condition = condition.repeat(1, 1, x.shape[-1])
        condition = torch.cat([x, condition], dim=1)
        film_scale = self.cond_scale(condition)
        film_bias = self.cond_bias(condition)
        x_out = film_scale * x + film_bias
        
        return x_out


class SplitMerge(nn.Module):
    """2S Process: Segmentation and Stitching(merge)."""
    def __init__(self, seg_size: int, seg_overlap: bool = True):
        self.seg_size = seg_size
        self.seg_overlap = seg_overlap

    @staticmethod
    def split(x: torch.Tensor, seg_size: int):
        """
        Args:
            input tensor x has shape [N, C, T]
        
        Returns:
            output tensor segment has shape [N, S, K, C] and padding size
        """
        seg_stride = seg_size // 2

        # padding
        batch, feat_size, seq_len = x.shape # [N, C, T]
        
        rest = seg_size - (seg_stride + seq_len % seg_size) % seg_size
        if rest > 0:
            pad = Variable(torch.zeros(batch, feat_size, rest)).type(x.type())
            x = torch.cat([x, pad], dim=-1)
        
        pad_aux = Variable(torch.zeros(batch, feat_size, seg_stride)).type(x.type())
        x = torch.cat([pad_aux, x, pad_aux], dim=-1)

        # splitting
        batch, feat_size, seq_len = x.shape

        seg_1 = x[:, :, :-seg_stride].contiguous().view(batch, feat_size, -1, seg_size)
        seg_2 = x[:, :, seg_stride:].contiguous().view(batch, feat_size, -1, seg_size)

        segments = torch.cat([seg_1, seg_2], dim=-1).view(batch, feat_size, -1, seg_size) # [N, C, S, K]
        segments = segments.permute(0, 2, 3, 1) # [N, S, K, C]
        
        return segments, rest
    
    @staticmethod
    def merge(x: torch.Tensor, rest: int):
        """
        Args:
            input tensor x has shape [N, S, K, C]
        
        Outputs:
            output tensor has shape [N, C, T]
        """
        batch, total_seg, seg_size, feat_size = x.shape
        seg_stride = seg_size // 2
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, feat_size, -1, seg_size*2)

        x1 = x[:, :, :, :seg_size].contiguous().view(batch, feat_size, -1)[:, :, seg_stride:]
        x2 = x[:, :, :, seg_size:].contiguous().view(batch, feat_size, -1)[:, :, :-seg_stride]

        output = (x1 + x2) / 2
        if rest > 0:
            output = output[..., :-rest]
        
        return output.contiguous()


class MovingAverage1D(nn.Module):
    """
    Simple moving average layer.

    Args:
        kerenl_size: moving window length
        stride: shift of window step
        add_padding: if true, add padding zeros
        causal: if true, padding only in past side, otherwise in both side
    """
    def __init__(self, kernel_size: int, stride: int, add_padding: bool = False, causal: bool = True):
        super().__init__()
        self.add_padding = add_padding
        self.causal = causal
        self.kernel_size = kernel_size
        self.stride = stride
        self.sma = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            input tensor x has shape [N, T]
        
        Returns:
            output tensor out has shape [N, T']
        """
        batch, _ = x.shape
        if self.add_padding:
            if self.causal:
                padd = torch.zeros(batch, self.kernel_size - 1).to(x.device)
                x_pad = torch.cat([padd, x], dim=-1)
            
            else:
                pre_padd = torch.zeros(batch, self.kernel_size//2).to(x.device)
                post_padd = torch.zeros(batch, self.kernel_size//2).to(x.device)
                x_pad = torch.cat([pre_padd, x, post_padd], dim=-1)
        
        else:
            x_pad = x

        x_pad = x_pad
        out = self.sma(x_pad)

        return out


def spectral_compression(x: torch.Tensor, alpha: float = 0.3, dim: int = 1):
    _re, _im = torch.chunk(x, 2, dim=dim)
    mag = _re.pow(2) + _im.pow(2)
    mag = (mag + 1e-8).sqrt()
    mag = mag.pow(alpha)
    phase = torch.atan2(_im+0., _re)
    
    return mag * torch.exp(1j * torch.angle(phase))
