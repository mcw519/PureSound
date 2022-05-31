from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import ChanLN, GlobLN


class DepthwiseSeparableConv1d(nn.Module):
    """
    Separable Conv1d with embed input.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        hid_channels: If not zero, applies a dimension transform from in_channels to hid_channels
        kernel: Kernel size of Conv1d
        stride: Stride step of Conv1d
        dilation: Dilation of Conv1d
        skip: Skip-connection between input to output
        causal: If true, all of operating would be causal
    """
    def __init__(self, in_channels: int, out_channels: int, hid_channels: Optional[int] = None,
                    kernel: int = 3, stride: int = 1, dilation: int = 1, skip: bool = False, causal: bool = True) -> None:
        super().__init__()

        self.causal = causal
        self.skip = skip
        self.transform = False

        if hid_channels is not None:
            self.transform = True
            if self.causal:
                self.in_conv = nn.Sequential(
                    nn.Conv1d(in_channels, hid_channels, 1),
                    nn.BatchNorm1d(hid_channels),
                    nn.PReLU()
                )
            else:
                self.in_conv = nn.Sequential(
                    nn.Conv1d(in_channels, hid_channels, 1),
                    nn.GroupNorm(1, hid_channels, eps=1e-8),
                    nn.PReLU()
                )

        self.hid_channels = hid_channels if hid_channels is not None else in_channels

        if self.causal:
            self.padding = (kernel - 1) * dilation
            self.depthwise = nn.Sequential(
                nn.Conv1d(self.hid_channels, self.hid_channels, kernel_size=kernel, stride=stride, dilation=dilation, padding=self.padding, groups=self.hid_channels),
                nn.BatchNorm1d(self.hid_channels),
                nn.PReLU())
            self.pointwise = nn.Sequential(
                nn.Conv1d(self.hid_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.PReLU())
        
        else:
            self.padding = ((kernel - 1) // 2 ) * dilation
            self.depthwise = nn.Sequential(
                nn.Conv1d(self.hid_channels, self.hid_channels, kernel_size=kernel, stride=stride, dilation=dilation, padding=self.padding, groups=self.hid_channels),
                nn.GroupNorm(1, self.hid_channels, eps=1e-8),
                nn.PReLU())
            self.pointwise = nn.Sequential(
                nn.Conv1d(self.hid_channels, out_channels, kernel_size=1, stride=1),
                nn.GroupNorm(1, out_channels, eps=1e-8),
                nn.PReLU())
        
        if self.skip:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, C, T]
        
        Returns:
            output tensor shape [N, C, T]
        """
        if self.transform:
            res = self.in_conv(x)
        else:
            res = x.clone()

        res = self.depthwise(res)
        res = self.pointwise(res)

        if self.causal:
            res = res[..., :-self.padding]

        if self.skip:
            res = res + self.skip_conv(x)
        
        return res


class SpectralTransform(nn.Module):
    """
    This is a part of FFC module.

    Args:
        in_channels: input channel dimension
        out_channels: output channel dimension
        kernel_size: (kernel_f, kernel_t)
        stride: (stride_f, stride_t)
        causal: true, for causal operation
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1), causal: bool = True):
        super().__init__()

        freq_pad = (kernel_size[0]//2, kernel_size[0]//2) # center padding in frequency axis
        time_pad = (kernel_size[1]-1, 0) if causal else (kernel_size[1]//2, kernel_size[1]//2)

        self.in_conv_bn_relu = nn.Sequential(
            nn.ZeroPad2d(time_pad + freq_pad),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.fft_conv_bn_relu = nn.Sequential(
            nn.Conv2d(2*out_channels, 2*out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU())

        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor shape is [N, CH, C, T]
        """
        x = self.in_conv_bn_relu(x)

        ffted = torch.fft.rfft(x, dim=2)
        ffted_re = ffted.real
        ffted_im = ffted.imag
        ffted = torch.cat([ffted_re, ffted_im], dim=1)
        ffted = self.fft_conv_bn_relu(ffted)
        ffted_re, ffted_im = torch.chunk(ffted, 2, dim=1)
        ffted = torch.stack([ffted_re, ffted_im], dim=-1)
        ffted = torch.view_as_complex(ffted)
        ffted = torch.fft.irfft(ffted, dim=2)

        x = x + ffted
        x = self.out_conv(x)

        return x


class FFC(nn.Module):
    """
    Fast Fourier Convolution.

    Args:
        in_channels: input channel dimension
        out_channels: output channel dimension
        alpha: the ratioi between global and local channels, `global_channels=in_channels*alpha`
        kernel_size: (kernel_f, kernel_t)
        stride: (stride_f, stride_t)
        causal: true, for causal operation

    Reference:
        [1] FFC-SE: Fast Fourier Convolution for Speech Enhancement
    """
    def __init__(self, in_channels: int, out_channels: int, alpha: float = 0.3, kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1), causal: bool = True):
        super().__init__()

        self.fft_in_ch = int(in_channels*alpha)
        self.fft_out_ch = int(out_channels*alpha)
        self.local_in_ch = in_channels - self.fft_in_ch
        self.local_out_ch = out_channels - self.fft_out_ch

        freq_pad = (kernel_size[0]//2, kernel_size[0]//2) # center padding in frequency axis
        time_pad = (kernel_size[1]-1, 0) if causal else (kernel_size[1]//2, kernel_size[1]//2)

        self.global_spec_trans = SpectralTransform(self.fft_in_ch, self.fft_out_ch, kernel_size=kernel_size, stride=stride, causal=causal)
        self.global_conv = nn.Sequential(
            nn.ZeroPad2d(time_pad + freq_pad),
            nn.Conv2d(self.fft_in_ch, self.local_out_ch, kernel_size=kernel_size, stride=stride))

        self.local_global_conv = nn.Sequential(
            nn.ZeroPad2d(time_pad + freq_pad),
            nn.Conv2d(self.local_in_ch, self.fft_out_ch, kernel_size=kernel_size, stride=stride))

        self.local_local_conv = nn.Sequential(
            nn.ZeroPad2d(time_pad + freq_pad),
            nn.Conv2d(self.local_in_ch, self.local_out_ch, kernel_size=kernel_size, stride=stride))

        self.global_norm_relu = nn.Sequential(
            nn.BatchNorm2d(self.fft_out_ch),
            nn.ReLU())

        self.local_norm_relu = nn.Sequential(
            nn.BatchNorm2d(self.local_out_ch),
            nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor shape is [N, CH, C, T]
        """
        global_in = x[:, :self.fft_in_ch, :, :]
        local_in = x[:, self.fft_in_ch:, :, :]

        ffted = self.global_spec_trans(global_in)
        global_to_local = self.global_conv(global_in)

        local_to_global = self.local_global_conv(local_in)
        local_to_local = self.local_local_conv(local_in)

        global_out = ffted + local_to_global
        local_out = global_to_local + local_to_local

        global_out = self.global_norm_relu(global_out)
        local_out = self.local_norm_relu(local_out)

        return torch.cat([local_out, global_out], dim=1)