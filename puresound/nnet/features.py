from typing import Dict, Optional

import torch
import torch.nn as nn

from .lobe.stft import mel_filterbank
from .lobe.trivial import LambdaLayer, Magnitude, SpecAugment


class MelBank(nn.Module):
    """
    This is a layer for converting stft-complex form to mel filter banks.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        n_banks: int = 80,
        utt_norm: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        self.uttnorm = utt_norm
        self.trainable = trainable
        mel_fb = mel_filterbank(sr, n_fft, n_banks)  # [n_mels, n_fft//2 +1]
        mel_fb = mel_fb.permute(1, 0)  # [n_fft//2 +1, n_mels]

        if trainable:
            mel_fb = nn.Parameter(mel_fb, requires_grad=True)
            self.register_parameter("filterbank", mel_fb)
        else:
            self.register_buffer("filterbank", mel_fb)

    def forward(self, x: torch.Tensor):
        """
        Convert a batch of complex spectrum to mel-spectrograms.

        Args
            input tensor x has shape [N, C, T, 2]
        """
        spec_imag = x[..., 0]
        spec_real = x[..., 1]
        spec = spec_real.pow(2) + spec_imag.pow(2)
        mag = torch.sqrt(spec + 1e-8) if self.trainable else torch.sqrt(spec)
        mag = mag.permute(0, 2, 1)  # [N, T, C]
        melspec = torch.matmul(mag, self.filterbank)

        if self.uttnorm:
            melspec = melspec - melspec.mean(dim=1, keepdim=True)

        return melspec.permute(0, 2, 1)  # [N, C, T]


class WeightedSum(nn.Module):
    def __init__(self, n_samples: int, trainable: bool = True):
        super().__init__()

        w = torch.ones(n_samples) / n_samples
        w = torch.nn.Parameter(w, requires_grad=True)
        if trainable:
            self.register_parameter("w", w)
        else:
            self.register_buffer("w", w)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should has shape [..., n_samples]
        """
        out = x * self.w
        out = out.sum(dim=-1)

        return out


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        feats_type: str = "complex",
        drop_stft_first_bin: bool = True,
        include_specaug: bool = False,
        specaug_args: Optional[Dict] = None,
        trainable: bool = False,
    ):
        super().__init__()

        self.feats_type = feats_type.lower()
        self.drop_stft_first_bin = drop_stft_first_bin
        self.include_specaug = include_specaug
        assert self.feats_type in [
            "free",
            "complex",
            "magnitude",
            "log1p",
            "fbank80_16k",
            "fbank128_16k",
        ]

        if self.feats_type == "complex":
            if drop_stft_first_bin:
                self.transform = LambdaLayer(
                    lambda x: x[:, 1:, ...].permute(0, 3, 1, 2)
                )
            else:
                self.transform = LambdaLayer(lambda x: x.permute(0, 3, 1, 2))
        elif self.feats_type == "magnitude":
            self.transform = Magnitude(drop_first=drop_stft_first_bin)
        elif self.feats_type == "log1p":
            self.transform = Magnitude(drop_first=drop_stft_first_bin, log1p=True)
        elif self.feats_type == "fbank80_16k":
            self.transform = MelBank(
                sr=16000, n_fft=512, n_banks=80, trainable=trainable
            )
        elif self.feats_type == "fbank128_16k":
            self.transform = MelBank(
                sr=16000, n_fft=512, n_banks=128, trainable=trainable
            )
        elif self.feats_type == "free":
            self.transform = nn.Identity

        if include_specaug:
            self.specaug = SpecAugment(**specaug_args)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: All input shape from encoder should be [N, C, T, 2] or [N, C, T]

        Returns:
            return features should has shape [N, CH, C, T]
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        feats_for_enhanced = self.transform(x)
        if self.include_specaug:
            feats = self.specaug(feats_for_enhanced)
        else:
            feats = feats_for_enhanced.clone()

        return feats, feats_for_enhanced

    def back_forward(self, x: torch.Tensor):
        # [N, CH, C, T]
        if self.feats_type in ["complex", "magnitude", "log1p"]:
            if self.drop_stft_first_bin:
                batch, channels, _, nframes = x.shape
                padding = torch.zeros(
                    (batch, channels, 1, nframes), device=x.device, dtype=x.dtype
                )
                x = torch.cat([padding, x], dim=2)

        return x
