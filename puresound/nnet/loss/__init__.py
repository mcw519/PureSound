import torch
import torch.nn as nn

from .sdr import SDRLoss
from .spk import AAMsoftmax, SphereFace2
from .residual import ResidualReferenceLoss
from .asr_feature import ASRFeatureLoss
from .stft_loss import MultiResolutionSTFTLoss, OverSuppressionLoss, SpectralLoss
from .vad import BackgroundVADHeadBCELoss, VADActivityLoss, VADHeadBCELoss

__all__ = [
    "AAMsoftmax",
    "ASRFeatureLoss",
    "MultiResolutionSTFTLoss",
    "OverSuppressionLoss",
    "ResidualReferenceLoss",
    "SDRLoss",
    "SpectralLoss",
    "SphereFace2",
    "TimeDomainBasicLoss",
    "BackgroundVADHeadBCELoss",
    "VADActivityLoss",
    "VADHeadBCELoss",
]


class TimeDomainBasicLoss(nn.Module):
    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__()
        self.name = name.lower()
        self.reduction = reduction
        if self.name == "l1":
            self.loss_func = nn.functional.l1_loss
        elif self.name == "mse":
            self.loss_func = nn.functional.mse_loss
        else:
            raise NotImplementedError

    def forward(self, enh: torch.Tensor, ref: torch.Tensor):
        return self.loss_func(input=enh, target=ref, reduction=self.reduction)
