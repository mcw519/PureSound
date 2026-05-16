import torch
import torch.nn as nn

from .sdr import SDRLoss
from .spk import AAMsoftmax, SphereFace2
from .stft_loss import MultiResolutionSTFTLoss, OverSuppressionLoss, SpectralLoss
from .vad import VADActivityLoss

__all__ = [
    "AAMsoftmax",
    "MultiResolutionSTFTLoss",
    "OverSuppressionLoss",
    "SDRLoss",
    "SpectralLoss",
    "SphereFace2",
    "TimeDomainBasicLoss",
    "VADActivityLoss",
]


class TimeDomainBasicLoss(nn.Module):
    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__()
        self.name = name.lower()
        self.reduction = reduction
        if self.name == "l1":
            self.loss_func = nn.functional.l1_loss
        elif self.name in ["l1", "mse"]:
            self.loss_func = nn.functional.mse_loss
        else:
            raise NotImplementedError

    def forward(self, enh: torch.Tensor, ref: torch.Tensor):
        return self.loss_func(input=enh, target=ref, reduction=self.reduction)
