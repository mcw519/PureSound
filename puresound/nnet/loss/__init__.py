import torch
import torch.nn as nn

from .asr_feature import ASRFeatureLoss
from .dist import DistHeadRegressionLoss
from .residual import ResidualReferenceLoss
from .sdr import SDRLoss
from .spk import AAMsoftmax, GE2ELoss, SphereFace2, TripletLoss
from .stft_loss import MultiResolutionSTFTLoss, OverSuppressionLoss, SpectralLoss
from .vad import BackgroundVADHeadBCELoss, F1_loss, VADActivityLoss, VADHeadBCELoss

# Every loss the config loader can name via ``loss_func[].type`` (resolved with
# ``getattr(loss, type)`` in recipes.py and the egs mains). Keep in sync with the
# imports above so each loss in the library stays reachable from a recipe config.
__all__ = [
    "AAMsoftmax",
    "ASRFeatureLoss",
    "BackgroundVADHeadBCELoss",
    "DistHeadRegressionLoss",
    "F1_loss",
    "GE2ELoss",
    "MultiResolutionSTFTLoss",
    "OverSuppressionLoss",
    "ResidualReferenceLoss",
    "SDRLoss",
    "SpectralLoss",
    "SphereFace2",
    "TimeDomainBasicLoss",
    "TripletLoss",
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
