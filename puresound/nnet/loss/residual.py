import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdr import SDRLoss


class ResidualReferenceLoss(nn.Module):
    """Supervise the residual left after subtracting the enhanced target.

    The voice-isolation dataloader already emits ``consistency_noise`` as
    ``noisy_speech - clean_speech``. This loss keeps the deployed model
    single-output while giving training an auxiliary "where did the suppressed
    energy go?" signal:

        residual_pred = noisy_speech - enhanced
        residual_ref = batch[reference_key]
    """

    uses_batch = True

    def __init__(
        self,
        reference_key: str = "consistency_noise",
        loss: str = "l1",
        target_present_only: bool = False,
        sdr_args: dict | None = None,
    ):
        super().__init__()
        self.reference_key = reference_key
        self.loss = loss.lower()
        self.target_present_only = bool(target_present_only)
        if self.loss == "sdr":
            self.loss_func = SDRLoss(**(sdr_args or {}))
        elif self.loss not in ("l1", "mse"):
            raise NotImplementedError(f"Unsupported residual loss: {loss}")

    @staticmethod
    def _match_rank(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.dim() == ref.dim() + 1 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() + 1 == ref.dim() and ref.shape[1] == 1:
            x = x.unsqueeze(1)
        return x

    def forward(
        self,
        enhanced: torch.Tensor,
        target: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        if self.reference_key not in batch or "noisy_speech" not in batch:
            raise KeyError(
                f"ResidualReferenceLoss requires batch['noisy_speech'] and "
                f"batch['{self.reference_key}']"
            )

        noisy = batch["noisy_speech"].to(device=enhanced.device, dtype=enhanced.dtype)
        ref = batch[self.reference_key].to(device=enhanced.device, dtype=enhanced.dtype)
        noisy = self._match_rank(noisy, enhanced)
        ref = self._match_rank(ref, enhanced)

        length = min(enhanced.shape[-1], noisy.shape[-1], ref.shape[-1])
        enhanced = enhanced[..., :length]
        noisy = noisy[..., :length]
        ref = ref[..., :length]

        if self.target_present_only and "target_present" in batch:
            keep = batch["target_present"].to(enhanced.device).view(-1) > 0.5
            if keep.numel() == enhanced.shape[0] and keep.any():
                enhanced = enhanced[keep]
                noisy = noisy[keep]
                ref = ref[keep]
            elif keep.numel() == enhanced.shape[0]:
                return enhanced.sum() * 0.0

        residual = noisy - enhanced
        if self.loss == "l1":
            return F.l1_loss(residual, ref)
        if self.loss == "mse":
            return F.mse_loss(residual, ref)
        return self.loss_func(residual, ref)
