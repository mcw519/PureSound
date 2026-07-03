"""Explicit far-parent supervision (DISTANCE_PARENT proposal P1).

The backbone grows a second ("far") decoder that predicts the far/background
speech. The system runs that decoder output through the same iSTFT pipeline as
the near output and stashes the waveform as ``last_far_wav``; ``compute_loss``
routes it here. Deployment still emits only the near output -- these losses are
training-only, a more direct version of ResidualReferenceLoss's implicit
``noisy - near_hat`` residual.
"""
import torch
import torch.nn as nn

from .sdr import SDRLoss


def _match_rank(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.dim() == ref.dim() + 1 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.dim() + 1 == ref.dim() and ref.shape[1] == 1:
        x = x.unsqueeze(1)
    return x


class FarReconstructionLoss(nn.Module):
    """Supervise the far decoder output ``far_hat`` against the far speech.

    ``far_target`` (``batch[target_key]``) is the summed full-RIR interferer
    speech. Rows with no interferer (silent far, e.g. N-only) are routed through
    inactive (iSDR) handling so SDR does not explode on a zero reference.
    Returns ``None`` when the far decoder is disabled so ``compute_loss`` skips it.
    """

    uses_far_output = True

    def __init__(
        self,
        target_key: str = "far_target",
        loss: str = "sdr",
        sdr_args: dict | None = None,
    ):
        super().__init__()
        self.target_key = target_key
        self.loss = loss.lower()
        if self.loss == "sdr":
            self.loss_func = SDRLoss(**(sdr_args or {}))
        elif self.loss not in ("l1", "mse"):
            raise NotImplementedError(f"Unsupported far loss: {loss}")

    def forward(self, far_wav: torch.Tensor | None, batch: dict):
        if far_wav is None:
            return None  # far decoder disabled / eval -> compute_loss skips
        ref = batch.get(self.target_key)
        if ref is None:
            return far_wav.sum() * 0.0
        ref = ref.to(device=far_wav.device, dtype=far_wav.dtype)
        far_wav = _match_rank(far_wav, ref)
        length = min(far_wav.shape[-1], ref.shape[-1])
        far_wav = far_wav[..., :length]
        ref = ref[..., :length]
        if self.loss == "sdr":
            inactive = ref.abs().amax(dim=-1) == 0
            return self.loss_func(far_wav, ref, inactive_labels=inactive)
        if self.loss == "l1":
            return nn.functional.l1_loss(far_wav, ref)
        return nn.functional.mse_loss(far_wav, ref)


class MixtureConsistencyLoss(nn.Module):
    """``near_hat + far_hat`` should reconstruct the (early-near + full-far) speech.

    Target is ``clean_speech + far_target`` -- NOT the raw noisy mixture: noisy
    carries the near talker at FULL reverb, so anchoring ``near_hat`` (an EARLY,
    de-reverbed target) to noisy would fight the de-reverb goal. Low-weight
    coupling regulariser that pushes the two decoders to PARTITION the speech
    rather than double-count it. Returns ``None`` when far is disabled.
    """

    uses_mixture_consistency = True

    def __init__(
        self,
        target_key: str = "far_target",
        clean_key: str = "clean_speech",
        loss: str = "l1",
    ):
        super().__init__()
        self.target_key = target_key
        self.clean_key = clean_key
        self.loss = loss.lower()

    def forward(self, near_wav: torch.Tensor, far_wav: torch.Tensor | None, batch: dict):
        if far_wav is None:
            return None
        clean = batch.get(self.clean_key)
        far_ref = batch.get(self.target_key)
        if clean is None or far_ref is None:
            return far_wav.sum() * 0.0
        clean = clean.to(device=near_wav.device, dtype=near_wav.dtype)
        far_ref = far_ref.to(device=near_wav.device, dtype=near_wav.dtype)
        near_wav = _match_rank(near_wav, clean)
        far_wav = _match_rank(far_wav, far_ref)
        length = min(
            near_wav.shape[-1], far_wav.shape[-1], clean.shape[-1], far_ref.shape[-1]
        )
        recon = near_wav[..., :length] + far_wav[..., :length]
        target = clean[..., :length] + far_ref[..., :length]
        if self.loss == "mse":
            return nn.functional.mse_loss(recon, target)
        return nn.functional.l1_loss(recon, target)
