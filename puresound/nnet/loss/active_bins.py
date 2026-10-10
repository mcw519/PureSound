"""Log-magnitude error counted only where the clean target has speech.

The usual enhancement losses barely ask for the spectral fine structure
*inside* speech. MultiResolutionSTFTLoss and ASRFeatureLoss put almost all of
their gradient on noise and silence bins; SDRLoss, the one that mostly lands on
speech, is energy-weighted and cannot see per-bin structure. An output can
therefore improve on all of them while mid-level harmonic peaks inside speech
are shaved. The missing piece is the training signal, not model capacity.

This term asks for that structure directly: per-bin log-magnitude L1, averaged
over bins where the clean target sits within ``dynamic_range_db`` of the
utterance's peak bin. Silence and noise-only regions contribute nothing, so
their (large, noisy) log errors cannot outvote the speech interior.
"""

from __future__ import annotations

import torch


class ActiveBinLogMagLoss(torch.nn.Module):
    """L1 on log-magnitude, over the clean target's active bins only.

    Args:
        n_fft, hop: analysis grid. Defaults match the DPCRN recipes' encoder so
            "a bin" here is a bin the mask acts on.
        dynamic_range_db: a bin counts as active when the clean target there is
            within this many dB of that utterance's loudest bin. 50 spans loud
            vowels down to word-final fricatives.
        under_weight: multiplier on the error where the output is *below* the
            target. 1.0 is symmetric. The damage this term targets is undershoot
            (shaved peaks), so a value above 1 leans the term against it.
        silence_floor_db: a bin below this absolute level is never active, whatever
            the utterance peak. Without it a fully silent target row (the
            target-absent training path) has peak == floor, every bin passes the
            relative test, and the row costs |enh - (-100 dB)| ~ 100 -- a term
            two orders louder than anything else in the objective, pushing those
            rows to digital zero. -80 dB sits far below any speech bin of a
            normally levelled row and well above the log floor (-100 dB).
    """

    required_inputs = ("enhanced", "target")

    def __init__(
        self,
        n_fft: int = 512,
        hop: int = 160,
        dynamic_range_db: float = 50.0,
        under_weight: float = 1.0,
        silence_floor_db: float = -80.0,
    ):
        super().__init__()
        if dynamic_range_db <= 0.0:
            raise ValueError(f"dynamic_range_db must be positive, got {dynamic_range_db}")
        if under_weight <= 0.0:
            raise ValueError(f"under_weight must be positive, got {under_weight}")
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.dynamic_range_db = float(dynamic_range_db)
        self.under_weight = float(under_weight)
        self.silence_floor_db = float(silence_floor_db)
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def _log_mag(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wav, self.n_fft, self.hop, self.n_fft, self.window.to(wav.device),
            return_complex=True, center=True,
        )
        return 10.0 * torch.log10(spec.real**2 + spec.imag**2 + 1e-10)

    def forward(self, enh: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if enh.dim() == 3 and enh.shape[1] == 1:
            enh = enh.squeeze(1)
        if ref.dim() == 3 and ref.shape[1] == 1:
            ref = ref.squeeze(1)
        length = min(enh.shape[-1], ref.shape[-1])
        enh, ref = enh[..., :length], ref[..., :length]
        if length < self.n_fft:
            return enh.sum() * 0.0
        ref_db = self._log_mag(ref)
        enh_db = self._log_mag(enh)
        peak = ref_db.amax(dim=(-2, -1), keepdim=True)
        active = (
            (ref_db >= peak - self.dynamic_range_db) & (ref_db > self.silence_floor_db)
        ).float()
        diff = enh_db - ref_db
        weight = torch.where(diff < 0, torch.full_like(diff, self.under_weight), torch.ones_like(diff))
        per_row = (diff.abs() * weight * active).sum(dim=(-2, -1)) / active.sum(dim=(-2, -1)).clamp_min(1.0)
        has_speech = (active.sum(dim=(-2, -1)) > 0).float()
        return (per_row * has_speech).sum() / has_speech.sum().clamp_min(1.0)
