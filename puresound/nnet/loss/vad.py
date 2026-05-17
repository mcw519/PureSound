import torch
import torch.nn as nn
import torch.nn.functional as F


class VADActivityLoss(nn.Module):
    """
    Differentiable VAD-style activity loss.

    When an external VAD target is provided, that target is used as the label.
    Otherwise the target activity falls back to clean foreground energy.
    """

    uses_vad_target = True

    def __init__(
        self,
        frame_length: int = 400,
        hop_length: int = 160,
        activity_threshold_db: float = -40.0,
        logit_scale: float = 0.25,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
        require_vad_target: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.activity_threshold_db = float(activity_threshold_db)
        self.logit_scale = float(logit_scale)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)
        self.require_vad_target = bool(require_vad_target)
        self.eps = float(eps)

    def _as_batch_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3:
            wav = wav.mean(dim=1)
        if wav.dim() != 2:
            raise ValueError(
                f"Expected waveform shape [B, T] or [B, C, T], got {wav.shape}"
            )
        return wav

    def _frame_power(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[-1] < self.frame_length:
            wav = F.pad(wav, (0, self.frame_length - wav.shape[-1]))
        frames = wav.unfold(-1, self.frame_length, self.hop_length)
        return frames.square().mean(dim=-1)

    def _power_to_db(
        self, power: torch.Tensor, reference_power: torch.Tensor
    ) -> torch.Tensor:
        reference_power = reference_power.clamp_min(self.eps)
        return 10.0 * torch.log10(power.clamp_min(self.eps) / reference_power)

    def forward(
        self,
        enh: torch.Tensor,
        ref: torch.Tensor,
        vad_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enh = self._as_batch_waveform(enh)
        ref = self._as_batch_waveform(ref)
        min_len = min(enh.shape[-1], ref.shape[-1])
        enh = enh[..., :min_len]
        ref = ref[..., :min_len]

        enh_power = self._frame_power(enh)
        if vad_target is None:
            if self.require_vad_target:
                raise ValueError(
                    "VADActivityLoss requires `vad_target`, but the batch did not "
                    "provide one. Enable `vad_label` in the config to generate VAD "
                    "labels, for example with `backend: silero`."
                )
            ref_power = self._frame_power(ref)
            reference_power = ref_power.amax(dim=-1, keepdim=True)
            silent_utterance = reference_power <= self.eps
            reference_power = torch.where(
                silent_utterance, torch.ones_like(reference_power), reference_power
            )
            ref_db = self._power_to_db(ref_power, reference_power)
            target_activity = (ref_db > self.activity_threshold_db).float()
            target_activity = torch.where(
                silent_utterance.expand_as(target_activity),
                torch.zeros_like(target_activity),
                target_activity,
            )
        else:
            target_activity = vad_target.to(device=enh.device, dtype=enh.dtype)
            if target_activity.dim() == 1:
                target_activity = target_activity.unsqueeze(0)
            n_frames = enh_power.shape[-1]
            if target_activity.shape[-1] > n_frames:
                target_activity = target_activity[..., :n_frames]
            elif target_activity.shape[-1] < n_frames:
                target_activity = F.pad(
                    target_activity, (0, n_frames - target_activity.shape[-1])
                )
            ref_power = self._frame_power(ref)
            reference_power = ref_power.amax(dim=-1, keepdim=True).clamp_min(self.eps)

        enh_db = self._power_to_db(enh_power, reference_power)
        logits = (enh_db - self.activity_threshold_db) * self.logit_scale
        weights = torch.where(
            target_activity > 0.5,
            torch.full_like(target_activity, self.false_negative_weight),
            torch.full_like(target_activity, self.false_positive_weight),
        )
        return F.binary_cross_entropy_with_logits(
            logits, target_activity, weight=weights
        )
