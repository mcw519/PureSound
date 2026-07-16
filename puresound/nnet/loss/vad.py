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
            # Anchor to 0 dBFS so activity_threshold_db is an absolute dBFS
            # threshold, invariant to volume / clipping perturbations on the
            # reference signal. Model output is clamped to [-1, 1] so enh_db
            # lies in (-inf, 0] dBFS.
            reference_power = enh_power.new_ones(enh_power.shape[0], 1)

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


class VADHeadBCELoss(nn.Module):
    """BCE on a backbone VAD-head's frame-level logits against a VAD target.

    Unlike VADActivityLoss (which derives activity from the enhanced waveform
    energy), this supervises an explicit per-frame logit head exposed by the
    backbone. EncDecMaskBase routes ``backbone.last_vad_logits`` here via the
    ``uses_vad_logits`` dispatch flag. ``false_positive_weight`` upweights
    silence frames so background speech is less likely to trigger the head.
    ``balance_per_batch`` makes positive and negative frames contribute equal
    total BCE mass, preventing an imbalanced batch from rewarding an all-speech
    or all-silence constant. Calibrate the deployment threshold separately.
    """

    uses_vad_logits = True

    def __init__(
        self,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
        balance_per_batch: bool = False,
    ):
        super().__init__()
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)
        self.balance_per_batch = bool(balance_per_batch)

    def forward(
        self,
        vad_logits: torch.Tensor | None,
        vad_target: torch.Tensor | None,
    ) -> torch.Tensor:
        if vad_logits is None:
            raise ValueError(
                "VADHeadBCELoss requires the backbone to expose `last_vad_logits`; "
                "enable a vad_head in the backbone config."
            )
        if vad_target is None:
            raise ValueError(
                "VADHeadBCELoss requires `vad_target`; enable `vad_label` in the "
                "config to generate VAD labels."
            )

        if vad_logits.dim() == 1:
            vad_logits = vad_logits.unsqueeze(0)
        target = vad_target.to(device=vad_logits.device, dtype=vad_logits.dtype)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        # align frame counts (STFT framing vs label framing can differ by 1)
        n = min(vad_logits.shape[-1], target.shape[-1])
        vad_logits = vad_logits[..., :n]
        target = target[..., :n]

        weights = torch.where(
            target > 0.5,
            torch.full_like(target, self.false_negative_weight),
            torch.full_like(target, self.false_positive_weight),
        )
        if self.balance_per_batch:
            positive = target > 0.5
            n_positive = positive.sum()
            n_negative = positive.numel() - n_positive
            # An all-active or all-silent batch has no class trade-off to
            # balance; retain its configured static weight in that case.
            if n_positive > 0 and n_negative > 0:
                total = target.new_tensor(float(positive.numel()))
                pos_scale = total / (2.0 * n_positive.to(target.dtype))
                neg_scale = total / (2.0 * n_negative.to(target.dtype))
                weights = weights * torch.where(
                    positive,
                    torch.full_like(target, pos_scale),
                    torch.full_like(target, neg_scale),
                )
        return F.binary_cross_entropy_with_logits(vad_logits, target, weight=weights)


class BackgroundVADHeadBCELoss(VADHeadBCELoss):
    """BCE on a background-speech activity head.

    The foreground VAD head learns "should ASR listen now"; this companion head
    learns "is non-target speech present now". It gives the bottleneck an
    explicit representation for background talkers without making background
    speech part of the enhanced output.
    """

    uses_vad_logits = False
    uses_background_vad_logits = True
