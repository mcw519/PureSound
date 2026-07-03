"""Speaker-consistency loss for enhancement training.

Runs a frozen pretrained puresound speaker model (e.g. PS-spk-v1-1) over the
enhancement model's output and the clean target, then penalises cosine
distance between the two embeddings. The intuition: the enhancement model
should preserve the foreground speaker's identity, not just match the
waveform shape. Useful for breaking the "loudest = target" shortcut and for
nudging the model to lock onto a specific voice in BSS-style mixtures.

Per-sample silence masking: target-absent training rows (clean_speech ≈ 0)
produce meaningless reference embeddings, so they are dropped from the loss
rather than dominating it with noise-vs-anything cosine scores.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: str) -> str:
    """Resolve a config-supplied path. Absolute paths and CWD-relative paths
    that already exist are returned untouched; otherwise fall back to the
    same path interpreted relative to the puresound repo root so configs can
    keep using ``egs/...`` regardless of the script's CWD."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = _REPO_ROOT / p
    if candidate.exists():
        return str(candidate)
    return str(p)


class SpeakerConsistencyLoss(nn.Module):
    """1 − cos(emb(enh), emb(ref)) with a frozen speaker encoder.

    Args:
        encoder_config_path: YAML config describing the speaker model
            (encoder / features / backbone). Used to rebuild the architecture
            via ``puresound.recipes.init_siso_model``.
        checkpoint_path: state_dict for that model. ``loss_func_list`` keys
            are skipped (AAMsoftmax head is irrelevant for inference).
        target_sample_rate: rate at which the encoder was trained. Only used
            as documentation today — the loss assumes ``enh``/``ref`` already
            share this rate (DPARN 16k recipe is the intended caller).
        silence_threshold: any row whose reference max-abs falls below this
            is treated as target-absent and excluded from the mean. Set to
            ``0`` to disable masking.
    """

    def __init__(
        self,
        encoder_config_path: str,
        checkpoint_path: str,
        target_sample_rate: int = 16000,
        silence_threshold: float = 1e-4,
    ):
        super().__init__()
        self.encoder_config_path = _resolve_repo_path(str(encoder_config_path))
        self.checkpoint_path = _resolve_repo_path(str(checkpoint_path))
        self.target_sample_rate = int(target_sample_rate)
        self.silence_threshold = float(silence_threshold)
        # Lazy-load: building the encoder pulls torchaudio / lightning and
        # touches the filesystem, so defer until the first training step.
        # Stored via object.__setattr__ to bypass nn.Module auto-registration
        # so the frozen ECAPA params never enter the optimizer's loss-param
        # group (see EncDecMaskBase.get_total_param_groups).
        object.__setattr__(self, "_encoder_module", None)
        object.__setattr__(self, "_encoder_device", None)

    def _load_encoder(self) -> None:
        from puresound.recipes import init_siso_model
        from puresound.utils import load_hparam

        cfg = load_hparam(file_path=self.encoder_config_path)
        encoder = init_siso_model(cfg["model"])
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        if hasattr(encoder, "reload_checkpoint"):
            encoder.reload_checkpoint(state, load_loss_func=False)
        else:
            encoder.load_state_dict(state, strict=False)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_encoder_module", encoder)

    def _ensure_on_device(self, device: torch.device) -> None:
        if self._encoder_device != device:
            self._encoder_module.to(device)
            object.__setattr__(self, "_encoder_device", device)

    def _embed(self, wav: torch.Tensor) -> torch.Tensor:
        # Speaker model's EncPredClassBase.forward expects [B, T]; collapse
        # the channel dim if present (DPARN emits [B, 1, T]).
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return self._encoder_module(wav)

    def forward(self, enh: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self._encoder_module is None:
            self._load_encoder()
        self._ensure_on_device(enh.device)

        # Per-row silence mask. Reshape to [B, -1] so we can amax over time
        # regardless of whether tensors are [B, T] or [B, 1, T].
        ref_flat = ref.reshape(ref.shape[0], -1)
        valid = ref_flat.abs().amax(dim=-1) > self.silence_threshold
        if not bool(valid.any()):
            return enh.new_zeros(())

        emb_enh = self._embed(enh)
        with torch.no_grad():
            emb_ref = self._embed(ref)

        # Safety net: if anything upstream produced a non-finite embedding,
        # return zero rather than poisoning the whole batch's gradient. The
        # underlying MelBank sqrt(0) gradient bug is fixed; this guard catches
        # future regressions (e.g. enhanced output diverging to Inf).
        if not torch.isfinite(emb_enh).all() or not torch.isfinite(emb_ref).all():
            return enh.new_zeros(())

        cos = F.cosine_similarity(emb_enh, emb_ref, dim=-1)
        per_sample = (1.0 - cos) * valid.to(cos.dtype)
        denom = valid.to(cos.dtype).sum().clamp_min(1.0)
        return per_sample.sum() / denom
