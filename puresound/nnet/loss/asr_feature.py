"""ASR-aware perceptual loss.

Signal losses (SI-SDR / MR-STFT) reward suppressing interference but do NOT
penalise destroying intelligibility, so a model tuned on them alone can
over-suppress real speech and raise WER. This loss matches the
enhanced output's features to the clean target's features inside a FROZEN,
self-supervised speech encoder (torchaudio wav2vec2 / HuBERT, 16 kHz,
differentiable). Those encoders are trained on large amounts of REAL speech, so
their features encode phonetic content robustly across domains -- the bet is
that matching them transfers to the real-domain WER better than waveform SI-SDR.

WER itself is non-differentiable; this optimises a differentiable ASR proxy.
The encoder is frozen and kept OUT of the module registry (list-wrapped) so it
is not saved into checkpoints nor synced by DDP; it is moved to the input device
lazily and run in fp32 (autocast disabled) for stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRFeatureLoss(nn.Module):
    """Feature-matching loss through a frozen SSL/ASR encoder.

    Args:
        bundle: a ``torchaudio.pipelines`` bundle name, e.g. ``HUBERT_BASE``,
            ``WAV2VEC2_BASE``, ``WAV2VEC2_ASR_BASE_960H`` (all 16 kHz).
        layers: transformer-layer output indices to match (mid layers carry the
            most phonetic information). Distances are averaged over these.
        loss: ``l1`` | ``mse`` | ``cosine`` (1 - cosine similarity).
    """

    def __init__(
        self,
        bundle: str = "HUBERT_BASE",
        layers=(6, 9),
        loss: str = "l1",
    ):
        super().__init__()
        import torchaudio

        b = getattr(torchaudio.pipelines, bundle)
        self.sample_rate = int(b.sample_rate)
        model = b.get_model().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # List-wrap so the encoder is NOT a registered submodule: keeps it out of
        # state_dict / DDP and out of the optimizer. Moved to device lazily.
        self._ssl = [model]
        self.bundle = bundle
        self.layers = tuple(int(x) for x in layers)
        self.num_layers = max(self.layers) + 1
        self.loss = loss.lower()
        if self.loss not in ("l1", "mse", "cosine"):
            raise NotImplementedError(f"Unsupported ASR feature loss: {loss}")

    @staticmethod
    def _to_bt(x: torch.Tensor) -> torch.Tensor:
        """Coerce to [B, T]."""
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _features(self, wav: torch.Tensor):
        feats, _ = self._ssl[0].extract_features(wav, num_layers=self.num_layers)
        return [feats[i] for i in self.layers]

    def forward(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        enhanced = self._to_bt(enhanced)
        target = self._to_bt(target)
        length = min(enhanced.shape[-1], target.shape[-1])
        enhanced = enhanced[..., :length]
        target = target[..., :length]

        ssl = self._ssl[0]
        if next(ssl.parameters()).device != enhanced.device:
            ssl.to(enhanced.device)

        # Frozen pretrained encoder: run in fp32 with autocast disabled (stable),
        # grad still flows back to `enhanced`; target is a fixed reference.
        with torch.autocast(device_type=enhanced.device.type, enabled=False):
            enh_f = self._features(enhanced.float())
            with torch.no_grad():
                tgt_f = self._features(target.float())

        total = enhanced.new_zeros(())
        for ef, tf in zip(enh_f, tgt_f):
            tf = tf.to(ef.dtype)
            if self.loss == "l1":
                total = total + F.l1_loss(ef, tf)
            elif self.loss == "mse":
                total = total + F.mse_loss(ef, tf)
            else:  # cosine
                total = total + (1.0 - F.cosine_similarity(ef, tf, dim=-1)).mean()
        return total / len(enh_f)
