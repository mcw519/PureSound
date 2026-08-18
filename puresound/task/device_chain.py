"""What happens to a finished mixture on its way to the model.

Once the near and far sources are mixed and the noise is in, everything left is
*capture and transmission*: the resampler a codec ran through, the microphone's
tilt, a high-pass, an automatic gain, a VoIP codec, dropped packets. None of it
knows anything about speakers or rooms, which is why it lives here rather than
in the middle of ``NoiseSuppressionDataset.__getitem__``.

Two contracts this class exists to hold:

**Stage order is load-bearing, not cosmetic.** Every stage draws from the shared
RNG stream, so reordering two of them changes what a seeded recipe produces even
though each stage is individually unchanged. The order below is the order the
released checkpoints were trained with.

**A disabled stage must not touch the RNG stream at all.** Each guard is
``block is not None and block.used and torch.rand(1) < block.prob`` -- the
probability draw is *inside* the short circuit on purpose, so turning a stage
off leaves every later stage drawing exactly what it drew before. That is what
lets an old recipe regenerate bit-identically after a new knob is added.

Which signals a stage touches is the other half of the contract:

* SRC, IIR and HPF are *linear channel* effects -- they hit the mixture and the
  clean target with the same parameters, because the target is what the model is
  asked to recover *through* that channel.
* volume likewise, so the pair keeps its level relationship.
* codec and packet loss hit the mixture only. They are transmission damage; the
  target stays the undamaged reference the model is scored against.
"""

from __future__ import annotations

import random
from typing import NamedTuple, Optional

import torch

from puresound.audio.dsp import wav_resampling


class ChainResult(NamedTuple):
    """The pair after the chain. Both are needed: several stages move them
    together, and reading only one back is how they drift apart."""

    noisy: torch.Tensor
    target: torch.Tensor


class DeviceChain:
    """Applies the capture/transmission stages to one mixture-target pair.

    Built once per dataset from the already-validated config blocks; the sample
    rate is per call because it follows the item when a recipe has no
    ``target_sample_rate``.
    """

    def __init__(
        self,
        augmentor,
        *,
        src=None,
        ir_response=None,
        hpf=None,
        volume=None,
        codec=None,
        packet_loss=None,
    ):
        self.augmentor = augmentor
        self.src = src
        self.ir_response = ir_response
        self.hpf = hpf
        self.volume = volume
        self.codec = codec
        self.packet_loss = packet_loss

    @staticmethod
    def _fires(block, probability_draw: bool = True) -> bool:
        """Guard shared by every stage.

        The draw is inside the short circuit deliberately -- see the module
        docstring on why a disabled stage must consume no randomness.
        """
        if block is None or not block.used:
            return False
        return not probability_draw or bool(torch.rand(1) < block.prob)

    def apply(
        self, noisy: torch.Tensor, target: torch.Tensor, *, sample_rate: int
    ) -> ChainResult:
        noisy, target = self._sample_rate_conversion(noisy, target, sample_rate)
        noisy, target = self._second_order_iir(noisy, target)
        noisy, target = self._high_pass(noisy, target, sample_rate)
        noisy, target = self._volume(noisy, target, sample_rate)
        noisy = self._codec(noisy, sample_rate)
        noisy = self._packet_loss(noisy, sample_rate)
        return ChainResult(*self._overload_guard(noisy, target))

    # ------------------------------------------------------------------ #
    # Linear channel: the target follows the mixture through each of these
    # ------------------------------------------------------------------ #

    def _sample_rate_conversion(self, noisy, target, sample_rate):
        if not self._fires(self.src):
            return noisy, target
        src_target = random.choices(
            self.src.src_range, weights=self.src.prob_each
        )[0]
        # Two resamplers with audibly different filters; picking between them
        # widens the artifact distribution rather than baking in one vendor's.
        src_backend = "sox" if torch.rand(1) < 0.5 else "torchaudio"

        noisy, src_info = self.augmentor.apply_src_effect(
            wav=noisy, sr=sample_rate, src_sr=src_target, src_backend=src_backend
        )
        if src_backend == "sox":
            target, _ = wav_resampling(
                wav=target, origin_sr=sample_rate, target_sr=src_target, backend="sox"
            )
            target, _ = wav_resampling(
                wav=target, origin_sr=src_target, target_sr=sample_rate, backend="sox"
            )
        else:
            # Reuse the mixture's resampler state so the target gets the same
            # filter, not merely the same rates.
            target, *src_info = wav_resampling(
                wav=target,
                origin_sr=sample_rate,
                target_sr=src_target,
                backend="torchaudio",
                torch_backend_params=src_info[-1],
            )
            target, *src_info = wav_resampling(
                wav=target,
                origin_sr=src_target,
                target_sr=sample_rate,
                backend="torchaudio",
                torch_backend_params=src_info[-1],
            )
        return noisy, target

    def _second_order_iir(self, noisy, target):
        if not self._fires(self.ir_response):
            return noisy, target
        noisy, (a_coeffs, b_coeffs) = self.augmentor.apply_2nd_iir_response(wav=noisy)
        target, _ = self.augmentor.apply_2nd_iir_response(
            wav=target, a_coeffs=a_coeffs, b_coeffs=b_coeffs
        )
        return noisy, target

    def _high_pass(self, noisy, target, sample_rate):
        if not self._fires(self.hpf):
            return noisy, target
        cutoff = random.choices(self.hpf.cutoff, weights=self.hpf.prob_each)[0]
        q_factor = torch.FloatTensor(1).normal_(mean=0.707, std=0.1).clip(0.3, 1.3)
        noisy, _ = self.augmentor.apply_hpf(
            wav=noisy, sr=sample_rate, cutoff_freq=cutoff, q_factor=q_factor
        )
        target, _ = self.augmentor.apply_hpf(
            wav=target, sr=sample_rate, cutoff_freq=cutoff, q_factor=q_factor
        )
        return noisy, target

    def _volume(self, noisy, target, sample_rate):
        if not self._fires(self.volume):
            return noisy, target
        if torch.rand(1) < self.volume.clipping_prob:
            min_q = torch.FloatTensor(1).uniform_(
                self.volume.clipping_range.min[0], self.volume.clipping_range.min[1]
            )
            max_q = torch.FloatTensor(1).uniform_(
                self.volume.clipping_range.max[0], self.volume.clipping_range.max[1]
            )
            noisy, (min_quantile, max_quantile) = (
                self.augmentor.apply_clipping_distortion(
                    wav=noisy, min_quantile=min_q, max_quantile=max_q
                )
            )
            # The target is clipped at the mixture's REALIZED quantiles, not at
            # its own: the same analogue overload hit both.
            target, _ = self.augmentor.apply_clipping_distortion(
                wav=target, min_quantile=min_quantile, max_quantile=max_quantile
            )
        else:
            gain = (
                torch.FloatTensor(1)
                .uniform_(
                    self.volume.perturbed_range[0], self.volume.perturbed_range[1]
                )
                .item()
            )
            noisy, vol_ratio = self.augmentor.sox_volume_perturbed(
                wav=noisy, vol_ratio=gain, sr=sample_rate
            )
            target, _ = self.augmentor.sox_volume_perturbed(
                wav=target, vol_ratio=vol_ratio, sr=sample_rate
            )
        return noisy, target

    # ------------------------------------------------------------------ #
    # Transmission damage: mixture only, the target stays the clean reference
    # ------------------------------------------------------------------ #

    def _codec(self, noisy, sample_rate):
        if not self._fires(self.codec):
            return noisy
        codecs = self.codec.codecs
        prob_each = self.codec.prob_each
        if prob_each:
            codec_name = random.choices(codecs, weights=prob_each, k=1)[0]
        else:
            codec_name = random.choice(codecs)
        bitrate_range = self.codec.bitrate_range.get(codec_name)
        bit_rate = (
            random.randint(int(bitrate_range[0]), int(bitrate_range[1]))
            if bitrate_range
            else None
        )
        noisy, _ = self.augmentor.apply_codec(
            wav=noisy, sr=sample_rate, codec_name=codec_name, bit_rate=bit_rate
        )
        return noisy

    def _packet_loss(self, noisy, sample_rate):
        if not self._fires(self.packet_loss):
            return noisy
        packet_ms = random.choice(self.packet_loss.packet_ms_choices)
        lo, hi = self.packet_loss.loss_rate_range
        loss_rate = random.uniform(float(lo), float(hi))
        noisy, _ = self.augmentor.apply_packet_loss(
            wav=noisy,
            sr=sample_rate,
            packet_ms=int(packet_ms),
            loss_rate=loss_rate,
        )
        return noisy

    # ------------------------------------------------------------------ #

    @staticmethod
    def _overload_guard(noisy, target):
        """Rescale the pair together if the chain pushed it past full scale.

        The dataset clips once before the noise / volume / IIR stages, any of
        which can push it back over, and the model's own output is clamped to
        [-1, 1] -- so a target above full scale is one the model cannot reach.
        Both are divided by the same peak so the level relationship survives.
        Consumes no randomness.
        """
        peak = float(torch.maximum(noisy.abs().amax(), target.abs().amax()))
        if peak > 1.0:
            noisy = noisy / peak
            target = target / peak
        return noisy, target


def device_chain_from_blocks(augmentor, dataset) -> Optional[DeviceChain]:
    """Build the chain from a dataset's already-validated augmentation blocks."""
    return DeviceChain(
        augmentor,
        src=dataset.augmentation_src_args,
        ir_response=dataset.augmentation_ir_response_args,
        hpf=dataset.augmentation_hpf_args,
        volume=dataset.augmentation_volume_args,
        codec=getattr(dataset, "augmentation_codec_args", None),
        packet_loss=getattr(dataset, "augmentation_packet_loss_args", None),
    )
