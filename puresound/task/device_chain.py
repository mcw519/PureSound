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

Which signals a stage touches is the other half of the contract, and it follows
from where the stage sits relative to the converter:

* SRC, IIR, HPF and volume are the *analogue* path -- a transducer response, a
  rumble filter, a preamp gain. All linear, so all of them hit the mixture and
  the clean target with the same parameters: the target is what the model is
  asked to recover *through* that channel, and superposition is what makes the
  mixture still equal the sum of its parts at the SIR the recipe asked for.
* ``_analogue_to_digital`` is the boundary, and the only point in the chain
  where full scale means anything at all.
* codec and packet loss are *digital transmission* damage, mixture only; the
  target stays the undamaged reference the model is scored against.

Linear means linear. The DSP backends underneath every stage in the first group
saturate at full scale by default, which turns them into waveshapers on a hot
mixture while the quieter target sails through -- see
`puresound.audio.dsp.apply_linear`, which is what keeps them honest. The one
nonlinearity the analogue path is allowed is the overload the recipe asks for by
probability in ``_volume``, and that one clips the pair at the *same* absolute
thresholds.
"""

from __future__ import annotations

import random
from typing import NamedTuple, Optional

import torch

from puresound.audio.dsp import wav_resampling


#: Every stage records what it actually did, as plain floats, so eval can group
#: results by the channel a row went through -- the same thing
#: ``eval_indomain.py --by-bucket`` already does with SIR / overlap / DRR.
#:
#: Numbers only, and one entry per key on EVERY row: that is what lets them ride
#: the existing scalar collate (a `torch.cat` of 0-dim tensors) for free. The
#: string-valued `RIR_PROVENANCE_KEYS` take the other collate path, and they are
#: the cautionary example -- added for traceability, never read by anything.
#:
#: Convention matches the task scalars already emitted: `*_applied` is 0.0/1.0,
#: and a parameter is NaN on rows where its stage did not fire.
DEVICE_CHAIN_SCALARS = (
    "src_applied",
    "src_target_sr",
    "iir_applied",
    "hpf_applied",
    "hpf_cutoff",
    "volume_applied",
    "volume_clipped",
    "volume_gain",
    "codec_applied",
    "codec_kind",
    "codec_bitrate",
    "packet_loss_applied",
    "packet_loss_rate",
    "overload_rescaled",
)

#: Emitted as a float so it collates with the rest; mirrors MIX_MODE_CODES.
CODEC_CODES = {"libopus": 1.0, "g722": 2.0}

_NAN = float("nan")


def _blank_record() -> dict[str, float]:
    """A row where nothing fired still reports every key, or the batch cannot
    be collated into one tensor per key."""
    return {
        key: (0.0 if key.endswith(("_applied", "_clipped", "_rescaled")) else _NAN)
        for key in DEVICE_CHAIN_SCALARS
    }


class ChainResult(NamedTuple):
    """The pair after the chain, plus what the chain did to it.

    Both signals are needed: several stages move them together, and reading only
    one back is how they drift apart. ``applied`` is the per-row provenance --
    see DEVICE_CHAIN_SCALARS.
    """

    noisy: torch.Tensor
    target: torch.Tensor
    applied: dict


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
        overload_guard: bool = True,
    ):
        self.augmentor = augmentor
        # Off for TSE, which has never had this stage. Adding it there is a
        # behaviour change to what that task trains on, so it is a knob rather
        # than something a shared component quietly imposes.
        self.overload_guard = overload_guard
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
        # A plain accumulator scoped to this one call -- each stage writes what
        # it drew. Not cross-call state: the record leaves with the result.
        record = _blank_record()
        noisy, target = self._sample_rate_conversion(noisy, target, sample_rate, record)
        noisy, target = self._second_order_iir(noisy, target, record)
        noisy, target = self._high_pass(noisy, target, sample_rate, record)
        noisy, target = self._volume(noisy, target, sample_rate, record)
        if self.overload_guard:
            noisy, target = self._analogue_to_digital(noisy, target, record)
        noisy = self._codec(noisy, sample_rate, record)
        noisy = self._packet_loss(noisy, sample_rate, record)
        return ChainResult(noisy, target, record)

    # ------------------------------------------------------------------ #
    # Linear channel: the target follows the mixture through each of these
    # ------------------------------------------------------------------ #

    def _sample_rate_conversion(self, noisy, target, sample_rate, record):
        if not self._fires(self.src):
            return noisy, target
        src_target = random.choices(
            self.src.src_range, weights=self.src.prob_each
        )[0]
        record["src_applied"] = 1.0
        record["src_target_sr"] = float(src_target)
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

    def _second_order_iir(self, noisy, target, record):
        if not self._fires(self.ir_response):
            return noisy, target
        record["iir_applied"] = 1.0
        noisy, (a_coeffs, b_coeffs) = self.augmentor.apply_2nd_iir_response(wav=noisy)
        target, _ = self.augmentor.apply_2nd_iir_response(
            wav=target, a_coeffs=a_coeffs, b_coeffs=b_coeffs
        )
        return noisy, target

    def _high_pass(self, noisy, target, sample_rate, record):
        if not self._fires(self.hpf):
            return noisy, target
        cutoff = random.choices(self.hpf.cutoff, weights=self.hpf.prob_each)[0]
        record["hpf_applied"] = 1.0
        record["hpf_cutoff"] = float(cutoff)
        q_factor = torch.FloatTensor(1).normal_(mean=0.707, std=0.1).clip(0.3, 1.3)
        noisy, _ = self.augmentor.apply_hpf(
            wav=noisy, sr=sample_rate, cutoff_freq=cutoff, q_factor=q_factor
        )
        target, _ = self.augmentor.apply_hpf(
            wav=target, sr=sample_rate, cutoff_freq=cutoff, q_factor=q_factor
        )
        return noisy, target

    def _volume(self, noisy, target, sample_rate, record):
        if not self._fires(self.volume):
            return noisy, target
        record["volume_applied"] = 1.0
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
            record["volume_clipped"] = 1.0
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
            record["volume_gain"] = float(gain)
        return noisy, target

    # ------------------------------------------------------------------ #
    # Transmission damage: mixture only, the target stays the clean reference
    # ------------------------------------------------------------------ #

    def _codec(self, noisy, sample_rate, record):
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
        record["codec_applied"] = 1.0
        record["codec_kind"] = CODEC_CODES.get(codec_name, _NAN)
        record["codec_bitrate"] = _NAN if bit_rate is None else float(bit_rate)
        return noisy

    def _packet_loss(self, noisy, sample_rate, record):
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
        record["packet_loss_applied"] = 1.0
        record["packet_loss_rate"] = float(loss_rate)
        return noisy

    # ------------------------------------------------------------------ #

    @staticmethod
    def _analogue_to_digital(noisy, target, record):
        """The converter: where the signal stops being a pressure and becomes
        samples, and the only place in this chain where full scale exists.

        Upstream is the acoustic and analogue path. Level there is sound
        pressure through a transducer and a preamp, and pressure has no full
        scale -- a peak above 1.0 is not an error, it is a loud room. Nothing
        upstream may treat 1.0 as a ceiling, which is the whole point of
        `apply_linear`. Downstream is digital: a codec cannot encode past full
        scale, and the model's own output is clamped to [-1, 1], so a target
        above it is one the model cannot reach however well it separates.

        Crossing that boundary is *gain staging*, not clipping -- the engineer
        setting the preamp so the converter is not driven into its rails, which
        is what anyone recording a loud room actually does. Divide the pair by
        the same peak: linear, so superposition survives and the mixture is
        still its sources at the SIR the recipe asked for.

        Deliberate overload is a different thing and lives elsewhere on purpose:
        `_volume` clips at the mixture's realized quantiles, on request, with
        the same thresholds on both signals. Doing it here as well would clip
        the recipe's rows twice and label neither.

        Consumes no randomness.
        """
        peak = float(torch.maximum(noisy.abs().amax(), target.abs().amax()))
        if peak > 1.0:
            noisy = noisy / peak
            target = target / peak
            record["overload_rescaled"] = 1.0
        return noisy, target


def device_chain_from_blocks(
    augmentor, dataset, *, overload_guard: bool = True
) -> Optional[DeviceChain]:
    """Build the chain from a dataset's already-validated augmentation blocks.

    Stages the dataset has no block for are simply absent, and an absent stage
    costs nothing -- which is how a task with no codec knob shares this chain
    with one that has it.
    """
    return DeviceChain(
        augmentor,
        overload_guard=overload_guard,
        src=dataset.augmentation_src_args,
        ir_response=dataset.augmentation_ir_response_args,
        hpf=dataset.augmentation_hpf_args,
        volume=dataset.augmentation_volume_args,
        codec=getattr(dataset, "augmentation_codec_args", None),
        packet_loss=getattr(dataset, "augmentation_packet_loss_args", None),
    )
