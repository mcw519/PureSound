"""Over-suppression relief, applied after the mask and only at inference.

Two knobs that do the same job at different points in the signal path, and one
object so they travel together:

* **``spec_floor``** works in the spectral domain, before the iSTFT. It clamps
  each enhanced magnitude bin to at least ``spec_floor * |mix bin|`` while
  keeping the enhanced phase, so the mask cannot attenuate any bin below that
  fraction of what came in. Per-bin, so a bin the mask got right is untouched.
* **``dry_blend``** works on the finished waveform. It mixes the untouched input
  back in: ``dry_blend * enhanced + (1 - dry_blend) * input``. Broadband, so it
  relieves deletion everywhere at once and leaks interference everywhere at once.

Both default to a no-op and the training path never sets them, which is what
keeps this out of what the model learns.

**``dry_blend`` puts a hard ceiling under suppression, and it is worth knowing
the number.** Whatever the model does, the output keeps ``1 - dry_blend`` of the
input, so the deepest attenuation reachable is ``20*log10(1 - dry_blend)``: 0.9
caps it at exactly -20.0 dB, 0.95 at -26.0 dB. A far-field residual measured at
``dry_blend 0.9`` is measuring the blend as much as the model once it approaches
that. `suppression_ceiling_db` reports it rather than leaving it implicit.

This lives apart from the module for two reasons. It is not part of the model --
nothing here is learned, and an exported graph does not contain it, so a
deployment that runs the graph alone is running a *different* system than a
benchmark at ``dry_blend 0.9``. And keeping it in one place means the two knobs
are testable without building an encoder, and describable in a manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Postprocessor:
    """Inference-only over-suppression relief.

    Args:
        dry_blend: in ``(0, 1]``. 1.0 is a no-op.
        spec_floor: in ``[0, 1)``. 0.0 is a no-op.
    """

    dry_blend: float = 1.0
    spec_floor: float = 0.0

    #: Below this the enhanced magnitude is treated as zero rather than divided by.
    _EPS = 1e-8

    def __post_init__(self):
        if not 0.0 < self.dry_blend <= 1.0:
            raise ValueError(
                f"dry_blend must be in (0, 1], got {self.dry_blend}. 1.0 disables it; "
                "0 would mean the output is the input with the model discarded."
            )
        if not 0.0 <= self.spec_floor < 1.0:
            raise ValueError(
                f"spec_floor must be in [0, 1), got {self.spec_floor}. 0.0 disables "
                "it; 1.0 would floor every bin at the input and mask nothing."
            )

    @property
    def enabled(self) -> bool:
        return self.dry_blend < 1.0 or self.spec_floor > 0.0

    @property
    def suppression_ceiling_db(self) -> float:
        """The deepest attenuation reachable, in dB, given ``dry_blend``.

        ``-inf`` when the blend is off. This is arithmetic, not an estimate: the
        residual is ``1 - dry_blend`` of the input no matter what the model did.
        """
        if self.dry_blend >= 1.0:
            return -math.inf
        return 20.0 * math.log10(1.0 - self.dry_blend)

    def floor_spectrum(self, enh: torch.Tensor, mix_tf: torch.Tensor) -> torch.Tensor:
        """Floor the enhanced magnitude to ``>= spec_floor * |mix|`` per bin.

        Keeps the enhanced phase. ``enh`` and ``mix_tf`` are real/imag stacked on
        dim=1, shape ``[N, 2, C, T]``. A no-op where the mask did not
        over-suppress, and a no-op entirely when ``spec_floor`` is 0.

        **A bin the mask zeroed completely stays at zero.** The floor scales the
        enhanced value to reach the target magnitude, and scaling 0+0j reaches
        nothing -- there is no phase to keep. So this relieves partial
        over-suppression and cannot relieve total over-suppression, which is the
        harder case. Lifting those would mean borrowing the mix's phase, a
        different operation with a different failure mode; ``dry_blend`` is the
        knob that covers them today. Behaviour predates this class and is
        unchanged by it.
        """
        if self.spec_floor <= 0.0:
            return enh
        real, imag = torch.chunk(enh, chunks=2, dim=1)
        mix_real, mix_imag = torch.chunk(mix_tf, chunks=2, dim=1)
        enh_mag = torch.sqrt(real * real + imag * imag + self._EPS)
        mix_mag = torch.sqrt(mix_real * mix_real + mix_imag * mix_imag + self._EPS)
        scale = torch.maximum(enh_mag, self.spec_floor * mix_mag) / enh_mag
        return torch.cat([real * scale, imag * scale], dim=1)

    def blend_waveform(self, enh: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Mix the untouched input back into the enhanced waveform.

        Only the overlapping span is blended -- the STFT round trip can leave the
        two a few samples apart -- and the rest of the enhanced output is left as
        it is rather than truncated. Result is clamped to [-1, 1], the same range
        the module's own output is clamped to.
        """
        if self.dry_blend >= 1.0:
            return enh
        if mix.dim() == enh.dim() + 1 and mix.shape[0] == 1:
            mix = mix.squeeze(0)
        overlap = min(enh.shape[-1], mix.shape[-1])
        blended = (
            self.dry_blend * enh[..., :overlap]
            + (1.0 - self.dry_blend) * mix[..., :overlap]
        )
        out = enh.clone()
        out[..., :overlap] = torch.clamp(blended, min=-1.0, max=1.0)
        return out

    def reject_spec_floor(self, mask_type: str) -> None:
        """A spectral floor needs a magnitude to floor and a phase to keep.

        Only the complex-ratio path has both. The other mask types silently
        ignored the knob -- four of the five branches never mentioned it -- so a
        recipe asking for relief it could not get looked like relief that did not
        help.
        """
        if self.spec_floor > 0.0:
            raise ValueError(
                f"spec_floor={self.spec_floor} cannot be applied to mask_type "
                f"{mask_type!r}: it floors a magnitude while keeping the enhanced "
                "phase, which only the complex-ratio path produces. Use dry_blend "
                "for broadband relief on this model."
            )

    #: Manifest key. `recommended_inference` predates this class -- four of the
    #: five shipped DPCRN exports already carry `dry_blend: 0.9` under it, with a
    #: note spelling out the latency alignment and the -20 dB bound. It was a
    #: dangling contract: written for a consumer that did not exist. Reusing the
    #: name rather than inventing a second one means those artefacts start being
    #: honoured instead of needing migration.
    MANIFEST_KEY = "recommended_inference"

    def as_manifest(self) -> dict:
        """What an export records, so a deployment reproduces the configuration a
        benchmark measured instead of guessing at it."""
        return {
            "dry_blend": float(self.dry_blend),
            "spec_floor": float(self.spec_floor),
            "suppression_ceiling_db": self.suppression_ceiling_db,
        }


#: The no-op, shared so the common path allocates nothing.
IDENTITY = Postprocessor()


def resolve(
    postprocess: Optional[Postprocessor], dry_blend: float, spec_floor: float
) -> Postprocessor:
    """One `Postprocessor` from either calling convention.

    `forward` keeps its `dry_blend=` / `spec_floor=` keywords because the eval
    scripts pass them by name, and also takes a built `Postprocessor`. Passing
    both ways at once is a mistake rather than a precedence question.
    """
    if postprocess is None:
        if dry_blend >= 1.0 and spec_floor <= 0.0:
            return IDENTITY
        return Postprocessor(dry_blend=dry_blend, spec_floor=spec_floor)
    if dry_blend < 1.0 or spec_floor > 0.0:
        raise ValueError(
            "pass either a Postprocessor or dry_blend/spec_floor, not both: "
            f"got {postprocess!r} alongside dry_blend={dry_blend}, "
            f"spec_floor={spec_floor}"
        )
    return postprocess
