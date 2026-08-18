"""Everything that is not a voice.

Three sources, and they are three because they answer to different things:

* **Recorded noise at an SNR** relative to the speech already in the mixture --
  a cafe, a street, a fan. Optionally rendered through a channel of the *same*
  room the speech was rendered in, so the noise floor is spatially coherent with
  the mixture instead of arriving dry from nowhere.
* **White noise**, on a quarter of the rows the recorded noise did not take a
  dynamic type on. Cheap broadband cover the recorded pool does not always give.
* **A capture floor** at an absolute level that does *not* scale with the
  speech. A deployed mic's self-noise and the room's own tone sit where they sit
  whoever is talking, and an SNR-relative source cannot express that.

The first two are level-relative and go before the device chain; so does the
third, deliberately, because capsule noise picks up the device response the way
the speech does.

Contracts, the same ones `DeviceChain` and `OverlapGating` hold:

**RNG order is load-bearing.** Every branch draws from the shared stream, so
moving a draw changes what a seeded recipe produces.

**A source that does not fire draws nothing.** The probability draw is inside
each short circuit, so turning one off leaves every later stage seeing what it
saw before.

**Noise is added to the mixture only.** The target stays the reference the model
is scored against -- adding noise to both would be asking the model to
reproduce it.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class NoiseResult(NamedTuple):
    """The mixture with noise in it, plus what the row should report.

    ``snr`` is the recorded-noise SNR as drawn, NaN when that source did not
    fire. It is recorded before the white-noise branch, which draws its own SNR
    into the same name in the code this replaces -- reporting that one would
    label the row with the wrong number.
    """

    noisy: torch.Tensor
    snr: float


class NoiseStage:
    """Adds the non-speech sources to a finished speech mixture.

    Built once per dataset from the already-validated ``augmentation_noise``
    block; ``sample_rate`` and ``room_scene`` are per call because they follow
    the row.
    """

    def __init__(self, augmentor, config):
        self.augmentor = augmentor
        self.config = config

    def apply(
        self,
        noisy: torch.Tensor,
        *,
        sample_rate: int,
        room_scene=None,
    ) -> NoiseResult:
        """Add recorded noise, then white noise, then the capture floor."""
        snr = float("nan")
        config = self.config
        if config and config.used and torch.rand(1) < config.prob:
            noisy, snr = self._recorded(noisy, sample_rate, room_scene)
        if isinstance(noisy, list):
            noisy = noisy[0]
        noisy = self._capture_floor(noisy)
        return NoiseResult(noisy, snr)

    # ------------------------------------------------------------------ #

    def _recorded(self, noisy, sample_rate, room_scene) -> tuple:
        config = self.config
        dynamic_type = False
        snr = (
            torch.FloatTensor(1)
            .uniform_(config.snr_range[0], config.snr_range[1])
            .item()
        )
        # Captured before the white-noise branch below draws into the same name.
        reported = snr

        # 1 / 4 cases add dynamic noise type
        if torch.rand(1) < config.prob / 4:
            dynamic_type = True

        noise_transform = self._room_coloring(sample_rate, room_scene)
        noisy, _ = self.augmentor.add_bg_noise(
            wav=noisy,
            snr_list=[snr],
            dynamic_type=dynamic_type,
            sr=sample_rate,
            noise_transform=noise_transform,
        )
        # unwrap list
        noisy = noisy[0]

        # if dynamic is False, 1 / 4 add white noise
        if not dynamic_type and torch.rand(1) < config.prob_white_noise:
            snr = (
                torch.FloatTensor(1)
                .uniform_(
                    config.white_noise_snr_range[0],
                    config.white_noise_snr_range[1],
                )
                .item()
            )
            noisy, _ = self.augmentor.add_bg_white_noise(wav=noisy, snr_list=[snr])
        return noisy, reported

    def _room_coloring(self, sample_rate, room_scene) -> Optional[callable]:
        """A channel of the same room the speech was rendered in.

        Noise that arrives dry while the speech is reverberant tells the model
        which is which for free. Returns None when the block is off or the row
        has no room -- and consumes no randomness in that case.
        """
        config = self.config.room_coloring
        if not (
            config
            and config.used
            and room_scene is not None
            and torch.rand(1).item() < config.prob
        ):
            return None

        def transform(noise, _sr=sample_rate, _scene=room_scene):
            reverbed, _ = self.augmentor.apply_rir(
                wav=noise,
                rir_mode="full",
                sr=_sr,
                room_scene=_scene,
                source_role="interferer",
            )
            return reverbed

        return transform

    def _capture_floor(self, noisy: torch.Tensor) -> torch.Tensor:
        """A noise floor at an absolute level rather than an SNR.

        "dBFS" is the level as *drawn*, not as delivered. The converter at the
        end of the device chain gain-stages the whole row, so a floor drawn at
        -45 arrives lower by however much that row was turned down -- on the
        shipped recipe, the 29% of rows it touches move by a median of 1.0 dB
        and 6.3 dB at p5. That is correct for what this models: capsule
        self-noise and room tone both sit upstream of the preamp and both follow
        it. A converter's *own* electronic noise would not, and would have to be
        added after `DeviceChain._analogue_to_digital` -- at roughly -90 dBFS it
        is 40 dB below anything this range draws, which is why there is no such
        stage.
        """
        config = self.config.absolute_floor if self.config else None
        if not (config and config.used and torch.rand(1).item() < config.prob):
            return noisy
        low, high = config.level_dbfs_range
        level_dbfs = torch.empty(1).uniform_(float(low), float(high)).item()
        floor = torch.randn_like(noisy) * (10.0 ** (level_dbfs / 20.0))
        return noisy + floor
