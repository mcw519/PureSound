"""Pydantic models for cross-task audio-pipeline capabilities.

Field defaults here are the *behavioural* defaults, not placeholders: they are
the values the synthesis path used to supply inline at each read site
(``cfg.get("fade_samples", 400)``). Now that datasets read these models by
attribute, a field left as ``None`` would silently replace a real default with
nothing -- so anything with an inline default carries it here instead, and
``test_config_defaults_match_the_pipeline`` pins the pairing.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from .base import (
    FloatRange,
    IntRange,
    NonNegativeFloat,
    Probability,
    StrictConfig,
    require_fields_when_enabled,
)


class Toggle(StrictConfig):
    used: StrictBool


class MediaVoiceConfig(StrictConfig):
    used: StrictBool = False
    prob: Probability = 0.0
    hp_cutoff_range: FloatRange = (200.0, 400.0)
    lp_cutoff_range: FloatRange = (3500.0, 7000.0)
    compress_power_range: FloatRange = (0.6, 0.9)


class EchoPlaybackConfig(StrictConfig):
    used: StrictBool = False
    prob: Probability = 0.0
    distance_range: FloatRange = (0.2, 1.0)
    erle_db_range: FloatRange = (20.0, 35.0)


class OverlapControlConfig(StrictConfig):
    used: StrictBool = False
    no_overlap_prob: Probability = 0.25
    high_overlap_prob: Probability = 0.25
    mid_overlap_range: FloatRange = (0.1, 0.5)
    high_overlap_range: FloatRange = (0.5, 1.0)
    fill_on_silent_range: FloatRange = (0.3, 0.5)
    fade_samples: Annotated[int, Field(ge=0)] = 400
    turn_taking_prob: Probability = 0.0
    turn_near_seconds: FloatRange = (1.5, 3.0)
    turn_far_seconds: FloatRange = (2.0, 4.5)
    turn_gap_seconds: FloatRange = (0.0, 0.4)
    turn_overlap_seconds: FloatRange = (0.0, 0.3)
    far_first_prob: Probability = 0.5


class MixModeEntry(StrictConfig):
    name: str = "physical"
    prob: NonNegativeFloat = 0.0
    physical: StrictBool = False
    distance_level: StrictBool = False
    sir_range: FloatRange | None = None
    jitter_db: FloatRange = (-3.0, 3.0)

    @model_validator(mode="after")
    def explicit_level_modes_need_a_range(self):
        if not self.physical and not self.distance_level and self.sir_range is None:
            raise ValueError(
                "sir_range is required unless physical or distance_level is enabled"
            )
        return self


class MixModeConfig(StrictConfig):
    used: StrictBool = False
    modes: list[MixModeEntry] = Field(default_factory=list)


class SpeechAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None
    is_target: StrictBool | None = None
    add_n_cases: int | tuple[int, int] | None = None
    snr_range: FloatRange | None = None
    media_voice: MediaVoiceConfig | None = None
    echo_playback: EchoPlaybackConfig | None = None
    overlap_control: OverlapControlConfig | None = None
    mix_mode: MixModeConfig | None = None

    @field_validator("add_n_cases", mode="before")
    @classmethod
    def normalize_case_range(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def enabled_contract(self):
        require_fields_when_enabled(
            self, "prob", "is_target", "add_n_cases", "snr_range"
        )
        if (
            isinstance(self.add_n_cases, tuple)
            and self.add_n_cases[0] > self.add_n_cases[1]
        ):
            raise ValueError("add_n_cases lower bound must not exceed upper bound")
        return self


class RoomColoringConfig(StrictConfig):
    used: StrictBool = False
    prob: Probability = 0.0


class AbsoluteFloorConfig(StrictConfig):
    """A capture noise floor that does not scale with the speech.

    `level_dbfs_range` is the level as drawn, upstream of the device chain --
    not the level in the delivered row. The chain's converter gain-stages the
    whole row when the analogue path runs hot, which carries the floor down with
    it, and that is the point: capsule self-noise and room tone both sit before
    the preamp and both follow it. Read the realized level off the row, not off
    this knob. See `NoiseSuppressionDataset.__getitem__` for why a converter's
    own electronic noise would need a different stage entirely.
    """

    used: StrictBool = False
    prob: Probability = 0.0
    level_dbfs_range: FloatRange = (-55.0, -35.0)


class NoiseAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None
    noise_folder: str | None = None
    snr_range: FloatRange | None = None
    prob_white_noise: Probability | None = None
    white_noise_snr_range: FloatRange | None = None
    room_coloring: RoomColoringConfig | None = None
    absolute_floor: AbsoluteFloorConfig | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(
            self,
            "prob",
            "noise_folder",
            "snr_range",
            "prob_white_noise",
            "white_noise_snr_range",
        )


class RoomBankMemberConfig(StrictConfig):
    name: str | None = None
    weight: Annotated[float, Field(gt=0.0)] = 1.0
    bank_type: Literal["room", "release"] | None = None
    folder: str
    recipe_id: str | None = None
    split: Literal["train", "validation", "test"] | None = None
    near_labels: list[str] | None = None
    far_labels: list[str] | None = None
    drr_window_ms: Annotated[float, Field(gt=0.0)] = 2.5
    wav_name: str | None = None
    meta_name: str | None = None
    cache_size: Annotated[int, Field(gt=0)] = 64
    manifest_name: str | None = None
    include_failed_qc: StrictBool | None = None
    allow_legacy_layout: StrictBool | None = None
    release_manifest_name: str | None = None
    require_production: StrictBool | None = None
    production_decision_name: str | None = None
    audit: StrictBool | None = None
    audit_cache: StrictBool | None = None

    @model_validator(mode="after")
    def release_contract(self):
        if self.bank_type == "release":
            if not self.recipe_id or not self.split:
                raise ValueError("release bank requires recipe_id and split")
        return self


class PreGeneratedBankConfig(RoomBankMemberConfig):
    used: StrictBool
    usage_role: Literal["train", "validation", "test"] | None = None
    banks: list[RoomBankMemberConfig] | None = None
    folder: str | None = None

    @model_validator(mode="after")
    def source_contract(self):
        if not self.used:
            return self
        if bool(self.folder) == bool(self.banks):
            raise ValueError(
                "enabled pregenerated bank needs exactly one of folder or banks"
            )
        return self


class RoomSimulatorConfig(StrictConfig):
    used: StrictBool
    source_level: StrictBool = False
    pregenerated: PreGeneratedBankConfig | None = None
    room_dim_range: list[FloatRange] | None = None
    rt60_range: FloatRange | None = None
    source_receiver_distance_range: FloatRange | None = None
    foreground_distance_range: FloatRange | None = None
    interferer_distance_range: FloatRange | None = None
    media_distance_range: FloatRange | None = None
    receiver_margin: Annotated[float, Field(gt=0.0)] = 0.4
    source_margin: Annotated[float, Field(gt=0.0)] = 0.4
    media_wall_offset_max: Annotated[float, Field(ge=0.0)] = 0.2
    sound_speed: Annotated[float, Field(gt=0.0)] = 343.0
    nsample: Annotated[int, Field(gt=0)] | None = None
    order: int = -1
    hp_filter: StrictBool = True

    @model_validator(mode="after")
    def simulator_contract(self):
        if not self.used:
            return self
        if self.pregenerated and self.pregenerated.used:
            return self
        return require_fields_when_enabled(
            self,
            "room_dim_range",
            "rt60_range",
            "source_receiver_distance_range",
        )


class DrrContrastConfig(StrictConfig):
    used: StrictBool
    mode: Literal["random", "deterministic"] = "random"
    direct_window_ms: Annotated[float, Field(gt=0.0)] = 2.5
    prob: Probability | None = None
    near_boost_db: FloatRange | None = None
    far_cut_db: FloatRange | None = None
    pivot_m: Annotated[float, Field(gt=0.0)] | None = None
    extra_db_per_decade: float | None = None

    @model_validator(mode="after")
    def mode_contract(self):
        if not self.used:
            return self
        if self.mode == "deterministic" and self.extra_db_per_decade is None:
            raise ValueError("deterministic mode requires extra_db_per_decade")
        if self.mode == "random":
            if self.prob is not None and self.prob <= 0.0:
                raise ValueError("random mode prob must be greater than zero")
            for name in ("near_boost_db", "far_cut_db"):
                bounds = getattr(self, name)
                if bounds is not None and bounds[0] < 0.0:
                    raise ValueError(f"{name} must be non-negative")
        return self


class DirectSmearConfig(StrictConfig):
    """``augmentation_reverb.direct_smear`` -- scramble the direct arrival's timing.

    The distance readout depends most on the direct window's fine timing:
    noise-replacing the first 2.5 ms keeps 45% of the near/far separation
    (early reflections 62%, late tail 60%), a 5 ms smear leaves 1%, and
    destroying timing alone keeps 35% against 81% for flattening the spectrum
    (probes/dist_cue_anatomy_README.md). One cue, and reverberation masks it --
    which is why deletion rises monotonically with RT60.

    Removing it on a fraction of rows asks the model to find a substitute.
    Whether one exists to find is open: the same study measured timing and
    spectrum as read jointly, not summed (destroying both keeps 54%, more than
    timing alone), so spectral tilt is a candidate, not a guarantee. No recipe
    sets this knob yet; it ships dark.

    Applied to the RIR before it is sliced, so the ``full`` mixture and the
    ``early`` target inherit the same smear. ``used: False`` needs no other
    field; ``prob`` must sit in (0, 1].
    """

    used: StrictBool
    prob: Probability | None = None
    smear_ms_range: FloatRange | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        require_fields_when_enabled(self, "prob", "smear_ms_range")
        if self.used and (self.smear_ms_range or [1.0])[0] <= 0.0:
            raise ValueError(
                f"smear_ms_range must start above 0 ms (0 is a no-op and is "
                f"already covered by prob < 1), got {self.smear_ms_range}"
            )
        return self


class ReverbAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None
    target_rir_type: Literal["full", "early", "direct", "anechoic"] | None = None
    rir_folder: str | None = None
    simulator: RoomSimulatorConfig | None = None
    drr_contrast: DrrContrastConfig | None = None
    direct_smear: DirectSmearConfig | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        require_fields_when_enabled(self, "prob", "target_rir_type")
        if not self.used:
            return self
        simulator_enabled = bool(self.simulator and self.simulator.used)
        if not simulator_enabled and not self.rir_folder:
            raise ValueError("enabled reverb needs rir_folder or an enabled simulator")
        return self


class ContinuousSpeedAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None
    speed_range: FloatRange | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(self, "prob", "speed_range")


class DiscreteSpeedAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None
    speed_change: list[Annotated[float, Field(gt=0.0)]] | None = None
    treat_as_new_speaker: StrictBool | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(
            self, "prob", "speed_change", "treat_as_new_speaker"
        )


class SimpleProbAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(self, "prob")


class SourceRateAugmentation(SimpleProbAugmentation):
    src_range: list[Annotated[int, Field(gt=0)]] | None = None
    prob_each: list[NonNegativeFloat] | None = None

    @model_validator(mode="after")
    def choices_contract(self):
        require_fields_when_enabled(self, "src_range", "prob_each")
        if self.used and len(self.src_range or []) != len(self.prob_each or []):
            raise ValueError("src_range and prob_each must have equal lengths")
        if self.used and not any(self.prob_each or []):
            raise ValueError("prob_each must contain a positive weight")
        return self


class HighPassAugmentation(SimpleProbAugmentation):
    cutoff: list[Annotated[float, Field(gt=0.0)]] | None = None
    prob_each: list[NonNegativeFloat] | None = None

    @model_validator(mode="after")
    def choices_contract(self):
        require_fields_when_enabled(self, "cutoff", "prob_each")
        if self.used and len(self.cutoff or []) != len(self.prob_each or []):
            raise ValueError("cutoff and prob_each must have equal lengths")
        if self.used and not any(self.prob_each or []):
            raise ValueError("prob_each must contain a positive weight")
        return self


class ClippingRange(StrictConfig):
    min: FloatRange
    max: FloatRange


class VolumeAugmentation(SimpleProbAugmentation):
    perturbed_range: FloatRange | None = None
    clipping_prob: Probability | None = None
    clipping_range: ClippingRange | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(
            self, "perturbed_range", "clipping_prob", "clipping_range"
        )


class CompressorAugmentation(SimpleProbAugmentation):
    """``augmentation_compressor`` -- broadcast-style dynamic-range compression.

    Models "the recording went through a compressor", which publication and
    conferencing chains routinely do. Measured on the QVF publication clips: at
    heavy settings this moves the model's own DRR readout from -0.61 to +5.19 dB
    on our device recordings, landing on the +5.33 those clips actually read --
    so envelope flattening is one of the things that makes a distant talker read
    as a near one. Absolute level was ruled out separately (a -28 dB
    renormalisation moved the estimate by 0.00 m).

    A time-varying gain, so it sits in the device chain's LINEAR group and the
    same curve goes on the mixture and the target. `prob` at 0 draws nothing and
    leaves a recipe bit-identical.
    """

    threshold_db_range: FloatRange | None = None
    ratio_range: FloatRange | None = None
    attack_ms_range: FloatRange | None = None
    release_ms_range: FloatRange | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        require_fields_when_enabled(
            self, "threshold_db_range", "ratio_range",
            "attack_ms_range", "release_ms_range",
        )
        if self.used and (self.ratio_range or [1.0])[0] < 1.0:
            raise ValueError(
                f"ratio_range must start at >= 1.0 (1.0 = no compression), "
                f"got {self.ratio_range}"
            )
        if self.used and (self.attack_ms_range or [1.0])[0] <= 0.0:
            raise ValueError(f"attack_ms_range must be > 0, got {self.attack_ms_range}")
        return self


class CodecAugmentation(SimpleProbAugmentation):
    codecs: list[Literal["libopus", "g722"]] | None = None
    prob_each: list[NonNegativeFloat] | None = None
    bitrate_range: dict[str, IntRange] = Field(default_factory=dict)

    @model_validator(mode="after")
    def codec_contract(self):
        require_fields_when_enabled(self, "codecs")
        if not self.used:
            return self
        codecs = self.codecs or []
        if not codecs:
            raise ValueError("codecs must not be empty")
        if self.prob_each is not None and len(self.prob_each) != len(codecs):
            raise ValueError("codecs and prob_each must have equal lengths")
        unknown = sorted(set(self.bitrate_range) - set(codecs))
        if unknown:
            raise ValueError(
                "bitrate_range contains codecs not selected by codecs: "
                + ", ".join(unknown)
            )
        for name, bounds in self.bitrate_range.items():
            if bounds[0] > bounds[1] or bounds[0] <= 0:
                raise ValueError(f"invalid bitrate range for {name}")
        return self


class PacketLossAugmentation(SimpleProbAugmentation):
    packet_ms_choices: list[Annotated[int, Field(gt=0)]] | None = None
    loss_rate_range: FloatRange | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        require_fields_when_enabled(self, "packet_ms_choices", "loss_rate_range")
        if self.used and self.loss_rate_range is not None:
            if self.loss_rate_range[0] < 0.0 or self.loss_rate_range[1] > 1.0:
                raise ValueError("loss_rate_range must lie within [0, 1]")
        return self


class TargetAbsentAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability = 0.0
    force_interferer: StrictBool = False


class RealFarAugmentation(StrictConfig):
    used: StrictBool
    pool_manifest: str | None = None
    prob: Probability = 0.0
    lone_far_prob: Probability = 0.0
    turn_taking_prob: Probability | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(self, "pool_manifest")


class RealNearAugmentation(StrictConfig):
    used: StrictBool
    pool_manifest: str | None = None
    prob: Probability = 0.0
    turn_taking_prob: Probability | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(self, "pool_manifest")


class VadLabelConfig(StrictConfig):
    used: StrictBool
    backend: Literal["energy", "silero"] = "energy"
    frame_length: Annotated[int, Field(gt=0)] = 400
    hop_length: Annotated[int, Field(gt=0)] = 160
    # Backend constructors own these options; they are an explicit extension
    # boundary rather than an accidental catch-all for the recipe itself.
    args: dict[str, object] = Field(default_factory=dict)
