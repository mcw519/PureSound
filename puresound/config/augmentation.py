"""Pydantic models for cross-task audio-pipeline capabilities.

Field defaults here are the *behavioural* defaults, not placeholders: datasets
read these models by attribute and use the value as given, so a field left as
``None`` would silently mean "nothing" where the synthesis path needs a real
value. Anything the pipeline needs a default for carries it here, and
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


class NoiseSourceConfig(StrictConfig):
    """One noise corpus in a weighted pool."""

    name: str
    folder: str
    weight: Annotated[float, Field(gt=0.0)] = 1.0


class SnrBand(StrictConfig):
    """One piece of a piecewise-uniform SNR distribution."""

    low: float
    high: float
    prob: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def ordered(self):
        if self.high <= self.low:
            raise ValueError(f"snr band needs high > low, got [{self.low}, {self.high}]")
        return self


class NoiseAugmentation(StrictConfig):
    used: StrictBool
    prob: Probability | None = None
    noise_folder: str | None = None
    # Several corpora at chosen shares: each draw picks a source by `weight`,
    # then a file uniformly inside it. With `noise_folder` alone every FILE is
    # equally likely, so a corpus's share is its file count, not a chosen share.
    # Either this or `noise_folder`, not both.
    noise_sources: list[NoiseSourceConfig] | None = None
    snr_range: FloatRange | None = None
    # Piecewise-uniform SNR draw inside `snr_range`: a band is picked by `prob`,
    # then the SNR uniformly inside it. It exists so the range can reach high
    # SNRs without thinning the hard mixtures -- uniform over [-5, 40] would put
    # only 44 % of the draws below 15 dB.
    # None = uniform over `snr_range`.
    snr_bands: list[SnrBand] | None = None
    prob_white_noise: Probability | None = None
    white_noise_snr_range: FloatRange | None = None
    room_coloring: RoomColoringConfig | None = None
    absolute_floor: AbsoluteFloorConfig | None = None

    @model_validator(mode="after")
    def enabled_contract(self):
        if self.noise_folder is not None and self.noise_sources is not None:
            raise ValueError("give noise_folder or noise_sources, not both")
        if self.noise_sources is not None:
            names = [source.name for source in self.noise_sources]
            if len(set(names)) != len(names):
                raise ValueError(f"noise_sources names must be unique, got {names}")
        if self.used and self.noise_folder is None and not self.noise_sources:
            raise ValueError("required when enabled: noise_folder or noise_sources")
        if self.snr_bands is not None:
            if self.snr_range is None:
                raise ValueError("snr_bands needs snr_range as the envelope it lies in")
            total = sum(band.prob for band in self.snr_bands)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"snr_bands probabilities must sum to 1, got {total}")
            low, high = self.snr_range
            outside = [b for b in self.snr_bands if b.low < low or b.high > high]
            if outside:
                raise ValueError(f"snr_bands {outside} fall outside snr_range {list(self.snr_range)}")
        return require_fields_when_enabled(
            self,
            "prob",
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

    The model's distance readout depends most on the fine timing of the few
    milliseconds after the direct arrival, and reverberation masks that cue.
    Removing it on a fraction of rows asks the model to find a substitute.
    Whether one exists is open -- timing and spectrum are read jointly, not
    independently, so spectral tilt is a candidate, not a guarantee -- and no
    recipe sets this knob. See
    ``puresound.audio.impulse_response.smear_direct_arrival``.

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
    """``augmentation_volume`` -- a gain, or with ``clipping_prob`` an overload.

    ``clipping_range`` gives the quantiles of the mixture's samples to clip at.
    ``target_clipping`` says what happens to the target on those rows:

    ``mixture_level``  clipped at the mixture's thresholds, the same absolute
                       levels, so a target quieter than them is left untouched
    ``own_quantile``   clipped at the same quantiles of its own samples; the
                       target of a quiet talker in loud noise is clipped as hard
                       as the mixture. What the dpcrn_mamba v1/v2 recipes were
                       trained with.
    """

    perturbed_range: FloatRange | None = None
    clipping_prob: Probability | None = None
    clipping_range: ClippingRange | None = None
    target_clipping: Literal["mixture_level", "own_quantile"] = "mixture_level"

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(
            self, "perturbed_range", "clipping_prob", "clipping_range"
        )


class CompressorAugmentation(SimpleProbAugmentation):
    """``augmentation_compressor`` -- broadcast-style dynamic-range compression.

    Models "the recording went through a compressor", which publication and
    conferencing chains routinely do. Envelope flattening raises the DRR the
    model reads off a recording, so it is one of the things that makes a
    distant talker read as a near one; absolute level, by contrast, does not
    move the model's distance estimate.

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


class RowInitialAmbientAugmentation(StrictConfig):
    """Open a row with scene sound only -- no speech -- for the first 1-4 s.

    Why: a stretch of ambience ahead of the first speech changes how strongly
    the model suppresses on a cold start, and without this block every training
    row has speech from its first frames, so that situation is never trained.
    The lead is applied to BOTH keep and suppress rows (class-blind, so the lead
    itself can never carry label information): all speech (foreground AND
    interferers, target included) is masked out of the lead window before the
    noise stage, which then fills it with the row's own room-coloured noise and
    capture floor. Rows whose noise draw is off get a silent lead instead --
    also a real deployment condition (stream start in a dead-quiet room).
    """

    used: StrictBool
    prob: Probability = 0.0
    #: lead length drawn uniformly, seconds
    lead_seconds_range: list[float] = [1.0, 4.0]
    #: raised-cosine ramp at the lead edge; a hard edge is a click the model
    #: could key on
    fade_ms: float = 50.0

    @model_validator(mode="after")
    def sane_range(self):
        if len(self.lead_seconds_range) != 2:
            raise ValueError("lead_seconds_range must be [lo, hi]")
        lo, hi = self.lead_seconds_range
        if not (0.0 < lo <= hi):
            raise ValueError(f"lead_seconds_range must be 0 < lo <= hi, got {(lo, hi)}")
        return self


class SessionTurnShapeConfig(StrictConfig):
    """How often each conversational shape is drawn on a session row.

    Weights, not probabilities: they are normalised at draw time, so raising one
    does not force the others to be edited. All four are non-zero by default --
    the point of the block is that the four shapes coexist.

    * ``user_first`` -- U opens, then U and the bystanders alternate.
    * ``bystander_first`` -- a bystander holds the floor for
      ``bystander_open_seconds`` and U enters afterwards. This is the wrong-anchor
      shape, which ordinary rows rarely produce.
    * ``user_gap`` -- U speaks, then ``user_gap_seconds`` of the row's own floor
      (a bystander may talk inside it), then U returns. Ordinary rows almost
      never contain such a re-entry; a long gap is NEVER labelled target-absent.
    * ``overlap`` -- the same alternation with the turn boundaries overlapping
      instead of separated, i.e. double talk (the KEEP guard case).
    """

    user_first: NonNegativeFloat = 0.25
    bystander_first: NonNegativeFloat = 0.30
    user_gap: NonNegativeFloat = 0.30
    overlap: NonNegativeFloat = 0.15

    @model_validator(mode="after")
    def at_least_one_shape(self):
        if self.user_first + self.bystander_first + self.user_gap + self.overlap <= 0:
            raise ValueError("shape_probs must contain a positive weight")
        return self


class SessionRowsConfig(StrictConfig):
    """``session_rows`` -- multi-turn conversational rows with per-turn identity.

    A session row is one user U (near draw), one or two bystanders B (far draw)
    and the row's own noise floor, arranged as a *script* of turns instead of two
    continuously-talking sources. It exists because ordinary rows make the
    foreground decision cheap: the user almost always starts within the first
    half second, no row renders the same talker twice, and nothing in the
    objective mentions time.

    Contracts this block keeps:

    * **One device-chain draw per row**, applied jointly to the mixture and the
      target exactly as ``ns.py`` does for every other row type.
      Explicit ``paired_view_prob`` applies an independent chain to the exact
      same completed mixture, collated separately for consistency supervision.
    * **The user's gap is never digital silence** and never target-absent. The
      gap carries the row's own floor (``floor_dbfs_range`` guarantees one even
      on rows where every noise source declines to fire), and the row keeps
      ``target_present = 1`` throughout, because a higher target-absent rate
      teaches the model to delete the user's speech.
    * **Disabled costs nothing.** ``enabled: False`` (or ``prob: 0``) draws no
      randomness at all, and a row shorter than ``min_seconds`` is checked before
      the draw -- so the short length buckets of a recipe with this block on stay
      bit-identical to the same recipe without it.
    """

    #: Named ``enabled`` rather than ``used`` to match the block's design
    #: spelling; ``used`` below is what the shared recipe plumbing reads
    #: (``BaseRecipe.augmentation_kwargs``).
    enabled: StrictBool = False
    prob: Probability = 0.0
    #: Sessions need room for several turns. Rows shorter than this are never
    #: session rows, and the test is made BEFORE the probability draw.
    min_seconds: float = Field(default=12.0, gt=0.0)
    #: Cap on the scripted span. The remainder of a longer row is floor only.
    #: Set above the longest length bucket to make it a no-op (the default:
    #: sessions are long rows inside the existing schedule, not a new bucket).
    max_seconds: float = Field(default=60.0, gt=0.0)
    n_bystanders: IntRange = (1, 2)
    user_distance_range: FloatRange = (0.3, 1.0)
    bystander_distance_range: FloatRange = (1.5, 4.0)
    shape_probs: SessionTurnShapeConfig = SessionTurnShapeConfig()
    user_turn_seconds: FloatRange = (1.5, 4.0)
    bystander_turn_seconds: FloatRange = (2.0, 4.5)
    bystander_open_seconds: FloatRange = (2.0, 8.0)
    user_gap_seconds: FloatRange = (5.0, 20.0)
    turn_gap_seconds: FloatRange = (0.2, 0.8)
    overlap_seconds: FloatRange = (0.5, 3.0)
    #: Boundary double-talk on the shapes that are not the ``overlap`` shape.
    boundary_overlap_prob: Probability = 0.25
    #: A bystander talking inside the user's gap -- the case where "the user is
    #: away" and "nobody is near" have to stay distinguishable.
    bystander_in_gap_prob: Probability = 0.7
    min_turn_seconds: float = Field(default=0.5, gt=0.0)
    max_turns: int = Field(default=16, gt=0)
    #: The user moves: later turns run through a SECOND near-range channel of the
    #: same room. An RIR change only -- the chain draw stays single.
    rir_move_prob: Probability = 0.3
    #: A bystander drawn in the user's own distance class, so proximity alone
    #: cannot separate the roles (the near/far shortcut).
    distance_matched_bystander_prob: Probability = 0.2
    #: SIR for the summed bystander bus, mixed by the same ``add_bg_noise`` the
    #: other row types use. ``sir_low_tail_*`` is the heavier low tail, because
    #: loud bystanders are rare in the ordinary rows.
    sir_range: FloatRange = (-5.0, 10.0)
    sir_low_tail_prob: Probability = 0.30
    sir_low_tail_range: FloatRange = (-10.0, -5.0)
    #: Absolute capture floor forced onto session rows, mirroring
    #: ``augmentation_noise.absolute_floor`` semantics (level as drawn, upstream
    #: of the device chain). Low by design: it is a guarantee against digital
    #: silence in a gap, not a second noise source.
    floor_dbfs_range: FloatRange = (-60.0, -45.0)
    #: Raised-cosine gate width, in samples, same units as
    #: ``overlap_control.fade_samples``; a hard turn edge is a click.
    fade_samples: Annotated[int, Field(ge=0)] = 400
    #: Cross-chain pairs: a paired row draws a slot and renders the material that
    #: slot determines, so two rows carrying the same ``row_source_id`` are the
    #: same source through two independent chain draws. ``pair_pool_size`` is the
    #: number of distinct paired sources -- small means frequent pairs and
    #: frequent repetition, large means the opposite.
    pair_prob: Probability = 0.0
    pair_pool_size: Annotated[int, Field(gt=0)] = 1024
    pair_seed_base: Annotated[int, Field(ge=0)] = 20260905
    #: Explicit second chain view of this exact post-noise, post-speed mixture.
    #: It is collated separately and only supervises chain consistency.
    paired_view_prob: Probability = 0.0
    paired_view_min_seconds: float = Field(default=12.0, gt=0.0)

    @property
    def used(self) -> bool:
        """What ``BaseRecipe.augmentation_kwargs`` and the dataset registry read."""
        return bool(self.enabled)

    @model_validator(mode="after")
    def session_contract(self):
        if not self.enabled:
            return self
        if self.paired_view_prob > 0 and self.pair_prob > 0:
            raise ValueError("explicit paired views cannot use legacy finite-pool pairing")
        if self.max_seconds < self.min_seconds:
            raise ValueError("max_seconds must not be below min_seconds")
        lo, hi = self.n_bystanders
        if lo < 1 or lo > hi:
            raise ValueError(f"n_bystanders must be 1 <= lo <= hi, got {(lo, hi)}")
        if self.user_gap_seconds[0] < 1.0:
            raise ValueError("user_gap_seconds must start at >= 1 s")
        if self.min_turn_seconds >= self.user_turn_seconds[1]:
            raise ValueError("min_turn_seconds must be below user_turn_seconds top")
        return self


class RealFarAugmentation(StrictConfig):
    used: StrictBool
    pool_manifest: str | None = None
    prob: Probability = 0.0
    lone_far_prob: Probability = 0.0
    turn_taking_prob: Probability | None = None
    # A pool recording is one take (VOiCES: ~16 s). Past the training length it
    # is cropped; SHORT of it the aligner zero-pads, which on a keep row makes
    # the target half silence. With this on, extra takes from the same
    # (speaker, room, mic) -- same talker, same chain -- are concatenated until
    # the row length is covered. Off by default; at a row length one take
    # already covers it changes nothing, RNG included.
    stitch_to_length: StrictBool = False

    @model_validator(mode="after")
    def enabled_contract(self):
        return require_fields_when_enabled(self, "pool_manifest")


class RealNearAugmentation(StrictConfig):
    used: StrictBool
    pool_manifest: str | None = None
    prob: Probability = 0.0
    turn_taking_prob: Probability | None = None
    #: see RealFarAugmentation.stitch_to_length -- it matters more here, because
    #: this pool supplies the KEEP row's target.
    stitch_to_length: StrictBool = False

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
