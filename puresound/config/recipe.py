"""Task-level recipe models assembled from reusable pipeline capabilities."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from .augmentation import (
    CodecAugmentation,
    ContinuousSpeedAugmentation,
    DiscreteSpeedAugmentation,
    HighPassAugmentation,
    NoiseAugmentation,
    PacketLossAugmentation,
    RealFarAugmentation,
    RealNearAugmentation,
    RowInitialAmbientAugmentation,
    ReverbAugmentation,
    SessionRowsConfig,
    SimpleProbAugmentation,
    SourceRateAugmentation,
    SpeechAugmentation,
    TargetAbsentAugmentation,
    VadLabelConfig,
    VolumeAugmentation,
    CompressorAugmentation,
)
from .base import Probability, StrictConfig
from .curriculum import CurriculumConfig, split_track_path


TaskName = Literal[
    "noise_suppression",
    "voice_isolation",
    "target_speaker_extraction",
    "speaker_embedding",
]


class DatasetConfig(StrictConfig):
    train_metafile: str
    valid_metafile: str
    test_folder: str
    proc_output_folder: str
    target_sample_rate: int | None
    gain_normalized_to: float | None
    training_length_seconds: float = Field(gt=0.0)
    filter_min_utterance_length: float = Field(ge=0.0)
    filter_min_utterance_per_speaker: int = Field(gt=0)
    train_pipeline_role: Literal["train", "validation", "test"] = "train"
    validation_pipeline_role: Literal["train", "validation", "test"] = "validation"


class LengthBucket(StrictConfig):
    """One (row length, batch size) pair a training batch can be drawn as."""

    seconds: float = Field(gt=0.0)
    n_spk: int = Field(gt=0)
    prob: float = Field(gt=0.0, le=1.0)


class TrainerConfig(StrictConfig):
    lightning_trainer_args: dict[str, Any]
    train_iter_per_epoch: int = Field(gt=0)
    valid_iter_per_epoch: int = Field(gt=0)
    n_spk_per_batch: int = Field(gt=0)
    n_utt_per_speaker: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    num_gpus: int = Field(ge=0)
    work_folder: str
    valid_seed: int = 1234
    find_unused_parameters: StrictBool = False
    # Mixed row lengths. Each TRAINING batch draws one bucket, so the model is
    # supervised at several context lengths instead of one. Why it exists: a row
    # is scored as a whole, so its evidence-poor opening frames carry 1/N of the
    # loss -- train at one long length and behaviour on short context decays,
    # train at one short length and long-context behaviour never develops. The
    # batch size travels WITH the length because activation memory does, so
    # every bucket fits the same card. Validation always uses n_spk_per_batch /
    # training_length_seconds, so valid loss stays comparable across epochs.
    # None = single length, unchanged.
    length_schedule: list[LengthBucket] | None = None

    @model_validator(mode="after")
    def length_schedule_probabilities(self):
        if self.length_schedule:
            total = sum(b.prob for b in self.length_schedule)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"length_schedule probabilities must sum to 1, got {total}"
                )
        return self


class OptimizerConfig(StrictConfig):
    type: str
    learning_rate: float = Field(gt=0.0)
    args: dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(StrictConfig):
    type: str
    warmup_step: int
    args: dict[str, Any] = Field(default_factory=dict)


class LossConfig(StrictConfig):
    type: str
    weighted: float
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("args", mode="before")
    @classmethod
    def normalize_empty_args(cls, value):
        return {} if value is None else value


class InactiveTargetConfig(StrictConfig):
    used: StrictBool
    prob: Probability = 0.0


class EnrollmentConfig(StrictConfig):
    enroll_length_seconds: float = Field(gt=0.0)
    gain_normalized_to: float | None
    add_inactive_target: InactiveTargetConfig
    add_noise: NoiseAugmentation
    add_reverb: ReverbAugmentation
    add_volume: VolumeAugmentation


def union_bank_member_names(reverb: ReverbAugmentation | None) -> list[str]:
    """The names a union RIR bank's members are served under.

    Mirrors ``AudioEffectAugmentor.init_room_bank``: a member without a ``name``
    is served as ``bank<index>``, so a curriculum can address it either way and
    the two sides cannot drift into different naming.
    """
    simulator = reverb.simulator if reverb is not None else None
    bank = simulator.pregenerated if simulator is not None else None
    members = bank.banks if bank is not None else None
    if not members:
        return []
    return [
        member.name if member.name else f"bank{index}"
        for index, member in enumerate(members)
    ]


def _check_augmentation_target(recipe: "BaseRecipe", reference: str) -> None:
    block_name, _, remainder = reference.partition(".")
    block = getattr(recipe, block_name, None)
    if block is None:
        raise ValueError(
            f"curriculum schedules {reference!r} but this recipe has no {block_name}"
        )
    if not block.used:
        raise ValueError(
            f"curriculum schedules {reference!r} but {block_name}.used is False. "
            "Keep the block enabled and ramp its prob from 0 -- a disabled block "
            "is skipped before its random draw, so switching it on mid-run moves "
            "every draw after it."
        )
    cursor: Any = block
    for field in remainder.split("."):
        cursor = getattr(cursor, field, None)
        if cursor is None:
            raise ValueError(
                f"curriculum schedules {reference!r} but {block_name} does not set "
                f"{field!r}"
            )


def _check_bank_target(recipe: "BaseRecipe", reference: str) -> None:
    names = union_bank_member_names(getattr(recipe, "augmentation_reverb", None))
    if not names:
        raise ValueError(
            f"curriculum schedules bank weight {reference!r}, but this recipe's "
            "reverb block has no `banks:` union to weight"
        )
    if reference not in names:
        raise ValueError(
            f"curriculum schedules bank weight {reference!r}, not a member of the "
            "union: " + ", ".join(names)
        )


def _check_loss_target(recipe: "BaseRecipe", reference: str) -> None:
    losses = getattr(recipe, "loss_func", None)
    if not losses:
        raise ValueError(
            f"curriculum schedules loss {reference!r} but this recipe registers "
            "no loss_func"
        )
    if reference.startswith("#"):
        index = int(reference[1:])
        if index >= len(losses):
            raise ValueError(
                f"curriculum schedules loss {reference!r} but only {len(losses)} "
                "losses are registered"
            )
        return
    matches = [index for index, loss in enumerate(losses) if loss.type == reference]
    if not matches:
        raise ValueError(
            f"curriculum schedules loss {reference!r}, which this recipe does not "
            "register: " + ", ".join(sorted({loss.type for loss in losses}))
        )
    if len(matches) > 1:
        raise ValueError(
            f"curriculum schedules loss {reference!r}, but this recipe registers it "
            f"{len(matches)} times -- address one by index, e.g. '#{matches[0]}'"
        )


class BaseRecipe(StrictConfig):
    """Fields shared by every training recipe, independent of task."""

    schema_version: Literal[2]
    purpose: Literal["train"]
    task: TaskName
    dataset: DatasetConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    model: dict[str, Any]

    # A recipe model may intentionally leave component kwargs open: those dicts
    # are validated by the selected model/loss constructors. Pipeline control
    # flow, on the other hand, is fully typed below.
    augmentation_speech: SpeechAugmentation | None = None
    augmentation_noise: NoiseAugmentation | None = None
    augmentation_reverb: ReverbAugmentation | None = None
    augmentation_ir_response: SimpleProbAugmentation | None = None
    augmentation_src: SourceRateAugmentation | None = None
    augmentation_hpf: HighPassAugmentation | None = None
    augmentation_volume: VolumeAugmentation | None = None
    augmentation_compressor: CompressorAugmentation | None = None
    vad_label: VadLabelConfig | None = None

    # Knobs that move with the epoch instead of staying where the file put them
    # (puresound.config.curriculum). None: every knob is a constant.
    curriculum: CurriculumConfig | None = None

    @model_validator(mode="after")
    def curriculum_targets_exist(self):
        """Every track must name something this recipe actually has.

        A curriculum that points at a disabled block, a bank member that is not
        in the union, or a loss that is not registered would train perfectly
        happily and change nothing -- the whole reason this block exists is to
        stop the recipe and the run from disagreeing, so the mismatch is an
        error at load time.
        """
        curriculum = self.curriculum
        if curriculum is None or not curriculum.used:
            return self
        for track in curriculum.tracks:
            kind, reference = split_track_path(track.path)
            if kind == "aug":
                _check_augmentation_target(self, reference)
            elif kind == "bank":
                _check_bank_target(self, reference)
            else:
                _check_loss_target(self, reference)
        return self

    def augmentation_kwargs(self) -> dict[str, Any]:
        """The augmentation arguments a dataset constructor takes.

        Derived from the model's own fields, so a task that adds a block gets it
        forwarded without a second list to keep in sync -- the gap that let
        ``augmentation_realfar`` need edits in six files.
        """
        kwargs: dict[str, Any] = {}
        for name in type(self).model_fields:
            if not (name.startswith("augmentation_") or name == "vad_label"):
                continue
            block = getattr(self, name)
            if block is not None and not block.used:
                block = None
            kwargs[f"{name}_args"] = block
        return kwargs

    def curriculum_loss_indices(self) -> dict[str, int]:
        """Loss reference -> position in ``loss_func``, for the applier.

        Resolved here, once, off the same list the losses are built from: the
        thing that applies a weight at runtime holds an index, never a name it
        has to look up again against a different list.
        """
        curriculum = self.curriculum
        losses = getattr(self, "loss_func", None) or []
        if curriculum is None or not curriculum.used:
            return {}
        indices: dict[str, int] = {}
        for track in curriculum.tracks:
            kind, reference = split_track_path(track.path)
            if kind != "loss":
                continue
            if reference.startswith("#"):
                indices[reference] = int(reference[1:])
            else:
                indices[reference] = next(
                    index for index, loss in enumerate(losses) if loss.type == reference
                )
        return indices


class SisoRecipe(BaseRecipe):
    loss_func: list[LossConfig]


class NoiseSuppressionRecipe(SisoRecipe):
    task: Literal["noise_suppression"]
    augmentation_speed: ContinuousSpeedAugmentation | None = None
    augmentation_codec: CodecAugmentation | None = None
    augmentation_packet_loss: PacketLossAugmentation | None = None
    augmentation_target_absent: TargetAbsentAugmentation | None = None
    augmentation_row_initial_ambient: RowInitialAmbientAugmentation | None = None

    @model_validator(mode="after")
    def reject_isolation_mix_modes(self):
        mix_mode = self.augmentation_speech and self.augmentation_speech.mix_mode
        if mix_mode and mix_mode.used:
            raise ValueError(
                "augmentation_speech.mix_mode is only valid for voice_isolation"
            )
        return self


class VoiceIsolationRecipe(SisoRecipe):
    task: Literal["voice_isolation"]
    augmentation_speed: ContinuousSpeedAugmentation | None = None
    augmentation_codec: CodecAugmentation | None = None
    augmentation_packet_loss: PacketLossAugmentation | None = None
    augmentation_target_absent: TargetAbsentAugmentation | None = None
    augmentation_realfar: RealFarAugmentation | None = None
    augmentation_realnear: RealNearAugmentation | None = None
    augmentation_row_initial_ambient: RowInitialAmbientAugmentation | None = None
    augmentation_session_rows: SessionRowsConfig | None = None

    @model_validator(mode="after")
    def session_rows_preconditions(self):
        """Two blocks a session row cannot survive, refused at config load.

        ``is_target`` replaces the target with the mixture, which would make the
        bystanders part of what the model must keep -- the opposite of the row's
        purpose. ``augmentation_row_initial_ambient`` masks every speech
        component out of the row's opening AFTER the script has been written, so
        the turn spans would claim speech the mixture no longer has.

        The block carries the ``augmentation_`` prefix because that is what
        ``augmentation_kwargs`` forwards to a dataset as ``<name>_args``, while
        the knob inside it is spelled ``enabled``; ``SessionRowsConfig.used``
        bridges the two for the shared plumbing.
        """
        block = self.augmentation_session_rows
        if block is None or not block.enabled:
            return self
        speech = self.augmentation_speech
        if speech is not None and speech.used and speech.is_target:
            raise ValueError(
                "session_rows needs augmentation_speech.is_target: False"
            )
        ambient = self.augmentation_row_initial_ambient
        if ambient is not None and ambient.used:
            raise ValueError(
                "session_rows and augmentation_row_initial_ambient are exclusive"
            )
        return self


class SpeakerEmbeddingRecipe(SisoRecipe):
    task: Literal["speaker_embedding"]
    augmentation_speed: DiscreteSpeedAugmentation | None = None


class TargetSpeakerExtractionRecipe(BaseRecipe):
    task: Literal["target_speaker_extraction"]
    enroll_speech: EnrollmentConfig
    signal_loss_func: list[LossConfig]
    class_loss_func: list[LossConfig] | None = None
    augmentation_speed: ContinuousSpeedAugmentation | None = None


class InferenceDatasetConfig(StrictConfig):
    target_sample_rate: int = Field(gt=0)


class InferenceTrainerConfig(StrictConfig):
    work_folder: str


class InferenceRecipe(StrictConfig):
    """Model-only recipe used by demos, benchmarks and streaming exporters."""

    schema_version: Literal[2]
    purpose: Literal["inference"]
    task: TaskName
    dataset: InferenceDatasetConfig
    trainer: InferenceTrainerConfig
    model: dict[str, Any]


TrainingRecipe = (
    NoiseSuppressionRecipe
    | VoiceIsolationRecipe
    | SpeakerEmbeddingRecipe
    | TargetSpeakerExtractionRecipe
)

Recipe = TrainingRecipe | InferenceRecipe


TASK_SCHEMAS: dict[str, type[BaseRecipe]] = {
    "noise_suppression": NoiseSuppressionRecipe,
    "voice_isolation": VoiceIsolationRecipe,
    "speaker_embedding": SpeakerEmbeddingRecipe,
    "target_speaker_extraction": TargetSpeakerExtractionRecipe,
}
