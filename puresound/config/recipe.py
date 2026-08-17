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
    ReverbAugmentation,
    SimpleProbAugmentation,
    SourceRateAugmentation,
    SpeechAugmentation,
    TargetAbsentAugmentation,
    VadLabelConfig,
    VolumeAugmentation,
)
from .base import Probability, StrictConfig


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
    vad_label: VadLabelConfig | None = None

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


class SisoRecipe(BaseRecipe):
    loss_func: list[LossConfig]


class NoiseSuppressionRecipe(SisoRecipe):
    task: Literal["noise_suppression"]
    augmentation_speed: ContinuousSpeedAugmentation | None = None
    augmentation_codec: CodecAugmentation | None = None
    augmentation_packet_loss: PacketLossAugmentation | None = None
    augmentation_target_absent: TargetAbsentAugmentation | None = None

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
