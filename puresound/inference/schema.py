"""Strict, serialisable schemas used by the local PureSound model zoo.

The catalog deliberately contains metadata and relative paths only.  Model
weights remain in the task-specific ``egs`` directories, while this module
provides one validated description of how each artifact is consumed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from puresound.config.base import StrictConfig


TaskName = Literal[
    "noise_suppression",
    "voice_isolation",
    "target_speaker_extraction",
    "speaker_embedding",
]
Lifecycle = Literal[
    "released",
    "candidate",
    "reference",
    "experimental",
    "historical",
]
Role = Literal[
    "default",
    "alternative",
    "fallback",
    "reference",
    "candidate",
    "diagnostic",
    "historical",
]
ProcessorName = Literal["stft_frame_ort", "waveform_embedding_ort"]


class ParameterSpec(StrictConfig):
    """One request-overridable inference parameter."""

    type: Literal["float", "int", "bool", "string"]
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: list[Any] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ParameterSpec":
        valid_type = {
            "float": (float, int),
            "int": (int,),
            "bool": (bool,),
            "string": (str,),
        }[self.type]
        if isinstance(self.default, bool) and self.type in {"float", "int"}:
            raise ValueError(f"parameter default for {self.type} must not be boolean")
        if not isinstance(self.default, valid_type):
            raise ValueError(
                f"parameter default must be {self.type}, got {type(self.default).__name__}"
            )
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("parameter minimum must not exceed maximum")
        if self.choices is not None and self.default not in self.choices:
            raise ValueError("parameter default must be one of choices")
        return self


class AudioSpec(StrictConfig):
    sample_rate: int = Field(gt=0)
    channels: int = Field(default=1, gt=0)
    target_dbfs: float | None = None
    accepted_sample_rates: list[int] = Field(default_factory=list)


class ArtifactSpec(StrictConfig):
    """A concrete ONNX file and the processor contract that consumes it."""

    format: Literal["onnx"] = "onnx"
    path: str = Field(min_length=1)
    variant: str = Field(default="default", min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    processor: ProcessorName
    manifest: str | None = None
    input_names: list[str] = Field(default_factory=list)
    output_names: list[str] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def validate_ports(self) -> "ArtifactSpec":
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("artifact input_names must be unique")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("artifact output_names must be unique")
        if self.processor == "stft_frame_ort" and not self.manifest:
            raise ValueError("stft_frame_ort artifacts require a sidecar manifest")
        return self

    @property
    def sidecar_manifest(self) -> str | None:
        """Readable alias for clients that use the catalog terminology."""
        return self.manifest


class ModelSpec(StrictConfig):
    """Logical model metadata; one model may have multiple artifact variants."""

    id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    display_name: str = Field(min_length=1)
    task: TaskName
    lifecycle: Lifecycle
    roles: list[Role] = Field(default_factory=list)
    description: str | None = None
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    audio: AudioSpec | None = None
    capabilities: dict[str, bool] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    postprocessing: dict[str, Any] = Field(default_factory=dict)
    recommended_inference: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    source_checkpoint: str | None = None
    source_config: str | None = None
    benchmark_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_model(self) -> "ModelSpec":
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("model roles must be unique")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("model inputs must be unique")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("model outputs must be unique")
        variants = [artifact.variant for artifact in self.artifacts]
        if len(set(variants)) != len(variants):
            raise ValueError("artifact variants must be unique within a model")
        if "default" in self.roles and not self.artifacts:
            raise ValueError("a default model must expose at least one artifact")
        return self

    @property
    def sample_rate(self) -> int | None:
        return self.audio.sample_rate if self.audio else None

    @property
    def channels(self) -> int | None:
        return self.audio.channels if self.audio else None

    @property
    def recommended_parameters(self) -> dict[str, Any]:
        return self.recommended_inference


class Catalog(StrictConfig):
    """Top-level model-zoo document."""

    schema_version: Literal[1] = 1
    models: list[ModelSpec] = Field(default_factory=list, min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> "Catalog":
        ids = [model.id for model in self.models]
        if len(set(ids)) != len(ids):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError("duplicate model id(s): " + ", ".join(duplicates))
        for task in {
            "noise_suppression",
            "voice_isolation",
            "target_speaker_extraction",
            "speaker_embedding",
        }:
            defaults = [
                model.id
                for model in self.models
                if model.task == task and "default" in model.roles and model.artifacts
            ]
            if len(defaults) > 1:
                raise ValueError(
                    f"task {task!r} has more than one default model: "
                    + ", ".join(defaults)
                )
        return self


# Descriptive aliases keep the schema pleasant to use in integrations while
# preserving the short names used by the YAML loader.
ModelZooCatalog = Catalog
LogicalModelSpec = ModelSpec


__all__ = [
    "ArtifactSpec",
    "AudioSpec",
    "Catalog",
    "LogicalModelSpec",
    "ModelSpec",
    "ModelZooCatalog",
    "ParameterSpec",
]
