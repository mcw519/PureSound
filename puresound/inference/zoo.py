"""Versioned, local Model Zoo loading and validation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import ArtifactSpec, Catalog, ModelSpec


class ModelZooError(RuntimeError):
    """Base error raised while loading or resolving the catalog."""


class ModelZooValidationError(ModelZooError):
    """Raised when a catalog or one of its artifacts is not deployable."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("Model Zoo validation failed:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class ValidationReport:
    models: int
    artifacts: int
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok


def _repository_root(catalog_path: Path) -> Path:
    """Resolve paths relative to the repository containing the catalog.

    The checked-in catalog lives in ``<repo>/model_zoo/catalog.yaml``.  For
    tests and downstream packages a catalog may be placed directly in a root
    directory; both layouts are accepted without changing the serialized
    paths in the document.
    """

    return catalog_path.parent.parent if catalog_path.parent.name == "model_zoo" else catalog_path.parent


class ModelZoo:
    """Read-only access to the repository-local model registry."""

    def __init__(self, catalog: Catalog, catalog_path: str | Path):
        self.catalog = catalog
        self.catalog_path = Path(catalog_path).resolve()
        self.root = _repository_root(self.catalog_path)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelZoo":
        catalog_path = Path(path).expanduser().resolve()
        if not catalog_path.is_file():
            raise ModelZooError(f"model zoo catalog not found: {catalog_path}")
        try:
            raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog = Catalog.model_validate(raw or {})
        except Exception as exc:  # pydantic/yaml errors are part of the public failure
            raise ModelZooError(f"invalid model zoo catalog {catalog_path}: {exc}") from exc
        return cls(catalog, catalog_path)

    load = from_file

    @classmethod
    def default(cls) -> "ModelZoo":
        """Load the checked-in catalog, optionally overridden for deployments."""

        configured = os.environ.get("PURESOUND_MODEL_ZOO")
        path = Path(configured).expanduser() if configured else None
        if path is None:
            path = Path(__file__).resolve().parents[2] / "model_zoo" / "catalog.yaml"
            if not path.is_file():
                cwd_path = Path.cwd() / "model_zoo" / "catalog.yaml"
                if cwd_path.is_file():
                    path = cwd_path
        return cls.from_file(path)

    @property
    def schema_version(self) -> int:
        return self.catalog.schema_version

    @property
    def models(self) -> list[ModelSpec]:
        return list(self.catalog.models)

    def list(
        self,
        task: str | None = None,
        *,
        runnable_only: bool = True,
    ) -> list[ModelSpec]:
        task = {
            "sv": "speaker_embedding",
            "speaker_verification": "speaker_embedding",
            "ns": "noise_suppression",
            "tse": "target_speaker_extraction",
        }.get(task, task)
        models = [model for model in self.catalog.models if task is None or model.task == task]
        if runnable_only:
            models = [model for model in models if model.artifacts]
        return sorted(models, key=lambda model: model.id)

    def get(self, model_id: str) -> ModelSpec:
        aliases = {
            "dpcrn_v8": "voice-isolate-dpcrn-v8",
            "dpcrn_curriculum_v1": "voice-isolate-dpcrn-curriculum-v1",
            "PS-spk-v1": "speaker-verification-ps-spk-v1",
            "PS-spk-v1-1": "speaker-verification-ps-spk-v1-1",
            "ps-spk-v1": "speaker-verification-ps-spk-v1",
            "ps-spk-v1-1": "speaker-verification-ps-spk-v1-1",
            "sv-ps-spk-v1": "speaker-verification-ps-spk-v1",
            "sv-ps-spk-v1-1": "speaker-verification-ps-spk-v1-1",
        }
        model_id = aliases.get(model_id, model_id)
        for model in self.catalog.models:
            if model.id == model_id:
                return model
        raise KeyError(f"unknown model id: {model_id}")

    def default_model(self, task: str) -> ModelSpec:
        """Return the logical model marked ``default`` for a task."""
        models = [model for model in self.list(task=task) if "default" in model.roles]
        if not models:
            raise ModelZooError(f"no default runnable model registered for task {task!r}")
        if len(models) > 1:
            raise ModelZooError(
                f"multiple default runnable models registered for task {task!r}: "
                + ", ".join(model.id for model in models)
            )
        return models[0]

    get_default = default_model

    def resolve_artifact(self, model_id: str, variant: str | None = None) -> ArtifactSpec:
        model = self.get(model_id)
        if not model.artifacts:
            raise ModelZooError(f"model {model_id!r} has no runnable artifact")
        if variant is not None:
            for artifact in model.artifacts:
                if artifact.variant == variant:
                    return artifact
            raise KeyError(f"model {model_id!r} has no artifact variant {variant!r}")
        for artifact in model.artifacts:
            if artifact.variant == "default":
                return artifact
        return model.artifacts[0]

    def path_for(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self.root / path

    def artifact_path(self, model_id: str, variant: str | None = None) -> Path:
        return self.path_for(self.resolve_artifact(model_id, variant).path)

    def inspect(self, model_id: str) -> ModelSpec:
        """Return one logical model (CLI uses ``get`` internally)."""
        return self.get(model_id)

    def manifest_path(self, model_id: str, variant: str | None = None) -> Path | None:
        manifest = self.resolve_artifact(model_id, variant).manifest
        return self.path_for(manifest) if manifest else None

    def find_by_artifact(self, path: str | Path) -> tuple[ModelSpec, ArtifactSpec] | None:
        """Find a model/variant by ONNX or source-checkpoint path.

        This bridge is intentionally small and exists for existing Gradio code
        that stores paths in component values.  New callers should use a
        logical model id.
        """

        candidate = Path(path)
        candidates = {candidate.resolve()}
        if not candidate.is_absolute():
            candidates.add((self.root / candidate).resolve())
        for model in self.catalog.models:
            if model.source_checkpoint:
                source = self.path_for(model.source_checkpoint).resolve()
                if source in candidates and model.artifacts:
                    return model, model.artifacts[0]
            for artifact in model.artifacts:
                if self.path_for(artifact.path).resolve() in candidates:
                    return model, artifact
        return None

    def validate(
        self,
        *,
        verify_hash: bool = True,
        check_graph: bool = True,
        raise_on_error: bool = True,
    ) -> ValidationReport:
        """Validate paths, sidecars, hashes and ONNX tensor contracts.

        ``check_graph=False`` is useful for lightweight catalog linting where
        ONNX Runtime is not installed.  The command-line validator keeps it
        enabled by default so a green result means the artifacts can load.
        """

        errors: list[str] = []
        artifact_count = 0
        known_processors = {"stft_frame_ort", "waveform_embedding_ort"}
        seen_artifact_paths: dict[Path, str] = {}
        for model in self.catalog.models:
            for provenance_name in ("source_checkpoint", "source_config"):
                provenance = getattr(model, provenance_name)
                if provenance and not self.path_for(provenance).is_file():
                    errors.append(
                        f"{model.id}: {provenance_name.replace('_', ' ')} not found: "
                        f"{self.path_for(provenance)}"
                    )
            for artifact in model.artifacts:
                artifact_count += 1
                prefix = f"{model.id}/{artifact.variant}"
                if artifact.processor not in known_processors:
                    errors.append(f"{prefix}: unsupported processor {artifact.processor!r}")
                expected_processor = {
                    "voice_isolation": "stft_frame_ort",
                    "speaker_embedding": "waveform_embedding_ort",
                }.get(model.task)
                if expected_processor and artifact.processor != expected_processor:
                    errors.append(
                        f"{prefix}: processor {artifact.processor!r} is incompatible "
                        f"with task {model.task!r} (expected {expected_processor!r})"
                    )
                if artifact.processor == "stft_frame_ort" and not model.capabilities.get(
                    "realtime_streaming", False
                ):
                    errors.append(
                        f"{prefix}: stft_frame_ort requires realtime_streaming capability"
                    )
                if artifact.processor == "waveform_embedding_ort" and not model.capabilities.get(
                    "speaker_verification", False
                ):
                    errors.append(
                        f"{prefix}: waveform_embedding_ort requires speaker_verification capability"
                    )
                onnx_path = self.path_for(artifact.path)
                resolved_onnx = onnx_path.resolve()
                previous = seen_artifact_paths.get(resolved_onnx)
                if previous is not None:
                    errors.append(
                        f"{prefix}: artifact path duplicates {previous}: {resolved_onnx}"
                    )
                else:
                    seen_artifact_paths[resolved_onnx] = prefix
                if not onnx_path.is_file():
                    errors.append(f"{prefix}: ONNX file not found: {onnx_path}")
                    continue
                if verify_hash and artifact.sha256:
                    digest = _sha256(onnx_path)
                    if digest.lower() != artifact.sha256.lower():
                        errors.append(
                            f"{prefix}: SHA256 mismatch (catalog {artifact.sha256}, file {digest})"
                        )
                manifest: dict[str, Any] | None = None
                if artifact.manifest:
                    manifest_path = self.path_for(artifact.manifest)
                    if not manifest_path.is_file():
                        errors.append(f"{prefix}: sidecar manifest not found: {manifest_path}")
                    else:
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        except Exception as exc:
                            errors.append(f"{prefix}: invalid sidecar manifest: {exc}")
                        if manifest is not None:
                            manifest_onnx = manifest.get("onnx_path")
                            if manifest_onnx and Path(str(manifest_onnx)).name != onnx_path.name:
                                errors.append(
                                    f"{prefix}: sidecar onnx_path {manifest_onnx!r} "
                                    f"does not describe {onnx_path.name!r}"
                                )
                            if manifest.get("processor") != artifact.processor:
                                errors.append(
                                    f"{prefix}: sidecar processor {manifest.get('processor')!r} "
                                    f"does not match {artifact.processor!r}"
                                )
                            if model.audio and manifest.get("sample_rate") is not None:
                                if int(manifest["sample_rate"]) != model.audio.sample_rate:
                                    errors.append(
                                        f"{prefix}: sidecar sample_rate {manifest.get('sample_rate')} "
                                        f"does not match catalog {model.audio.sample_rate}"
                                    )
                            catalog_dry = model.recommended_inference.get("dry_blend")
                            sidecar_dry = manifest.get("recommended_inference", {}).get("dry_blend")
                            if catalog_dry is not None and sidecar_dry is not None:
                                if abs(float(catalog_dry) - float(sidecar_dry)) > 1e-9:
                                    errors.append(
                                        f"{prefix}: catalog dry_blend {catalog_dry} does not "
                                        f"match sidecar {sidecar_dry}"
                                    )
                if not check_graph:
                    continue
                try:
                    import onnxruntime

                    session = onnxruntime.InferenceSession(
                        str(onnx_path), providers=["CPUExecutionProvider"]
                    )
                    session_inputs = list(session.get_inputs())
                    session_outputs = list(session.get_outputs())
                    actual_inputs = [item.name for item in session_inputs]
                    actual_outputs = [item.name for item in session_outputs]
                    expected_inputs = artifact.input_names or (
                        list(manifest.get("input_names", [])) if manifest else []
                    )
                    expected_outputs = artifact.output_names or (
                        list(manifest.get("output_names", [])) if manifest else []
                    )
                    if expected_inputs and actual_inputs != expected_inputs:
                        errors.append(
                            f"{prefix}: input names mismatch (catalog {expected_inputs}, graph {actual_inputs})"
                        )
                    if expected_outputs and actual_outputs != expected_outputs:
                        errors.append(
                            f"{prefix}: output names mismatch (catalog {expected_outputs}, graph {actual_outputs})"
                        )
                    if artifact.processor == "stft_frame_ort" and manifest:
                        for key in ("state_input_names", "state_output_names", "state_shapes"):
                            if key not in manifest:
                                errors.append(f"{prefix}: sidecar missing {key}")
                        if session_inputs:
                            noisy_shape = session_inputs[0].shape
                            if len(noisy_shape) != 3 or noisy_shape[-1] not in (2, "2"):
                                errors.append(
                                    f"{prefix}: noisy_frame must have shape [batch, freq, 2], got {noisy_shape}"
                                )
                        if session_outputs:
                            enhanced_shape = session_outputs[0].shape
                            if len(enhanced_shape) != 3 or enhanced_shape[-1] not in (2, "2"):
                                errors.append(
                                    f"{prefix}: enhanced_frame must have shape [batch, freq, 2], got {enhanced_shape}"
                                )
                    if artifact.processor == "waveform_embedding_ort":
                        if actual_inputs != ["Audio"] and not artifact.input_names:
                            errors.append(f"{prefix}: waveform embedding must expose input 'Audio'")
                        if actual_outputs != ["Embedding"] and not artifact.output_names:
                            errors.append(f"{prefix}: waveform embedding must expose output 'Embedding'")
                        if session_inputs and len(session_inputs[0].shape) != 2:
                            errors.append(
                                f"{prefix}: Audio input must have shape [batch, samples], got {session_inputs[0].shape}"
                            )
                        if session_outputs:
                            embedding_shape = session_outputs[0].shape
                            if len(embedding_shape) != 2 or embedding_shape[-1] not in (192, "192"):
                                errors.append(
                                    f"{prefix}: Embedding output must have shape [batch, 192], got {embedding_shape}"
                                )
                except Exception as exc:
                    errors.append(f"{prefix}: ONNX Runtime load failed: {exc}")
        report = ValidationReport(
            models=len(self.catalog.models),
            artifacts=artifact_count,
            errors=tuple(errors),
        )
        if errors and raise_on_error:
            raise ModelZooValidationError(errors)
        return report


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_catalog(
    path: str | Path | None = None,
    *,
    verify_hash: bool = True,
    check_graph: bool = True,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Convenience wrapper used by scripts and integrations."""
    zoo = ModelZoo.default() if path is None else ModelZoo.from_file(path)
    return zoo.validate(
        verify_hash=verify_hash,
        check_graph=check_graph,
        raise_on_error=raise_on_error,
    )


__all__ = [
    "ModelZoo",
    "ModelZooError",
    "ModelZooValidationError",
    "ValidationReport",
    "validate_catalog",
]
