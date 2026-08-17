"""One canonical v2 parsing and validation entry point for every recipe task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .recipe import InferenceRecipe, Recipe, TASK_SCHEMAS, TaskName


class RecipeConfigError(ValueError):
    """Raised when a canonical recipe cannot be parsed or validated."""


def _read_recipe_yaml(path: str | Path) -> dict[str, Any]:
    """Read one canonical YAML mapping with the safe loader."""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise RecipeConfigError(f"recipe document in {source} must be a mapping")
    return dict(document)


def _parse_recipe(
    data: Mapping[str, Any],
    *,
    source: str | None = None,
    expected_task: TaskName | None = None,
    expected_purpose: str | None = None,
) -> Recipe:
    """Validate an already-loaded canonical v2 recipe mapping."""

    try:
        task = data.get("task")
        purpose = data.get("purpose")
        if expected_task is not None and task != expected_task:
            raise RecipeConfigError(
                f"recipe declares task {task!r}, expected {expected_task!r}"
            )
        if expected_purpose is not None and purpose != expected_purpose:
            raise RecipeConfigError(
                f"recipe declares purpose {purpose!r}, expected {expected_purpose!r}"
            )
        if purpose == "inference":
            schema = InferenceRecipe
        elif purpose == "train" and task in TASK_SCHEMAS:
            schema = TASK_SCHEMAS[task]
        else:
            raise RecipeConfigError(
                "recipe requires canonical schema_version, purpose and task discriminators"
            )
        return schema.model_validate(data)
    except (RecipeConfigError, ValidationError) as exc:
        where = f" in {source}" if source else ""
        raise RecipeConfigError(f"invalid recipe config{where}:\n{exc}") from exc


def load_recipe(
    path: str | Path,
    *,
    expected_task: TaskName | None = None,
    expected_purpose: str | None = None,
) -> Recipe:
    """Load a recipe into its task-specific typed model."""

    return _parse_recipe(
        _read_recipe_yaml(path),
        source=str(path),
        expected_task=expected_task,
        expected_purpose=expected_purpose,
    )
