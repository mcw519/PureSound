"""Small, dependency-light building blocks for recipe configuration models."""

from __future__ import annotations

from typing import Annotated, Any, Mapping, TypeVar

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field


class StrictConfig(BaseModel):
    """Base for every public config object.

    Config modules deliberately import neither torch nor any task implementation,
    so recipes can be linted and JSON schemas can be generated in lightweight
    tooling.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


def _yaml_pair(value: Any) -> Any:
    """YAML has sequences but no tuple syntax; normalize only that container."""

    return tuple(value) if isinstance(value, list) else value


def _ordered_pair(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError("range lower bound must not exceed upper bound")
    return value


FloatRange = Annotated[
    tuple[float, float], BeforeValidator(_yaml_pair), AfterValidator(_ordered_pair)
]
IntRange = Annotated[tuple[int, int], BeforeValidator(_yaml_pair)]

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


def delegated_kwargs(model: BaseModel) -> dict[str, Any]:
    """Keyword arguments for a component that owns its own defaults.

    Some blocks are handed wholesale to a constructor that validates and
    defaults them itself -- the room simulator, the RIR bank loaders, the VAD
    labelers, the DRR-contrast knob. ``exclude_unset`` is what makes that safe:
    only keys the recipe actually wrote are forwarded, so the component's own
    defaults still apply to everything else instead of being overwritten by this
    layer's idea of them.
    """
    return model.model_dump(mode="python", exclude_unset=True)


def with_overrides(model: ConfigModel, **changes: Any) -> ConfigModel:
    """A validated copy with these fields replaced.

    Typed control-flow models are frozen, so eval and audit scripts cannot poke
    at their fields the way they poked at the old dicts. Explicit constructor
    extension boundaries (for example ``recipe.model``) remain dictionaries,
    and object builders treat them as read-only. An override through this helper
    goes through the same validation a config file does, so
    ``training_length_seconds=-1`` fails here instead of deep inside synthesis.

    Nested blocks take a mapping and are merged field-wise, not replaced
    wholesale. Fields the recipe never set stay unset, so
    :func:`delegated_kwargs` keeps forwarding only what was written.
    """
    payload = model.model_dump(mode="python", exclude_unset=True)
    for key, value in changes.items():
        current = getattr(model, key, None)
        if isinstance(value, Mapping) and isinstance(current, BaseModel):
            payload[key] = with_overrides(current, **value).model_dump(
                mode="python", exclude_unset=True
            )
        else:
            payload[key] = value
    return type(model).model_validate(payload)


def require_fields_when_enabled(model: Any, *names: str) -> Any:
    """Shared conditional requirement for capability blocks with ``used``."""

    if not model.used:
        return model
    missing = [name for name in names if getattr(model, name) is None]
    if missing:
        raise ValueError("required when enabled: " + ", ".join(sorted(missing)))
    return model
