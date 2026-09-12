"""Typed recipe configuration API."""

from .base import delegated_kwargs, with_overrides
from .curriculum import CurriculumConfig, CurriculumTrack
from .loader import RecipeConfigError, load_recipe
from .recipe import (
    BaseRecipe,
    InferenceRecipe,
    NoiseSuppressionRecipe,
    Recipe,
    SisoRecipe,
    SpeakerEmbeddingRecipe,
    TargetSpeakerExtractionRecipe,
    TrainingRecipe,
    VoiceIsolationRecipe,
)

__all__ = [
    "BaseRecipe",
    "CurriculumConfig",
    "CurriculumTrack",
    "delegated_kwargs",
    "with_overrides",
    "InferenceRecipe",
    "NoiseSuppressionRecipe",
    "Recipe",
    "RecipeConfigError",
    "SisoRecipe",
    "SpeakerEmbeddingRecipe",
    "TargetSpeakerExtractionRecipe",
    "TrainingRecipe",
    "VoiceIsolationRecipe",
    "load_recipe",
]
