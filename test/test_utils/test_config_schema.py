"""The recipe config contract: what it must reject, and what it must not break.

The regression that motivates it: `augmentation_query_distance` outlived the
mechanism it configured (380da2e) in 33 configs, because nothing checked whether
a config key was real. The same hole made `porb: 0.5` a silent
probability-zero experiment rather than an error.

Everything here goes through the real loader. There is no fragment-validation
side door any more: a recipe is validated as a whole, by the model its task
selects.
"""

import copy
import pathlib

import pytest
import yaml

from puresound.config import (
    InferenceRecipe,
    NoiseSuppressionRecipe,
    Recipe,
    RecipeConfigError,
    SpeakerEmbeddingRecipe,
    VoiceIsolationRecipe,
    load_recipe,
)
from puresound.config.loader import _parse_recipe as parse_recipe
from puresound.config.recipe import TASK_SCHEMAS
from puresound.config.augmentation import (
    ContinuousSpeedAugmentation,
    DiscreteSpeedAugmentation,
    NoiseAugmentation,
)
from puresound.dataset.dynamic_base import as_block
from puresound.recipes import init_siso_model
from puresound.system import runner

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Configs a user is expected to run as-is. These must stay clean, or the
#: schema is wrong rather than the config.
CORE_CONFIGS = (
    "egs/default_config.yaml",
    "egs/voice_isolate/config/train_dpcrn.yaml",
    "egs/noise_suppression/config/dpcrn.yaml",
    "egs/noise_suppression/config/dparn.yaml",
    "egs/speaker_embedding/conf/PS-spk-v1.yaml",
    "egs/speaker_embedding/conf/PS-spk-v1-1.yaml",
    "egs/target_speaker_extraction/config/default_config.yaml",
    "egs/voice_isolate/config/infer_dpcrn.yaml",
)

ACTIVE_CONFIGS = CORE_CONFIGS + tuple(
    str(path.relative_to(REPO_ROOT))
    for path in sorted((REPO_ROOT / "egs/voice_isolate/config/exp").glob("*.yaml"))
)

VOICE_ISOLATION = "egs/voice_isolate/config/train_dpcrn.yaml"
SPEAKER_EMBEDDING = "egs/speaker_embedding/conf/PS-spk-v1.yaml"


def _load(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text())


def _with(rel: str, **blocks) -> dict:
    """A shipped config with whole blocks replaced -- the way a real recipe is
    edited, so a fragment is always checked in the context of its task."""
    config = _load(rel)
    config.update(blocks)
    return config


# --------------------------------------------------------------------------- #
# What must keep working
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel", ACTIVE_CONFIGS)
def test_every_active_config_loads(rel):
    assert isinstance(load_recipe(REPO_ROOT / rel), Recipe)


def test_active_config_inventory_is_complete():
    assert len(ACTIVE_CONFIGS) == 35


def test_the_default_recipe_still_builds_its_model():
    # Proves the typed recipe reaches the constructors, not just the validator.
    recipe = load_recipe(REPO_ROOT / VOICE_ISOLATION)
    assert init_siso_model(recipe.model) is not None


def test_the_loader_dispatches_on_task():
    assert isinstance(load_recipe(REPO_ROOT / VOICE_ISOLATION), VoiceIsolationRecipe)
    assert isinstance(load_recipe(REPO_ROOT / SPEAKER_EMBEDDING), SpeakerEmbeddingRecipe)
    assert isinstance(
        load_recipe(REPO_ROOT / "egs/noise_suppression/config/dpcrn.yaml"),
        NoiseSuppressionRecipe,
    )
    assert isinstance(
        load_recipe(REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn.yaml"),
        InferenceRecipe,
    )


def test_model_only_recipe_contains_no_training_placeholders():
    recipe = load_recipe(REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn.yaml")
    assert not hasattr(recipe, "optimizer")
    assert not hasattr(recipe, "scheduler")
    assert not hasattr(recipe, "loss_func")


def test_discriminators_are_required_and_expectations_never_override_them():
    inference = _load("egs/voice_isolate/config/infer_dpcrn.yaml")
    inference.pop("task")
    with pytest.raises(RecipeConfigError, match="task None"):
        parse_recipe(inference, expected_task="voice_isolation")
    inference["task"] = "noise_suppression"
    with pytest.raises(RecipeConfigError, match="expected 'voice_isolation'"):
        parse_recipe(inference, expected_task="voice_isolation")

    training = _load(VOICE_ISOLATION)
    with pytest.raises(RecipeConfigError, match="expected 'inference'"):
        parse_recipe(training, expected_purpose="inference")


@pytest.mark.parametrize("missing", ["schema_version", "purpose", "task"])
def test_each_canonical_discriminator_is_required(missing):
    config = _load(VOICE_ISOLATION)
    config.pop(missing)
    with pytest.raises(RecipeConfigError):
        parse_recipe(config)


# --------------------------------------------------------------------------- #
# What must be rejected
# --------------------------------------------------------------------------- #


def test_a_retired_block_is_rejected_even_when_disabled():
    """`used: False` must not buy an exemption.

    All 33 configs carrying the retired block have it disabled. If a disabled
    block were exempt, copying one of them to start a new experiment would keep
    propagating knobs that configure nothing -- exactly the failure this exists
    to stop.
    """
    config = _with(VOICE_ISOLATION, augmentation_query_distance={"used": False})
    with pytest.raises(RecipeConfigError, match="augmentation_query_distance"):
        parse_recipe(config)


def test_unused_tse_enrollment_ir_block_is_rejected():
    config = _load("egs/target_speaker_extraction/config/default_config.yaml")
    config["enroll_speech"]["add_ir_response"] = {"used": False}
    with pytest.raises(RecipeConfigError, match="add_ir_response"):
        parse_recipe(config)


def test_a_typo_is_rejected_rather_than_silently_defaulting():
    config = _load(VOICE_ISOLATION)
    config["augmentation_speech"]["porb"] = 0.5
    with pytest.raises(RecipeConfigError, match="porb"):
        parse_recipe(config)


def test_a_typo_inside_a_nested_block_is_rejected():
    config = _load(VOICE_ISOLATION)
    config["augmentation_speech"]["overlap_control"]["turn_taking_porb"] = 0.3
    with pytest.raises(RecipeConfigError, match="turn_taking_porb"):
        parse_recipe(config)


def test_a_typo_inside_a_mix_mode_entry_is_rejected():
    config = _load(VOICE_ISOLATION)
    config["augmentation_speech"]["mix_mode"]["modes"][0]["sir_rnage"] = [0, 1]
    with pytest.raises(RecipeConfigError, match="sir_rnage"):
        parse_recipe(config)


def test_a_typo_in_a_control_section_is_rejected_too():
    config = _load(VOICE_ISOLATION)
    config["trainer"]["valid_sede"] = 1234
    with pytest.raises(RecipeConfigError, match="valid_sede"):
        parse_recipe(config)


def test_every_problem_is_reported_at_once():
    """A config with several dead knobs should say so in one run, not one per run."""
    config = _load(VOICE_ISOLATION)
    config["augmentation_speech"]["porb"] = 0.5
    config["augmentation_noise"]["snr_rnage"] = [0, 1]
    with pytest.raises(RecipeConfigError) as excinfo:
        parse_recipe(config)
    message = str(excinfo.value)
    assert "porb" in message and "snr_rnage" in message


def test_a_missing_required_key_is_rejected_when_the_block_is_on():
    config = _load(VOICE_ISOLATION)
    del config["augmentation_speech"]["snr_range"]
    with pytest.raises(RecipeConfigError, match="required when enabled: snr_range"):
        parse_recipe(config)


def test_a_missing_required_key_is_tolerated_when_the_block_is_off():
    """A disabled block never reaches the dataset, so demanding its keys would
    reject valid configs."""
    config = _load(VOICE_ISOLATION)
    config["augmentation_speech"]["used"] = False
    del config["augmentation_speech"]["snr_range"]
    parse_recipe(config)


def test_scalar_types_and_probability_ranges_are_validated():
    config = _with(
        VOICE_ISOLATION,
        augmentation_packet_loss={
            "used": "false",
            "prob": 2,
            "packet_ms_choices": "20",
            "loss_rate_range": "low-high",
        },
    )
    with pytest.raises(RecipeConfigError, match="valid boolean"):
        parse_recipe(config)


def test_numeric_strings_are_not_coerced():
    config = _load(VOICE_ISOLATION)
    config["optimizer"]["learning_rate"] = "0.001"
    config["dataset"]["filter_min_utterance_per_speaker"] = "1"
    with pytest.raises(RecipeConfigError) as excinfo:
        parse_recipe(config)
    message = str(excinfo.value)
    assert "learning_rate" in message
    assert "filter_min_utterance_per_speaker" in message


def test_codec_bitrate_keys_must_match_the_selected_codecs():
    config = _with(
        VOICE_ISOLATION,
        augmentation_codec={
            "used": True,
            "prob": 1.0,
            "codecs": ["libopus"],
            "bitrate_range": {"libops": [6000, 12000]},
        },
    )
    with pytest.raises(RecipeConfigError, match="libops"):
        parse_recipe(config)


def test_reverb_requires_one_active_source_when_enabled():
    config = _with(
        VOICE_ISOLATION,
        augmentation_reverb={"used": True, "prob": 0.5, "target_rir_type": "full"},
    )
    with pytest.raises(RecipeConfigError, match="rir_folder or an enabled simulator"):
        parse_recipe(config)


def test_a_delegated_subblock_is_still_typed_by_its_own_model():
    config = _load(VOICE_ISOLATION)
    config["augmentation_reverb"]["simulator"]["pregenerated"]["anything"] = 1
    with pytest.raises(RecipeConfigError, match="anything"):
        parse_recipe(config)


# --------------------------------------------------------------------------- #
# Task dispatch
# --------------------------------------------------------------------------- #


def test_speed_dialect_is_selected_by_task():
    """`augmentation_speed` means different things to the two tasks, and each
    accepts only the representation its dataset consumes."""
    continuous = {"used": True, "prob": 0.5, "speed_range": [0.9, 1.1]}
    discrete = {
        "used": True,
        "prob": 0.5,
        "speed_change": [0.9, 1.1],
        "treat_as_new_speaker": True,
    }
    parse_recipe(_with(VOICE_ISOLATION, augmentation_speed=continuous))
    parse_recipe(_with(SPEAKER_EMBEDDING, augmentation_speed=discrete))
    with pytest.raises(RecipeConfigError, match="speed_change"):
        parse_recipe(_with(VOICE_ISOLATION, augmentation_speed=discrete))
    with pytest.raises(RecipeConfigError, match="speed_range"):
        parse_recipe(_with(SPEAKER_EMBEDDING, augmentation_speed=continuous))


def test_a_voice_isolation_only_block_is_rejected_for_noise_suppression():
    config = _with(
        "egs/noise_suppression/config/dpcrn.yaml",
        augmentation_realfar={"used": True, "pool_manifest": "x.jsonl"},
    )
    with pytest.raises(RecipeConfigError, match="augmentation_realfar"):
        parse_recipe(config)


def test_mix_mode_is_rejected_outside_voice_isolation():
    config = _load("egs/noise_suppression/config/dpcrn.yaml")
    config["augmentation_speech"]["mix_mode"] = {
        "used": True,
        "modes": [{"name": "physical", "prob": 1.0, "physical": True}],
    }
    with pytest.raises(RecipeConfigError, match="voice_isolation"):
        parse_recipe(config)


# --------------------------------------------------------------------------- #
# The bridge into the datasets
# --------------------------------------------------------------------------- #


def test_augmentation_kwargs_covers_every_block_each_task_declares():
    """A task that adds a block must get it forwarded to its dataset without a
    second list to keep in sync -- the gap that made `augmentation_realfar` need
    edits in six files."""
    for task, model in TASK_SCHEMAS.items():
        declared = {
            name
            for name in model.model_fields
            if name.startswith("augmentation_") or name == "vad_label"
        }
        forwarded = set(model.augmentation_kwargs(model.model_construct()))
        assert forwarded == {f"{name}_args" for name in declared}, task


def test_a_disabled_block_reaches_the_dataset_as_none():
    """Several init_* steps key off "is this argument present at all" -- the
    augmentor loads a noise folder for any non-None noise block, enabled or not."""
    config = _load(VOICE_ISOLATION)
    config["augmentation_noise"]["used"] = False
    kwargs = parse_recipe(config).augmentation_kwargs()
    assert kwargs["augmentation_noise_args"] is None


def test_every_disabled_block_reaches_the_dataset_as_none():
    config = _load(VOICE_ISOLATION)
    for block in ("augmentation_hpf", "augmentation_volume", "vad_label"):
        config[block]["used"] = False
    kwargs = parse_recipe(config).augmentation_kwargs()
    for block in ("augmentation_hpf", "augmentation_volume", "vad_label"):
        assert kwargs[f"{block}_args"] is None, block


def test_as_block_revalidates_a_different_pydantic_model():
    wrong = NoiseAugmentation(used=False)
    with pytest.raises(Exception, match="noise_folder|snr_range|Extra inputs"):
        as_block(wrong, ContinuousSpeedAugmentation)


def test_each_speed_dialect_accepts_direct_mappings_at_the_dataset_boundary():
    continuous = as_block(
        {"used": True, "prob": 0.5, "speed_range": [0.9, 1.1]},
        ContinuousSpeedAugmentation,
    )
    discrete = as_block(
        {
            "used": True,
            "prob": 0.5,
            "speed_change": [0.9, 1.1],
            "treat_as_new_speaker": True,
        },
        DiscreteSpeedAugmentation,
    )
    assert continuous.speed_range == (0.9, 1.1)
    assert discrete.speed_change == [0.9, 1.1]


def test_config_defaults_match_what_the_pipeline_used_to_supply_inline():
    """Datasets read these models by attribute now, so a field left at None
    would silently replace a real default with nothing. These are the values the
    read sites used to pass to `.get(key, default)`."""
    recipe = parse_recipe(
        _with(
            VOICE_ISOLATION,
            augmentation_speech={
                "used": True,
                "prob": 1.0,
                "is_target": False,
                "add_n_cases": [1, 2],
                "snr_range": [-5, 10],
                "overlap_control": {"used": True},
                "media_voice": {"used": True},
                "echo_playback": {"used": True},
            },
        )
    )
    overlap = recipe.augmentation_speech.overlap_control
    assert overlap.fade_samples == 400
    assert overlap.no_overlap_prob == 0.25
    assert overlap.high_overlap_prob == 0.25
    assert overlap.mid_overlap_range == (0.1, 0.5)
    assert overlap.high_overlap_range == (0.5, 1.0)
    assert overlap.fill_on_silent_range == (0.3, 0.5)
    assert overlap.turn_taking_prob == 0.0
    assert overlap.turn_near_seconds == (1.5, 3.0)
    assert overlap.turn_far_seconds == (2.0, 4.5)
    assert overlap.turn_gap_seconds == (0.0, 0.4)
    assert overlap.turn_overlap_seconds == (0.0, 0.3)
    assert overlap.far_first_prob == 0.5

    media = recipe.augmentation_speech.media_voice
    assert media.hp_cutoff_range == (200.0, 400.0)
    assert media.lp_cutoff_range == (3500.0, 7000.0)
    assert media.compress_power_range == (0.6, 0.9)

    echo = recipe.augmentation_speech.echo_playback
    assert echo.distance_range == (0.2, 1.0)
    assert echo.erle_db_range == (20.0, 35.0)

    assert recipe.vad_label.frame_length == 400
    assert recipe.vad_label.hop_length == 160


def test_parsing_does_not_mutate_the_input():
    config = _load(VOICE_ISOLATION)
    before = copy.deepcopy(config)
    parse_recipe(config)
    assert config == before


def test_model_construction_does_not_mutate_the_recipe_payload():
    recipe = load_recipe(REPO_ROOT / "egs/noise_suppression/config/dparn.yaml")
    before = copy.deepcopy(recipe.model)
    init_siso_model(recipe.model)
    assert recipe.model == before


def test_shared_builder_separates_dataset_and_pipeline_roles(monkeypatch):
    instances = []

    class FakeDataset:
        def __init__(self, metafile_path, **kwargs):
            self.meta = {}
            instances.append((metafile_path, kwargs))

    monkeypatch.setattr(runner, "SpeakerSampler", lambda **kwargs: object())
    monkeypatch.setattr(
        runner.torch.utils.data, "DataLoader", lambda dataset, **kwargs: dataset
    )
    recipe = load_recipe(REPO_ROOT / VOICE_ISOLATION)
    runner.build_dataloaders(dataset_cls=FakeDataset, collate_fn=None, recipe=recipe)

    assert instances[0][1]["dataset_role"] == "train"
    assert instances[0][1]["pipeline_role"] == "train"
    assert instances[1][1]["dataset_role"] == "validation"
    assert instances[1][1]["pipeline_role"] == "validation"


def test_pipeline_role_can_explicitly_reuse_the_training_distribution():
    recipe = load_recipe(
        REPO_ROOT
        / "egs/voice_isolate/config/exp/train_dpcrn_m6bank_scratch.yaml"
    )
    assert recipe.dataset.validation_pipeline_role == "train"
