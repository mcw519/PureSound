"""The recipe config contract (`puresound.config`): what it must reject, and
what it must not break.

A config key that configures nothing -- a retired block, a typo like
`porb: 0.5` -- is an error, not a silent default. Everything here goes through
the real loader: a recipe is validated as a whole, by the model its task
selects.
"""

import copy
import importlib
import pathlib

import pytest
import yaml
from pydantic import ValidationError

from puresound.config import (
    InferenceRecipe,
    NoiseSuppressionRecipe,
    Recipe,
    RecipeConfigError,
    SpeakerEmbeddingRecipe,
    VoiceIsolationRecipe,
    load_recipe,
)
from puresound.config.augmentation import ContinuousSpeedAugmentation, NoiseAugmentation
from puresound.config.loader import _parse_recipe as parse_recipe
from puresound.config.recipe import TASK_SCHEMAS
from puresound.dataset.dynamic_base import as_block
from puresound.recipes import MODEL_FACTORY_FOR_TASK, init_model_for_task, init_siso_model
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

#: Everything else that ships and must still load: the follow-on recipes, the
#: evaluation configs the benchmark drives, and the recipe fixtures the test
#: suite uses.
ACTIVE_CONFIGS = CORE_CONFIGS + tuple(
    str(path.relative_to(REPO_ROOT))
    for directory in (
        "egs/voice_isolate/config",
        "egs/voice_isolate/config/eval",
        "test/fixtures/recipes",
    )
    for path in sorted((REPO_ROOT / directory).glob("*.yaml"))
    if str(path.relative_to(REPO_ROOT)) not in CORE_CONFIGS
)

VOICE_ISOLATION = "egs/voice_isolate/config/train_dpcrn.yaml"
INFERENCE = "egs/voice_isolate/config/infer_dpcrn.yaml"
NOISE_SUPPRESSION = "egs/noise_suppression/config/dpcrn.yaml"
SPEAKER_EMBEDDING = "egs/speaker_embedding/conf/PS-spk-v1.yaml"
TSE = "egs/target_speaker_extraction/config/default_config.yaml"

#: The default recipe is a curriculum: its schedules name keys inside
#: `augmentation_speech`, so that block cannot be switched off without the
#: schedule dangling. Tests about what a DISABLED block does start from a plain
#: recipe instead.
NO_CURRICULUM = "test/fixtures/recipes/train_dpcrn_no_curriculum.yaml"

DATASET_FOR_TASK = {
    "noise_suppression": ("puresound.task.ns", "NoiseSuppressionDataset"),
    "voice_isolation": ("puresound.task.voice_isolation", "VoiceIsolationDataset"),
    "target_speaker_extraction": ("puresound.task.tse", "TargetSpeakerExtractDataset"),
    "speaker_embedding": ("puresound.task.sv", "SpeakerEmbeddingDataset"),
}

CONTINUOUS_SPEED = {"used": True, "prob": 0.5, "speed_range": [0.9, 1.1]}
DISCRETE_SPEED = {
    "used": True,
    "prob": 0.5,
    "speed_change": [0.9, 1.1],
    "treat_as_new_speaker": True,
}


def _load(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text())


def _dataset_class(task):
    module, class_name = DATASET_FOR_TASK[task]
    return getattr(importlib.import_module(module), class_name)


# --------------------------------------------------------------------------- #
# What must keep working
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel", ACTIVE_CONFIGS)
def test_every_active_config_loads(rel):
    assert isinstance(load_recipe(REPO_ROOT / rel), Recipe)


@pytest.mark.parametrize(
    "rel,recipe_cls",
    [
        (VOICE_ISOLATION, VoiceIsolationRecipe),
        (SPEAKER_EMBEDDING, SpeakerEmbeddingRecipe),
        (NOISE_SUPPRESSION, NoiseSuppressionRecipe),
        (INFERENCE, InferenceRecipe),
    ],
)
def test_the_loader_dispatches_on_task_and_purpose(rel, recipe_cls):
    recipe = load_recipe(REPO_ROOT / rel)
    assert isinstance(recipe, recipe_cls)
    if recipe_cls is InferenceRecipe:
        # a model-only recipe carries no training placeholders
        for field in ("optimizer", "scheduler", "loss_func"):
            assert not hasattr(recipe, field)


def test_a_missing_required_key_is_tolerated_when_the_block_is_off():
    """A disabled block never reaches the dataset, so demanding its keys would
    reject valid configs."""
    config = _load(NO_CURRICULUM)
    config["augmentation_speech"]["used"] = False
    del config["augmentation_speech"]["snr_range"]
    parse_recipe(config)


def test_each_task_accepts_its_own_speed_dialect():
    """`augmentation_speed` means different things to the two task families,
    and each accepts only the representation its dataset consumes; the
    crossed combinations are in the rejection table below."""
    for rel, block in ((VOICE_ISOLATION, CONTINUOUS_SPEED), (SPEAKER_EMBEDDING, DISCRETE_SPEED)):
        config = _load(rel)
        config["augmentation_speed"] = block
        parse_recipe(config)


def test_neither_parsing_nor_model_construction_mutates_its_input():
    config = _load(VOICE_ISOLATION)
    before = copy.deepcopy(config)
    parse_recipe(config)
    assert config == before

    recipe = load_recipe(REPO_ROOT / "egs/noise_suppression/config/dparn.yaml")
    before = copy.deepcopy(recipe.model)
    init_siso_model(recipe.model)
    assert recipe.model == before


# --------------------------------------------------------------------------- #
# What must be rejected
# --------------------------------------------------------------------------- #


def _set(*path, value):
    def mutate(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return mutate


def _del(*path):
    def mutate(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return mutate


def _pop(key):
    return lambda config: config.pop(key)


def _all(*mutations):
    def mutate(config):
        for mutation in mutations:
            mutation(config)

    return mutate


REJECTED = {
    # `used: False` must not buy an exemption, or copying an old config keeps
    # propagating knobs that configure nothing
    "retired-block": (VOICE_ISOLATION, _set("augmentation_query_distance", value={"used": False}), {}, ["augmentation_query_distance"]),
    "retired-tse-subblock": (TSE, _set("enroll_speech", "add_ir_response", value={"used": False}), {}, ["add_ir_response"]),
    "typo": (VOICE_ISOLATION, _set("augmentation_speech", "porb", value=0.5), {}, ["porb"]),
    "typo-nested": (VOICE_ISOLATION, _set("augmentation_speech", "overlap_control", "turn_taking_porb", value=0.3), {}, ["turn_taking_porb"]),
    "typo-mix-mode-entry": (VOICE_ISOLATION, _set("augmentation_speech", "mix_mode", "modes", 0, "sir_rnage", value=[0, 1]), {}, ["sir_rnage"]),
    "typo-control-section": (VOICE_ISOLATION, _set("trainer", "valid_sede", value=1234), {}, ["valid_sede"]),
    # several dead knobs are reported in one run, not one per run
    "every-problem-at-once": (
        VOICE_ISOLATION,
        _all(_set("augmentation_speech", "porb", value=0.5),
             _set("augmentation_noise", "snr_rnage", value=[0, 1])),
        {}, ["porb", "snr_rnage"],
    ),
    "missing-required-key": (VOICE_ISOLATION, _del("augmentation_speech", "snr_range"), {}, ["required when enabled: snr_range"]),
    "scalar-types": (
        VOICE_ISOLATION,
        _set("augmentation_packet_loss", value={
            "used": "false", "prob": 2, "packet_ms_choices": "20", "loss_rate_range": "low-high",
        }),
        {}, ["valid boolean"],
    ),
    "numeric-strings-not-coerced": (
        VOICE_ISOLATION,
        _all(_set("optimizer", "learning_rate", value="0.001"),
             _set("dataset", "filter_min_utterance_per_speaker", value="1")),
        {}, ["learning_rate", "filter_min_utterance_per_speaker"],
    ),
    "codec-bitrate-keys": (
        VOICE_ISOLATION,
        _set("augmentation_codec", value={
            "used": True, "prob": 1.0, "codecs": ["libopus"],
            "bitrate_range": {"libops": [6000, 12000]},
        }),
        {}, ["libops"],
    ),
    "reverb-without-a-source": (
        VOICE_ISOLATION,
        _set("augmentation_reverb", value={"used": True, "prob": 0.5, "target_rir_type": "full"}),
        {}, ["rir_folder or an enabled simulator"],
    ),
    "delegated-subblock": (VOICE_ISOLATION, _set("augmentation_reverb", "simulator", "pregenerated", "anything", value=1), {}, ["anything"]),
    "discrete-speed-on-voice-isolation": (VOICE_ISOLATION, _set("augmentation_speed", value=DISCRETE_SPEED), {}, ["speed_change"]),
    "continuous-speed-on-speaker-embedding": (SPEAKER_EMBEDDING, _set("augmentation_speed", value=CONTINUOUS_SPEED), {}, ["speed_range"]),
    "voice-isolation-block-on-ns": (NOISE_SUPPRESSION, _set("augmentation_realfar", value={"used": True, "pool_manifest": "x.jsonl"}), {}, ["augmentation_realfar"]),
    "mix-mode-on-ns": (
        NOISE_SUPPRESSION,
        _set("augmentation_speech", "mix_mode", value={
            "used": True, "modes": [{"name": "physical", "prob": 1.0, "physical": True}],
        }),
        {}, ["voice_isolation"],
    ),
    # the discriminators are required, and an expectation never overrides them
    "no-schema-version": (VOICE_ISOLATION, _pop("schema_version"), {}, []),
    "no-purpose": (VOICE_ISOLATION, _pop("purpose"), {}, []),
    "no-task": (VOICE_ISOLATION, _pop("task"), {}, []),
    "no-task-but-expected": (INFERENCE, _pop("task"), {"expected_task": "voice_isolation"}, ["task None"]),
    "wrong-task": (INFERENCE, _set("task", value="noise_suppression"), {"expected_task": "voice_isolation"}, ["expected 'voice_isolation'"]),
    "wrong-purpose": (VOICE_ISOLATION, lambda config: None, {"expected_purpose": "inference"}, ["expected 'inference'"]),
}


@pytest.mark.parametrize("case", sorted(REJECTED))
def test_an_invalid_recipe_is_rejected_by_name(case):
    rel, mutate, parse_kwargs, expected = REJECTED[case]
    config = _load(rel)
    mutate(config)
    with pytest.raises(RecipeConfigError) as excinfo:
        parse_recipe(config, **parse_kwargs)
    message = str(excinfo.value)
    for fragment in expected:
        assert fragment in message, (fragment, message)


# --------------------------------------------------------------------------- #
# The bridge into the datasets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task", sorted(TASK_SCHEMAS))
def test_the_recipe_emits_exactly_the_blocks_its_dataset_accepts(task):
    """`runner.build_dataloaders` forwards `recipe.augmentation_kwargs()` whole.

    A block the recipe emits and the dataset does not accept is a `TypeError`
    on the first real run, not at import; a block the dataset accepts and no
    recipe emits is a knob no recipe can turn on. `augmentation_kwargs` must
    forward every declared block without a second list to keep in sync.
    """
    recipe_cls = TASK_SCHEMAS[task]
    emitted = {
        f"{name}_args"
        for name in recipe_cls.model_fields
        if name.startswith("augmentation_") or name == "vad_label"
    }
    assert set(recipe_cls.augmentation_kwargs(recipe_cls.model_construct())) == emitted

    dataset_cls = _dataset_class(task)
    accepted = set(dataset_cls.AUGMENTATION_BLOCKS)
    assert emitted - accepted == set(), (
        f"{dataset_cls.__name__} rejects {sorted(emitted - accepted)}, which "
        f"{recipe_cls.__name__} emits -- build_dataloaders would raise"
    )
    assert accepted - emitted == set(), (
        f"{dataset_cls.__name__} accepts {sorted(accepted - emitted)}, which no "
        f"{recipe_cls.__name__} can set"
    )


def test_a_disabled_block_reaches_the_dataset_as_none():
    """Several init_* steps key off "is this argument present at all" -- the
    augmentor loads a noise folder for any non-None noise block, enabled or not."""
    blocks = ("augmentation_noise", "augmentation_hpf", "augmentation_volume", "vad_label")
    config = _load(NO_CURRICULUM)
    for block in blocks:
        config[block]["used"] = False
    kwargs = parse_recipe(config).augmentation_kwargs()
    for block in blocks:
        assert kwargs[f"{block}_args"] is None, block


def test_the_dataset_registry_types_each_task_speed_dialect():
    """Same block name, disjoint keys by task: speaker embedding perturbs in
    discrete steps and may treat one as a new speaker, the separation tasks draw
    a continuous range. The registry entry decides which model a block is
    checked against, and `as_block` re-validates even a model instance of the
    wrong type."""
    key = "augmentation_speed_args"
    sv_model = _dataset_class("speaker_embedding").AUGMENTATION_BLOCKS[key]
    ns_model = _dataset_class("noise_suppression").AUGMENTATION_BLOCKS[key]

    assert as_block(DISCRETE_SPEED, sv_model).speed_change == [0.9, 1.1]
    assert as_block(CONTINUOUS_SPEED, ns_model).speed_range == (0.9, 1.1)
    with pytest.raises(ValidationError):
        as_block(CONTINUOUS_SPEED, sv_model)
    with pytest.raises(ValidationError):
        as_block(DISCRETE_SPEED, ns_model)
    with pytest.raises(Exception, match="noise_folder|snr_range|Extra inputs"):
        as_block(NoiseAugmentation(used=False), ContinuousSpeedAugmentation)


def test_a_block_the_caller_omitted_is_none_and_an_unknown_one_is_named(
    tmp_path, write_puresound_metafile
):
    """Read sites say `if self.augmentation_noise_args:`, so every registered
    block needs an attribute even on a dataset built with no augmentation. The
    registry replaced a parameter list, so a misspelling is still a TypeError
    that says what was wrong."""
    from puresound.task.voice_isolation import VoiceIsolationDataset

    dataset = VoiceIsolationDataset(
        metafile_path=str(write_puresound_metafile(tmp_path / "meta.csv")),
        min_utt_length_in_seconds=0.05,
        min_utts_in_each_speaker=1,
        target_sr=16000,
        training_sample_length_in_seconds=0.2,
    )
    for name in VoiceIsolationDataset.AUGMENTATION_BLOCKS:
        assert getattr(dataset, name) is None, name

    with pytest.raises(TypeError, match="augmentation_reverbb_args"):
        _dataset_class("speaker_embedding")(
            metafile_path="unused", augmentation_reverbb_args={}
        )


@pytest.mark.parametrize(
    "rel,validation_pipeline_role",
    [
        (VOICE_ISOLATION, "validation"),
        # a recipe can explicitly validate on the training distribution
        ("test/fixtures/recipes/train_dpcrn_validation_pipeline_train.yaml", "train"),
    ],
)
def test_the_shared_builder_separates_dataset_and_pipeline_roles(
    monkeypatch, rel, validation_pipeline_role
):
    instances = []

    class FakeDataset:
        def __init__(self, metafile_path, **kwargs):
            self.meta = {}
            instances.append(kwargs)

    monkeypatch.setattr(runner, "SpeakerSampler", lambda **kwargs: object())
    monkeypatch.setattr(runner.torch.utils.data, "DataLoader", lambda dataset, **kwargs: dataset)
    runner.build_dataloaders(
        dataset_cls=FakeDataset, collate_fn=None, recipe=load_recipe(REPO_ROOT / rel)
    )

    assert (instances[0]["dataset_role"], instances[0]["pipeline_role"]) == ("train", "train")
    assert instances[1]["dataset_role"] == "validation"
    assert instances[1]["pipeline_role"] == validation_pipeline_role


@pytest.mark.parametrize(
    "rel", [NOISE_SUPPRESSION, VOICE_ISOLATION, TSE, SPEAKER_EMBEDDING]
)
def test_the_task_resolves_the_one_model_factory_that_builds_its_model(rel):
    """A task's `model` block fits only one of the two module shapes -- the
    conditioned one reads `c_encoder` / `c_backbone` -- so the factory is
    resolved from `recipe.task`, and the other factory must not also work, or
    the table would pin nothing."""
    assert set(MODEL_FACTORY_FOR_TASK) == set(TASK_SCHEMAS)
    with pytest.raises(KeyError, match="no model factory"):
        init_model_for_task("not_a_task")

    recipe = load_recipe(str(REPO_ROOT / rel))
    factory = init_model_for_task(recipe.task)
    assert factory(recipe.model) is not None
    for other in set(MODEL_FACTORY_FOR_TASK.values()) - {factory}:
        with pytest.raises((KeyError, TypeError)):
            other(recipe.model)


def test_dataset_kwargs_are_the_constructor_arguments_the_runner_passes():
    from puresound.config import load_recipe

    recipe = load_recipe("egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml", expected_task="noise_suppression")
    kwargs = recipe.dataset_kwargs()
    assert kwargs["target_sr"] == 16000 and kwargs["training_sample_length_in_seconds"] == 6.0
    assert kwargs["augmentation_noise_args"] is recipe.augmentation_noise
    assert {"min_utt_length_in_seconds", "min_utts_in_each_speaker", "audio_gain_normalized_to"} <= set(kwargs)
