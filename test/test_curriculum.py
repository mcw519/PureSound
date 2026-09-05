"""``curriculum`` -- knobs that move with the epoch.

The block lets one run change a knob where a recipe otherwise states it once, so
the things worth guarding are the ones that would make that substitution
dishonest:

1. **A recipe without a curriculum is untouched**, item shape and RNG stream
   included -- otherwise no run before this is comparable with one after it.
2. **A schedule that resolves to the recipe's own constants changes nothing.**
   The ramp is the only difference between the two, never the plumbing.
3. **The epoch is told, not counted.** A run resumed at epoch N must not replay
   the schedule from 0, which is what an internally counted epoch would do.
4. **A target that does not exist is an error at load time.** A curriculum
   pointing at a disabled block or a missing loss trains perfectly happily and
   does nothing at all; that is the failure this block exists to prevent.
"""

import pathlib
import random

import numpy as np
import pytest
import soundfile as sf
import yaml
from pydantic import ValidationError

from puresound.audio.rir.bank.loader import UnionRoomBank
from puresound.config.curriculum import CurriculumConfig
from puresound.config.loader import _parse_recipe as parse_recipe
from puresound.config import RecipeConfigError
from puresound.task.sampler import SpeakerSampler
from puresound.task.voice_isolation import VoiceIsolationDataset


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOICE_ISOLATION = "egs/voice_isolate/config/train_dpcrn.yaml"
SR = 16000


def _curriculum(*tracks, used=True):
    return CurriculumConfig.model_validate({"used": used, "tracks": list(tracks)})


# --------------------------------------------------------------------------- #
# the schedule itself
# --------------------------------------------------------------------------- #


def test_linear_ramps_between_points_and_holds_outside_them():
    track = _curriculum(
        {"path": "loss:OverSuppressionLoss", "points": [[10, 0], [30, 3]]}
    ).tracks[0]
    assert track.value_at(0) == 0.0, "before the first point the value is held"
    assert track.value_at(10) == 0.0
    assert track.value_at(20) == pytest.approx(1.5)
    assert track.value_at(30) == 3.0
    assert track.value_at(999) == 3.0, "a track never extrapolates"


def test_step_holds_each_value_until_the_next_point_is_reached():
    track = _curriculum(
        {
            "path": "aug:augmentation_realfar.prob",
            "interp": "step",
            "points": [[0, 0.0], [40, 0.2]],
        }
    ).tracks[0]
    assert [track.value_at(e) for e in (0, 39, 40, 41)] == [0.0, 0.0, 0.2, 0.2]


def test_a_single_point_is_a_constant():
    track = _curriculum({"path": "bank:core", "points": [[0, 0.5]]}).tracks[0]
    assert track.value_at(0) == track.value_at(500) == 0.5


def test_values_are_grouped_by_who_applies_them():
    values = _curriculum(
        {"path": "aug:augmentation_speech.media_voice.prob", "points": [[0, 0.3]]},
        {"path": "bank:core", "points": [[0, 0.5]]},
        {"path": "loss:SDRLoss", "points": [[0, 1.0]]},
    ).resolve(0)
    assert values.augmentation == {"augmentation_speech.media_voice.prob": 0.3}
    assert values.bank_weights == {"core": 0.5}
    assert values.loss_weights == {"SDRLoss": 1.0}
    assert values.augmentation_overrides() == {
        "augmentation_speech": {"media_voice": {"prob": 0.3}}
    }


def test_a_disabled_curriculum_resolves_to_nothing():
    values = _curriculum(
        {"path": "loss:SDRLoss", "points": [[0, 1.0]]}, used=False
    ).resolve(0)
    assert values.loss_weights == {}


@pytest.mark.parametrize(
    "track, message",
    [
        ({"path": "augmentation_realfar.prob", "points": [[0, 0]]}, "must be"),
        ({"path": "rir:core", "points": [[0, 0]]}, "unknown curriculum path kind"),
        ({"path": "aug:dataset.training_length_seconds", "points": [[0, 6]]},
         "not a schedulable augmentation knob"),
        ({"path": "aug:augmentation_realfar.used", "points": [[0, 1]]},
         "not a schedulable augmentation knob"),
        ({"path": "loss:#x", "points": [[0, 1]]}, "loss index must be"),
        ({"path": "bank:core", "points": [[30, 1], [10, 2]]}, "ascend by epoch"),
        ({"path": "bank:core", "points": [[0, "loud"]]}, "must be a number"),
        ({"path": "bank:core", "points": [[0, 1, 2]]}, "point pair"),
    ],
)
def test_a_malformed_track_is_refused(track, message):
    with pytest.raises(ValidationError, match=message):
        _curriculum(track)


def test_the_same_knob_cannot_be_scheduled_twice():
    with pytest.raises(ValidationError, match="scheduled twice"):
        _curriculum(
            {"path": "bank:core", "points": [[0, 1]]},
            {"path": "bank:core", "points": [[0, 2]]},
        )


# --------------------------------------------------------------------------- #
# targets are checked against the recipe that owns them
# --------------------------------------------------------------------------- #


def _recipe_dict(*tracks):
    config = yaml.safe_load((REPO_ROOT / VOICE_ISOLATION).read_text())
    config["curriculum"] = {"used": True, "tracks": list(tracks)}
    return config


def test_a_recipe_accepts_a_curriculum_over_knobs_it_has():
    recipe = parse_recipe(
        _recipe_dict(
            {"path": "aug:augmentation_realfar.prob", "points": [[0, 0.0], [40, 0.2]]},
            {"path": "loss:OverSuppressionLoss", "points": [[0, 0.0], [30, 3.0]]},
        )
    )
    assert recipe.curriculum is not None
    weights = [loss.weighted for loss in recipe.loss_func]
    index = recipe.curriculum_loss_indices()["OverSuppressionLoss"]
    assert weights[index] == 3.0, "the index must point at the loss it names"


def test_scheduling_a_disabled_block_is_refused():
    # augmentation_target_absent ships disabled, and a disabled block never
    # reaches its probability draw -- ramping its prob would do nothing.
    with pytest.raises(RecipeConfigError, match="used is False"):
        parse_recipe(
            _recipe_dict(
                {"path": "aug:augmentation_target_absent.prob", "points": [[0, 0.1]]}
            )
        )


def test_scheduling_a_bank_weight_without_a_union_is_refused():
    with pytest.raises(RecipeConfigError, match="no `banks:` union"):
        parse_recipe(_recipe_dict({"path": "bank:core", "points": [[0, 0.5]]}))


def test_bank_members_are_addressable_by_name_and_by_position():
    config = _recipe_dict({"path": "bank:core", "points": [[0, 0.5], [40, 0.1]]})
    bank = config["augmentation_reverb"]["simulator"]["pregenerated"]
    folder = bank.pop("folder")
    bank["banks"] = [
        {"name": "core", "folder": folder, "near_labels": ["near_0"],
         "far_labels": ["far_0"]},
        {"folder": folder, "near_labels": ["near_0"], "far_labels": ["far_0"]},
    ]
    assert parse_recipe(config).curriculum is not None

    config["curriculum"]["tracks"] = [{"path": "bank:bank1", "points": [[0, 0.5]]}]
    assert parse_recipe(config).curriculum is not None, "unnamed member is bank<index>"

    config["curriculum"]["tracks"] = [{"path": "bank:wide", "points": [[0, 0.5]]}]
    with pytest.raises(RecipeConfigError, match="not a member of the union"):
        parse_recipe(config)


def test_scheduling_a_loss_the_recipe_does_not_register_is_refused():
    with pytest.raises(RecipeConfigError, match="does not\n?\\s*register"):
        parse_recipe(_recipe_dict({"path": "loss:HuberLoss", "points": [[0, 1.0]]}))


def test_an_ambiguous_loss_reference_is_refused_with_the_index_to_use():
    # The shipped recipe stacks two ASRFeatureLoss terms; naming the type alone
    # would silently weight whichever came first.
    with pytest.raises(RecipeConfigError, match="address one by index"):
        parse_recipe(_recipe_dict({"path": "loss:ASRFeatureLoss", "points": [[0, 1.0]]}))

    recipe = parse_recipe(
        _recipe_dict({"path": "loss:#5", "points": [[0, 1.0], [20, 0.5]]})
    )
    assert recipe.curriculum_loss_indices() == {"#5": 5}


def test_a_recipe_without_a_curriculum_still_loads():
    recipe = parse_recipe(yaml.safe_load((REPO_ROOT / VOICE_ISOLATION).read_text()))
    assert recipe.curriculum is None
    assert recipe.curriculum_loss_indices() == {}


# --------------------------------------------------------------------------- #
# the epoch reaches the workers
# --------------------------------------------------------------------------- #


def _meta(n_spk=32):
    return {
        f"spk{i}": {"utts": {f"u{i}_{j}": {"sr": SR} for j in range(2)}}
        for i in range(n_spk)
    }


def _sampler(**kw):
    return SpeakerSampler(data=_meta(), total_batch=4, n_spks=8, n_per=1, **kw)


def test_without_a_curriculum_the_item_shape_is_unchanged():
    random.seed(0)
    assert all(len(e) == 2 for b in _sampler() for e in b)


def test_every_item_carries_the_epoch_when_one_is_scheduled():
    sampler = _sampler(emit_epoch=True)
    for expected in (0, 1, 2):
        for batch in sampler:
            assert all(len(e) == 5 and e[4] == expected for e in batch)
            assert all(e[3] is None for e in batch), "no length schedule, empty slot"


def test_a_length_schedule_and_a_curriculum_share_one_item():
    sampler = _sampler(emit_epoch=True, length_schedule=[(3.0, 8, 1.0)])
    for batch in sampler:
        assert all(e[3] == 3.0 and e[4] == 0 for e in batch)


def test_the_epoch_is_told_not_counted():
    """The resume case: counting passes internally replays the schedule."""
    sampler = _sampler(emit_epoch=True)
    sampler.set_epoch(40)
    assert [b[0][4] for b in sampler] == [40] * 4
    assert [b[0][4] for b in sampler] == [40] * 4, "no drift while nobody says so"
    sampler.set_epoch(41)
    assert [b[0][4] for b in sampler] == [41] * 4


def test_lightning_reaches_set_epoch_through_the_batch_sampler():
    """The contract that makes a resumed run follow its schedule.

    Lightning calls ``set_epoch`` on ``dataloader.sampler`` and
    ``dataloader.batch_sampler.sampler`` before an epoch's iterator is consumed.
    This sampler IS the batch sampler, so it exposes itself as its own inner
    sampler; if that ever stops working, a resumed run silently trains epoch 0's
    recipe.
    """
    import torch
    from lightning.fabric.utilities.data import _set_sampler_epoch

    sampler = _sampler(emit_epoch=True)
    loader = torch.utils.data.DataLoader(
        dataset=[0] * 64, batch_sampler=sampler, collate_fn=lambda b: b
    )
    _set_sampler_epoch(loader, 7)
    assert next(iter(sampler))[0][4] == 7


# --------------------------------------------------------------------------- #
# the knobs actually move (and cost nothing when they do not)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("curriculum_corpus")
    rows = ["uttid, spkid, gender, path, length, sample rate, channels"]
    for speaker in range(4):
        path = root / f"spk{speaker}.wav"
        time = np.arange(int(SR * 8.0)) / SR
        wav = 0.05 * np.sin(2 * np.pi * (180.0 + 40.0 * speaker) * time)
        sf.write(str(path), wav.astype("float32"), SR)
        rows.append(f"spk{speaker}_0, spk{speaker}, m, {path}, {int(SR * 8.0)}, {SR}, 1")
    meta = root / "meta.csv"
    meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return meta


_SPEECH_ARGS = {
    "used": True,
    "is_target": False,
    "prob": 0.7,
    "add_n_cases": [1, 1],
    "snr_range": [-5.0, 5.0],
    "media_voice": {"used": True, "prob": 0.3},
    "overlap_control": {"used": True, "no_overlap_prob": 0.1},
}


def _dataset(corpus, curriculum=None, **blocks):
    args = {"augmentation_speech_args": _SPEECH_ARGS}
    args.update({f"augmentation_{name}_args": block for name, block in blocks.items()})
    return VoiceIsolationDataset(
        metafile_path=str(corpus),
        min_utt_length_in_seconds=1.0,
        min_utts_in_each_speaker=1,
        target_sr=SR,
        training_sample_length_in_seconds=2.0,
        audio_gain_normalized_to=-28,
        curriculum=curriculum,
        **args,
    )


def test_the_scheduled_knob_holds_the_value_the_epoch_asks_for(corpus):
    dataset = _dataset(
        corpus,
        _curriculum(
            {
                "path": "aug:augmentation_speech.media_voice.prob",
                "points": [[0, 0.0], [20, 1.0]],
            }
        ),
    )
    assert dataset.augmentation_speech_args.media_voice.prob == 0.0

    dataset[("spk0", SR, 11, None, 10)]
    assert dataset.augmentation_speech_args.media_voice.prob == pytest.approx(0.5)

    dataset[("spk0", SR, 11, None, 20)]
    assert dataset.augmentation_speech_args.media_voice.prob == 1.0
    assert dataset.augmentation_speech_args.prob == 0.7, "untouched knobs stay put"


def test_a_schedule_at_the_recipe_s_own_values_changes_no_audio(corpus):
    """The plumbing is not allowed to cost anything -- only the ramp may."""
    plain = _dataset(corpus)
    scheduled = _dataset(
        corpus,
        _curriculum({"path": "aug:augmentation_speech.prob", "points": [[0, 0.7]]}),
    )
    for seed in (3, 4, 5):
        reference = plain[("spk0", SR, seed)]
        row = scheduled[("spk0", SR, seed, None, 7)]
        assert np.allclose(reference["noisy_speech"], row["noisy_speech"])
        assert np.allclose(reference["clean_speech"], row["clean_speech"])


def test_an_item_without_an_epoch_leaves_the_knobs_where_they_are(corpus):
    """Validation draws 3-tuples: its distribution must not follow the ramp."""
    dataset = _dataset(
        corpus,
        _curriculum({"path": "aug:augmentation_speech.prob", "points": [[0, 0.1], [20, 1.0]]}),
    )
    dataset[("spk0", SR, 1)]
    assert dataset.augmentation_speech_args.prob == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# bank weights
# --------------------------------------------------------------------------- #


class _StubBank:
    def __len__(self):
        return 10


def _union(*names):
    return UnionRoomBank([{"name": n, "weight": 1.0, "bank": _StubBank()} for n in names])


def test_reweighting_names_only_the_members_that_move():
    bank = _union("core", "expand", "wide")
    bank.set_weights({"core": 2.0})
    assert bank.weights == pytest.approx([0.5, 0.25, 0.25])
    assert sum(bank.weights) == pytest.approx(1.0)


def test_a_retired_member_may_go_to_zero_but_not_all_of_them():
    bank = _union("core", "realfar")
    bank.set_weights({"core": 0.0})
    assert bank.weights == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="cannot all be zero"):
        bank.set_weights({"realfar": 0.0})


@pytest.mark.parametrize(
    "weights, message",
    [({"nope": 1.0}, "unknown union bank member"), ({"core": -1.0}, "not be negative")],
)
def test_a_bad_reweight_is_refused(weights, message):
    with pytest.raises(ValueError, match=message):
        _union("core", "realfar").set_weights(weights)


# --------------------------------------------------------------------------- #
# the training-process half
# --------------------------------------------------------------------------- #


class _StubModule:
    def __init__(self, weights):
        self.loss_func_list_w = list(weights)
        self.logged = {}

    def log_dict(self, values, **_):
        self.logged.update(values)


class _StubTrainer:
    is_global_zero = True

    def __init__(self, epoch, sampler=None):
        self.current_epoch = epoch
        self.train_dataloader = type("_L", (), {"batch_sampler": sampler})()


def test_the_callback_moves_the_weight_the_epoch_asks_for():
    from puresound.system.curriculum import CurriculumCallback

    curriculum = _curriculum(
        {"path": "loss:OverSuppressionLoss", "points": [[0, 0.0], [20, 3.0]]},
        {"path": "aug:augmentation_realfar.prob", "points": [[0, 0.2]]},
    )
    callback = CurriculumCallback(curriculum, {"OverSuppressionLoss": 2})
    module = _StubModule([1.0, 0.5, 99.0, 0.2])

    callback.on_train_epoch_start(_StubTrainer(10), module)
    assert module.loss_func_list_w == [1.0, 0.5, pytest.approx(1.5), 0.2]

    callback.on_train_epoch_start(_StubTrainer(20), module)
    assert module.loss_func_list_w[2] == 3.0
    assert module.logged["curriculum/aug.augmentation_realfar.prob"] == 0.2, (
        "a data-side knob is logged here too -- the log is the run's only record "
        "of the schedule it trained under"
    )


def test_the_callback_also_tells_the_sampler_which_epoch_it_is():
    from puresound.system.curriculum import CurriculumCallback

    sampler = _sampler(emit_epoch=True)
    callback = CurriculumCallback(_curriculum({"path": "bank:core", "points": [[0, 1]]}), {})
    callback.on_train_epoch_start(_StubTrainer(31, sampler), _StubModule([]))
    assert next(iter(sampler))[0][4] == 31


def test_only_the_training_dataset_follows_the_schedule():
    """Validation keeps the recipe's constants, or its loss stops being a series."""
    from puresound.system import runner

    captured = {}

    class _StubDataset:
        meta = {f"spk{i}": {"utts": {f"u{i}": {"sr": SR}}} for i in range(20)}

        def __init__(self, **kwargs):
            captured[kwargs["dataset_role"]] = kwargs.get("curriculum")

    recipe = parse_recipe(
        _recipe_dict({"path": "aug:augmentation_realfar.prob", "points": [[0, 0.0], [40, 0.2]]})
    )
    train, valid = runner.build_dataloaders(
        dataset_cls=_StubDataset, collate_fn=lambda b: b, recipe=recipe
    )
    assert captured["train"] is recipe.curriculum
    assert captured["validation"] is None
    assert train.batch_sampler.emit_epoch is True
    assert valid.batch_sampler.emit_epoch is False


# --------------------------------------------------------------------------- #
# one item key, every task
# --------------------------------------------------------------------------- #


def test_the_key_parser_reads_every_shape_a_sampler_emits(corpus):
    dataset = _dataset(corpus)
    assert dataset.parse_item_key(("spk0", SR)) == (
        "spk0", SR, None, None, None
    )
    assert dataset.parse_item_key(("spk0", SR, 7)).seed == 7
    assert dataset.parse_item_key(("spk0", SR, 7, 3.0)).seconds == 3.0
    key = dataset.parse_item_key(("spk0", SR, None, None, 12))
    assert (key.seed, key.seconds, key.epoch) == (None, None, 12)


def test_a_key_that_is_not_one_says_so(corpus):
    with pytest.raises(TypeError, match="an item key is"):
        _dataset(corpus).parse_item_key(("spk0",))


def test_the_row_length_comes_from_the_key_and_is_cleared_after(corpus):
    dataset = _dataset(corpus)
    dataset.parse_item_key(("spk0", SR, 1, 3.0))
    assert dataset.sample_length == 3 * SR
    dataset.parse_item_key(("spk0", SR, 1))
    assert dataset.sample_length == 2 * SR, "a stale length must not leak forward"


def test_every_dynamic_dataset_reads_the_same_key(corpus):
    """A key shape one task understands and another does not is how a seeded
    validation pass, a length schedule or a curriculum ends up task-specific."""
    from puresound.dataset.dynamic_base import DynamicBaseDataset
    from puresound.task.ns import NoiseSuppressionDataset
    from puresound.task.sv import SpeakerEmbeddingDataset
    from puresound.task.tse import TargetSpeakerExtractDataset

    for cls in (NoiseSuppressionDataset, SpeakerEmbeddingDataset,
                TargetSpeakerExtractDataset, VoiceIsolationDataset):
        source = cls.__getitem__.__code__.co_names
        assert "parse_item_key" in source, f"{cls.__name__} unpacks its own key"
    assert callable(DynamicBaseDataset.parse_item_key)


# --------------------------------------------------------------------------- #
# knobs held by a composed component
# --------------------------------------------------------------------------- #


def test_a_scheduled_knob_reaches_the_components_built_from_it(corpus):
    """The trap this guards: a block is read per row, but the pieces composed
    FROM it are built once. A schedule that moved only the block would leave the
    capture chain, the noise stage and the gate serving their original values."""
    dataset = _dataset(
        corpus,
        _curriculum(
            {"path": "aug:augmentation_noise.prob", "points": [[0, 0.0], [10, 0.8]]},
            {"path": "aug:augmentation_hpf.prob", "points": [[0, 0.0], [10, 0.5]]},
            {
                "path": "aug:augmentation_speech.overlap_control.no_overlap_prob",
                "points": [[0, 0.1], [10, 0.4]],
            },
        ),
        noise={
            "used": True, "prob": 0.0, "snr_range": [0.0, 20.0],
            "noise_folder": str(corpus.parent), "prob_white_noise": 0.0,
            "white_noise_snr_range": [10.0, 30.0],
        },
        hpf={"used": True, "prob": 0.0, "cutoff": [100], "prob_each": [1.0]},
    )
    assert dataset.noise_stage.config.prob == 0.0
    assert dataset.device_chain.hpf.prob == 0.0
    assert dataset.overlap_gating.config.no_overlap_prob == pytest.approx(0.1)

    dataset[("spk0", SR, 2, None, 10)]
    assert dataset.noise_stage.config.prob == pytest.approx(0.8)
    assert dataset.device_chain.hpf.prob == pytest.approx(0.5)
    assert dataset.overlap_gating.config.no_overlap_prob == pytest.approx(0.4)


def test_rebinding_is_not_needed_when_only_bank_weights_move(corpus):
    """Composing costs nothing here, but a no-op rebuild every epoch would be a
    lie about what changed -- the components stay the objects they were."""
    dataset = _dataset(
        corpus, _curriculum({"path": "bank:core", "points": [[0, 1.0], [10, 0.5]]})
    )
    before = dataset.device_chain
    dataset.apply_curriculum_epoch(10)
    assert dataset.device_chain is before
