"""The speaker sampler's two knobs: a length schedule and per-source weights.

The contracts that matter:
  * every DDP rank draws the SAME length in the same step (ranks average
    gradients; different lengths there means different batch sizes averaged
    together and a stall on every sync), while the speakers still differ;
  * a corpus's batch share follows its weight, not its speaker count, and a
    speaker no weight claims is an error rather than a speaker never drawn;
  * a recipe without either knob is bit-for-bit what it was, RNG stream included.
"""
import random
from collections import Counter

import pytest
from pydantic import ValidationError

from puresound.config.recipe import TrainerConfig
from puresound.task.sampler import SpeakerSampler

SCHEDULE = [(3.0, 20, 0.25), (6.0, 12, 0.30), (12.0, 6, 0.25), (30.0, 2, 0.20)]


def _meta(counts):
    """``{"prefix_": n_speakers}`` -> sampler metadata, three utterances each."""
    return {
        f"{prefix}{i}": {"utts": {f"{prefix}{i}_{j}": {"sr": 16000} for j in range(3)}}
        for prefix, n in counts.items()
        for i in range(n)
    }


def _scheduled(**kw):
    return SpeakerSampler(data=_meta({"spk": 64}), total_batch=kw.pop("total_batch", 40),
                          n_spks=12, n_per=1, length_schedule=SCHEDULE, **kw)


def test_without_either_knob_the_batches_and_the_stream_are_unchanged():
    data = _meta({"a_": 30})
    random.seed(0)
    before = list(SpeakerSampler(data=data, total_batch=5, n_spks=4, n_per=1))
    after_default = random.random()
    random.seed(0)
    after = list(SpeakerSampler(data=data, total_batch=5, n_spks=4, n_per=1,
                                source_weights=None, length_schedule=None))
    assert before == after and random.random() == after_default
    assert all(len(batch) == 4 and all(len(entry) == 2 for entry in batch) for batch in before)


def test_scheduled_batches_carry_one_length_and_its_batch_size():
    sizes = {s: n for s, n, _ in SCHEDULE}
    for batch in _scheduled():
        lengths = {e[3] for e in batch}
        assert len(lengths) == 1, "a batch is stacked -- one length or it cannot collate"
        seconds = lengths.pop()
        assert len(batch) == sizes[seconds]
        assert all(len(e) == 4 for e in batch)


def test_every_rank_draws_the_same_length_sequence_but_its_own_speakers():
    """The length sequence is the one that would silently halve DDP throughput if
    it regressed; sharing the speakers too would waste a rank."""
    samplers = [_scheduled(rank=rank, world_size=3) for rank in (0, 1, 2)]
    lengths = [[b[0][3] for b in s] for s in samplers]
    assert lengths[0] == lengths[1] == lengths[2]
    assert len(set(lengths[0])) > 1, "a schedule that never varies is not a schedule"
    speakers = [[b[0][0] for b in s] for s in samplers[:2]]
    assert speakers[0] != speakers[1]


def test_the_length_mix_follows_the_declared_probabilities_and_moves_between_epochs():
    sampler = _scheduled(total_batch=4000)
    first = [b[0][3] for b in sampler]
    for seconds, _, prob in SCHEDULE:
        assert abs(first.count(seconds) / len(first) - prob) < 0.03
    assert [b[0][3] for b in sampler] != first


def test_schedule_probabilities_must_sum_to_one():
    base = dict(lightning_trainer_args={}, train_iter_per_epoch=10,
                valid_iter_per_epoch=10, n_spk_per_batch=12, n_utt_per_speaker=1,
                num_workers=0, num_gpus=1, work_folder="/tmp/x")
    with pytest.raises(ValidationError):
        TrainerConfig(**base, length_schedule=[{"seconds": 6.0, "n_spk": 12, "prob": 0.5}])
    TrainerConfig(**base, length_schedule=[{"seconds": 6.0, "n_spk": 12, "prob": 1.0}])


@pytest.mark.parametrize(
    "counts, weights, total_batch, n_spks, seed, minority, share",
    [
        # Uniform over speakers would give the small source ~1%.
        ({"big_": 1000, "small_": 10}, {"big_": 1.0, "small_": 1.0}, 500, 8, 7, "small_", (0.45, 0.55)),
        # The longest matching prefix owns a speaker.
        ({"dns5_": 4, "dns5de_": 4}, {"dns5": 1.0, "dns5de_": 3.0}, 200, 4, 1, "dns5de_", (0.7, 0.8)),
    ],
    ids=["weights-not-speaker-counts", "longest-prefix"],
)
def test_the_share_of_slots_per_source_follows_the_weights(counts, weights, total_batch, n_spks,
                                                           seed, minority, share):
    sampler = SpeakerSampler(data=_meta(counts), total_batch=total_batch, n_spks=n_spks,
                             n_per=1, seed=seed, source_weights=weights)
    slots = Counter(spk.startswith(minority) for batch in sampler for spk, *_ in batch)
    assert share[0] < slots[True] / sum(slots.values()) < share[1]


@pytest.mark.parametrize(
    "counts, weights, match",
    [
        ({"a_": 3, "b_": 3}, {"a_": 1.0}, "match no source prefix"),
        ({"a_": 3}, {"a_": 1.0, "typo_": 1.0}, "match no speaker"),
    ],
    ids=["unclaimed-speaker", "prefix-matching-nothing"],
)
def test_a_weight_table_that_does_not_cover_the_speakers_exactly_is_an_error(counts, weights, match):
    with pytest.raises(ValueError, match=match):
        SpeakerSampler(data=_meta(counts), total_batch=1, n_spks=2, n_per=1,
                       source_weights=weights)


def test_speakers_in_a_weighted_batch_stay_distinct_when_a_source_has_enough():
    sampler = SpeakerSampler(
        data=_meta({"a_": 50, "b_": 50}), total_batch=50, n_spks=12, n_per=1, seed=3,
        source_weights={"a_": 1.0, "b_": 1.0},
    )
    for batch in sampler:
        assert len({spk for spk, *_ in batch}) == 12
