"""`CoverageSampler`: queues instead of fresh picks, continued across runs.

Contracts: a pass visits every speaker (and a speaker every utterance) before a
repeat; ranks deal disjoint slots of one walk and share the row length; the walk
is the same however it is cut into epochs, resumes and stages; the expected mix
follows the weights; a row never outlasts its utterance.
"""
import random
from collections import Counter

import pytest
from pydantic import ValidationError

from puresound.config.recipe import TrainerConfig
from puresound.task.sampler import CoverageSampler

SR = 16000


def _meta(counts, seconds=(5.0, 6.0, 12.0)):
    """``{"prefix_": n_speakers}`` -> sampler metadata, one utterance per length."""
    return {
        f"{prefix}{i}": {
            "utts": {
                f"{prefix}{i}_{j}": {"sr": SR, "length": int(s * SR)}
                for j, s in enumerate(seconds)
            }
        }
        for prefix, n in counts.items()
        for i in range(n)
    }


def _sampler(data, **kw):
    kw.setdefault("total_batch", 10)
    kw.setdefault("n_items", 4)
    kw.setdefault("default_seconds", 4.0)
    kw.setdefault("seed", 11)
    return CoverageSampler(data=data, **kw)


def _items(sampler, epochs):
    out = []
    for epoch in epochs:
        sampler.set_epoch(epoch)
        out += [item for batch in sampler for item in batch]
    return out


def test_a_pass_visits_every_speaker_once_and_a_speaker_cycles_its_utterances():
    data = _meta({"a_": 7})
    items = _items(_sampler(data, total_batch=21, n_items=1), [0])
    speakers = [item[0] for item in items]
    for start in range(0, 21, 7):
        assert sorted(speakers[start:start + 7]) == sorted(data), "one pass = each speaker once"
    for spk in data:
        visits = [item[5] for item in items if item[0] == spk]
        assert sorted(visits) == sorted(data[spk]["utts"]), "all utterances before a repeat"


def test_the_ranks_take_disjoint_slots_of_one_walk_and_share_the_row_length():
    data = _meta({"a_": 40})
    schedule = [(4.0, 3, 0.5), (10.0, 2, 0.5)]
    ranks = [_sampler(data, total_batch=6, rank=r, world_size=2, length_schedule=schedule)
             for r in (0, 1)]
    batches = [list(s) for s in ranks]
    assert [b[0][3] for b in batches[0]] == [b[0][3] for b in batches[1]]
    keys = [(item[0], item[3]) for rank in batches for batch in rank for item in batch]
    assert len(keys) == len(set(keys)), "a slot is drawn by exactly one rank"
    seeds = [item[2] for rank in batches for batch in rank for item in batch]
    assert len(seeds) == len(set(seeds))


def test_a_resumed_run_continues_where_the_epoch_left_off():
    data = _meta({"a_": 30, "b_": 30})
    weights = {"a_": 1.0, "b_": 3.0}
    straight = _sampler(data, source_weights=weights)
    through = [_items(straight, [e]) for e in range(3)]

    resumed = _sampler(data, source_weights=weights)
    resumed.start_from(straight.checkpoint_state(epochs_done=2), resume=True)
    assert _items(resumed, [2]) == through[2]


def test_the_walk_is_the_same_however_it_is_cut_into_epochs_and_stages():
    """Sources, lengths, speakers, utterances and item seeds: one long epoch, three
    short ones, and a two-stage ladder all deal the same items."""
    data = _meta({"a_": 12, "b_": 5})
    kw = dict(source_weights={"a_": 1.0, "b_": 1.0},
              length_schedule=[(4.0, 3, 0.6), (10.0, 2, 0.4)], rank=1, world_size=2)
    whole = _items(_sampler(data, total_batch=30, **kw), [0])
    epochs = _items(_sampler(data, total_batch=10, **kw), [0, 1, 2])
    first = _sampler(data, total_batch=10, **kw)
    staged = _items(first, [0, 1])
    second = _sampler(data, total_batch=10, **kw)
    second.start_from(first.checkpoint_state(epochs_done=2), resume=False)
    staged += _items(second, [0])
    assert whole == epochs == staged


def test_the_next_stage_continues_the_walk_instead_of_replaying_it():
    data = _meta({"a_": 50})
    stage1 = _sampler(data, total_batch=5, n_items=4)
    first = _items(stage1, [0, 1])
    stage2 = _sampler(data, total_batch=5, n_items=4, seed=999)  # its own --set_seed
    stage2.start_from(stage1.checkpoint_state(epochs_done=2), resume=False)
    second = _items(stage2, [0])
    assert [i[0] for i in first + second[:10]] == [i[0] for i in _items(
        _sampler(data, total_batch=15, n_items=4), [0])][:50], "one continuous walk"
    assert not {i[0] for i in first} & {i[0] for i in second[:10]}, "no speaker repeats inside a pass"

    fresh = _sampler(data, total_batch=5, n_items=4, seed=11)
    assert _items(fresh, [0]) != second, "a warm start does not restart the walk"


def test_the_share_of_slots_per_source_follows_the_weights():
    data = _meta({"big_": 1000, "small_": 10})
    sampler = _sampler(data, total_batch=500, n_items=8,
                       source_weights={"big_": 1.0, "small_": 1.0})
    share = Counter(item[0].startswith("small_") for item in _items(sampler, [0]))
    assert 0.45 < share[True] / sum(share.values()) < 0.55


def test_speakers_within_a_source_are_drawn_evenly():
    data = _meta({"a_": 9})
    counts = Counter(item[0] for item in _items(_sampler(data, total_batch=90, n_items=1), [0]))
    assert set(counts.values()) == {10}


def test_long_rows_only_draw_utterances_that_fill_them():
    data = _meta({"a_": 20}, seconds=(4.5, 8.0, 12.0))
    data["short_only"] = {"utts": {"s_0": {"sr": SR, "length": 5 * SR},
                                   "s_1": {"sr": SR, "length": 5 * SR}}}
    schedule = [(4.0, 6, 0.5), (10.0, 2, 0.5)]
    sampler = _sampler(data, total_batch=200, length_schedule=schedule)
    lengths = {utt: meta["length"] / SR for spk in data.values() for utt, meta in spk["utts"].items()}
    batches = list(sampler)
    for batch in batches:
        assert len({item[3] for item in batch}) == 1
        assert len(batch) == {4.0: 6, 10.0: 2}[batch[0][3]]
    long_items = [item for batch in batches for item in batch if item[3] == 10.0]
    assert long_items and all(lengths[item[5]] >= 10.0 for item in long_items)
    assert all(item[0] != "short_only" for item in long_items)


def test_the_length_mix_follows_the_declared_probabilities():
    schedule = [(4.0, 3, 0.8), (10.0, 1, 0.2)]
    sampler = _sampler(_meta({"a_": 30}), total_batch=3000, length_schedule=schedule)
    lengths = [batch[0][3] for batch in sampler]
    assert abs(lengths.count(10.0) / len(lengths) - 0.2) < 0.03


def test_the_key_names_the_utterance_and_carries_the_epoch_only_when_asked():
    data = _meta({"a_": 5})
    plain = next(iter(_sampler(data)))[0]
    assert len(plain) == 6 and plain[1] is None and plain[3] is None and plain[4] is None
    assert plain[5] in data[plain[0]]["utts"]
    sampler = _sampler(data, emit_epoch=True)
    sampler.set_epoch(3)
    assert next(iter(sampler))[0][4] == 3


def test_iterating_leaves_the_global_random_stream_alone():
    random.seed(5)
    expected = random.random()
    random.seed(5)
    list(_sampler(_meta({"a_": 5})))
    assert random.random() == expected


def test_a_checkpoint_from_another_world_size_cannot_resume_a_stage():
    sampler = _sampler(_meta({"a_": 5}), rank=0, world_size=2)
    state = sampler.checkpoint_state(epochs_done=1)
    other = _sampler(_meta({"a_": 5}), rank=0, world_size=1)
    other.start_from(state, resume=True)
    with pytest.raises(ValueError, match="world size"):
        next(iter(other))
    other.start_from(state, resume=False)  # a new stage may change it
    next(iter(other))


def test_the_coverage_sampler_takes_one_utterance_per_speaker_slot():
    base = dict(lightning_trainer_args={}, train_iter_per_epoch=10, valid_iter_per_epoch=10,
                n_spk_per_batch=12, num_workers=0, num_gpus=1, work_folder="/tmp/x")
    with pytest.raises(ValidationError, match="n_utt_per_speaker"):
        TrainerConfig(**base, n_utt_per_speaker=4, train_sampler="coverage")
    assert TrainerConfig(**base, n_utt_per_speaker=1, train_sampler="coverage").train_sampler == "coverage"
    assert TrainerConfig(**base, n_utt_per_speaker=4).train_sampler == "speaker"


def test_nearby_row_lengths_keep_independent_coverage_queues():
    data = _meta({"a_": 40})
    lengths = (4.000001, 4.000002)
    sampler = _sampler(data, total_batch=200, n_items=1,
                       length_schedule=[(s, 1, 0.5) for s in lengths])
    items = _items(sampler, [0])
    for length in lengths:
        picked = [item[0] for item in items if item[3] == length][:40]
        assert len(picked) == len(set(picked)) == 40


def test_resume_rejects_a_changed_stage_but_warm_start_allows_new_batch_sizes():
    data = _meta({"a_": 30})
    first = _sampler(data)
    state = first.checkpoint_state(epochs_done=2)
    for kwargs in ({"total_batch": 5}, {"n_items": 2}, {"emit_epoch": True}):
        changed = _sampler(data, **kwargs)
        with pytest.raises(ValueError, match="stage settings changed"):
            changed.start_from(state, resume=True)
        changed.start_from(state, resume=False)
        assert next(iter(changed))[0][2] == first._item_seed(state["next"]["slot"])


@pytest.mark.parametrize("resume", [False, True])
def test_checkpoint_continuation_rejects_changed_eligible_utterances(resume):
    data = _meta({"a_": 30})
    state = _sampler(data).checkpoint_state(epochs_done=2)
    data["a_0"]["utts"].pop("a_0_1")
    changed = _sampler(data)
    with pytest.raises(ValueError, match="eligible queues changed"):
        changed.start_from(state, resume=resume)


def test_checkpoint_replay_is_independent_of_advance_chunk_size(monkeypatch):
    import puresound.task.sampler as module
    data = _meta({"a_": 30, "b_": 12})
    kwargs = dict(total_batch=37, rank=1, world_size=2,
                  source_weights={"a_": 1., "b_": 2.},
                  length_schedule=[(4., 3, 0.6), (10., 2, 0.4)])
    first = _sampler(data, **kwargs)
    expected = _items(first, [5])
    state = first.checkpoint_state(epochs_done=5)
    monkeypatch.setattr(module, "_DRAW_BATCH_CHUNK", 7)
    resumed = _sampler(data, **kwargs)
    resumed.start_from(state, resume=True)
    assert _items(resumed, [5]) == expected


@pytest.mark.parametrize("kwargs", [
    {"default_seconds": float("inf")}, {"n_items": 0}, {"seed": -1},
    {"source_weights": {"a_": float("nan")}},
    {"source_weights": {"a_": float("inf")}},
])
def test_invalid_sampling_settings_fail_before_drawing(kwargs):
    with pytest.raises(ValueError):
        _sampler(_meta({"a_": 30}), **kwargs)
