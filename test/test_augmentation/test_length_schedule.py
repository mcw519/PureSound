"""trainer.length_schedule -- mixed row lengths in one training run.

The two contracts that matter are not "it produces varied lengths" but:
  * every rank draws the SAME length in the same step (DDP averages gradients
    across ranks; different lengths there means different batch sizes averaged
    together and a stall on every sync), and
  * a recipe without a schedule is bit-for-bit what it was, RNG stream included.
"""
import random

import pytest

from puresound.task.sampler import SpeakerSampler

SCHEDULE = [(3.0, 20, 0.25), (6.0, 12, 0.30), (12.0, 6, 0.25), (30.0, 2, 0.20)]


def _meta(n_spk=64):
    return {f"spk{i}": {"utts": {f"u{i}_{j}": {"sr": 16000} for j in range(3)}}
            for i in range(n_spk)}


def _sampler(**kw):
    return SpeakerSampler(data=_meta(), total_batch=40, n_spks=12, n_per=1, **kw)


def test_without_a_schedule_entries_keep_their_shape():
    random.seed(0)
    batches = list(_sampler())
    assert all(len(e) == 2 for b in batches for e in b)
    assert all(len(b) == 12 for b in batches)


def test_scheduled_batches_carry_one_length_and_its_batch_size():
    sizes = {s: n for s, n, _ in SCHEDULE}
    for batch in _sampler(length_schedule=SCHEDULE):
        lengths = {e[3] for e in batch}
        assert len(lengths) == 1, "a batch is stacked -- one length or it cannot collate"
        seconds = lengths.pop()
        assert len(batch) == sizes[seconds]
        assert all(len(e) == 4 for e in batch)


def test_every_rank_draws_the_same_length_sequence():
    """The one that would silently halve DDP throughput if it regressed."""
    seqs = []
    for rank in (0, 1, 2):
        s = _sampler(length_schedule=SCHEDULE, rank=rank, world_size=3)
        seqs.append([b[0][3] for b in s])
    assert seqs[0] == seqs[1] == seqs[2]
    assert len(set(seqs[0])) > 1, "a schedule that never varies is not a schedule"


def test_speakers_still_differ_across_ranks():
    """Only the length is shared -- sharing the speakers too would waste a rank."""
    a = [b[0][0] for b in _sampler(length_schedule=SCHEDULE, rank=0, world_size=2)]
    b = [b[0][0] for b in _sampler(length_schedule=SCHEDULE, rank=1, world_size=2)]
    assert a != b


def test_the_mix_follows_the_declared_probabilities():
    s = SpeakerSampler(data=_meta(), total_batch=4000, n_spks=12, n_per=1,
                       length_schedule=SCHEDULE)
    seen = [b[0][3] for b in s]
    for seconds, _, prob in SCHEDULE:
        assert abs(seen.count(seconds) / len(seen) - prob) < 0.03


def test_the_length_sequence_moves_between_epochs():
    s = _sampler(length_schedule=SCHEDULE)
    first = [b[0][3] for b in s]
    second = [b[0][3] for b in s]
    assert first != second


def test_probabilities_must_sum_to_one():
    from pydantic import ValidationError
    from puresound.config.recipe import TrainerConfig
    base = dict(lightning_trainer_args={}, train_iter_per_epoch=10,
                valid_iter_per_epoch=10, n_spk_per_batch=12, n_utt_per_speaker=1,
                num_workers=0, num_gpus=1, work_folder="/tmp/x")
    with pytest.raises(ValidationError):
        TrainerConfig(**base, length_schedule=[{"seconds": 6.0, "n_spk": 12, "prob": 0.5}])
    TrainerConfig(**base, length_schedule=[{"seconds": 6.0, "n_spk": 12, "prob": 1.0}])
