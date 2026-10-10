from puresound.task.sampler import SpeakerSampler


def test_speaker_sampler_uses_distinct_seeded_streams_per_rank():
    """Same (seed, rank) -> same batches; different ranks -> disjoint draws."""
    data = {f"spk{i}": {"utts": {f"utt{i}": {"sr": 16000}}} for i in range(8)}

    def batches(rank):
        return list(
            SpeakerSampler(
                data, total_batch=4, n_spks=2, n_per=1, seed=123, rank=rank, world_size=2
            )
        )

    rank0, rank1 = batches(0), batches(1)

    assert rank0 == batches(0)
    assert rank0 != rank1
    assert {item[2] for batch in rank0 for item in batch}.isdisjoint(
        {item[2] for batch in rank1 for item in batch}
    )


def test_fast_sampling_draws_batches_from_pools_of_any_size():
    """The remainder of a pool that does not divide into groups of 10000 joins the
    last group; a small pool is one group."""
    for n_speakers in (50, 10003):
        data = {f"spk{i}": {"utts": {}} for i in range(n_speakers)}
        sampler = SpeakerSampler(data, total_batch=40, n_spks=4, n_per=1, fast_sampling=True)
        assert all(len(batch) == 4 for batch in sampler)
