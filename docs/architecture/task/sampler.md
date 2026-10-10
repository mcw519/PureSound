# puresound.task.sampler

繁體中文版本：[`sampler.zh-TW.md`](sampler.zh-TW.md)

**Batch** samplers used as a `DataLoader`'s `batch_sampler`: each `__iter__`
yield is already a full batch of item keys, handed straight to the task
dataset's `__getitem__` (see [task.ns](ns.md)), which reads every key shape
through one parser (`DynamicBaseDataset.parse_item_key`).

The key grows with what the run asked for, and only with that:

| key | when |
| --- | --- |
| `(speaker, sr)` | always |
| `(speaker, sr, item_seed)` | the sampler is seeded (deterministic validation) |
| `(speaker, sr, item_seed_or_None, seconds)` | `length_schedule` is set |
| `(speaker, sr, item_seed_or_None, seconds_or_None, epoch)` | `emit_epoch` is on |
| `(speaker, sr, item_seed, seconds_or_None, epoch_or_None, utterance)` | `CoverageSampler` (it chooses the utterance) |

Empty slots stay in place as `None`, so the arity alone says which shape a key
is. `sr` is `None` unless `select_by_sr_first` is on. A run that asks for none
of these gets the two-element key.

## Class: `SpeakerSampler`

Every batch contains `n_spks` speakers times `n_per` items per speaker. That is
what lets a batch (a) supply enough distinct speakers for interferer sampling
to draw "some other speaker", and (b) supply several items per speaker for
embedding losses that need more than one sample per class per step.

### Constructor

```python
SpeakerSampler(
    data: Dict,
    total_batch: int,
    n_spks: int,
    n_per: int,
    fast_sampling: bool = False,
    select_by_sr_first: bool = False,
    seed: Optional[int] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    length_schedule: Optional[List[Tuple[float, int, float]]] = None,
    emit_epoch: bool = False,
    source_weights: Optional[Dict[str, float]] = None,
)
```

**Parameters:**
- `data` -- the dataset's `meta` dict (`SpeakerSampler(data=train_dataset.meta, ...)`):
  keyed by speaker id, each value holding at least an `"utts"` sub-dict. Only the
  speaker ids are read, plus `data[spk]["utts"][utt]["sr"]` when
  `select_by_sr_first=True`. The sampler drops its reference to `data` after
  construction, so it never holds a second copy of the corpus metadata.
- `total_batch` -- batches per epoch (`__len__`).
- `n_spks` -- distinct speakers per batch. If it exceeds the speaker pool, the
  constructor clamps `n_spks = len(pool)` and raises
  `n_per = ceil(n_spks * n_per / len(pool))` to keep the batch size about the
  same, and logs a warning.
- `n_per` -- items per speaker per batch.
- `fast_sampling` -- shuffle the speaker pool once and split it into groups of
  10,000 (the remainder joins the last group); each batch picks a group, then
  `n_spks` speakers inside it. A speed trade-off for very large pools, at the
  cost of exact uniformity. It takes precedence over `select_by_sr_first` during
  iteration.
- `select_by_sr_first` -- each batch first picks one sample rate, then `n_spks`
  speakers from that rate's pool, so the whole batch shares one `sr`. The runner
  sets it whenever the recipe has no `target_sample_rate`, because a batch of
  mixed native rates cannot be stacked into one tensor.
- `seed` -- enables deterministic mode (below).
- `rank` / `world_size` -- override the values read from `torch.distributed`
  (`(0, 1)` when it is not initialised). For tests and callers that want to
  reproduce one rank's stream.
- `length_schedule` -- `(seconds, n_spks, prob)` buckets. Each batch draws one,
  so the model is supervised at several context lengths in one run; the bucket's
  `n_spks` replaces the constructor's, because activation memory follows the
  length. The draw is seeded by `(seed, epoch, batch index)` and never by rank:
  under DDP the ranks step in lockstep, and two ranks with different row lengths
  in the same step would average gradients over different batch sizes and stall
  on every sync.
- `emit_epoch` -- append the epoch index to every key. That is how a recipe's
  `curriculum` reaches a dataset copy living in a worker process. Off unless
  something is scheduled, so an unscheduled run keeps its key shape and its RNG
  stream.
- `source_weights` -- `{speaker-id prefix: weight}`. Each batch slot first draws
  a source by weight, then a speaker inside it (without replacement unless a
  source is drawn more often than it has speakers). Without it every speaker is
  equally likely, so a corpus's share of the batches is its share of the
  speakers, not of the hours. Every speaker must match exactly one prefix (the
  longest wins); an unmatched speaker or an empty prefix raises `ValueError`, as
  does combining this with `select_by_sr_first` or `fast_sampling`.

### `set_epoch(epoch)`

Which epoch the next pass is, told from outside. Counting passes internally is
wrong on the path that matters: a run resumed at epoch N would restart the count
at 0 and replay the beginning of any epoch-dependent schedule. Lightning calls
this on `dataloader.batch_sampler.sampler` before each epoch's iterator is
consumed, including the resumed one; the `sampler` property returns the sampler
itself so the call lands. Without a caller the internal count applies, so a
plain PyTorch loop is unaffected.

### Random streams

- **Seeded**: `__iter__` uses a private `random.Random(seed + rank * 1_000_003)`,
  so re-creating the sampler with the same `(seed, rank, world_size)` reproduces
  the same batches -- speaker choices, `sr`, order -- whatever else in the
  process has touched the global RNG, and different ranks get independent
  streams.
- **Unseeded, DDP** (`world_size > 1`): a private stream per rank, offset by
  rank from one draw of the global RNG.
- **Unseeded, single process**: the global `random` module.

### Deterministic validation

The runner (`puresound.system.runner.build_dataloaders`) seeds only the
validation sampler, so the training sampler keeps exploring new draws:

```python
valid_sampler = SpeakerSampler(
    data=valid_dataset.meta,
    total_batch=trainer.valid_iter_per_epoch,
    n_spks=trainer.n_spk_per_batch,
    n_per=trainer.n_utt_per_speaker,
    select_by_sr_first=not corpus.target_sample_rate,
    seed=trainer.valid_seed,          # recipe default 1234
)
```

In seeded mode each key carries an `item_seed` derived from
`(seed, rank, batch index, speaker slot, replica index)`. `parse_item_key`
reseeds `random`, `numpy` and `torch` with it before anything else, so the whole
on-the-fly synthesis of that item -- utterance choice, room placement, SIR,
every augmentation coin-flip -- regenerates bit-identically regardless of epoch,
run or DataLoader worker layout. Validation loss and metrics are therefore
computed on the same synthetic examples every epoch and every run.
`test/task/test_sampler.py` pins that the same `(seed, rank)`
reproduces identical batches and that two ranks' item seeds do not overlap.

### `__len__() -> int`

Returns `total_batch`.

### `__iter__() -> Iterator[List[Tuple]]`

Yields `total_batch` batches, each a list of `n_spks * n_per` keys (after any
clamping, or with the bucket's `n_spks` under a length schedule), shuffled so
items from different speakers interleave.

### Recipe wiring

```python
from puresound.task.sampler import SpeakerSampler

train_sampler = SpeakerSampler(
    data=train_dataset.meta,
    total_batch=trainer.train_iter_per_epoch,
    n_spks=trainer.n_spk_per_batch,
    n_per=trainer.n_utt_per_speaker,
    select_by_sr_first=not corpus.target_sample_rate,
    length_schedule=[(b.seconds, b.n_spk, b.prob) for b in trainer.length_schedule]
    if trainer.length_schedule else None,
    emit_epoch=curriculum is not None,
    source_weights=trainer.speaker_source_weights,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,   # batch_sampler, not sampler
    pin_memory=True,
    num_workers=trainer.num_workers,
    collate_fn=collate_fn,
    worker_init_fn=seed_worker,    # puresound.system.runner.seed_worker
)
```

`puresound.system.runner.build_dataloaders` is the one place that builds these;
every recipe driver (`egs/*/main.py`) calls it.

## Class: `CoverageSampler`

The same mix as `SpeakerSampler` with `n_per=1` and `source_weights` -- a slot
draws a source by weight, then a speaker, then an utterance -- but speakers and
utterances come from shuffled queues instead of fresh picks: every speaker of a
source is drawn once before any repeats, and every utterance of a speaker before
any repeats. Selected by `trainer.train_sampler: coverage`; validation keeps the
seeded `SpeakerSampler`.

```python
CoverageSampler(
    data: Dict,
    total_batch: int,
    n_items: int,
    default_seconds: float,
    length_schedule: Optional[List[Tuple[float, int, float]]] = None,
    source_weights: Optional[Dict[str, float]] = None,
    emit_epoch: bool = False,
    seed: int = 0,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
)
```

- `data` -- the dataset's `meta`; reads `data[spk]["utts"][utt]["length"]` and `["sr"]`.
- `n_items` -- items per batch without a length schedule.
- `default_seconds` -- the row length without a schedule (`training_length_seconds`).
- `length_schedule`, `source_weights`, `emit_epoch`, `rank`, `world_size` -- as in `SpeakerSampler`.

**Row lengths.** Each exact length has its own queues, holding only utterances at
least that long, so the selected foreground does not need padding for a short
input file. A speaker (or a whole source) with no such utterance sits that length
out, with a log line. The length is drawn per batch and is the same on every rank;
NS, SV, and TSE honor it for target crops and interference alignment. Augmentation
can still insert silence or replace a foreground for task-specific absent rows.

**Ranks.** Every rank computes the whole batch's slots and keeps its own share,
so the ranks deal disjoint parts of one walk.

**Position.** Every draw -- length, source, queue order, per-item synthesis seed
-- is a function of the seed and the position in the walk, read from its own
PCG64 stream, never from the global RNG. With unchanged draw settings, splitting
the walk into epochs and runs preserves the selected items and synthesis seeds:

- `checkpoint_state(epochs_done, batches_done=None)` -- the position after that
  many full epochs, optionally followed by completed batches of the current
  epoch. The callback counts batches consumed by training, rather than sampler
  yields that may have been prefetched. Reconstruction uses bounded chunks and
  caches recent positions.
- `start_from(state, resume=True)` -- resumes the same stage at an epoch
  boundary. World size, batch counts, length/source probabilities, epoch-key
  settings, and eligible queue contents/order must match the checkpoint.
- `start_from(state, resume=False)` -- starts a new stage at the saved position,
  including a partial epoch stopped by `max_steps`. Batch counts, probabilities,
  and world size may change; shared row-length queues must still have the same
  eligible speakers and utterances in the same order. New lengths get new queues.

The recorded seed is kept either way. Checkpoints contain queue signatures and a
stage signature, so incompatible settings raise an error rather than silently
replaying or skipping data. A partial-epoch checkpoint requires a warm start:
Lightning's non-stateful dataloader cannot reliably resume inside an epoch.
The callback also checks Lightning's restored epoch before training. If an
iteration-based `max_steps` loop restores the preceding epoch, use epoch-based
training or a warm start instead.

`puresound.system.sampling` writes the state into every checkpoint
(`checkpoint["coverage_sampler"]`) and reads it back for `--ckpt_path` and
`--pretrained_ckpt_path`; a run with neither starts a new walk at `--set_seed`.
Requires `n_utt_per_speaker: 1` and a `target_sample_rate`. If a named utterance is
empty, silent, or nonfinite, loading raises an error rather than substituting a
different utterance outside the queue.

## Class: `SpeakerGenderSampler`

A separate, simpler batch sampler for gender-balanced batches: each batch draws
`n_spks / 2` speakers from `spk_list_male` and `n_spks / 2` from
`spk_list_female` (or `n_spks / 3` from each of the three lists when
`spk_list_other` is given, for speakers without gender metadata), and repeats
each drawn speaker id `n_per` times. `n_spks` must divide evenly across the 2 or
3 groups (asserted in `__init__`). `__len__` is `total_batch`.

```python
SpeakerGenderSampler(
    total_batch: int,
    n_spks: int,
    n_per: int,
    spk_list_male: List,
    spk_list_female: List,
    spk_list_other: Optional[List] = None,
)
```

Its batches are bare speaker ids, not the `(speaker, sr, ...)` keys the dynamic
datasets parse, and it has no seeded or DDP mode; nothing in the repository
calls it. It is kept as a library asset that pairs with the gender metadata
(`utt2gender_path`) of [`puresound.dataset.corpus.kaldi`](../../usage/data_preparation.md).
