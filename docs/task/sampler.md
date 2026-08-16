# puresound.task.sampler

繁體中文版本：[`sampler.zh-TW.md`](sampler.zh-TW.md)

`torch.utils.data.Sampler`-style **batch** samplers used as a `DataLoader`'s
`batch_sampler`: each `__iter__` yield is already a full batch's worth of
`(speaker, sr[, item_seed])` keys, handed straight to the task dataset's
`__getitem__` (see [task.ns](ns.md), [task.tse](tse.md), [task.sv](sv.md)).

## Class: `SpeakerSampler`

Every batch contains `n_spks` speakers times `n_per` utterances per speaker,
which is what lets a batch (a) supply enough distinct speakers for the
interferer-sampling code in `task.ns`/`task.tse`/`task.voice_isolation` to
draw "some other speaker in this batch's pool", and (b) supply multiple
utterances per speaker for embedding losses that need several samples per
class per step (GE2E-style batches, `AAMsoftmax`; see
`egs/speaker_embedding/conf/PS-spk-v1.yaml`'s `n_spk_per_batch: 64,
n_utt_per_speaker: 2`).

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
)
```

**Parameters:**
- `data` -- the `meta` dict a `DynamicBaseDataset` subclass builds from the
  metafile (in practice always `dataset.meta`, e.g.
  `SpeakerSampler(data=train_dataset.meta, ...)`): keyed by speaker id, each
  value a dict with at least an `"utts"` sub-dict (utterance id ->
  utterance info). Only the top-level keys (speaker ids) are read unless
  `select_by_sr_first=True`, in which case `data[spk]["utts"][utt]["sr"]` is
  also read to build the per-SR speaker index. The constructor discards its
  reference to `data` (`del self.data`) right after extracting what it needs,
  so it never holds a second copy of the corpus metadata alongside the
  dataset.
- `total_batch` -- number of batches `__iter__` yields per epoch (`__len__`).
- `n_spks` -- distinct speakers per batch. If `n_spks` exceeds the speaker
  pool size, the constructor does **not** raise: it clamps
  `n_spks = len(spk_pool)` and inflates `n_per = ceil(n_spks * n_per /
  len(spk_pool))` to approximately preserve the requested batch size,
  printing a message.
- `n_per` -- utterances per speaker per batch (see the clamping above).
- `fast_sampling` -- if `True`, shuffle the speaker pool once and split it
  into groups of 10,000; each batch samples one group first, then `n_spks`
  speakers from within that group, instead of `random.sample`-ing the whole
  pool every batch. A speed trade-off for very large speaker pools, at the
  cost of exact uniformity. **Takes precedence over `select_by_sr_first`
  during iteration** -- if both are `True`, the SR-grouping path is silently
  never taken and every yielded `sr` stays `None`.
- `select_by_sr_first` -- if `True`, each batch first samples one sample rate
  (uniformly among the rates present in `data`), then samples `n_spks`
  speakers from only that rate's speaker pool, so every item in the batch
  shares one common `sr`. Recipes set this whenever the dataset does not
  force a single `target_sample_rate` (`select_by_sr_first=False if
  corpus_dict["target_sample_rate"] else True`), since a batch of mixed
  native sample rates cannot otherwise be stacked into one tensor.
- `seed` -- enables **deterministic validation** (see below).
- `rank` / `world_size` -- override the DDP rank/world size that would
  otherwise be read from `torch.distributed` (falls back to `(0, 1)` when
  distributed is not initialized). Mainly for tests and non-DDP callers that
  still want to reproduce a specific rank's seeded stream.

### Deterministic validation (seeded mode)

Recipes seed only the **validation** sampler, leaving the training sampler
unseeded so it keeps exploring new draws:

```python
valid_sampler = SpeakerSampler(
    data=valid_dataset.meta,
    total_batch=trainer_dict["valid_iter_per_epoch"],
    n_spks=trainer_dict["n_spk_per_batch"],
    n_per=trainer_dict["n_utt_per_speaker"],
    select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
    seed=trainer_dict.get("valid_seed", 1234),
)
```

When `seed` is set:

- **Per-rank private RNG.** `__iter__` builds its own
  `random.Random(seed + rank * 1_000_003)` instead of drawing from the shared
  `random` module, so re-creating a `SpeakerSampler` with the same
  `(seed, rank, world_size)` reproduces the exact same sequence of batches
  (speaker choices, `sr`, post-shuffle order) every time, independent of
  whatever else in the process has touched the global RNG. Different ranks
  get disjoint, independent streams (never colliding on the same items).
- **Per-item seed.** Each yielded tuple gains a third element, `item_seed`,
  deterministically derived from `(seed, rank, batch_idx, speaker slot,
  replica index)`. The task dataset's `__getitem__` (`task.ns`,
  `task.voice_isolation`) detects the 3-tuple form and reseeds `random`,
  `numpy`, and `torch`'s global RNGs with `item_seed` before doing anything
  else, so the *entire* on-the-fly synthesis chain for that item -- which
  utterance is drawn, room placement, SIR, every augmentation coin-flip --
  regenerates bit-identically regardless of epoch, run, or DataLoader worker
  layout.

Net effect: validation loss/metrics are computed on the literal same
synthetic examples every epoch and every run, so epoch-over-epoch and
run-over-run comparisons are not confounded by sampling noise. This is
exercised directly in
`test/test_utils/test_voice_isolate_recipe.py::test_speaker_sampler_uses_distinct_seeded_streams_per_rank`,
which asserts that the same `(seed, rank)` reproduces identical batches and
that two ranks' item seeds never overlap.

### `__len__() -> int`

Returns `total_batch`.

### `__iter__() -> Iterator[List[Tuple]]`

Yields `total_batch` batches. Each batch is a `List` of
`Tuple[str, Optional[int]]` (or `Tuple[str, Optional[int], int]` in seeded
mode) of length `n_spks * n_per` (after any clamping from an oversized
request) -- `(speaker_id, sr)` or `(speaker_id, sr, item_seed)` -- shuffled so
items from different speakers interleave rather than group by speaker.

### Recipe wiring

```python
from puresound.task.sampler import SpeakerSampler

train_sampler = SpeakerSampler(
    data=train_dataset.meta,
    total_batch=trainer_dict["train_iter_per_epoch"],
    n_spks=trainer_dict["n_spk_per_batch"],
    n_per=trainer_dict["n_utt_per_speaker"],
    select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,   # note: batch_sampler, not sampler
    pin_memory=True,
    num_workers=trainer_dict["num_workers"],
    collate_fn=collate_fn,
)
```

Used this way (as `batch_sampler=`) by all three egs mains that build a
`DynamicBaseDataset` subclass: `puresound/system/runner.py`,
`egs/target_speaker_extraction/main.py`, `egs/speaker_embedding/main.py`.

## Class: `SpeakerGenderSampler`

A separate, simpler batch sampler that yields gender-balanced batches: each
batch draws `n_spks / 2` speakers from `spk_list_male` and `n_spks / 2` from
`spk_list_female` (or `n_spks / 3` from each of the three lists when the
optional `spk_list_other` is supplied, for speakers with missing gender
metadata), then `n_per` utterances per drawn speaker. `n_spks` must divide
evenly across the 2 or 3 groups -- asserted in `__init__`. `__len__` is
`total_batch`.

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

It does not implement the seeded/DDP `item_seed` mechanism that
[`SpeakerSampler`](#class-speakersampler) does, and it has no call site in
this repo today. It is kept as a library asset: it pairs with the
`--utt2gender_path` gender-metadata path in the `prepare_metafile.py` scripts
(`egs/noise_suppression`, `egs/target_speaker_extraction`,
`egs/speaker_embedding`), which is what produces the per-speaker gender lists
this sampler consumes.
