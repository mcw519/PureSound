import hashlib
import json
import logging
import math
import random
import struct
import zlib
from collections import OrderedDict, defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch.distributed as dist


logger = logging.getLogger(__name__)


def _distributed_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _group_by_source(
    speakers: List[str], weights: Dict[str, float]
) -> List[Tuple[str, List[str], float]]:
    """``[(prefix, speakers, normalised weight)]``; longest matching prefix wins."""
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError(f"source weights must be positive, got {weights}")
    prefixes = sorted(weights, key=len, reverse=True)
    groups: Dict[str, List[str]] = {prefix: [] for prefix in weights}
    unmatched = []
    for spk in speakers:
        owner = next((p for p in prefixes if spk.startswith(p)), None)
        if owner is None:
            unmatched.append(spk)
        else:
            groups[owner].append(spk)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} speaker(s) match no source prefix in {sorted(weights)}, "
            f"e.g. {unmatched[:3]}; they would never be drawn."
        )
    empty = [p for p, spks in groups.items() if not spks]
    if empty:
        raise ValueError(f"source prefix(es) {empty} match no speaker in the metafile.")
    total = sum(weights.values())
    if not math.isfinite(total):
        scale = max(weights.values())
        weights = {p: w / scale for p, w in weights.items()}
        total = sum(weights.values())
    return [(p, sorted(groups[p]), weights[p] / total) for p in sorted(weights)]


def _draw_by_source(rng, sources, n_spks: int) -> List[str]:
    """``n_spks`` speakers: a source per slot by weight, then speakers inside it.

    Speakers are drawn without replacement inside a source, like the unweighted
    path, unless a source was drawn more times than it has speakers.
    """
    picks = rng.choices(range(len(sources)), weights=[w for _, _, w in sources], k=n_spks)
    classes: List[str] = []
    for index in sorted(set(picks)):
        count = picks.count(index)
        spks = sources[index][1]
        if count <= len(spks):
            classes += rng.sample(spks, count)
        else:
            classes += [rng.choice(spks) for _ in range(count)]
    return classes


class SpeakerSampler:
    def __init__(
        self,
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
    ):
        """
        Sample a batch of data for specific speaker number and per-speaker's utterance.

        Args:
            data: Dict with key as spk-id and item is dataset's List[uttid]
            total_batch: how much batchs
            n_spks: In each batch contain N speakers.
            n_per: Numbers of utterance per speaker.
            fast_sampling: If True, sample speaker by group first, then sample speaker from group.
            select_by_sr_first: If True, sample speaker by SR group first, then sample speaker from group.
            seed: If set, yields the same batches every epoch and attaches a per-item
                seed to each entry, i.e. (spk, sr, item_seed) instead of (spk, sr), so
                the dataset can regenerate identical samples (deterministic validation).
            emit_epoch: Append this epoch's index to every entry, so a dataset in a
                worker process can move its knobs with the epoch (a recipe's
                `curriculum` block). Workers are re-created per epoch and hold their
                own copy of the dataset, so the epoch has to travel with the item;
                this is the same channel `length_schedule` uses for row length.
            source_weights: ``{spkid prefix: weight}``. Each batch slot first draws
                a source by weight, then a speaker uniformly inside it. Without
                it every speaker is equally likely, so a corpus's share of the
                batches is its share of the *speakers* -- which is how a subset
                that names articles as speakers can outweigh one with more real
                readers and more hours. Every speaker must fall under exactly
                one prefix (the longest one wins); one that falls under none
                would never be drawn.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.n_batch = total_batch
        self.n_spks = n_spks
        # (seconds, n_spks, prob) per bucket. The draw MUST NOT depend on rank:
        # under DDP the ranks step in lockstep, and two ranks running different
        # row lengths in the same step would average gradients over different
        # batch sizes and stall on every sync. Speaker choice stays rank-offset,
        # only the length is shared.
        self.length_schedule = list(length_schedule) if length_schedule else None
        self.emit_epoch = bool(emit_epoch)
        self._epoch = -1
        self._epoch_set_externally = False
        self.n_per = n_per
        self.data = data
        self.spk_pool = list(data.keys())
        self.fast_sampling = fast_sampling
        self.select_by_sr_first = select_by_sr_first
        if select_by_sr_first:
            self.sr_meta = defaultdict(lambda: defaultdict(list))
            for spk in sorted(data.keys()):
                for utt in list(data[spk]["utts"].keys()):
                    _sr = data[spk]["utts"][utt]["sr"]
                    self.sr_meta[_sr][spk].append(utt)

        del self.data

        self.sources = None
        if source_weights:
            if select_by_sr_first or fast_sampling:
                raise ValueError(
                    "source_weights cannot be combined with select_by_sr_first or "
                    "fast_sampling; both pick speakers another way."
                )
            self.sources = _group_by_source(self.spk_pool, source_weights)
            for name, spks, weight in self.sources:
                logger.info(
                    "speaker source %s: %d speakers, %.1f%% of batch slots",
                    name, len(spks), 100.0 * weight,
                )

        if n_spks > len(self.spk_pool):
            self.n_per = math.ceil((n_spks * n_per) / len(self.spk_pool))
            self.n_spks = len(self.spk_pool)
            logger.warning(
                "Sample larger than population, reset it to n_spk=%s and n_per=%s.",
                self.n_spks, self.n_per,
            )

        if fast_sampling:
            # shuffle the speaker pool first
            random.shuffle(self.spk_pool)

            # Groups of 10000; the remainder joins the last group, so no group
            # is too small to draw a batch from (or empty).
            self.n_group = max(1, len(self.spk_pool) // 10000)
            self.spk_pool_group = [
                self.spk_pool[i * 10000 : (i + 1) * 10000] for i in range(self.n_group - 1)
            ]
            self.spk_pool_group.append(self.spk_pool[(self.n_group - 1) * 10000 :])

    def __len__(self):
        return self.n_batch

    def set_epoch(self, epoch: int) -> None:
        """Which epoch the next pass is, told from outside.

        Counting passes internally is wrong on the one path that matters:
        a run resumed at epoch N starts the count at 0 again, and anything
        derived from the epoch -- the length draw, a curriculum's knob values --
        then silently replays the beginning of the schedule. Lightning calls this on
        `dataloader.batch_sampler.sampler` before each epoch's iterator is
        consumed (`_set_sampler_epoch`), including on the resumed one, which is
        what `sampler` below exists to make reachable. Without a caller the
        internal count still applies, so a plain PyTorch loop is unaffected.
        """
        self._epoch = int(epoch)
        self._epoch_set_externally = True

    @property
    def sampler(self):
        """Self, so Lightning's ``_set_sampler_epoch`` reaches ``set_epoch``.

        It looks at ``dataloader.sampler`` and ``dataloader.batch_sampler.sampler``;
        this class is the batch sampler, and a batch sampler that carries no inner
        sampler would be skipped.
        """
        return self

    def __iter__(self):
        rank, world_size = _distributed_rank_world()
        if self.rank is not None:
            rank = int(self.rank)
        if self.world_size is not None:
            world_size = int(self.world_size)
        if self.seed is not None:
            rng = random.Random(self.seed + rank * 1_000_003)
        elif world_size > 1:
            rng = random.Random(random.getrandbits(63) + rank * 1_000_003)
        else:
            rng = random
        if not self._epoch_set_externally:
            self._epoch += 1
        for batch_idx in range(self.n_batch):
            batch = []
            sr = None

            n_spks, row_seconds = self.n_spks, None
            if self.length_schedule is not None:
                # Seeded by (epoch, batch index) only -- identical on every rank.
                draw = random.Random(
                    (self.seed or 0) * 7_919 + self._epoch * 104_729 + batch_idx
                )
                r, acc = draw.random(), 0.0
                for seconds, spks, prob in self.length_schedule:
                    acc += prob
                    if r <= acc:
                        n_spks, row_seconds = spks, seconds
                        break
                else:
                    seconds, spks, _ = self.length_schedule[-1]
                    n_spks, row_seconds = spks, seconds

            if not self.fast_sampling:
                if self.select_by_sr_first:
                    sr = rng.sample(list(self.sr_meta.keys()), 1)[0]
                    classes = rng.sample(list(self.sr_meta[sr].keys()), n_spks)
                elif self.sources is not None:
                    classes = _draw_by_source(rng, self.sources, n_spks)
                else:
                    classes = rng.sample(self.spk_pool, n_spks)
            else:
                # sample group first
                group = rng.sample(self.spk_pool_group, 1)[0]
                classes = rng.sample(group, n_spks)

            for i, c in enumerate(classes):
                if self.seed is not None:
                    item_seed = (
                        self.seed * 1_000_003
                        + rank * self.n_batch * 1_009
                        + batch_idx * 1_009
                        + i * self.n_per
                    )
                    batch += [(c, sr, item_seed + j) for j in range(self.n_per)]
                else:
                    batch += [(c, sr)] * self.n_per

            if row_seconds is not None or self.emit_epoch:
                # 4-tuple = "this row is N seconds long", 5-tuple adds "and it
                # belongs to epoch N". Empty slots stay in place (None when
                # unseeded, None when there is no length schedule) so the arity
                # alone says which shape this is; see
                # DynamicBaseDataset.parse_item_key, which reads every shape.
                batch = [
                    (e[0], e[1], e[2] if len(e) == 3 else None, row_seconds)
                    for e in batch
                ]
            if self.emit_epoch:
                batch = [e + (self._epoch,) for e in batch]

            # shuffling the sequence
            rng.shuffle(batch)
            yield batch


# One PCG64 stream per tag, so reading one stream never shifts another.
_TAG_LENGTH, _TAG_SOURCE, _TAG_SPEAKER, _TAG_UTTERANCE = 0x4C454E, 0x535243, 0x53504B, 0x555454
_DRAW_BATCH_CHUNK = 4096


def _uniforms(seed: int, tag: int, start: int, n: int) -> np.ndarray:
    """Draws ``[start, start + n)`` of the (seed, tag) stream.

    PCG64 spends one 64-bit output per double, so ``advance`` skips the prefix exactly.
    """
    bits = np.random.PCG64(np.random.SeedSequence([seed, tag]))
    if start:
        bits.advance(int(start))
    return np.random.Generator(bits).random(int(n))


def _stable_id(name: str) -> int:
    """A process-independent integer for a name (``hash`` is salted per process)."""
    return zlib.crc32(name.encode("utf-8"))


class _Bucket(NamedTuple):
    seconds: Optional[float]  # emitted row length; None = the dataset's own
    min_seconds: float        # shortest utterance that fills the row
    n_items: int
    prob: float
    key: str                  # checkpoint key of this bucket's queues
    sources: List[Tuple[str, List[str], float]]  # (name, speakers, weight)
    utts: Dict[str, List[str]]                   # speaker -> utterances that fill the row


class CoverageSampler:
    """Training batches drawn from shuffled queues instead of fresh picks.

    A slot draws a source by weight, then takes the next speaker from that
    source's speaker queue, then that speaker's next utterance from its
    utterance queue. Every speaker of a source comes up once before any repeats,
    and every utterance of a speaker before any repeats; the expected mix is
    `SpeakerSampler`'s with ``n_per`` 1.

    Queues are kept per row length and hold only utterances at least that long,
    avoiding foreground padding caused by a short input. The row length is
    drawn per batch and is the same on every rank; the ranks deal disjoint slots
    of one walk. Task augmentation can still insert silence or replace a target.

    Every draw is a function of the seed and the position in the walk, so a run
    restarted from `checkpoint_state` continues where it stopped (`start_from`),
    and each item carries its own synthesis seed.

    Items are ``(speaker, None, item_seed, seconds, epoch, utterance)``;
    ``seconds`` is None without a length schedule and ``epoch`` is None unless
    ``emit_epoch``.
    """

    def __init__(
        self,
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
    ):
        self.n_batch = int(total_batch)
        self.emit_epoch = bool(emit_epoch)
        self.seed = int(seed)
        self.rank = rank
        self.world_size = world_size
        self._epoch = -1
        self._epoch_set_externally = False
        self._stage_start = {"batch": 0, "slot": 0, "counts": {}}
        self._resumed_world_size: Optional[int] = None

        speakers = sorted(data)
        groups = (
            _group_by_source(speakers, source_weights)
            if source_weights
            else [("", speakers, 1.0)]
        )
        spec = (
            [(float(s), float(s), int(n), float(p)) for s, n, p in length_schedule]
            if length_schedule
            else [(None, float(default_seconds), int(n_items), 1.0)]
        )
        if self.n_batch < 1 or int(n_items) < 1:
            raise ValueError("coverage sampler batch counts must be positive")
        if self.seed < 0:
            raise ValueError("coverage sampler seed must be non-negative")
        if any(not math.isfinite(s) or s <= 0 or n < 1 or not math.isfinite(p) or p <= 0
               for _, s, n, p in spec):
            raise ValueError("coverage sampler lengths, batch sizes and probabilities must be finite and positive")
        self.buckets = [self._bucket(data, groups, *row) for row in spec]
        self._cum_probs = np.cumsum([b.prob for b in self.buckets])
        self._cum_probs /= self._cum_probs[-1]
        self._cum_weights = [np.cumsum([w for _, _, w in b.sources]) for b in self.buckets]
        for cum in self._cum_weights:
            cum /= cum[-1]
        self._sizes = np.array([b.n_items for b in self.buckets])
        self._perm_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._position_cache = OrderedDict()
        self._queue_signatures = {}
        for bucket in self.buckets:
            digest = hashlib.sha256()
            for name, spks, _ in bucket.sources:
                digest.update(json.dumps([name, spks], ensure_ascii=False).encode())
                for spk in spks:
                    digest.update(json.dumps([spk, bucket.utts[spk]], ensure_ascii=False).encode())
            self._queue_signatures[bucket.key] = digest.hexdigest()
        self._queue_history = dict(self._queue_signatures)
        stage = [self.n_batch, self.emit_epoch,
                 [(b.key, b.seconds, b.n_items, b.prob,
                   [(name, weight) for name, _, weight in b.sources]) for b in self.buckets]]
        self._stage_signature = hashlib.sha256(json.dumps(stage).encode()).hexdigest()

    @staticmethod
    def _bucket(data, groups, seconds, min_seconds, n_items, prob) -> _Bucket:
        utts: Dict[str, List[str]] = {}
        sources = []
        for name, spks, weight in groups:
            kept = []
            for spk in spks:
                # Metafile order: the queues must index the same list in every process.
                fitting = [
                    utt for utt, meta in data[spk]["utts"].items()
                    if float(meta["length"]) / float(meta["sr"]) >= min_seconds - 1e-9
                ]
                if fitting:
                    utts[spk] = fitting
                    kept.append(spk)
            if kept:
                sources.append((name, kept, weight))
            else:
                logger.warning(
                    "coverage sampler: source %r has no utterance of %.1f s or more",
                    name, min_seconds,
                )
        if not sources:
            raise ValueError(f"no utterance fills a {min_seconds:.1f} s row")
        n_utts = sum(len(v) for v in utts.values())
        logger.info(
            "coverage sampler: %s-second rows x %d, prob %.2f -- %d speakers, %d utterances",
            seconds, n_items, prob, len(utts), n_utts,
        )
        return _Bucket(seconds, min_seconds, n_items, prob, min_seconds.hex(),
                       sources, utts)

    def __len__(self):
        return self.n_batch

    def set_seed(self, seed: int) -> None:
        """Seed a new walk; `start_from` replaces it with the recorded seed."""
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("coverage sampler seed must be non-negative")
        self._perm_cache.clear()
        self._position_cache.clear()

    def set_epoch(self, epoch: int) -> None:
        """Which epoch of the stage the next pass is; see `SpeakerSampler.set_epoch`."""
        self._epoch = int(epoch)
        self._epoch_set_externally = True

    @property
    def sampler(self):
        """Self, so Lightning's ``_set_sampler_epoch`` reaches ``set_epoch``."""
        return self

    # ------------------------------------------------------------------ #
    # Position in the walk
    # ------------------------------------------------------------------ #

    def _world(self) -> Tuple[int, int]:
        rank, world_size = _distributed_rank_world()
        if self.rank is not None:
            rank = int(self.rank)
        if self.world_size is not None:
            world_size = int(self.world_size)
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("coverage sampler rank must be inside a positive world size")
        return rank, world_size

    def _buckets_of(self, first_batch: int, n: int) -> np.ndarray:
        drawn = _uniforms(self.seed, _TAG_LENGTH, first_batch, n)
        index = np.searchsorted(self._cum_probs, drawn, side="right")
        return np.minimum(index, len(self.buckets) - 1)

    def _advance(self, state: Dict, n_batches: int, world_size: int) -> Dict:
        """The position ``n_batches`` batches after ``state``, without building any item."""
        counts = dict(state["counts"])
        if n_batches <= 0:
            return {"batch": state["batch"], "slot": state["slot"], "counts": counts}
        slot = state["slot"]
        for first in range(0, n_batches, _DRAW_BATCH_CHUNK):
            buckets = self._buckets_of(state["batch"] + first, min(_DRAW_BATCH_CHUNK, n_batches - first))
            slots = self._sizes[buckets] * world_size
            drawn = _uniforms(self.seed, _TAG_SOURCE, slot, int(slots.sum()))
            slot_bucket = np.repeat(buckets, slots)
            for b, bucket in enumerate(self.buckets):
                picked = np.searchsorted(self._cum_weights[b], drawn[slot_bucket == b], side="right")
                picked = np.minimum(picked, len(bucket.sources) - 1)
                for s, n in enumerate(np.bincount(picked, minlength=len(bucket.sources))):
                    if n:
                        key = f"{bucket.key}|{bucket.sources[s][0]}"
                        counts[key] = counts.get(key, 0) + int(n)
            slot += int(slots.sum())
        return {"batch": state["batch"] + n_batches, "slot": slot,
                "counts": counts}

    def _position(self, n_batches: int, world_size: int) -> Dict:
        if n_batches < 0:
            raise ValueError("coverage sampler position must be non-negative")
        previous = max((n for w, n in self._position_cache if w == world_size and n <= n_batches), default=0)
        start = self._position_cache.get((world_size, previous), self._stage_start)
        state = self._advance(start, n_batches - previous, world_size)
        self._position_cache[(world_size, n_batches)] = state
        if len(self._position_cache) > 16:
            self._position_cache.popitem(last=False)
        return {**state, "counts": dict(state["counts"])}

    def checkpoint_state(self, epochs_done: int, *, batches_done: Optional[int] = None) -> Dict:
        """Position after full epochs and any consumed batches of the current one.

        Consumers record completed batches, not prefetched sampler yields.
        Partial-epoch checkpoints can warm start; same-stage resumes require
        an epoch boundary because Lightning's dataloader is not stateful.
        """
        _, world_size = self._world()
        if epochs_done < 0 or batches_done is not None and not 0 <= batches_done <= self.n_batch:
            raise ValueError("invalid coverage sampler checkpoint position")
        return {
            "version": 2,
            "seed": self.seed,
            "world_size": world_size,
            "queue_signatures": dict(self._queue_history),
            "stage_signature": self._stage_signature,
            "epoch_boundary": batches_done in (None, 0, self.n_batch),
            "stage_start": {**self._stage_start, "counts": dict(self._stage_start["counts"])},
            "next": self._position(epochs_done * self.n_batch + (batches_done or 0), world_size),
        }

    def start_from(self, state: Dict, *, resume: bool) -> None:
        """Continue a walk recorded by `checkpoint_state`.

        ``resume``: the same stage again, placed by its epoch. Otherwise a new
        stage that begins where the recorded one ended. The recorded seed is kept
        either way, since the queues are shuffled by it.
        """
        if int(state.get("version", 0)) != 2:
            raise ValueError(f"unknown coverage sampler state: {state!r}")
        recorded = state.get("queue_signatures", {})
        if any(recorded.get(key) != signature for key, signature in self._queue_signatures.items()
               if resume or key in recorded):
            raise ValueError("coverage sampler eligible queues changed; start a new walk")
        if resume and state.get("stage_signature") != self._stage_signature:
            raise ValueError("coverage sampler stage settings changed; use a warm start instead of resume")
        if resume and not state.get("epoch_boundary", False):
            raise ValueError("coverage sampler resume requires an epoch-boundary checkpoint; use a warm start")
        # Checked on the first pass: the process group may not exist yet here.
        self._resumed_world_size = int(state["world_size"]) if resume else None
        if int(state["seed"]) != self.seed:
            logger.info("coverage sampler: continuing the walk of seed %d (not %d)",
                        state["seed"], self.seed)
        self.seed = int(state["seed"])
        self._perm_cache.clear()
        self._position_cache.clear()
        self._queue_history = {**recorded, **self._queue_signatures}
        chosen = state["stage_start"] if resume else state["next"]
        self._stage_start = {"batch": int(chosen["batch"]), "slot": int(chosen["slot"]),
                             "counts": dict(chosen["counts"])}

    # ------------------------------------------------------------------ #
    # Items
    # ------------------------------------------------------------------ #

    def _permutation(self, key: tuple, n: int) -> np.ndarray:
        perm = self._perm_cache.get(key)
        if perm is None:
            perm = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([self.seed, *key]))
            ).permutation(n)
            self._perm_cache[key] = perm
            if len(self._perm_cache) > 50_000:
                self._perm_cache.popitem(last=False)
        return perm

    def _item_seed(self, slot: int) -> int:
        # Odd multipliers are invertible mod 2**63 and mod 2**32 (numpy's seed
        # width), so distinct slots get distinct seeds.
        return (self.seed * 0x9E3779B97F4A7C15 + slot * 0xBF58476D1CE4E5B9) % (1 << 63)

    def _item(self, bucket: _Bucket, source: int, k: int, slot: int, epoch: int) -> tuple:
        name, speakers, _ = bucket.sources[source]
        passes, position = divmod(k, len(speakers))
        rows = struct.unpack(">Q", struct.pack(">d", bucket.min_seconds))[0]
        order = self._permutation((_TAG_SPEAKER, rows, _stable_id(name), passes), len(speakers))
        speaker = speakers[order[position]]
        # A pass visits each speaker once, so ``passes`` is this speaker's visit count.
        pool = bucket.utts[speaker]
        cycle, index = divmod(passes, len(pool))
        utt_order = self._permutation((_TAG_UTTERANCE, rows, _stable_id(speaker), cycle), len(pool))
        return (
            speaker,
            None,
            self._item_seed(slot),
            bucket.seconds,
            epoch if self.emit_epoch else None,
            pool[utt_order[index]],
        )

    def __iter__(self):
        rank, world_size = self._world()
        if self._resumed_world_size not in (None, world_size):
            raise ValueError(
                f"the stage ran at world size {self._resumed_world_size}, not {world_size}"
            )
        if not self._epoch_set_externally:
            self._epoch += 1
        epoch = self._epoch
        state = self._position(epoch * self.n_batch, world_size)
        buckets = self._buckets_of(state["batch"], self.n_batch)
        slots = self._sizes[buckets] * world_size
        drawn = _uniforms(self.seed, _TAG_SOURCE, state["slot"], int(slots.sum()))
        counts = state["counts"]
        slot, offset = state["slot"], 0
        for b in buckets:
            bucket = self.buckets[b]
            n = bucket.n_items
            picked = np.searchsorted(self._cum_weights[b], drawn[offset:offset + n * world_size],
                                     side="right")
            batch = []
            # Every rank advances the queues over all slots of the batch and keeps
            # its own share.
            for j, s in enumerate(np.minimum(picked, len(bucket.sources) - 1)):
                key = f"{bucket.key}|{bucket.sources[s][0]}"
                k = counts.get(key, 0)
                counts[key] = k + 1
                if rank * n <= j < (rank + 1) * n:
                    batch.append(self._item(bucket, int(s), k, slot + j, epoch))
            slot += n * world_size
            offset += n * world_size
            yield batch


class SpeakerGenderSampler:
    def __init__(
        self,
        total_batch: int,
        n_spks: int,
        n_per: int,
        spk_list_male: List,
        spk_list_female: List,
        spk_list_other: Optional[List] = None,
    ):
        """
        Sample a batch of data for specific speaker number and per-speaker's utterance.

        Args:
            total_batch: how much batchs
            n_spks: In each batch contain N speakers
            n_per: Numbers of utterance per speaker
            spk_list_male: speaker list of male speaker
            spk_list_female: speaker list of female speaker
            spk_list_other: speaker list of missed gender information
        """
        self.n_batch = total_batch
        self.n_spks = n_spks
        self.n_per = n_per
        self.spk_list_m = spk_list_male
        self.spk_list_f = spk_list_female
        self.spk_list_other = spk_list_other
        if spk_list_other is None:
            assert n_spks % 2 == 0
        else:
            assert n_spks % 3 == 0

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = []

            if self.spk_list_other is None:
                classes += random.sample(
                    self.spk_list_m, k=self.n_spks // 2
                )  # [choosed spks, ....]
                classes += random.sample(
                    self.spk_list_f, k=self.n_spks // 2
                )  # [choosed spks, ....]
            else:
                classes += random.sample(
                    self.spk_list_m, k=self.n_spks // 3
                )  # [choosed spks, ....]
                classes += random.sample(
                    self.spk_list_f, k=self.n_spks // 3
                )  # [choosed spks, ....]
                classes += random.sample(
                    self.spk_list_other, k=self.n_spks // 3
                )  # [choosed spks, ....]

            random.shuffle(classes)
            for c in classes:
                batch += [c] * self.n_per

            yield batch
