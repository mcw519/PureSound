import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist


logger = logging.getLogger(__name__)


def _distributed_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


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
    ):
        """
        Sample a batch of data for specific speaker number and per-speaker's utterance.

        Args:
            data: Dict with key as spk-id and item is dataset's List[uttid]
            total_batch: how much batchs
            n_spks: In each batch contain N speakers.
            n_spks: Numbers of utterance per speaker.
            fast_sampling: If True, sample speaker by group first, then sample speaker from group.
            select_by_sr_first: If True, sample speaker by SR group first, then sample speaker from group.
            seed: If set, yields the same batches every epoch and attaches a per-item
                seed to each entry, i.e. (spk, sr, item_seed) instead of (spk, sr), so
                the dataset can regenerate identical samples (deterministic validation).
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
        self._epoch = -1
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

            # compute the number of groups
            self.n_group = len(self.spk_pool) // 10000
            self.spk_pool_group = []
            for i in range(self.n_group):
                self.spk_pool_group.append(self.spk_pool[i * 10000 : (i + 1) * 10000])

            self.spk_pool_group.append(self.spk_pool[(i + 1) * 10000 :])

    def __len__(self):
        return self.n_batch

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

            if row_seconds is not None:
                # 4-tuple = "this row is N seconds long". The seed slot stays in
                # place (None when unseeded) so the arity alone says which shape
                # this is; see NoiseSuppressionDataset.__getitem__.
                batch = [
                    (e[0], e[1], e[2] if len(e) == 3 else None, row_seconds)
                    for e in batch
                ]

            # shuffling the sequence
            rng.shuffle(batch)
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
