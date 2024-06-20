import random
from typing import Dict, List, Optional


class SpeakerSampler:
    def __init__(
        self,
        data: Dict,
        total_batch: int,
        n_spks: int,
        n_per: int,
        fast_sampling: bool = False,
    ):
        """
        Sample a batch of data for specific speaker number and per-speaker's utterance.

        Args:
            data: Dict with key as spk-id and item is dataset's List[uttid]
            total_batch: how much batchs
            n_spks: In each batch contain N speakers.
            n_spks: Numbers of utterance per speaker.
            fast_sampling: If True, sample speaker by group first, then sample speaker from group.
        """
        self.n_batch = total_batch
        self.n_spks = n_spks
        self.n_per = n_per
        self.data = data
        self.spk_pool = list(data.keys())
        self.fast_sampling = fast_sampling
        del self.data

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
        for _ in range(self.n_batch):
            batch = []

            if not self.fast_sampling:
                classes = random.sample(
                    self.spk_pool, self.n_spks
                )  # [choosed spks, ....]

            else:
                # sample group first
                group = random.sample(self.spk_pool_group, 1)[0]
                classes = random.sample(group, self.n_spks)

            for c in classes:
                batch += [c] * self.n_per

            # shuffling the sequence
            random.shuffle(batch)
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
