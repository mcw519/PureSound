import random


class SpeakerSampler:
    def __init__(self, data, total_batch, n_spks, n_per):
        """
        Sample batch data for specific speaker number and per-speaker's utterance.

        Args:
            data: Dict with key as spk-id and item is dataset's List[uttid]
            total_batch: how much batchs
            n_spks: In each batch contain N speakers.
            n_spks: Numbers of utterance per speaker.
        """
        self.n_batch = total_batch
        self.n_spks = n_spks
        self.n_per = n_per
        self.data = data
        self.spk_pool = list(data.keys())

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = random.sample(self.spk_pool, self.n_spks)  # [choosed spks, ....]
            for c in classes:
                utt_pool = self.data[c]
                utts = random.sample(utt_pool, self.n_per)
                batch += utts

            yield batch
