from typing import Optional

import torch

from puresound.audio.augmentaion import AudioEffectAugmentor
from puresound.audio.spectrum import wav_to_stft
from puresound.dataset.parser import MetafileParser


class DynamicBaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metafile_path: str,
        noise_folder: Optional[str] = None,
        rir_folder: Optional[str] = None,
    ):
        self.noise_folder = noise_folder
        self.rir_folder = rir_folder

        self.init_necessary()
    
    def init_necessary(self):
        raise NotImplementedError

    def __len__(self):
        pass

    def gen_meta(
        self, metafile_path: str, min_utts_in_spk: int = 10, min_utt_length: float = 3.0
    ):
        """
        Generating the metadata, and also removed speaker whos utterances number less than min_utts_in_spk
        """
        print("----" * 30)
        print("Dataset creating and filtering")

        meta = MetafileParser.read_from_metafile(
            f_path=metafile_path, use_speaker_as_key=True
        )

        remove_key_by_length = 0
        for spk in sorted(meta.keys()):
            for utt in list(meta[spk]["utts"].keys()):
                _s = float(meta[spk]["utts"][utt]["length"]) / float(meta[spk]["sr"])
                if _s < float(min_utt_length):
                    del meta[spk]["utts"][utt]
                    remove_key_by_length += 1

        print(
            f"remove {remove_key_by_length} utts which utts duration less than {min_utt_length} seconds."
        )

        delete_spk = []
        for spk in sorted(meta.keys()):
            if len(meta[spk]["utts"]) < min_utts_in_spk:
                delete_spk.append(spk)

        for spk in delete_spk:
            del meta[spk]

        print(
            f"Delete {len(delete_spk)} speakers which less than {min_utts_in_spk} utterances."
        )

        # Add corpus id in meta file
        # We assume the speaker named as {corpus_name}_{speaker_name}
        all_corpus_id = set()
        for spk in sorted(meta.keys()):
            corpus_id = spk.strip().split("_")[0]
            meta[spk]["corpus_id"] = corpus_id
            all_corpus_id.add(corpus_id)

        gender_meta = {"m": {}, "f": {}, "other": {}}
        gender_spks = {"m": [], "f": [], "other": []}
        for cid in all_corpus_id:
            gender_meta["m"][cid] = []
            gender_meta["f"][cid] = []
            gender_meta["other"][cid] = []

        missed_gender_info = 0
        for spk in meta.keys():
            cid = meta[spk]["corpus_id"]
            if (
                meta[spk]["gender"].lower() == "m"
                or meta[spk]["gender"].lower() == "male"
            ):
                gender_meta["m"][cid].append(spk)
                gender_spks["m"].append(spk)

            elif (
                meta[spk]["gender"].lower() == "f"
                or meta[spk]["gender"].lower() == "female"
            ):
                gender_meta["f"][cid].append(spk)
                gender_spks["f"].append(spk)

            else:
                missed_gender_info += 1
                gender_meta["other"][cid].append(spk)
                gender_spks["other"].append(spk)

        _new_corpus_id = set()
        for cid in all_corpus_id:
            print(f"{cid:>10}:           male speakers = {len(gender_meta['m'][cid])}")
            print(f"{cid:>10}:         female speakers = {len(gender_meta['f'][cid])}")
            print(
                f"{cid:>10}:  unknow gender speakers = {len(gender_meta['other'][cid])}"
            )

            if len(gender_meta["other"][cid]) != 0:
                if len(gender_meta["other"][cid]) < 0:
                    print(
                        f"remove corpus id {cid} because speaker numbers less than 4."
                    )
                else:
                    _new_corpus_id.add(cid)
            else:
                if len(gender_meta["m"][cid]) < 0 or len(gender_meta["f"][cid]) < 0:
                    print(
                        f"remove corpus id {cid} because speaker numbers less than 4."
                    )
                else:
                    _new_corpus_id.add(cid)

        self.all_corpus_id = _new_corpus_id

        total_male_speakers = 0
        total_female_speakers = 0
        for cid in self.all_corpus_id:
            total_male_speakers += len(gender_meta["m"][cid])
            total_female_speakers += len(gender_meta["f"][cid])

        print(f"Overall have {len(self.all_corpus_id)} corpus, {self.all_corpus_id}")
        print(f"Overall have {total_male_speakers} male speakers")
        print(f"Overall have {total_female_speakers} female speakers")
        print(f"Missed gender information: {missed_gender_info}")
        print(f"Total speakers: {len(meta.keys())}")
        print("----" * 30)

        return meta, gender_meta, gender_spks

    def init_augmentor(self):
        self.augmentor = AudioEffectAugmentor()
        if self.noise_folder:
            self.augmentor.load_bg_noise_from_folder(self.noise_folder)
            print(
                f"Augmentor finished load {len(self.augmentor.bg_noise.keys())} noises"
            )

        if self.rir_folder:
            self.augmentor.load_rir_from_folder(self.rir_folder)
            print(f"Augmentor finished load {len(self.augmentor.rir.keys())} rirs")
        
        print("----" * 30)
    
