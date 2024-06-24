import random
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch

from puresound.audio.augmentaion import AudioEffectAugmentor
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_white_noise
from puresound.dataset.parser import MetafileParser


class DynamicBaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metafile_path: str,
        min_utt_length_in_seconds: float = 3.0,
        min_utts_in_each_speaker: int = 5,
        target_sr: Optional[int] = None,
        training_sample_length_in_seconds: float = 6.0,
        audio_gain_nomalized_to: Optional[int] = None,
        augmentation_speech_args: Optional[Dict] = None,
        augmentation_noise_args: Optional[Dict] = None,
        augmentation_reverb_args: Optional[Dict] = None,
        augmentation_speed_args: Optional[Dict] = None,
        augmentation_ir_response_args: Optional[Dict] = None,
        augmentation_src_args: Optional[Dict] = None,
        augmentation_hpf_args: Optional[Dict] = None,
        augmentation_volume_args: Optional[Dict] = None,
    ):
        super().__init__()
        # Matafile related
        self.metafile_path = metafile_path
        self.min_utt_length_in_seconds = min_utt_length_in_seconds
        self.min_utts_in_each_speaker = min_utts_in_each_speaker

        # Audio related
        self.target_sr = target_sr
        self.audio_gain_nomalized_to = audio_gain_nomalized_to
        self.training_sample_length_in_seconds = training_sample_length_in_seconds
        self.training_sample_length = int(
            self.target_sr * self.training_sample_length_in_seconds
        )

        # Augmentation related
        self.augmentation_speech_args = augmentation_speech_args
        self.augmentation_noise_args = augmentation_noise_args
        self.augmentation_reverb_args = augmentation_reverb_args
        self.augmentation_speed_args = augmentation_speed_args
        self.augmentation_ir_response_args = augmentation_ir_response_args
        self.augmentation_src_args = augmentation_src_args
        self.augmentation_hpf_args = augmentation_hpf_args
        self.augmentation_volume_args = augmentation_volume_args

        self.init_necessary()

    def init_necessary(self):
        self.meta, self.gender_meta, self.gender_spks = self.gen_meta(
            metafile_path=self.metafile_path,
            min_utt_length=self.min_utt_length_in_seconds,
            min_utts_in_spk=self.min_utts_in_each_speaker,
        )

        self.total_spks = sorted(list(self.meta.keys()))
        # add speaker index
        self.spk2idx = {}
        for idx, spkid in enumerate(self.total_spks):
            self.spk2idx[spkid] = idx

        self.init_augmentor()

    def __len__(self):
        pass

    def __getitem__(self):
        raise NotImplementedError

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
        if self.augmentation_noise_args is not None:
            self.augmentor.load_bg_noise_from_folder(
                self.augmentation_noise_args["noise_folder"]
            )
            print(
                f"Augmentor finished load {len(self.augmentor.bg_noise.keys())} noises"
            )

        if self.augmentation_reverb_args is not None:
            self.augmentor.load_rir_from_folder(
                self.augmentation_reverb_args["rir_folder"]
            )
            print(f"Augmentor finished load {len(self.augmentor.rir.keys())} rirs")

        print("----" * 30)

    def choose_an_utterance_by_speaker_name(
        self,
        target_speaker_name: str,
        ignoring_utt_list: Optional[List[str]] = None,
        select_channel: Optional[int] = None,
    ):
        """
        Random select an utterance in dataset by given a target speaker name

        Args:
            target_speaker_name: the key (name) of target speaker
            ignoring_utt_list: a list of utterance name for those we don't want to pick it
            select_channel: if given, only used the selected channel, it would make sure is single channel

        Returns:
            target speech tensor and its manifest information
        """
        timeout = 0
        target_speech_pool = deepcopy(self.meta[target_speaker_name]["utts"])
        check_key_list = list(target_speech_pool.keys())

        if ignoring_utt_list is not None:
            for ignored_key in ignoring_utt_list:
                if ignored_key in check_key_list:
                    del target_speech_pool[ignored_key]

        target_speech_pool = list(target_speech_pool.keys())
        # Chooce only one utterance
        tgt_key = random.sample(target_speech_pool, k=1)[0]

        target_speech, sr = AudioIO.open(
            f_path=self.meta[target_speaker_name]["utts"][tgt_key]["path"],
            target_lvl=self.audio_gain_nomalized_to,
            resample_to=self.target_sr,
        )

        # Check the waveform is not empty
        while target_speech.abs().mean() == 0:
            if timeout < 5:
                timeout += 1
                print(
                    f"Open an empty segment: {self.meta[target_speaker_name]['utts'][tgt_key]['path']}."
                )
                print(f"Retry {timeout} times.")
                target_speech, sr, (target_speaker_name, tgt_key) = (
                    self.choose_an_utterance_by_speaker_name(
                        target_speaker_name=target_speaker_name,
                        ignoring_utt_list=ignoring_utt_list,
                    )
                )
            else:
                raise RuntimeError(f"Timeout, can't find a useful utterance.")

        if select_channel is not None:
            if target_speech.shape[0] > select_channel:
                target_speech = target_speech[select_channel].reshape(1, -1)

        return target_speech, sr, (target_speaker_name, tgt_key)

    def apply_audio_augmentation(self):
        raise NotImplementedError

    def align_audio_list(
        self, wav_list: List[torch.Tensor], length: int, padding_type: str = "zero"
    ):
        """Randome shift, padding or truncate to match target length."""
        padding_type = padding_type.lower()
        assert padding_type in ["zero", "normal"]
        out_list = []

        for wav in wav_list:
            if wav.shape[-1] >= length:
                offset = random.randint(0, int(wav.shape[-1]) - length)
                # Avoid choice the zero tensor
                while wav[:, offset : offset + length].abs().mean() == 0:
                    # print(
                    #     f"This segment {wav[..., offset : offset + length]} from {offset} to {offset + length} is a empty tensor. RETRY."
                    # )
                    offset = random.randint(0, int(wav.shape[-1]) - length)
                    if wav[:, offset : offset + length].abs().mean() != 0:
                        break
                clips_wav = wav[:, offset : offset + length]
                assert clips_wav.shape[-1] == length

            else:
                ch, l = wav.shape
                offset = random.randint(0, length - int(wav.shape[-1]))
                pre_padd = torch.zeros(ch, offset, device=wav.device, dtype=wav.dtype)
                suf_padd = torch.zeros(
                    ch, length - l - offset, device=wav.device, dtype=wav.dtype
                )
                clips_wav = torch.cat([pre_padd, wav, suf_padd], dim=-1)
                assert clips_wav.shape[-1] == length

            if padding_type == "zero":
                pass

            elif padding_type == "normal":
                snr = [40]
                clips_wav = add_bg_white_noise(wav=clips_wav, snr_list=snr)[0]

            out_list.append(clips_wav)
        return out_list

    def avoid_audio_clipping(self, wav_list: List[torch.Tensor]):
        max_sample = np.max([x.abs().max().item() for x in wav_list])
        if max_sample > 1:
            out_list = []
            for wav in wav_list:
                out_list.append(wav / max_sample)

            return out_list
        else:
            return wav_list
