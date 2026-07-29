import os
from typing import Dict, Optional

import torch

from puresound.audio.io import AudioIO, wav_resampling
from puresound.utils import load_text_as_dict


class KaldiFormBaseDataset(torch.utils.data.Dataset):
    """
    Basic dataset follow Kaldi data preparation *.scp format.\n
    handcraft data folder must include:
        -- wav2scp.txt: audio path file
        -- [options] wav2ref.txt: audio for which reference audio, used in noisy to clean mapping

    Include:
        df: data frame
        idx_df: mapping idx to an unique df's key

    Args:
        folder: manifest folder
        resample_to: if not None, open waveform will resample to this value
    """

    def __init__(
        self,
        folder,
        resample_to: Optional[int] = None,
        mode: str = "train",
        audio_gain_normalized_to: Optional[int] = None,
        split_to_chunks_with_size: Optional[float] = None,
    ):
        super().__init__()
        self.folder = folder
        self.resample_to = resample_to
        assert mode.lower() in ["train", "dev", "eval"]
        self.mode = mode.lower()
        self.audio_gain_normalized_to = audio_gain_normalized_to
        self.split_to_chunks_with_size = split_to_chunks_with_size

        # Basic contents
        self._folder_content = {"wav2scp": "wav2scp.txt", "wav2ref": "wav2ref.txt"}
        self._load_df(self.folder)

    def __len__(self):
        return len(self.idx_df)

    def __getitem__(self, index: int):
        clean_speech = torch.empty(0)
        enroll_speech = torch.empty(0)

        key = self.idx_df[index]
        noisy_speech, sr = AudioIO.open(
            f_path=self.df[key]["wav2scp"],
            resample_to=self.resample_to,
            target_lvl=self.audio_gain_normalized_to,
        )
        noisy_speech = noisy_speech.squeeze()

        if "wav2enroll" in self.df[key]:
            enroll_speech, sr = AudioIO.open(
                f_path=self.df[key]["wav2enroll"],
                resample_to=self.resample_to,
                target_lvl=self.audio_gain_normalized_to,
            )
            enroll_speech = enroll_speech.squeeze()

        if self.mode == "eval":
            if self.split_to_chunks_with_size:
                chunk_length = int(sr * self.split_to_chunks_with_size)
                if noisy_speech.shape[-1] > chunk_length:
                    noisy_speech = noisy_speech.view(1, 1, 1, -1)
                    noisy_speech = torch.nn.functional.unfold(
                        noisy_speech,
                        kernel_size=(1, chunk_length),
                        stride=(1, chunk_length // 2),
                    )
                    noisy_speech = noisy_speech.squeeze(0).permute(1, 0)
                    if noisy_speech.dim() != 2:
                        noisy_speech = noisy_speech.view(1, -1)

        else:
            clean_speech, _sr = AudioIO.open(
                f_path=self.df[key]["wav2ref"],
                resample_to=self.resample_to,
                target_lvl=self.audio_gain_normalized_to,
            )
            if _sr != sr:
                print(
                    f"Reference audio samplerate {_sr} isn't same as Noisy audio {sr}, resampling to {sr} by Sox backend."
                )
                clean_speech, _ = wav_resampling(
                    wav=clean_speech, origin_sr=_sr, target_sr=sr, backend="sox"
                )
            clean_speech = clean_speech.squeeze()

        return {
            "noisy_speech": noisy_speech,
            "clean_speech": clean_speech,
            "conditional_speech": enroll_speech,
            "sr": sr,
            "name": key,
        }

    @property
    def folder_content(self):
        """
        Set like:
            'wav2scp': wav2scp.txt
            'wav2class': wav2class.txt
            'wav2ref': wav2ref.txt
            etc.
        """
        self._folder_content = {"wav2scp": "wav2scp.txt", "wav2ref": "wav2ref.txt"}
        return self._folder_content

    @folder_content.setter
    def folder_content(self, dct: Dict):
        self._folder_content.update(dct)
        print(f"Updated the content: {self._folder_content.keys()}")
        self._load_df(self.folder)

    def _load_df(self, folder: str) -> Dict:
        """method about loading manifest information."""
        self.df = {}
        load_dct = self._folder_content.copy()

        # check file, wav2scp is must needed
        if not os.path.isfile(f"{folder}/{self._folder_content['wav2scp']}"):
            raise FileNotFoundError(f"{self._folder_content['wav2scp']} is not found")

        else:
            _wav2scp = load_text_as_dict(f"{folder}/wav2scp.txt")
            for key in sorted(_wav2scp.keys()):
                self.df[key] = {"wav2scp": _wav2scp[key][0]}

            del load_dct["wav2scp"]

        if load_dct.keys != {}:
            for f in load_dct.keys():
                if not os.path.isfile(f"{folder}/{load_dct[f]}"):
                    # raise FileNotFoundError(f"{load_dct[f]} is not found")
                    print("Only incerece mode doesn't need wav2ref file")
                else:
                    _temp = load_text_as_dict(f"{folder}/{load_dct[f]}")
                    for key in sorted(_temp.keys()):
                        try:
                            if len(_temp[key]) != 1:
                                self.df[key].update({f: _temp[key][:]})
                            else:
                                self.df[key].update({f: _temp[key][0]})
                        except KeyError:
                            print(f"Non match key {key}")

        self.idx_df = self._idx2key(self.df)

    def _idx2key(self, df) -> Dict:
        """mapping df.keys to idx."""
        _idx_key = {}
        idx = 0
        for key in sorted(df.keys()):
            _idx_key[idx] = key
            idx += 1
        return _idx_key
