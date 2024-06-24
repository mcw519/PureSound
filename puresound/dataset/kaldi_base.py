import os
from typing import Dict, Optional

import torch

from puresound.audio.io import AudioIO
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

    def __init__(self, folder, resample_to: Optional[int] = None):
        super().__init__()
        self.folder = folder
        self.resample_to = resample_to
        self.df = self._load_df(self.folder)
        self.idx_df = self._idx2key(self.df)

    def __len__(self):
        return len(self.idx_df)

    def __getitem__(self, index: int):
        key = self.idx_df[index]
        noisy_speech, sr = AudioIO.open(f_path=self.df[key]["wav2scp"])
        noisy_speech = noisy_speech.squeeze()
        if "wav2ref" in self.df[key]:
            clean_speech, sr = AudioIO.open(f_path=self.df[key]["wav2ref"])
            clean_speech = clean_speech.squeeze()
        else:
            clean_speech = None
        return {
            "noisy_speech": noisy_speech,
            "clean_speech": clean_speech,
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
        return {"wav2scp": "wav2scp.txt", "wav2ref": "wav2ref.txt"}

    def _load_df(self, folder: str) -> Dict:
        """method about loading manifest information."""
        _df = {}
        load_dct = self.folder_content

        # check file, wav2scp is must needed
        if not os.path.isfile(f"{folder}/{self.folder_content['wav2scp']}"):
            raise FileNotFoundError(f"{self.folder_content['wav2scp']} is not found")

        else:
            _wav2scp = load_text_as_dict(f"{folder}/wav2scp.txt")
            for key in sorted(_wav2scp.keys()):
                _df[key] = {"wav2scp": _wav2scp[key][0]}

            del load_dct["wav2scp"]

        if load_dct.keys != {}:
            for f in load_dct.keys():
                if not os.path.isfile(f"{folder}/{load_dct[f]}"):
                    raise FileNotFoundError(f"{load_dct[f]} is not found")
                else:
                    _temp = load_text_as_dict(f"{folder}/{load_dct[f]}")
                    for key in sorted(_temp.keys()):
                        try:
                            if len(_temp[key]) != 1:
                                _df[key].update({f: _temp[key][:]})
                            else:
                                _df[key].update({f: _temp[key][0]})
                        except KeyError:
                            print(f"Non match key {key}")

        return _df

    def _idx2key(self, df) -> Dict:
        """mapping df.keys to idx."""
        _idx_key = {}
        idx = 0
        for key in sorted(df.keys()):
            _idx_key[idx] = key
            idx += 1
        return _idx_key
