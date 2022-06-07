import argparse
import io
import os
import random
from copy import deepcopy
from typing import Any, Dict, Optional

from puresound.src.utils import create_folder


class Parser():
    def __init__(self, config: Any) -> None:
        """
        Args:
            root_folder: corpus root path
        """
        self.config = config
    
    @staticmethod
    def read_librispeech_metadata(f_path: str, insert_root: Optional[str] = None) -> Dict:
        """
        Args:
            f_path: metadata path
            insert_root: if true, insert root path before audio path
        """
        meta_dict = {}
        with io.open(f_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == "":
                    continue

                uttid, spkid, gender, audio_path, length, sr, channels = line.strip().split(', ')
                if insert_root is not None:
                    audio_path = os.path.join(insert_root, audio_path)
                
                
                if spkid not in meta_dict.keys():
                    meta_dict[spkid] = {
                        'gender': gender,
                        'sr': sr,
                        'channels': channels,
                        'utts': {}}
                
                meta_dict[spkid]['utts'].update({uttid:{'path': audio_path, 'length': length}})
        
        return meta_dict


def pick_enroll(meta_pool: Dict, spk: str, uttid: str, n_enroll: int = 5) -> str:
    pool = deepcopy(meta_pool[spk])
    del pool['utts'][uttid]
    enroll_uttid = random.sample(list(pool['utts'].keys()), k=n_enroll) # smaple N utts
    enroll_audios_path = [pool['utts'][uttid]['path'] for uttid in enroll_uttid]
    return enroll_audios_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("corpus_metadata", type=str)
    parser.add_argument("corpus_root", type=str)
    config = parser.parse_args()

    create_folder(config.output_folder)
    
    meta_pool = Parser.read_librispeech_metadata(f_path=config.corpus_metadata, insert_root=config.corpus_root)
    dct = {}

    with io.open(config.meta_path, 'r') as f:
        for line in f.readlines()[1:]:
            uttid = line.strip().split(',')[0]
            uttid1, uttid2 = uttid.strip().split('_')
            mixed = line.strip().split(',')[1]
            s1 = line.strip().split(',')[2]
            s2 = line.strip().split(',')[3]
            spk1 = uttid.split('_')[0].split('-')[0]
            spk2 = uttid.split('_')[1].split('-')[0]
            enroll_spk1 = " ".join(pick_enroll(meta_pool, spk1, uttid1))
            enroll_spk2 = " ".join(pick_enroll(meta_pool, spk2, uttid2))
            dct[f"{uttid}_1"] = {'noisy': mixed, 'ref': s1, 'spk': spk1, 'all_spks': f"{spk1}-{spk2}", 'enroll': enroll_spk1}
            dct[f"{uttid}_2"] = {'noisy': mixed, 'ref': s2, 'spk': spk2, 'all_spks': f"{spk1}-{spk2}", 'enroll': enroll_spk2}

    for key in sorted(dct.keys()):
        with io.open(f'{config.output_folder}/wav2scp.txt', 'a+', encoding='utf-8') as f:
            f.writelines(f"{key} {dct[key]['noisy']}\n")

    for key in sorted(dct.keys()):
        with io.open(f'{config.output_folder}/wav2ref.txt', 'a+', encoding='utf-8') as f:
            f.writelines(f"{key} {dct[key]['ref']}\n")

    for key in sorted(dct.keys()):
        with io.open(f'{config.output_folder}/ref2spk.txt', 'a+', encoding='utf-8') as f:
            f.writelines(f"{key} {dct[key]['spk']}\n")

    for key in sorted(dct.keys()):
        with io.open(f'{config.output_folder}/wav2spk.txt', 'a+', encoding='utf-8') as f:
            f.writelines(f"{key} {dct[key]['all_spks']}\n")

    for key in sorted(dct.keys()):
        with io.open(f'{config.output_folder}/ref2list.txt', 'a+', encoding='utf-8') as f:
            f.writelines(f"{key} {dct[key]['enroll']}\n")
