import io
import os
from typing import Any, Dict, Optional

from puresound.src.utils import create_folder, str2bool


class MetafileParser:
    """
    Basic metafile looks like:
        uttid, spkid, gender, path, length, sample rate, channels
        116-288045-0000, 116, M, dev-other/116/288045/116-288045-0000.flac, 170400, 16000, 1
        116-288045-0001, 116, M, dev-other/116/288045/116-288045-0001.flac, 138160, 16000, 1

    Includes label path and start times:
        uttid, spkid, gender, path, length, sample rate, channels, label path, start time
        116-288045-0000, 116, M, dev-other/116/288045/116-288045-0000.flac, 170400, 16000, 1, dev-other/116/288045/116-288045-0000.label, 12345
        116-288045-0001, 116, M, dev-other/116/288045/116-288045-0001.flac, 138160, 16000, 1, dev-other/116/288045/116-288045-0001.label, 45623
    """

    @staticmethod
    def read_from_metafile(
        f_path: str,
        use_speaker_as_key: bool = False,
        insert_corpus_root_path: Optional[str] = None,
        with_label_column: bool = False,
        with_start_time_column: bool = False,
    ):
        """
        Args:
            f_path: the path of metafile
            use_speaker_as_ke: if true, set spkid as key, content as a dict with key is uttid
            insert_corpus_root_path: if not None, insert root path before audio path
            with_label_column: if true, use label information
            with_start_time_column: if ture, support start time information
        
        Returns:
            Dict
        """
        meta_dict = {}
        with io.open(f_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == "":
                    continue

                if not with_label_column:
                    (
                        uttid,
                        spkid,
                        gender,
                        audio_path,
                        length,
                        sr,
                        channels,
                    ) = line.strip().split(", ")

                else:
                    if not with_start_time_column:
                        (
                            uttid,
                            spkid,
                            gender,
                            audio_path,
                            length,
                            sr,
                            channels,
                            label_path,
                        ) = line.strip().split(", ")

                    else:
                        (
                            uttid,
                            spkid,
                            gender,
                            audio_path,
                            length,
                            sr,
                            channels,
                            label_path,
                            start_time,
                        ) = line.strip().split(", ")

                if insert_corpus_root_path is not None:
                    audio_path = os.path.join(insert_corpus_root_path, audio_path)
                    if with_label_column:
                        label_path = os.path.join(insert_corpus_root_path, label_path)

                if not use_speaker_as_key:
                    meta_dict[uttid] = {
                        "spkid": spkid,
                        "gender": gender,
                        "path": audio_path,
                        "length": length,
                        "sr": sr,
                        "channels": channels,
                    }

                    if with_label_column:
                        meta_dict[uttid].update({"label_path": label_path})

                    if with_start_time_column:
                        meta_dict[uttid].update({"start_time": start_time})

                else:
                    if spkid not in meta_dict.keys():
                        meta_dict[spkid] = {
                            "gender": gender,
                            "sr": sr,
                            "channels": channels,
                            "utts": {},
                        }

                    meta_dict[spkid]["utts"].update(
                        {
                            uttid: {
                                "path": audio_path,
                                "length": length,
                                "channels": channels,
                            }
                        }
                    )

                    if with_label_column:
                        meta_dict[spkid]["utts"][uttid].update(
                            {"label_path": label_path}
                        )

                    if with_start_time_column:
                        meta_dict[spkid]["utts"][uttid].update(
                            {"start_time": start_time}
                        )

        return meta_dict

    @staticmethod
    def create_scp_files(
        metafile_path: str,
        out_folder: str,
        insert_corpus_root_path: Optional[str] = None,
        with_label_column: bool = False,
        with_start_time_column: bool = False,
        rename_uttid: bool = False,
        add_prefix: Optional[str] = None,
    ) -> None:
        """
        Args:
            metafile_path: metadata path
            out_folder: output data folder
            insert_corpus_root_path: if not None, insert root path before audio path
            with_label_column: if true, use label information
            with_start_time_column: if ture, support start time information
            rename_uttid: rename uttid by sequential name start from 0
            add_prefix: add prefix in uttid, looks like {prefix}_{uttid}
        """
        create_folder(out_folder)
        wav2scp = io.open(f"{out_folder}/wav2scp.txt", "w", encoding="utf-8")
        wav2spk = io.open(f"{out_folder}/wav2spk.txt", "w", encoding="utf-8")
        wav2gender = io.open(f"{out_folder}/wav2gender.txt", "w", encoding="utf-8")
        wav2duration = io.open(f"{out_folder}/wav2duration.txt", "w", encoding="utf-8")
        if with_label_column:
            wav2label = io.open(f"{out_folder}/wav2label.txt", "w", encoding="utf-8")

        if with_start_time_column:
            wav2start = io.open(f"{out_folder}/wav2start.txt", "w", encoding="utf-8")

        meta_dct = MetafileParser.read_from_metafile(
            f_path=metafile_path,
            use_speaker_as_key=False,
            insert_corpus_root_path=insert_corpus_root_path,
            with_label_column=with_label_column,
            with_start_time_column=with_start_time_column,
        )

        for idx, key in enumerate(sorted(meta_dct.keys())):
            spkid = meta_dct[key]["spkid"]
            gender = meta_dct[key]["gender"]
            path = meta_dct[key]["path"]
            duration = float(meta_dct[key]["length"]) / float(meta_dct[key]["sr"])
            uttid = key if not rename_uttid else str(idx).zfill(5)
            uttid = uttid if add_prefix is None else f"{add_prefix}_{uttid}"

            wav2scp.writelines(f"{uttid} {path}\n")
            wav2spk.writelines(f"{uttid} {spkid}\n")
            wav2gender.writelines(f"{uttid} {gender}\n")
            wav2duration.writelines(f"{uttid} {duration}\n")
            if with_label_column:
                label_path = meta_dct[key]["label_path"]
                wav2label.writelines(f"{uttid} {label}\n")

            if with_start_time_column:
                s_time = meta_dct[key]["start_time"]
                wav2start.writelines(f"{uttid} {s_time}\n")

        wav2scp.close()
        wav2spk.close()
        wav2gender.close()
        wav2duration.close()
        if with_label_column:
            wav2label.close()
        if with_start_time_column:
            wav2start.close()
