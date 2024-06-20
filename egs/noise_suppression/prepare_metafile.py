import argparse
import io
import os
from typing import Optional

from puresound.audio.io import AudioIO
from puresound.utils import load_text_as_dict


def convert_kaldi_format_to_puresound_metafile(
    output_file_path: str,
    wav2scp_path: str,
    utt2spk_path: str,
    utt2gender: Optional[str] = None,
    separator: str = " ",
    insert_root_path: str = "",
):
    metafile = io.open(output_file_path, "w+", encoding="utf8")
    metafile.writelines("uttid, spkid, gender, path, length, sample rate, channels\n")
    wav2scp = load_text_as_dict(file_path=wav2scp_path, separator=separator)
    utt2spk = load_text_as_dict(file_path=utt2spk_path, separator=separator)
    if utt2gender is not None:
        utt2gender = load_text_as_dict(file_path=utt2gender, separator=separator)
    keys = sorted(wav2scp.keys())

    for utt_key in keys:
        gender = None

        if utt_key not in utt2spk:
            print(
                f"Can't find the {utt_key} speaker information in {utt2spk_path}, pass it."
            )
            continue

        if utt2gender is not None:
            if utt_key not in utt2gender:
                print(
                    f"Can't find the {utt_key} speaker information in {utt2spk}, pass it."
                )
                continue
            else:
                gender = utt2gender[utt2gender][0]

        speaker = utt2spk[utt_key][0]
        if insert_root_path is not None:
            audio_path = os.path.join(insert_root_path, wav2scp[utt_key][0])
        else:
            audio_path = wav2scp[utt_key][0]
        sample_rate, total_samples, _, num_channels = AudioIO.audio_info(
            f_path=audio_path
        )
        metafile.writelines(
            f"{utt_key}, {speaker}, {gender}, {audio_path}, {total_samples}, {sample_rate}, {num_channels}\n"
        )

    metafile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("wav2scp_path", type=str)
    parser.add_argument("utt2spk_path", type=str)
    parser.add_argument("--utt2gender_path", type=str, default=None)
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--insert_root_path", type=str, default=None)
    args = parser.parse_args()
    convert_kaldi_format_to_puresound_metafile(
        output_file_path=args.output_path,
        wav2scp_path=args.wav2scp_path,
        utt2spk_path=args.utt2spk_path,
        utt2gender=args.utt2gender_path,
        separator=args.separator,
        insert_root_path=args.insert_root_path,
    )
