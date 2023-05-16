import argparse
import io

import torch
from puresound.src.audio import AudioIO
from puresound.src.utils import create_folder, load_text_as_dict
from tqdm import tqdm


def main(args):
    # download example
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
    )
    (get_speech_timestamps, _, read_audio, *_) = utils
    sampling_rate = 16000  # also accepts 8000

    create_folder(args.output_folder)
    wav_list = load_text_as_dict(args.wav_scp)
    ref2vad = io.open(f"{args.output_folder}/ref2vad.txt", "w", encoding="utf8")

    for uttid in tqdm(wav_list):
        wav = read_audio(wav_list[uttid][0], sampling_rate=sampling_rate)
        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(
            wav, vad_model, sampling_rate=sampling_rate
        )
        wav_vad = torch.zeros_like(wav)  # This is a 1D tensor
        for segment in speech_timestamps:
            start, end = segment["start"], segment["end"]
            wav_vad[start:end].fill_(1)

        AudioIO.save(
            wav_vad.view(1, -1), f"{args.output_folder}/{uttid}_vad.wav", sampling_rate
        )
        ref2vad.writelines(f"{uttid} {args.output_folder}/{uttid}_vad.wav\n")

    ref2vad.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_scp", type=str)
    parser.add_argument("output_folder", type=str)
    config = parser.parse_args()
    main(config)
