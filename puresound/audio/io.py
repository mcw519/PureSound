import random
from typing import Optional, Tuple

import torch
import torchaudio

from .volume import normalize_waveform, rescale_waveform


class AudioIO:
    def __init__(self, verbose: bool = False) -> None:
        """
        Args:
            verbose: show details
        """
        self.verbose = verbose

    @staticmethod
    def audio_info(f_path: str):
        """Return audio's information"""
        metadata = torchaudio.info(f_path)
        sample_rate = metadata.sample_rate
        total_seconds = round(metadata.num_frames / sample_rate, 2)
        num_channels = metadata.num_channels
        return sample_rate, total_seconds, num_channels

    @staticmethod
    def open(
        f_path: str,
        normalized: bool = False,
        target_lvl: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Open an audio and then normalized or rescale the average amplitude.

        Args:
            f_path: Audio file path
            normalized: normalized waveform by average amplitude
            target_lvl: Target level in dB
            verbose: show detail

        Returns:
            waveform tensor and its sampling rate
        """
        wav, sr = torchaudio.load(f_path)
        avg_amp_ori = torch.mean(torch.abs(wav), dim=-1)

        if normalized:
            if target_lvl is not None and verbose:
                print(
                    f"You choosed the waveform nomalized, the target_lvl would not be used."
                )
                wav = normalize_waveform(wav=wav, amp_type="avg")

        elif target_lvl is not None:
            wav = rescale_waveform(
                wav=wav, target_lvl=target_lvl, amp_type="rms", scale="dB"
            )
            avg_amp_rescale = torch.mean(torch.abs(wav), dim=-1)

        if verbose:
            print(f"Open file: {f_path}")
            print(f"Avg_amp: {avg_amp_ori.item()}")
            if target_lvl is not None:
                print(f"RMS_rescale: {avg_amp_rescale.item()}")

        return wav, sr

    @staticmethod
    def save(wav: torch.Tensor, f_path: str, sr: int, **kwargs):
        """
        Save a waveform in disk.

        Args:
            wav: waveform tensor with shape [..., L]
            f_path: Audio file save path
            sr: sampling rate
        """
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(f"{f_path}", wav, sr, format="wav", **kwargs)

    @staticmethod
    def audio_cut(wav: torch.Tensor, sr: int, length_s: float):
        """Random cut audio in specific length, if not enough length padding zeros after sequences."""
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        wav, offset, end_offset = AudioIO.cut_audio(
            wav=wav, sr=sr, length_s=length_s, padding=True
        )
        return wav, (offset, end_offset)

    @staticmethod
    def cut_audio(
        wav: torch.Tensor, sr: int, length_s: int, padding: bool = False
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Random cut audio in specific length, if not enough length padding zeros after sequences.

        Args:
            wav: waveform tensor has last dimension is samples
            sr: sampling rate
            length_s: expect cutting length (in second)
            padding: if not enough length padding zeros after sequences

        Returns:
            return cutted audios and its slices index
        """
        audio_len = wav.shape[-1]
        target_len = sr * length_s

        if audio_len > target_len:
            offset = random.randint(0, int(audio_len) - target_len)
            wav = wav[:, offset : offset + target_len]
            end_offset = offset + target_len
        else:
            if padding:
                padding_zeros = torch.zeros(*wav.size()[:-1], target_len - audio_len)
                wav = torch.cat([wav, padding_zeros], dim=-1)
                offset = 0
                end_offset = offset + target_len

            else:
                offset = 0
                end_offset = wav.shape[-1]

        return wav, offset, end_offset
