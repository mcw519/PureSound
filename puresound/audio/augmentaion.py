import random
from typing import List, Optional

import torch
import torchaudio

from puresound.audio.dsp import wav_resampling
from puresound.audio.impluse_response import rand_add_2nd_filter_response, wav_apply_rir
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise, add_bg_white_noise
from puresound.audio.volume import rand_gain_distortion
from puresound.src.utils import recursive_read_folder


class AudioEffectAugmentor:
    """
    Audio data augmentation on waveform.

    Includes:
        Noise:
            Background noise
            White Gaussian noise
        Reverberation:
            Full / Early / Direct
        Volume:
            Gain distortion
        Sox effects:
            volume up/down
            speech up / slow down
            pitch shift
        Filters:
            Random Biquad Filter
            sample rate convert
            High-pass Filter

    Ex:
        augmentor = AudioAugmentor()
        augmentor._load_rir_from_folder(RIR_folder)
        wav = augmentor.apply_rir(wav)
    """

    def __init__(self):
        pass

    def load_bg_noise_from_folder(self, folder: str, suffix: str = ".wav"):
        """load bg-noise from folder path"""
        self.bg_noise = self._load_wav_folder(folder, suffix=suffix)

    def load_rir_from_folder(self, folder: str, suffix: str = ".wav"):
        """load RIR from folder path"""
        self.rir = self._load_wav_folder(folder, suffix=suffix)

    def _load_wav_folder(self, folder: str, suffix: str = ".wav"):
        """load all waveform in folder, and split the waveform id to be key"""
        temp = {}
        wav_list = []
        recursive_read_folder(folder, suffix, wav_list)
        for file in wav_list:
            file = file.strip().split(" ")[1]
            uttid = "_".join(file.split("/")[-1].split(".")[0:-1])
            temp[uttid] = {"wav_path": file}

        return temp

    def sox_volume_perturbed(self, wav: torch.Tensor, vol_ratio: float, sr: int):
        """
        Getting Sox volume adjustation by a specific parameter.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            vol_ratio: the ratio for volume up/down. The general range is in [0.125, 2]
            sr: waveform sampling rate

        Returns:
            waveform has been done volume adjusted.
        """
        effects = [["vol", str(vol_ratio)]]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)

        return wav, (vol_ratio)

    def sox_speed_perturbed(self, wav: torch.Tensor, speed: float, sr: int):
        """
        Getting Sox speed up/slow down adjustation by a specific parameter.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            speed: the ratio for speed up or slow down. The general range is in [0.8, 1.2]
            sr: waveform sampling rate

        Returns:
            waveform has been done speed up or slow down.
        """
        effects = [["speed", str(speed)], ["rate", str(sr)]]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)

        return wav, (speed)

    def sox_pitch_perturbed(self, wav: torch.Tensor, shift_ratio: int, sr: int):
        """
        Getting Sox pitch shift adjustation by a specific parameter.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            shift_ratio: the value for shifting. The general range is in [-100, 100]
            sr: waveform sampling rate

        Returns:
            waveform has been done pitch shifted.
        """
        effects = [["pitch", str(shift_ratio)]]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)

        return wav, (shift_ratio)

    def add_bg_noise(
        self,
        wav: torch.Tensor,
        snr_list: List,
        sr: int,
        dynamic_type: bool = False,
        noise_id: Optional[List[str]] = None,
    ):
        """
        Injected additive background noise with a SNR list.\n
        Numbers of augmented outputs must same as length of SNR list.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            snr_list: list of SNR ratio in dB
            sr: speech sample rate
            dynamic_type: if true, cascade 2 or more noises
            noise_id: which noise id would be used

        Returns:
            List of waveform has been add noise background
        """
        if noise_id is not None:
            assert isinstance(noise_id, list)
        else:
            if dynamic_type:
                noise_id = random.sample(self.bg_noise.keys(), k=2)
            else:
                noise_id = random.sample(self.bg_noise.keys(), k=1)[0]

        noise = []
        if dynamic_type:
            for i in range(len(noise_id)):
                bg_noise, noise_sr = AudioIO.open(
                    f_path=self.bg_noise[noise_id[i]]["wav_path"], normalized=False
                )
                if noise_sr != sr:
                    noise_sr = wav_resampling(
                        wav=bg_noise, origin_sr=noise_sr, target_sr=sr, backend="sox"
                    )

                noise.append(bg_noise)
        else:
            bg_noise, noise_sr = AudioIO.open(
                f_path=self.bg_noise[noise_id]["wav_path"], normalized=False
            )
            if noise_sr != sr:
                noise_sr, _ = wav_resampling(
                    wav=bg_noise, origin_sr=noise_sr, target_sr=sr, backend="sox"
                )
            noise.append(bg_noise)
            
        noisy_speech, added_noise = add_bg_noise(wav=wav, noise=noise, snr_list=snr_list)
        return noisy_speech, (added_noise, noise_id, snr_list)

    def add_bg_white_noise(
        self,
        wav: torch.Tensor,
        snr_list: List,
    ):
        """
        Injected additive background white noise with a SNR list.\n
        Numbers of augmented outputs must same as length of SNR list.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            snr_list: list of SNR ratio in dB
            sr: speech sample rate

        Returns:
            List of waveform has been add noise background
        """
        noisy_speech, noise = add_bg_white_noise(wav=wav, snr_list=snr_list)
        return noisy_speech, (noise, snr_list)

    def apply_rir(
        self,
        wav: torch.Tensor,
        rir_mode: str = "image",
        sr: int = 16000,
        rir_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Simulate reverberation data by convolue RIR in waveform by some specific paramters.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            key: RIR key in corpus
            choose_ch: if RIR channel not single, then used choose_ch as RIR channel
            rir_mode:
                image: input is reverberation, target is reverberation
                direct: input is reverberation, target is maximum peak impaulse to peak + 6ms
                early: input is reverberation, target is maximum peak impaulse to peak + 50ms

        Returns:
            waveform has been convolved with RIR

        Raises:
            NameError: if rir_mode not in (image, direct, early)
        """
        if rir_id is None:
            rir_id = random.choice(list(self.rir.keys()))

        impaulse, rir_sr = AudioIO.open(self.rir[rir_id]["wav_path"])
        if rir_sr != sr:
            impaulse, _ = wav_resampling(
                wav=impaulse, origin_sr=rir_sr, target_sr=sr, backend="sox"
            )

        reverb_wav = wav_apply_rir(
            wav=wav, impaulse=impaulse, sample_rate=sr, rir_mode=rir_mode
        )
        return reverb_wav, (rir_id, rir_mode)

    def apply_2nd_iir_response(
        self,
        wav: torch.Tensor,
        a_coeffs: Optional[torch.Tensor] = None,
        b_coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        wav_aug, a_coeffs, b_coeffs = rand_add_2nd_filter_response(wav=wav, a=a_coeffs, b=b_coeffs)
        return wav_aug, (a_coeffs, b_coeffs)

    def apply_gain_distortion(self, wav: torch.Tensor, sr: int):
        destroyed_wav, gain_distortion_info = rand_gain_distortion(
            wav=wav, sample_rate=sr, return_info=True
        )
        return destroyed_wav, (gain_distortion_info)

    def apply_src_effect(
        self, wav: torch.Tensor, sr: int, src_sr: int, src_backend: str
    ):
        src_wav, *src_info = wav_resampling(
            wav=wav, origin_sr=sr, target_sr=src_sr, backend=src_backend
        )
        if src_backend == "torchaudio":
            src_wav, *src_info = wav_resampling(
                wav=src_wav,
                origin_sr=src_sr,
                target_sr=sr,
                backend=src_backend,
                torch_backend_params=src_info[-1],
            )
        else:
            src_wav, *src_info = wav_resampling(
                wav=src_wav, origin_sr=src_sr, target_sr=sr, backend=src_backend
            )

        return src_wav, src_info

    def apply_hpf(self, wav: torch.Tensor, sr: int, cutoff_freq: int, q_factor: float):
        hpf_wav = torchaudio.functional.highpass_biquad(
            waveform=wav, sample_rate=sr, cutoff_freq=cutoff_freq, Q=q_factor
        )
        return hpf_wav, (cutoff_freq, q_factor)
