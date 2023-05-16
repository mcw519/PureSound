import random
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

from puresound.src.utils import convolve, fftconvolve, recursive_read_folder


class AudioIO:
    def __init__(self, verbose: bool = False) -> None:
        """
        Args:
            verbose: show detail
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
                wav = AudioIO.normalize_waveform(wav=wav, amp_type="avg")

        if target_lvl is not None:
            wav = AudioIO.rescale_waveform(
                wav=wav, target_lvl=target_lvl, amp_type="avg", scale="dB"
            )
            avg_amp_rescale = torch.mean(torch.abs(wav), dim=-1)

        if verbose:
            print(f"Open file: {f_path}")
            print(f"Avg_amp: {avg_amp_ori}")
            if target_lvl is not None:
                print(f"Avg_amp_rescale: {avg_amp_rescale}")

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
    def normalize_waveform(wav: torch.Tensor, amp_type: str = "avg") -> torch.Tensor:
        """
        This function normalizes a signal to unitary average or peak amplitude

        Args:
            wav: The waveform used for computing amplitude. Shape should be [..., L]
            amp_type: Whether to compute "avg" average or "peak" amplitude. Choose between ["avg", "peak"]
        
        Returns:
            Normalized level waveform
        """
        eps = 1e-14
        assert amp_type in ["avg", "peak"]

        if amp_type == "avg":
            den = torch.mean(torch.abs(wav), dim=-1, keepdim=True)
        elif amp_type == "peak":
            den = torch.max(torch.abs(wav), dim=-1, keepdim=True)[0]
        else:
            raise NotImplementedError

        den = den + eps

        return wav / den

    @staticmethod
    def rescale_waveform(
        wav: torch.Tensor,
        target_lvl: float,
        amp_type: str = "avg",
        scale: str = "linear",
    ) -> torch.Tensor:
        """
        This functions performs signal rescaling to a target level

        Args:
            wav: The waveform used for computing amplitude. Shape should be [..., L]
            target_lvl: Target lvl in dB or linear scale
            amp_type: Whether to compute "avg" average or "peak" amplitude. Choose between ["avg", "peak"]
            scale: whether target_lvl belongs to linear or dB scale. Choose between ["linear", "dB"]
        
        Returns:
            Rescaled waveform
        """
        assert amp_type in ["peak", "avg"]
        assert scale in ["linear", "dB"]

        wav = AudioIO.normalize_waveform(wav=wav, amp_type=amp_type)

        if scale == "linear":
            out = target_lvl * wav
        elif scale == "dB":
            target_lvl = 10 ** (target_lvl / 20)
            out = target_lvl * wav
        else:
            raise NotImplementedError

        return out

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


class AudioAugmentor:
    """
    Audio data augmentation on waveform.\n
    Ex:
        augmentor = AudioAugmentor()
        augmentor._load_rir_from_folder(RIR_folder)
        wav = augmentor.apply_rir(wav)
    """

    def __init__(self, sample_rate: int, convolve_mode: str = "convolution"):
        self.sr = sample_rate
        assert convolve_mode in ["convolution", "fft"]
        self.conv_mode = convolve_mode

    def sox_effect(self, wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """
        Getting Sox speed up/slow down and volumne adjustation.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            sr: waveform sampling rate

        Returns:
            waveform has been done speed up or slow down and volumn adjusted.
        """
        speed = float(torch.FloatTensor(1).uniform_(0.8, 1.2))
        vol = float(torch.FloatTensor(1).uniform_(0.125, 2))
        effects = [["speed", str(speed)], ["vol", str(vol)], ["rate", str(sr)]]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)

        return wav

    def sox_volumn_perturbed(
        self, wav: torch.Tensor, vol_ratio: float, sr: int = 16000
    ) -> torch.Tensor:
        """
        Getting Sox volumne adjustation by a specific parameter.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            vol_ratio: the ratio for volumn up/down. The general range is in [0.125, 2]
            sr: waveform sampling rate
        
        Returns:
            waveform has been done volumn adjusted.
        """
        effects = [["vol", str(vol_ratio)]]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)

        return wav

    def sox_speed_perturbed(
        self, wav: torch.Tensor, speed: float, sr: int = 16000
    ) -> Tuple[torch.Tensor, float]:
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

        return wav, speed

    def add_bg_noise(self, wav: torch.Tensor, snr_list: List) -> List:
        """
        Injected additive background noise with a SNR list.\n
        Numbers of augmented outputs must same as length of SNR list.

        Noisy = clean_wav + scale*noise,
        Here, we change the noise's scale to math the target SNR.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
            snr_list: list of SNR ratio in dB
        
        Returns:
            List of waveform has been add noise background
        """
        wav_power = wav.norm(p=2)
        noise_id = random.choice(list(self.bg_noise.keys()))
        noise, sr = AudioIO.open(self.bg_noise[noise_id]["wav_path"])

        if noise.shape[0] != 1:
            noise = noise[0, :].view(1, -1)
        if sr != self.sr:
            noise = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(
                noise
            )

        # check shape
        wav_length = wav.shape[-1]
        noise_length = noise.shape[-1]
        if wav_length <= noise_length:
            s = int(torch.randint(0, noise_length - wav_length, (1,)))
            noise = noise[:, s : s + wav_length]
            # noise = noise[:, :wav_length]
        else:
            noise = noise.repeat(1, round(wav_length / noise_length) + 1)
            noise = noise[:, :wav_length]

        noise_power = noise.norm(p=2)
        noisy_speech = []
        for snr_db in snr_list:
            snr = 10 ** (torch.Tensor([snr_db / 10]))
            scale = torch.sqrt(wav_power / (snr * noise_power + 1e-8))
            noisy_speech.append((wav + scale * noise))

        return noisy_speech

    def apply_rir(self, wav: torch.Tensor) -> Tuple[torch.Tensor, str, Optional[int]]:
        """
        Simulate reverberation data by convolue RIR in waveform.\n
        If provided RIR not single channel random choose one channel.

        Args:
            wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        
        Returns:
            waveform has been convolved with RIR
        """
        rir_key = random.choice(list(self.rir.keys()))
        impaulse, sr = AudioIO.open(self.rir[rir_key]["wav_path"])
        channel, _ = impaulse.shape
        choose_ch = None
        if channel != 1:
            choose_ch = random.randint(0, channel - 1)
            impaulse = impaulse[choose_ch, :].view(1, -1)

        if sr != self.sr:
            impaulse = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(
                impaulse
            )

        impaulse = impaulse / torch.norm(impaulse, p=2)
        if self.conv_mode == "fft":
            out = fftconvolve(wav, impaulse, mode="full")
            propagation_delays = impaulse.abs().argmax(dim=-1, keepdim=False)[0]
            out = out[..., propagation_delays : propagation_delays + wav.shape[-1]]
        else:
            impaulse = torch.flip(impaulse, [1])
            out = convolve(wav, impaulse)
        assert wav.shape[-1] == out.shape[-1]

        return out, rir_key, choose_ch

    def apply_rir_by_key(
        self,
        wav: torch.Tensor,
        key: str,
        choose_ch: int = None,
        rir_mode: str = "image",
        sr: int = 16000,
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
        impaulse, sr = AudioIO.open(self.rir[key]["wav_path"])
        channel, _ = impaulse.shape
        if channel != 1:
            if choose_ch is None:
                choose_ch = random.randint(0, channel - 1)
        else:
            choose_ch = 0

        impaulse = impaulse[choose_ch, :].view(1, -1)
        if sr != self.sr:
            impaulse = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(
                impaulse
            )

        if rir_mode.lower() == "image":
            pass
        elif rir_mode.lower() == "direct":
            peak_idx = impaulse.argmax().item()
            direct_range = peak_idx + int(sr * 0.006)  # 6ms range
            impaulse = impaulse[:, : int(direct_range)].view(1, -1)
        elif rir_mode.lower() == "early":
            peak_idx = impaulse.argmax().item()
            early_range = peak_idx + int(sr * 0.05)  # 50ms range
            impaulse = impaulse[:, : int(early_range)].view(1, -1)
        else:
            raise NameError

        impaulse = impaulse / torch.norm(impaulse, p=2)
        if self.conv_mode == "fft":
            out = fftconvolve(wav, impaulse, mode="full")
            propagation_delays = impaulse.abs().argmax(dim=-1, keepdim=False)[0]
            out = out[..., propagation_delays : propagation_delays + wav.shape[-1]]
        else:
            impaulse = torch.flip(impaulse, [1])
            out = convolve(wav, impaulse)

        assert wav.shape[-1] == out.shape[-1]

        return out

    def _get_white_noise_with_snr(self, wav: torch.Tensor, snr: int):
        """Simple Gaussian noise injection"""
        RMS_s = torch.sqrt(torch.mean(wav ** 2, dim=-1))
        RMS_n = torch.sqrt(RMS_s ** 2 / torch.pow(10, torch.Tensor([snr / 10])))
        STD_n = float(RMS_n)
        noise = torch.FloatTensor(wav.shape[-1]).normal_(mean=0, std=STD_n)
        return noise.view(1, -1)

    def load_bg_noise_from_folder(self, folder: str) -> None:
        """load bg-noise from folder path"""
        self.bg_noise = self._load_wav_folder(folder)

    def load_fg_noise_from_folder(self, folder: str) -> None:
        """load fg-noise from folder path"""
        self.fg_noise = self._load_wav_folder(folder)

    def load_rir_from_folder(self, folder: str) -> None:
        """load RIR from folder path"""
        self.rir = self._load_wav_folder(folder)

    def _load_wav_folder(self, folder: str) -> Dict:
        """load all waveform in folder, and split the waveform id to be key"""
        temp = {}
        wav_list = []
        recursive_read_folder(folder, ".wav", wav_list)
        for file in wav_list:
            file = file.strip().split(" ")[1]
            uttid = "_".join(file.split("/")[-1].split(".")[0:-1])
            temp[uttid] = {"wav_path": file}

        return temp

    def add_variaion_response(
        self,
        wav: torch.Tensor,
        a_coeffs: Optional[torch.Tensor] = None,
        b_coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reference:
            [1] A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement
        """
        if a_coeffs is None and b_coeffs is None:
            r = torch.Tensor(4).uniform_(-3 / 8, 3 / 8)
            a = torch.Tensor([1, r[0], r[1]])
            b = torch.Tensor([1, r[2], r[3]])

        wav = torchaudio.functional.lfilter(
            wav, a_coeffs=a.to(wav.device), b_coeffs=b.to(wav.device)
        )

        return wav, a_coeffs, b_coeffs
