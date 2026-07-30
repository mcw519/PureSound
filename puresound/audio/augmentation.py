import random
from itertools import count
from typing import List, Optional

import torch
import torchaudio

from puresound.audio.dsp import wav_resampling
from puresound.audio.impulse_response import rand_add_2nd_filter_response, wav_apply_rir
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise, add_bg_white_noise
from puresound.audio.room_simulator import RoomImpulseResponseSimulator
from puresound.audio.volume import rand_gain_distortion, wav_clipping
from puresound.utils import recursive_read_folder


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
            Clipping distortion
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
        self.bg_noise = {}
        self.rir = {}
        self.room_simulator = None
        self.room_bank = None
        self.simulated_rir = {}
        self.simulated_rir_counter = count()
        # Set by every apply_rir call so callers (e.g. eval set synthesis)
        # can read per-source RIR metadata such as DRR without changing the
        # public return signature. None for non-simulated RIRs.
        self._last_rir_meta: Optional[dict] = None

    def load_bg_noise_from_folder(self, folder: str, suffix: str = ".wav"):
        """load bg-noise from folder path"""
        self.bg_noise = self._load_wav_folder(folder, suffix=suffix)

    def load_rir_from_folder(self, folder: str, suffix: str = ".wav"):
        """load RIR from folder path"""
        self.rir = self._load_wav_folder(folder, suffix=suffix)

    def init_room_simulator(self, config: dict):
        """initialize physics-based shoebox RIR simulator."""
        simulator_config = dict(config)
        simulator_config.pop("used", None)
        simulator_config.pop("source_level", None)
        simulator_config.pop("pregenerated", None)
        self.room_simulator = RoomImpulseResponseSimulator(**simulator_config)

    def init_room_bank(self, config: dict):
        """initialize a bank of pre-generated multi-source room RIRs."""
        from puresound.audio.rir_bank import PreGeneratedRoomBank

        bank_config = dict(config)
        bank_config.pop("used", None)
        self.room_bank = PreGeneratedRoomBank(**bank_config)

    def sample_room_scene(self) -> Optional[dict]:
        if self.room_bank is not None:
            return self.room_bank.sample_scene()
        if self.room_simulator is None:
            return None
        return self.room_simulator.sample_scene()

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
        if hasattr(torchaudio, "sox_effects"):
            effects = [["vol", str(vol_ratio)]]
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
        else:
            wav = wav * vol_ratio

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
        if hasattr(torchaudio, "sox_effects"):
            effects = [["speed", str(speed)], ["rate", str(sr)]]
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
        else:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=max(1, int(sr * speed)),
                new_freq=sr,
            )

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
        if hasattr(torchaudio, "sox_effects"):
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
        noise_transform=None,
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
            noise_transform: optional callable applied to each loaded noise
                waveform BEFORE the SNR mixing (e.g. convolve it with a room
                channel so the noise shares the speech's acoustic space).
                None = the noise is mixed as loaded (dry).

        Returns:
            List of waveform has been add noise background
        """
        if noise_id is not None:
            assert isinstance(noise_id, list)
        else:
            if dynamic_type:
                noise_id = random.sample(list(self.bg_noise.keys()), k=2)
            else:
                noise_id = random.sample(list(self.bg_noise.keys()), k=1)[0]

        noise = []
        if dynamic_type:
            for i in range(len(noise_id)):
                bg_noise, noise_sr = AudioIO.open(
                    f_path=self.bg_noise[noise_id[i]]["wav_path"], normalized=False
                )
                if noise_sr != sr:
                    bg_noise, noise_sr = wav_resampling(
                        wav=bg_noise, origin_sr=noise_sr, target_sr=sr, backend="sox"
                    )

                noise.append(bg_noise)
        else:
            bg_noise, noise_sr = AudioIO.open(
                f_path=self.bg_noise[noise_id]["wav_path"], normalized=False
            )
            if noise_sr != sr:
                bg_noise, noise_sr = wav_resampling(
                    wav=bg_noise, origin_sr=noise_sr, target_sr=sr, backend="sox"
                )
            noise.append(bg_noise)

        if noise_transform is not None:
            noise = [noise_transform(n) for n in noise]

        noisy_speech, added_noise = add_bg_noise(
            wav=wav, noise=noise, snr_list=snr_list
        )
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
        room_scene: Optional[dict] = None,
        source_role: str = "source",
        distance_range_override: Optional[list[float]] = None,
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
        rir_metadata = None
        if self.room_bank is not None and rir_id is None:
            # Pre-generated multi-source room bank. A bank scene reuses one room
            # across the foreground + interferers; without a scene (e.g. the
            # non-source-level reverb path) draw an ad-hoc room/channel.
            if isinstance(room_scene, dict) and room_scene.get("_bank"):
                bank_scene = room_scene
            else:
                bank_scene = self.room_bank.sample_scene()
            impaulse, rir_metadata, rir_file_sr = self.room_bank.select_channel(
                scene=bank_scene,
                source_role=source_role,
                distance_range_override=distance_range_override,
            )
            if rir_file_sr != sr:
                impaulse, _ = wav_resampling(
                    wav=impaulse, origin_sr=rir_file_sr, target_sr=sr, backend="sox"
                )
            rir_id = f"bank-{next(self.simulated_rir_counter)}"
            self.simulated_rir[rir_id] = {
                "impulse": impaulse,
                "sample_rate": sr,
                "metadata": rir_metadata,
            }
        elif self.room_simulator is not None and rir_id is None:
            impaulse, rir_metadata = self.room_simulator.generate(
                sample_rate=sr,
                scene=room_scene,
                source_role=source_role,
                distance_range_override=distance_range_override,
            )
            rir_id = f"simulated-{next(self.simulated_rir_counter)}"
            self.simulated_rir[rir_id] = {
                "impulse": impaulse,
                "sample_rate": sr,
                "metadata": rir_metadata,
            }
        elif rir_id in self.simulated_rir:
            cached_rir = self.simulated_rir[rir_id]
            impaulse = cached_rir["impulse"]
            rir_metadata = cached_rir["metadata"]
            rir_sr = cached_rir["sample_rate"]
            if rir_sr != sr:
                impaulse, _ = wav_resampling(
                    wav=impaulse, origin_sr=rir_sr, target_sr=sr, backend="sox"
                )
        else:
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
        self._last_rir_meta = rir_metadata
        return reverb_wav, (rir_id, {"mode": rir_mode, "metadata": rir_metadata})

    def apply_2nd_iir_response(
        self,
        wav: torch.Tensor,
        a_coeffs: Optional[torch.Tensor] = None,
        b_coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        wav_aug, a_coeffs, b_coeffs = rand_add_2nd_filter_response(
            wav=wav, a=a_coeffs, b=b_coeffs
        )
        return wav_aug, (a_coeffs, b_coeffs)

    def apply_gain_distortion(self, wav: torch.Tensor, sr: int):
        destroyed_wav, gain_distortion_info = rand_gain_distortion(
            wav=wav, sample_rate=sr, return_info=True
        )
        return destroyed_wav, (gain_distortion_info)

    def apply_clipping_distortion(
        self, wav: torch.Tensor, min_quantile: float, max_quantile: float
    ):
        destored_wav = wav_clipping(
            wav=wav, min_quantile=min_quantile, max_quantile=max_quantile
        )
        return destored_wav, (min_quantile, max_quantile)

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

    def apply_media_coloring(
        self,
        wav: torch.Tensor,
        sr: int,
        hp_cutoff: float,
        lp_cutoff: float,
        compress_power: Optional[float] = None,
    ):
        """Spectral coloring for media-device speech (TV / loudspeaker playback):
        band-limit to the speaker's passband, then optional light dynamic-range
        compression (broadcast chains are compressed). RMS is restored so the
        downstream SIR scaling is unaffected by the coloring itself.

        Args:
            wav: [C, T] waveform.
            sr: sample rate of wav.
            hp_cutoff / lp_cutoff: passband edges in Hz.
            compress_power: in (0, 1]; |x|^p waveshaping on the peak-normalized
                signal, 1.0 or None = no compression.
        """
        rms_in = wav.pow(2).mean().sqrt().clamp_min(1e-8)
        colored = torchaudio.functional.highpass_biquad(
            waveform=wav, sample_rate=sr, cutoff_freq=hp_cutoff, Q=0.707
        )
        colored = torchaudio.functional.lowpass_biquad(
            waveform=colored, sample_rate=sr, cutoff_freq=lp_cutoff, Q=0.707
        )
        if compress_power is not None and compress_power < 1.0:
            peak = colored.abs().amax().clamp_min(1e-8)
            norm = colored / peak
            colored = torch.sign(norm) * norm.abs().pow(compress_power) * peak
        rms_out = colored.pow(2).mean().sqrt().clamp_min(1e-8)
        colored = colored * (rms_in / rms_out)
        return colored, (hp_cutoff, lp_cutoff, compress_power)

    # Format names AudioEffector requires per codec. Picked for round-trip
    # reliability in voice-bandwidth use (no Opus/AAC inside MKV quirks).
    _CODEC_FORMAT = {
        "libopus": "ogg",
        "g722": "matroska",
    }

    def apply_codec(
        self,
        wav: torch.Tensor,
        sr: int,
        codec_name: str,
        bit_rate: Optional[int] = None,
    ):
        """Simulate VoIP/telephony codec by encoding then decoding.

        Args:
            wav: [C, T] waveform (C=1 for mono).
            sr: input sample rate. The codec may internally resample
                (e.g. g722 forces 16 kHz); the output is returned at ``sr``
                with the original length preserved.
            codec_name: one of ``apply_codec.supported_codecs()``.
            bit_rate: optional override (bits-per-second). Ignored when the
                codec has no bitrate knob (g722).
        """
        from torchaudio.io import AudioEffector, CodecConfig

        if codec_name not in self._CODEC_FORMAT:
            raise ValueError(
                f"Unsupported codec '{codec_name}'. Choose from "
                f"{sorted(self._CODEC_FORMAT)}."
            )

        codec_config = CodecConfig(bit_rate=bit_rate) if bit_rate else None
        effector = AudioEffector(
            format=self._CODEC_FORMAT[codec_name],
            encoder=codec_name,
            codec_config=codec_config,
        )

        original_length = wav.shape[-1]
        wav_in = wav.float().transpose(0, 1).contiguous()
        out = effector.apply(wav_in, sample_rate=sr)
        out = out.transpose(0, 1).contiguous()
        if out.shape[-1] >= original_length:
            out = out[..., :original_length]
        else:
            out = torch.nn.functional.pad(
                out, (0, original_length - out.shape[-1])
            )
        return out, (codec_name, bit_rate)

    @staticmethod
    def supported_codecs():
        return list(AudioEffectAugmentor._CODEC_FORMAT.keys())

    def apply_packet_loss(
        self,
        wav: torch.Tensor,
        sr: int,
        packet_ms: int = 20,
        loss_rate: float = 0.05,
    ):
        """Zero out random packet-sized chunks to mimic VoIP drop-outs.

        Args:
            wav: [C, T] waveform.
            sr: sample rate (Hz).
            packet_ms: packet duration in milliseconds. 20 ms is the WebRTC
                default; 60 ms is typical for low-bandwidth Opus.
            loss_rate: per-packet Bernoulli drop probability.
        """
        packet_samples = max(1, int(round(sr * packet_ms / 1000)))
        T = wav.shape[-1]
        n_packets = T // packet_samples
        if n_packets == 0 or loss_rate <= 0:
            return wav.clone(), (packet_ms, loss_rate, 0)
        drop_mask = torch.rand(n_packets) < loss_rate
        n_dropped = int(drop_mask.sum().item())
        if n_dropped == 0:
            return wav.clone(), (packet_ms, loss_rate, 0)
        out = wav.clone()
        idx = torch.nonzero(drop_mask, as_tuple=False).flatten().tolist()
        for i in idx:
            s = i * packet_samples
            e = s + packet_samples
            out[..., s:e] = 0.0
        return out, (packet_ms, loss_rate, n_dropped)
