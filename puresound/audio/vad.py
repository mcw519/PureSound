from typing import Optional

import torch
import torch.nn.functional as F


def _as_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2 and wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() != 2:
        raise ValueError(f"Expected waveform shape [C, T] or [T], got {wav.shape}")
    return wav


def frame_count(length: int, frame_length: int, hop_length: int) -> int:
    if length < frame_length:
        return 1
    return ((length - frame_length) // hop_length) + 1


def activity_from_timestamps(
    timestamps: list[dict],
    length: int,
    frame_length: int,
    hop_length: int,
    min_overlap: float = 0.5,
) -> torch.Tensor:
    n_frames = frame_count(length, frame_length, hop_length)
    activity = torch.zeros(n_frames, dtype=torch.float32)
    for frame_idx in range(n_frames):
        frame_start = frame_idx * hop_length
        frame_end = min(frame_start + frame_length, length)
        frame_duration = max(1, frame_end - frame_start)
        for speech in timestamps:
            speech_start = int(speech["start"])
            speech_end = int(speech["end"])
            overlap = min(frame_end, speech_end) - max(frame_start, speech_start)
            if overlap / frame_duration >= min_overlap:
                activity[frame_idx] = 1.0
                break
    return activity


class EnergyVADLabeler:
    def __init__(
        self,
        frame_length: int = 400,
        hop_length: int = 160,
        activity_threshold_db: float = -40.0,
        eps: float = 1e-8,
    ):
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.activity_threshold_db = float(activity_threshold_db)
        self.eps = float(eps)

    def __call__(self, wav: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        del sample_rate
        wav = _as_mono(wav)
        if wav.shape[-1] < self.frame_length:
            wav = F.pad(wav, (0, self.frame_length - wav.shape[-1]))
        frames = wav.unfold(-1, self.frame_length, self.hop_length)
        power = frames.square().mean(dim=-1).squeeze(0)
        reference = power.max().clamp_min(self.eps)
        db = 10.0 * torch.log10(power.clamp_min(self.eps) / reference)
        return (db > self.activity_threshold_db).float()


class SileroVADLabeler:
    def __init__(
        self,
        frame_length: int = 400,
        hop_length: int = 160,
        threshold: float = 0.5,
        min_overlap: float = 0.5,
        model_sample_rate: int = 16000,
        **silero_kwargs,
    ):
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.threshold = float(threshold)
        self.min_overlap = float(min_overlap)
        self.model_sample_rate = int(model_sample_rate)
        self.silero_kwargs = silero_kwargs
        self.model = None
        self.get_speech_timestamps = None

    def _lazy_init(self):
        if self.model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD labels require the optional `silero-vad` package. "
                "Install it with `uv pip install silero-vad`, or set "
                "`vad_label.backend: energy`."
            ) from exc
        self.model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps

    def __call__(self, wav: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        self._lazy_init()
        sample_rate = int(sample_rate or self.model_sample_rate)
        wav = _as_mono(wav).squeeze(0).detach().cpu()
        original_length = wav.shape[-1]
        if sample_rate != self.model_sample_rate:
            wav_for_vad = torch.nn.functional.interpolate(
                wav.view(1, 1, -1),
                scale_factor=self.model_sample_rate / sample_rate,
                mode="linear",
                align_corners=False,
            ).view(-1)
        else:
            wav_for_vad = wav

        timestamps = self.get_speech_timestamps(
            wav_for_vad,
            self.model,
            sampling_rate=self.model_sample_rate,
            threshold=self.threshold,
            **self.silero_kwargs,
        )
        if sample_rate != self.model_sample_rate:
            scale = sample_rate / self.model_sample_rate
            timestamps = [
                {
                    "start": int(item["start"] * scale),
                    "end": int(item["end"] * scale),
                }
                for item in timestamps
            ]
        return activity_from_timestamps(
            timestamps=timestamps,
            length=original_length,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            min_overlap=self.min_overlap,
        )


def create_vad_labeler(config: Optional[dict]):
    if not config or not config.get("used"):
        return None
    labeler_args = dict(config.get("args", {}))
    labeler_args.setdefault("frame_length", config.get("frame_length", 400))
    labeler_args.setdefault("hop_length", config.get("hop_length", 160))
    backend = config.get("backend", "energy").lower()
    if backend == "energy":
        return EnergyVADLabeler(**labeler_args)
    if backend == "silero":
        return SileroVADLabeler(**labeler_args)
    raise ValueError(f"Unsupported VAD label backend: {backend}")
