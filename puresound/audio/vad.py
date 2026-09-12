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
        eps_mode: str = "absolute",
    ):
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.activity_threshold_db = float(activity_threshold_db)
        self.eps = float(eps)
        # "absolute" (historical): the numerator is clamped at a FIXED 1e-8 power,
        # so once the loudest frame drops below 1e-4 (peak-frame RMS < -40 dBFS)
        # every silent frame reads within -40 dB of the peak and the labeler
        # returns all-active (measured: 0.58 active at -40 dBFS, 1.00 at -45 and
        # below). "relative" clamps at eps * reference instead, so the dynamic
        # range is 80 dB below the row's own peak whatever its level.
        if eps_mode not in ("absolute", "relative"):
            raise ValueError(f"eps_mode must be absolute|relative, got {eps_mode!r}")
        self.eps_mode = eps_mode

    def __call__(self, wav: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        del sample_rate
        wav = _as_mono(wav)
        if wav.shape[-1] < self.frame_length:
            wav = F.pad(wav, (0, self.frame_length - wav.shape[-1]))
        frames = wav.unfold(-1, self.frame_length, self.hop_length)
        power = frames.square().mean(dim=-1).squeeze(0)
        reference = power.max().clamp_min(self.eps)
        floor = self.eps * reference if self.eps_mode == "relative" else self.eps
        db = 10.0 * torch.log10(power.clamp_min(floor) / reference)
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


class BatchedSileroVADLabeler:
    """GPU, batched equivalent of :class:`SileroVADLabeler`.

    Runs the Silero model once over a whole ``[B, T]`` batch via
    ``model.audio_forward`` (on GPU when the input is on GPU), then ports the
    exact post-processing state machine of ``silero_vad.get_speech_timestamps``
    so the per-sample frame labels match the CPU labeler.

    Only the default-parameter path is supported: ``max_speech_duration_s`` is
    assumed ``inf`` (our configs never override it), which removes the entire
    max-speech splitting branch and leaves a simple
    triggered/temp_end/min_silence/min_speech/speech_pad machine. A finite
    ``max_speech_duration_s`` raises, so we never silently diverge.

    Designed to be owned by the training module (not the dataset) and invoked
    after the batch lands on GPU, so Silero is lifted out of the DataLoader
    workers entirely.
    """

    def __init__(
        self,
        frame_length: int = 400,
        hop_length: int = 160,
        threshold: float = 0.5,
        min_overlap: float = 0.5,
        model_sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float = float("inf"),
        neg_threshold: Optional[float] = None,
        **unused,
    ):
        if max_speech_duration_s != float("inf"):
            raise ValueError(
                "BatchedSileroVADLabeler only supports max_speech_duration_s=inf "
                f"(the Silero default), got {max_speech_duration_s}."
            )
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.threshold = float(threshold)
        self.min_overlap = float(min_overlap)
        self.model_sample_rate = int(model_sample_rate)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.neg_threshold = neg_threshold
        self.model = None
        self._device = None

    def _lazy_init(self, device: torch.device):
        if self.model is None:
            try:
                from silero_vad import load_silero_vad
            except ImportError as exc:
                raise RuntimeError(
                    "Silero VAD labels require the optional `silero-vad` package. "
                    "Install it with `uv pip install silero-vad`, or set "
                    "`vad_label.backend: energy`."
                ) from exc
            self.model = load_silero_vad().eval()
        if self._device != device:
            self.model = self.model.to(device)
            self._device = device

    def _probs_to_timestamps(
        self, probs: list[float], window_size_samples: int, audio_length_samples: int
    ) -> list[dict]:
        """Port of get_speech_timestamps post-processing (max_speech=inf path)."""
        threshold = self.threshold
        neg_threshold = (
            self.neg_threshold
            if self.neg_threshold is not None
            else max(threshold - 0.15, 0.01)
        )
        sr = self.model_sample_rate
        min_speech_samples = sr * self.min_speech_duration_ms / 1000
        speech_pad_samples = sr * self.speech_pad_ms / 1000
        min_silence_samples = sr * self.min_silence_duration_ms / 1000

        triggered = False
        speeches: list[dict] = []
        current: dict = {}
        temp_end = 0

        for i, speech_prob in enumerate(probs):
            cur_sample = window_size_samples * i

            if (speech_prob >= threshold) and temp_end:
                temp_end = 0

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current["start"] = cur_sample
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = cur_sample
                if cur_sample - temp_end < min_silence_samples:
                    continue
                current["end"] = temp_end
                if (current["end"] - current["start"]) > min_speech_samples:
                    speeches.append(current)
                current = {}
                temp_end = 0
                triggered = False
                continue

        if current and (audio_length_samples - current["start"]) > min_speech_samples:
            current["end"] = audio_length_samples
            speeches.append(current)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] = int(
                        min(audio_length_samples, speech["end"] + speech_pad_samples)
                    )
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - speech_pad_samples)
                    )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
        return speeches

    @torch.no_grad()
    def __call__(
        self, wav: torch.Tensor, sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        sample_rate = int(sample_rate or self.model_sample_rate)
        if wav.dim() == 3:  # [B, C, T] -> mono
            wav = wav.mean(dim=1)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"Expected [B, T] / [B, C, T] / [T], got {wav.shape}")

        device = wav.device
        self._lazy_init(device)

        original_length = wav.shape[-1]
        if sample_rate != self.model_sample_rate:
            wav_for_vad = torch.nn.functional.interpolate(
                wav.unsqueeze(1),
                scale_factor=self.model_sample_rate / sample_rate,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        else:
            wav_for_vad = wav

        length_at_model_sr = wav_for_vad.shape[-1]
        window = 512 if self.model_sample_rate == 16000 else 256
        # [B, n_windows] speech probabilities; the only model-dependent step.
        probs = self.model.audio_forward(wav_for_vad, self.model_sample_rate)
        probs = probs.detach().cpu()
        # Silent rows (e.g. target-absent samples) map to all-zero activity,
        # matching DynamicBaseDataset.create_vad_target's explicit short-circuit.
        nonsilent = (wav.abs().amax(dim=-1) > 0).tolist()

        scale = sample_rate / self.model_sample_rate
        activities = []
        for b in range(wav.shape[0]):
            if not nonsilent[b]:
                timestamps = []
            else:
                timestamps = self._probs_to_timestamps(
                    probs[b].tolist(), window, length_at_model_sr
                )
                if sample_rate != self.model_sample_rate:
                    timestamps = [
                        {
                            "start": int(item["start"] * scale),
                            "end": int(item["end"] * scale),
                        }
                        for item in timestamps
                    ]
            activities.append(
                activity_from_timestamps(
                    timestamps=timestamps,
                    length=original_length,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                    min_overlap=self.min_overlap,
                )
            )
        return torch.stack(activities, dim=0).to(device)


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
