"""Processor implementations for the unified ONNX facade."""

from .base import AudioInputError, ProcessorProtocol, load_audio
from .stft_frame import StftFrameOrtProcessor, VoiceIsolationRuntime
from .waveform_embedding import SpeakerVerificationRuntime, WaveformEmbeddingOrtProcessor

PROCESSOR_REGISTRY = {
    "stft_frame_ort": VoiceIsolationRuntime,
    "waveform_embedding_ort": SpeakerVerificationRuntime,
}

__all__ = [
    "AudioInputError",
    "ProcessorProtocol",
    "PROCESSOR_REGISTRY",
    "load_audio",
    "SpeakerVerificationRuntime",
    "StftFrameOrtProcessor",
    "VoiceIsolationRuntime",
    "WaveformEmbeddingOrtProcessor",
]
