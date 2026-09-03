"""Small audio helpers shared by ONNX processors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch


class AudioInputError(ValueError):
    """Raised when an inference audio input cannot be normalised."""


class ProcessorProtocol(Protocol):
    """Structural contract implemented by each facade processor."""

    def infer(
        self,
        inputs: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


def load_audio(
    value: Any,
    *,
    sample_rate: int,
    target_dbfs: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load a path/tensor/array as mono float32 audio at ``sample_rate``.

    Arrays may be supplied as ``(samples, source_sample_rate)`` tuples.  A
    bare array is treated as already sampled at the model rate, which keeps
    the public facade convenient for realtime callers.
    """

    source_sr: int | None = None
    loaded_from_path = False
    if isinstance(value, tuple) and len(value) == 2:
        if isinstance(value[1], (int, np.integer)):
            value, source_sr = value
        elif isinstance(value[0], (int, np.integer)):
            source_sr, value = int(value[0]), value[1]
    if isinstance(value, (str, Path)):
        loaded_from_path = True
        from puresound.audio.io import AudioIO

        try:
            waveform, sr = AudioIO.open(
                str(value),
                resample_to=sample_rate,
                target_lvl=target_dbfs,
            )
        except Exception as exc:
            raise AudioInputError(f"failed to read audio input {value!r}: {exc}") from exc
        array = waveform.detach().cpu().numpy()
        source_sr = int(sr)
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise AudioInputError(f"unsupported audio input type: {type(value)!r}") from exc
    if array.size == 0:
        raise AudioInputError("audio input is empty")
    if not np.issubdtype(array.dtype, np.number):
        raise AudioInputError("audio input must contain numeric samples")
    integer_input = np.issubdtype(array.dtype, np.integer)
    integer_scale = float(np.iinfo(array.dtype).max) if integer_input else 1.0
    array = np.asarray(array, dtype=np.float32)
    if integer_input and integer_scale:
        array = array / integer_scale
    if array.ndim == 0:
        raise AudioInputError("audio input must be one- or two-dimensional")
    if array.ndim > 2:
        raise AudioInputError("audio input must be one- or two-dimensional")
    if array.ndim == 2:
        # AudioIO uses [channels, samples].  Numpy/Gradio often uses
        # [samples, channels], so infer the short axis as channels.
        if array.shape[0] <= 8:
            array = array.mean(axis=0)
        elif array.shape[1] <= 8:
            array = array.mean(axis=1)
        else:
            raise AudioInputError(
                "two-dimensional audio must have a channel axis of at most 8"
            )
    array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise AudioInputError("audio input contains NaN or infinity")
    # Bare arrays have no reliable sample-rate metadata.  Tuple inputs are
    # explicitly resampled when their source rate differs.
    if source_sr is not None and source_sr != sample_rate:
        try:
            import scipy.signal

            gcd = int(np.gcd(source_sr, sample_rate))
            array = scipy.signal.resample_poly(
                array,
                sample_rate // gcd,
                source_sr // gcd,
            ).astype(np.float32)
        except Exception as exc:
            raise AudioInputError(
                f"failed to resample audio from {source_sr} Hz to {sample_rate} Hz: {exc}"
            ) from exc
    if target_dbfs is not None and not loaded_from_path:
        rms = float(np.sqrt(np.mean(np.square(array))))
        if rms > 1e-12:
            array = array * (10.0 ** (float(target_dbfs) / 20.0) / rms)
    return array.astype(np.float32, copy=False), sample_rate


__all__ = ["AudioInputError", "ProcessorProtocol", "load_audio"]
