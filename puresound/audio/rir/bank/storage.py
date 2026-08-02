"""Filesystem adapter for generated RIR dataset items.

Writes the WAV/JSON pair that every downstream reader expects.  Kept apart
from the renderer so ``render`` stays free of filesystem concerns; moved out of
``puresound.audio.rir.render.hybrid`` in R2 of ``RIR_MODULARIZATION_PLAN.md``.

The WAV is 32-bit float (``encoding="PCM_F"``), matching
``puresound.audio.rir.contracts.RIR_WAV_SUBTYPE``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchaudio


def write_hybrid_rir_dataset_item(
    output_dir: str | Path,
    sample_id: str,
    rir: torch.Tensor,
    metadata: dict[str, Any],
    sample_rate: int,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    wav_path = sample_dir / "rir_5ch.wav"
    json_path = sample_dir / "metadata.json"

    wav = rir.detach().cpu()
    if wav.ndim != 2 or wav.shape[0] != 5:
        raise ValueError(f"Expected RIR tensor [5, samples], got {tuple(wav.shape)}")
    torchaudio.save(str(wav_path), wav, int(sample_rate), encoding="PCM_F")
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return wav_path, json_path


__all__ = ["write_hybrid_rir_dataset_item"]
