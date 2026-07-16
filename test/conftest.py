import sys
from pathlib import Path

import pytest
import torch

# Tests import repo-level namespace packages (e.g. egs.rir_generation); make the
# repo root importable regardless of the pytest invocation cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO  # noqa: E402


@pytest.fixture
def write_silent_wav():
    def _write(path: Path, sample_rate: int = 16000, duration: float = 0.1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        wav = torch.zeros(1, int(sample_rate * duration))
        AudioIO.save(wav, str(path), sample_rate)

    return _write


@pytest.fixture
def write_tone_wav():
    def _write(
        path: Path,
        sample_rate: int = 16000,
        duration: float = 0.25,
        freq: float = 220.0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        time = torch.arange(int(sample_rate * duration), dtype=torch.float32) / sample_rate
        wav = (0.1 * torch.sin(2 * torch.pi * freq * time)).view(1, -1)
        AudioIO.save(wav, str(path), sample_rate)

    return _write


@pytest.fixture
def write_puresound_metafile(write_tone_wav):
    def _write(
        path: Path,
        *,
        root: Path | None = None,
        speakers: int = 2,
        utterances_per_speaker: int = 2,
        sample_rate: int = 16000,
        duration: float = 0.25,
    ) -> Path:
        root = root or path.parent / "wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = ["uttid, spkid, gender, path, length, sample rate, channels"]
        for spk_idx in range(speakers):
            gender = "m" if spk_idx % 2 == 0 else "f"
            spkid = f"corpus_spk{spk_idx}"
            for utt_idx in range(utterances_per_speaker):
                uttid = f"{spkid}_utt{utt_idx}"
                wav_path = root / spkid / f"{uttid}.wav"
                write_tone_wav(
                    wav_path,
                    sample_rate=sample_rate,
                    duration=duration,
                    freq=180.0 + spk_idx * 80.0 + utt_idx * 10.0,
                )
                rows.append(
                    f"{uttid}, {spkid}, {gender}, {wav_path}, "
                    f"{int(sample_rate * duration)}, {sample_rate}, 1"
                )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    return _write
