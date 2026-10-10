"""Shared synthesis fixtures for the task tests."""

import pytest
import torch

from puresound.audio.io import AudioIO

SR = 16000


@pytest.fixture(scope="session")
def tone_corpus(tmp_path_factory):
    """A deterministic tone corpus plus a noise pool: ``(metafile, noise_dir)``.

    Tones rather than silence: several augmentation stages are no-ops on an
    all-zero signal, and a fingerprint over no-ops proves nothing.
    """
    root = tmp_path_factory.mktemp("tone_corpus")
    rows = ["uttid, spkid, gender, path, length, sample rate, channels"]
    n = int(SR * 1.5)
    time = torch.arange(n, dtype=torch.float32) / SR
    for speaker_index in range(4):
        speaker = f"corpus_spk{speaker_index}"
        for utt_index in range(4):
            path = root / "wav" / speaker / f"{speaker}_utt{utt_index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            freq = 180.0 + speaker_index * 80.0 + utt_index * 10.0
            wav = 0.1 * torch.sin(2 * torch.pi * freq * time)
            wav = wav + 0.01 * torch.sin(2 * torch.pi * freq * 3.7 * time)
            AudioIO.save(wav.view(1, -1), str(path), SR)
            gender = "m" if speaker_index % 2 == 0 else "f"
            rows.append(f"{speaker}_utt{utt_index}, {speaker}, {gender}, {path}, {n}, {SR}, 1")
    metafile = root / "meta.csv"
    metafile.write_text("\n".join(rows) + "\n", encoding="utf-8")

    noise_dir = root / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(7)
    for index in range(3):
        AudioIO.save(
            0.05 * torch.randn(1, SR * 2, generator=generator),
            str(noise_dir / f"n{index}.wav"),
            SR,
        )
    return metafile, noise_dir
