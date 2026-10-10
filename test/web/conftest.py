"""Shared fixtures for the web tests."""

from pathlib import Path

import pytest
import yaml


def _recipe(path: Path, **fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"schema_version": 2, **fields}))
    return path


@pytest.fixture
def pipeline_repo(tmp_path):
    """A repository root offering one noise-suppression and one voice-isolation
    training recipe, next to recipes the inspector must not offer."""
    _recipe(tmp_path / "egs/ns/config/train_a.yaml", purpose="train", task="noise_suppression",
            dataset={"training_length_seconds": 6.0})
    _recipe(tmp_path / "egs/vi/config/train_b.yaml", purpose="train", task="voice_isolation",
            curriculum={"used": True, "tracks": []})
    _recipe(tmp_path / "egs/ns/config/infer.yaml", purpose="infer", task="noise_suppression")
    _recipe(tmp_path / "egs/ns/config/exp/train_x.yaml", purpose="train", task="noise_suppression")
    _recipe(tmp_path / "egs/sv/config/train_sv.yaml", purpose="train", task="speaker_embedding")
    (tmp_path / "egs/ns/config/broken.yaml").write_text("purpose: [train")
    return tmp_path
