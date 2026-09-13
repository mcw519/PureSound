from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_project_declares_pytorch_lightning_distribution():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["dependencies"]

    assert "pytorch-lightning" in deps
    assert "lightning" not in deps


def test_requirements_file_uses_pytorch_lightning_distribution():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "pytorch-lightning" in requirements
    assert "lightning" not in requirements


def test_repo_source_no_longer_imports_lightning_namespace():
    python_files = [
        ROOT / "puresound/system/base.py",
        ROOT / "puresound/system/curriculum.py",
        ROOT / "puresound/system/runner.py",
        ROOT / "egs/noise_suppression/main.py",
        ROOT / "egs/speaker_embedding/main.py",
        ROOT / "egs/target_speaker_extraction/main.py",
        ROOT / "egs/voice_isolate/main.py",
        ROOT / "test/test_system/test_curriculum.py",
    ]

    contents = {path: path.read_text(encoding="utf-8") for path in python_files}

    assert any("pytorch_lightning" in text for text in contents.values())
    assert all("import lightning as L" not in text for text in contents.values())
    assert all("from lightning.pytorch" not in text for text in contents.values())
    assert all("from lightning.fabric" not in text for text in contents.values())
