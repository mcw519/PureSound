from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def test_project_declares_pytorch_lightning_distribution():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["dependencies"]

    assert "pytorch-lightning" in deps
    assert "lightning" not in deps


def test_requirements_file_uses_pytorch_lightning_distribution():
    requirements = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        match = NAME_RE.match(requirement)
        assert match, f"could not parse requirement line: {line!r}"
        requirements.append(match.group(1).lower())

    assert "pytorch-lightning" in requirements
    assert "lightning" not in requirements


def test_repo_source_no_longer_imports_lightning_namespace():
    roots = [ROOT / "puresound", ROOT / "egs", ROOT / "test"]
    python_files = [
        path
        for base in roots
        for path in base.rglob("*.py")
        if path != Path(__file__)
    ]
    contents = {path: path.read_text(encoding="utf-8") for path in python_files}

    assert any("pytorch_lightning" in text for text in contents.values())
    assert all("import lightning as L" not in text for text in contents.values())
    assert all("from lightning.pytorch" not in text for text in contents.values())
    assert all("from lightning.fabric" not in text for text in contents.values())
