from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requirement_name(requirement: str) -> str:
    match = NAME_RE.match(requirement.split("#", 1)[0].strip())
    assert match, f"could not parse requirement line: {requirement!r}"
    return match.group(1).lower()


def test_project_declares_pytorch_lightning_distribution():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = [_requirement_name(dep) for dep in project["project"]["dependencies"]]

    assert "pytorch-lightning" in deps
    assert "lightning" not in deps


def test_requirements_file_uses_pytorch_lightning_distribution():
    requirements = [
        _requirement_name(line)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]

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
    imported_modules = set()
    imported_from_modules = set()
    pytorch_lightning_seen = False

    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
                    if alias.name == "pytorch_lightning":
                        pytorch_lightning_seen = True
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_from_modules.add(node.module)
                if node.module == "pytorch_lightning":
                    pytorch_lightning_seen = True

    assert pytorch_lightning_seen
    assert "lightning" not in imported_modules
    assert "lightning.pytorch" not in imported_from_modules
    assert "lightning.fabric" not in imported_from_modules
