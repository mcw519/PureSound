from setuptools import find_packages, setup
from pathlib import Path

puresound_dir = Path(__file__).parent
requirements_path = puresound_dir / "requirements.txt"


def read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []

    requirements = []
    for line in path.read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#") or requirement.startswith("-"):
            continue
        requirements.append(requirement)
    return requirements


install_requires = read_requirements(requirements_path)

setup(
    name="puresound",
    version="0.1.0",
    python_requires=">=3.10.0",
    description="A Speech procssing toolkit based on PyTorch for speech research",
    author="Milo Wu",
    packages=find_packages(),
    install_requires=install_requires,
)
