from setuptools import find_packages, setup
from pathlib import Path

puresound_dir = Path(__file__).parent
requirements_path = puresound_dir / "requirements.txt"
install_requires = (
    requirements_path.read_text().splitlines() if requirements_path.exists() else []
)

setup(
    name="puresound",
    version="0.1.0",
    python_requires=">=3.10.0",
    description="A Speech procssing toolkit based on PyTorch for speech research",
    author="Milo Wu",
    packages=find_packages(),
    install_requires=install_requires,
)
