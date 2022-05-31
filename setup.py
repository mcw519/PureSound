from setuptools import find_packages, setup
from pathlib import Path

puresound_dir = Path(__file__).parent
install_requires = (puresound_dir / 'requirements.txt').read_text().splitlines()

setup(
    name = 'puresound',
    version = '0.1.0',
    python_requires='>=3.6.0',
    description='A Speech procssing toolkit based on PyTorch for speech research',
    author='Milo Wu',
    packages=find_packages(),
    install_requires=install_requires,
)