from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    runpy.run_module("egs.noise_suppression.prepare_metafile", run_name="__main__")
