import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

#: The four training entry points. A main is the only thing a user runs and
#: nothing else imports it, so a broken import or a renamed runner hook shows up
#: nowhere but here.
TRAINING_MAINS = {
    "noise_suppression": "egs/noise_suppression/main.py",
    "voice_isolate": "egs/voice_isolate/main.py",
    "target_speaker_extraction": "egs/target_speaker_extraction/main.py",
    "speaker_embedding": "egs/speaker_embedding/main.py",
}

#: The CLI every main shares, because they share `runner.build_arg_parser`. A
#: main that grows its own parser again fails here.
SHARED_FLAGS = (
    "--training",
    "--scoring",
    "--inference",
    "--dump_training_samples",
    "--ckpt_path",
    "--pretrained_ckpt_path",
    "--inference_sr",
    "--set_seed",
)


def _help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout
    return result.stdout


@pytest.mark.parametrize(
    "script",
    [
        "test/run_repo_checks.py",
        "egs/rir_generation/tools/audition/simulate_room_scene.py",
        "egs/rir_generation/generate_m6_bank.py",
    ],
)
def test_script_exposes_help(script):
    _help(script)


@pytest.mark.slow  # each main imports the whole training stack in a subprocess
@pytest.mark.parametrize("task", sorted(TRAINING_MAINS))
def test_every_training_main_exposes_the_shared_cli(task):
    """The shared flags are `store_true`, so argparse rejects `--training True`;
    the README next to each main must not document that form either."""
    usage = _help(TRAINING_MAINS[task])
    missing = [flag for flag in SHARED_FLAGS if flag not in usage]
    assert missing == [], f"{task} is missing {missing}"

    readme = REPO_ROOT / TRAINING_MAINS[task].replace("main.py", "README.md")
    if readme.exists():
        offenders = [
            line.strip()
            for line in readme.read_text(encoding="utf-8").splitlines()
            if "main.py" in line and "=True" in line
        ]
        assert offenders == [], offenders


def test_importing_the_cli_and_listing_models_does_not_load_torch():
    """The command line is usable on a machine that only runs ONNX: torch is
    imported by the commands that read or write audio files, not by the CLI."""
    code = (
        "import contextlib, io, sys\n"
        "from puresound.cli import main\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    assert main(['models', 'list']) == 0\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
