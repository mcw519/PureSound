import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout


def test_manual_test_scripts_expose_help():
    _assert_help("test/run_repo_checks.py")


def test_room_scene_script_exposes_help():
    _assert_help("egs/rir_generation/tools/audition/simulate_room_scene.py")


def test_m6_public_script_exposes_help():
    _assert_help("egs/rir_generation/generate_m6_bank.py")


# --------------------------------------------------------------------------- #
# The four training entry points
# --------------------------------------------------------------------------- #

TRAINING_MAINS = {
    "noise_suppression": "egs/noise_suppression/main.py",
    "voice_isolate": "egs/voice_isolate/main.py",
    "target_speaker_extraction": "egs/target_speaker_extraction/main.py",
    "speaker_embedding": "egs/speaker_embedding/main.py",
}

#: The CLI every main shares, because they share `runner.build_arg_parser`.
#: `speaker_embedding` and `target_speaker_extraction` used to hand-roll their
#: own parsers with `--training True`-style boolean values; these are the flags
#: that are now common, and a main that drifts back to its own parser fails here.
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


@pytest.mark.parametrize("task", sorted(TRAINING_MAINS))
def test_every_training_main_exposes_the_shared_cli(task):
    """A main is the only thing a user runs, and nothing else imports it, so a
    broken import or a renamed runner hook shows up nowhere but here."""
    result = subprocess.run(
        [sys.executable, TRAINING_MAINS[task], "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout
    missing = [flag for flag in SHARED_FLAGS if flag not in result.stdout]
    assert missing == [], f"{task} is missing {missing}"


@pytest.mark.parametrize("task", sorted(TRAINING_MAINS))
def test_no_main_takes_a_boolean_value_for_a_flag(task):
    """`--training True` was the old speaker-embedding / TSE form and argparse
    rejects it against a `store_true`. Documented invocations went with it, so
    this pins that the two do not drift apart again."""
    readme = REPO_ROOT / TRAINING_MAINS[task].replace("main.py", "README.md")
    if not readme.exists():
        pytest.skip(f"{readme} is not in this checkout")
    offenders = [
        line.strip()
        for line in readme.read_text(encoding="utf-8").splitlines()
        if "main.py" in line and "=True" in line
    ]
    assert offenders == [], offenders
