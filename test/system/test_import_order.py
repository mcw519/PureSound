"""pyarrow has to be imported before the audio/training stack, or it segfaults.

Loading `puresound.nnet` + `lightning` + `soundfile` and then importing pyarrow
kills the interpreter inside `pyarrow.lib` -- SIGSEGV, no Python traceback. The
reverse order is fine, so `test/conftest.py` imports pyarrow first.

Nothing in the suite imports pyarrow on purpose; it arrives through
gradio -> pandas when a test loads `egs/voice_isolate/scripts/demo.py`. Under
xdist every worker collects every module, so without the conftest import the
crash depends on which tests a worker draws and looks like a flaky demo test.

This runs in subprocesses because the damage is process-wide and the ordering
cannot be undone once the interpreter has loaded them.
"""

import subprocess
import sys

import pytest

AUDIO_STACK = "import puresound.nnet, lightning, soundfile\n"
PYARROW = "import pyarrow\n"
SURVIVED = "print('survived')\n"


def _run(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=300
    )


@pytest.mark.slow
def test_the_conftest_import_order_is_load_bearing():
    """pyarrow first survives; pyarrow last crashes.

    If the second half starts passing, the upstream conflict is gone and the
    pyarrow import at the top of `test/conftest.py` can go, with this test."""
    good = _run(PYARROW + AUDIO_STACK + SURVIVED)
    assert good.returncode == 0 and "survived" in good.stdout

    bad = _run(AUDIO_STACK + PYARROW + SURVIVED)
    if bad.returncode == 0:
        pytest.skip(
            "pyarrow no longer segfaults after the audio stack -- the conftest "
            "import and this test can both be removed"
        )
    assert bad.returncode < 0 or bad.returncode == 139, (
        f"expected a crash, got returncode={bad.returncode}: {bad.stderr[-400:]}"
    )
