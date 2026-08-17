"""Where the library's progress output goes.

Modules here log through ``logging.getLogger(__name__)`` rather than printing, so
an application can filter, redirect or silence what the library says.

Importing ``puresound`` still attaches one handler by default, which is not what
a library normally does. The reason is the shape of this repo: 84 files under
``egs/`` and ``tools/`` import the package and 20 of them build datasets, load
checkpoints or open audio -- none configure logging, and with only a
``NullHandler`` they would silently lose output that is read during runs (corpus
statistics, checkpoint-load reports). Attaching by default keeps every one of
those working while still making the output controllable:

* ``PURESOUND_LOG_AUTOCONFIG=0`` in the environment -- nothing is attached and
  the library is silent until you attach your own handler.
* ``logging.getLogger("puresound").setLevel(logging.WARNING)`` at runtime --
  keeps warnings, drops the progress chatter.
* ``configure_library_logging(level=..., stream=..., force=True)`` -- replace the
  default handler with your own.

The default handler writes to **stdout** with a bare ``%(message)s`` format, so
output lands where the ``print`` calls this replaced used to land and looks the
same; a shell redirect that captured it before still captures it.

Stdlib only, deliberately: ``puresound/__init__.py`` imports this module, and
``import puresound`` must not pull in torch --
``test/test_rir_r0_import_boundaries.py`` runs the RIR schema/metrics modules in
a subprocess and asserts torch never gets loaded.
"""

import logging
import os
import sys
from typing import Optional, TextIO

LOGGER_NAME = "puresound"

_AUTOCONFIG_ENV = "PURESOUND_LOG_AUTOCONFIG"
#: Marks the handler this module owns, so re-configuring replaces it instead of
#: stacking duplicates, and an application's own handlers are left alone.
_OWNED = "_puresound_default_handler"

#: Checked in order; the first one set wins. Same set Lightning's
#: ``rank_zero_only`` consults.
_RANK_ENV_KEYS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


def distributed_rank() -> int:
    """This process's rank, read from whatever the launcher set.

    From the environment rather than ``torch.distributed`` on purpose: the
    noisiest output (corpus statistics) is emitted while datasets are built,
    which happens before Lightning initializes the process group -- at that
    moment ``dist.is_initialized()`` is still False on every rank, so asking
    torch would report rank 0 everywhere and print N times.
    """
    for key in _RANK_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


class RankZeroFilter(logging.Filter):
    """Drop records on every rank but zero.

    Rank is read per record, not cached at construction: the default handler is
    installed at import time, which on a spawned worker precedes the launcher
    setting ``LOCAL_RANK``. A record can opt out with
    ``logger.info(..., extra={"all_ranks": True})`` when every rank genuinely
    needs to say it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "all_ranks", False)) or distributed_rank() == 0


def configure_library_logging(
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    rank_zero_only: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Attach the library's default handler to the ``puresound`` logger.

    Idempotent: calling it again is a no-op unless ``force=True``, which swaps
    the handler for one built with the new settings. Handlers this module does
    not own are never touched.
    """
    logger = logging.getLogger(LOGGER_NAME)
    owned = [handler for handler in logger.handlers if getattr(handler, _OWNED, False)]
    if owned and not force:
        return logger
    for handler in owned:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    if rank_zero_only:
        handler.addFilter(RankZeroFilter())
    setattr(handler, _OWNED, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def autoconfigure() -> None:
    """Install the default handler unless the environment opts out."""
    if os.environ.get(_AUTOCONFIG_ENV, "1") == "0":
        return
    configure_library_logging()
