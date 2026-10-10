import logging

from puresound.logging_setup import autoconfigure

__version__ = "0.1"

# A NullHandler so the library never warns about missing handlers when an
# application opts out of the default one; see puresound/logging_setup.py for
# why a default handler is attached at all and how to take it over.
logging.getLogger(__name__).addHandler(logging.NullHandler())
autoconfigure()
