"""Small, dependency-light web service for the PureSound Model Zoo.

The web layer intentionally stays thin: model discovery and inference remain
owned by :mod:`puresound.inference`, while this package only provides an HTTP
transport and a static browser client.
"""

from .server import WebService, WebServiceError, create_server, run

__all__ = ["WebService", "WebServiceError", "create_server", "run"]
