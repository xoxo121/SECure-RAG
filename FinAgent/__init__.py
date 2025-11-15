import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
from .schema import Model  # noqa: E402

__all__ = ["Model"]
