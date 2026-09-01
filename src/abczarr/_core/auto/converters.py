"""Hint-driven converters, from :mod:`bagof.converters`.

This module used to carry a home-grown converter engine; that surface now
lives in ``bagof.converters`` (the same one bagof-magic uses), so this
re-exports it. abczarr's own converters -- ``ToJson``, ``MetadataConverter``,
``DTypeConverter`` -- subclass ``Converter`` and register through
``register_converter`` from here, unchanged.
"""

from bagof.converters import *  # noqa: F401,F403
from bagof.converters import (  # noqa: F401
    Converter,
    get_converter,
    register_converter,
    wrap_converter,
)
from bagof.converters import __all__ as __all__  # noqa: F401,PLC0414
