__all__ = [
    "TypedConfig",
    "Extension",
    "MustUnderstandExtension",
    "ExtraField"
]

# stdlib
from functools import wraps

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, field, fields
from abczarr._core.metadata import JSONMetadata, Metadata


@autofrozen(extra_items=JSONMetadata)
class TypedConfig(Metadata):
    """This class is the base for the ``configuration`` object of an
    extension point."""


@autofrozen
class Extension(Metadata):
    """An `Extension` represents a Zarr v3 extension point: a named,
    configurable piece of metadata.

    Codecs, data types, chunk grids and chunk key encodings are all
    shaped this way. Each carries a ``name`` identifying which
    extension applies, its own ``configuration``, and
    ``must_understand``, which says whether a reader that does not
    recognize ``name`` must refuse to open the array instead of
    ignoring the extension.
    """

    name: str
    configuration: TypedConfig
    must_understand: bool = True

    def __init__(self, *args, **kwargs) -> None:
        if len(args) < 2 and "configuration" not in kwargs:
            config = kwargs
            kwargs = {}
            if "must_understand" in config:
                kwargs["must_understand"] = config.pop("must_understand")
            config = fields(self).configuration.type(**config)
            kwargs["configuration"] = config

        self.__attrs_init__(*args, **kwargs)

    def to_json(self) -> tz.JsonDict:
        # A default ``must_understand`` (True) is left implicit in the output.
        obj = super().to_json()
        if obj.get("must_understand", True) is True:
            obj.pop("must_understand")
        return obj


# Specify the __init__ wraps __attrs_init__ so that we get the correct
# signature and docstring.
Extension.__init__ = wraps(Extension.__attrs_init__)(Extension.__init__)


@autofrozen
class MustUnderstandExtension(Extension):
    """Represents an extension point that a reader may never silently
    ignore.

    ``must_understand`` is pinned to ``True``. The pin applies to
    extension points such as codecs, data types, chunk grids, and chunk
    key encodings, where an unrecognized value would make the array
    unreadable if skipped.
    """

    must_understand: tx.Literal[True] = field(repr=False)


@autofrozen(extra_items=JSONMetadata)
class ExtraField(Extension):
    """Represents a top-level extension field that a reader may safely
    skip if unrecognized.

    ``must_understand`` is pinned to ``False``.
    """

    must_understand: tx.Literal[False]
