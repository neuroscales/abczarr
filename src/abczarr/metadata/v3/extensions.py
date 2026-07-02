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
from abczarr._core.attrs import autofrozen, field, fields

# locals
from abczarr.metadata.base import Metadata, JSONMetadata


def _repr_if_false(value: bool) -> str:
    return None if value else repr(value)


@autofrozen(extra_items=JSONMetadata)
class TypedConfig(Metadata):
    ...


@autofrozen
class Extension(Metadata):
    name: str
    configuration: TypedConfig
    must_understand: bool = True

    def __init__(self, *args, **kwargs):
        if len(args) < 2 and "configuration" not in kwargs:
            config = kwargs
            kwargs = {}
            if "must_understand" in config:
                kwargs["must_understand"] = config.pop("must_understand")
            config = fields(self).configuration.type(**config)
            kwargs["configuration"] = config

        self.__attrs_init__(*args, **kwargs)

    def to_dict(self) -> tx.Union[str, tz.JSONDict]:
        if not self.configuration.to_dict() and self.must_understand:
            # We can serialize as a name
            return self.name
        obj = super().to_dict()
        if obj.get("must_understand", True) is True:
            obj.pop("must_understand")
        return obj


# Specify the __init__ wraps __attrs_init__ so that we get the correct
# signature and docstring.
Extension.__init__ = wraps(Extension.__attrs_init__)(Extension.__init__)


@autofrozen
class MustUnderstandExtension(Extension):
    must_understand: tx.Literal[True] = field(repr=False)


@autofrozen(extra_items=JSONMetadata)
class ExtraField(Extension):
    must_understand: tx.Literal[False]
