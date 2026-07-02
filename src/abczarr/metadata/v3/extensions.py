__all__ = [
    "TypedConfig",
    "Extension",
    "MustUnderstandExtension",
    "ExtraField"
]


# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen, field

# locals
from abczarr.metadata.base import Metadata, JSONMetadata


@autofrozen(extra_items=JSONMetadata)
class TypedConfig(Metadata):
    ...


@autofrozen
class Extension(Metadata):
    name: str
    configuration: TypedConfig
    must_understand: bool = True

    def to_dict(self) -> tx.Union[str, tz.JSONDict]:
        if not self.configuration.to_dict() and self.must_understand:
            # We can serialize as a name
            return self.name
        obj = super().to_dict()
        if obj.get("must_understand", True) is True:
            obj.pop("must_understand")
        return obj


@autofrozen
class MustUnderstandExtension(Extension):
    must_understand: tx.Literal[True] = field(repr=False)


@autofrozen
class ExtraField(Extension, extra_items=JSONMetadata):
    must_understand: tx.Literal[False]
