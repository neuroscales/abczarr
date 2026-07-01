__all__ = [
    "Extension",
    "ExtensionWithConfig",
    "ExtraField",
    "ExtraFieldWithConfig",
]

# dependencies
import typing_extensions as tx

# locals
from abczarr._core import typing as tz
from ..config import NamedConfig, NamedRequiredConfig


NamedJSONConfig = NamedConfig[str, tz.JSONDict]
NamedRequiredJSONConfig = NamedRequiredConfig[str, tz.JSONDict]


class Config(tx.TypedDict):
    ...


class ExtensionBase(NamedJSONConfig):
    must_understand: tx.NotRequired[bool]


class ExtensionBaseWithConfig(NamedRequiredJSONConfig, ExtensionBase):
    ...


class Extension(ExtensionBase):
    must_understand: tx.NotRequired[tx.Literal[True]]


class ExtensionWithConfig(NamedRequiredJSONConfig, Extension):
    ...


class ExtraField(ExtensionBase, extra_items=tz.JSON):
    must_understand: tx.Literal[False]


class ExtraFieldWithConfig(NamedRequiredJSONConfig, ExtraField):
    ...
