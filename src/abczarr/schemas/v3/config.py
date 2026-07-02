# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# typing
StrMapping = tx.Mapping[str, object]
TName = tx.TypeVar("TName", bound=str, default=str)
TConfig = tx.TypeVar("TConfig", bound=StrMapping, default=tz.JSONDict)


class NamedConfig(tx.TypedDict, tx.Generic[TName, TConfig]):
    """
    A typed dictionary representing an object with a name and configuration,
    where the configuration is an optional mapping of string keys to values,
    e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name
    (``TName``) and the type of the configuration (``TConfig``).
    """

    name: tx.ReadOnly[TName]
    """The name of the object."""

    configuration: tx.NotRequired[tx.ReadOnly[TConfig]]
    """The configuration of the object. Not required."""


class NamedRequiredConfig(tx.TypedDict, tx.Generic[TName, TConfig]):
    """
    A typed dictionary representing an object with a name and configuration,
    where the configuration is a mapping of string keys to values,
    e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name
    (``TName``) and the type of the configuration (``TConfig``).
    """

    name: tx.ReadOnly[TName]
    """The name of the object."""

    configuration: tx.ReadOnly[TConfig]
    """The configuration of the object."""
