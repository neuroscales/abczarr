# stdlib
import numbers
from enum import Enum

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from . import typing as tz
from .config import NamedConfig

# typing
E = tx.TypeVar("E", bound=Enum)

# contants
INTEGRAL_TYPES = (numbers.Integral, np.integer)


def enum_names(enum: tx.Type[E]) -> tx.Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum(data: object, cls: tx.Type[E]) -> E:
    if isinstance(data, cls):
        return data
    if not isinstance(data, str):
        raise TypeError(f"Expected str, got {type(data)}")
    if data in enum_names(cls):
        return cls(data)
    raise ValueError(
        f"Value must be one of {list(enum_names(cls))!r}. "
        f"Got {data} instead."
    )


def parse_name(data: tz.JSON, expected: str | None = None) -> str:
    if isinstance(data, str):
        if expected is None or data == expected:
            return data
        raise ValueError(f"Expected '{expected}'. Got {data} instead.")
    else:
        raise TypeError(f"Expected a string, got an instance of {type(data)}.")


def parse_configuration(data: tz.JSON) -> tx.JSON:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return data


@tx.overload
def parse_named_configuration(
    data: tx.Union[tz.JSON, NamedConfig[str, tx.Any]],
    expected_name: tx.Optional[str] = None
) -> tx.Tuple[str, tx.MutableMapping[str, tx.JSON]]: ...


@tx.overload
def parse_named_configuration(
    data: tz.JSON | NamedConfig[str, tx.Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
) -> tx.Tuple[str, tx.Optional[tx.MutableMapping[str, tx.JSON]]]: ...


def parse_named_configuration(
    data: tz.JSON | NamedConfig[str, tx.Any],
    expected_name: tx.Optional[str] = None,
    *,
    require_configuration: bool = True,
) -> tx.Tuple[str, tx.Optional[tx.MutableMapping[str, tx.JSON]]]:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    if "name" not in data:
        raise ValueError(
            f"Named configuration does not have a 'name' key. "
            f"Got {data}."
        )
    name_parsed = parse_name(data["name"], expected_name)
    if "configuration" in data:
        configuration_parsed = parse_configuration(data["configuration"])
    elif require_configuration:
        raise ValueError(
            f"Named configuration does not have a 'configuration' key. "
            f"Got {data}."
        )
    else:
        configuration_parsed = None
    return name_parsed, configuration_parsed


def parse_shapelike(data: tz.ShapeLike) -> tz.Shape:
    """
    Parse a shape-like input into an explicit shape.
    """
    if isinstance(data, INTEGRAL_TYPES):
        if data < 0:
            raise ValueError(
                f"Expected a non-negative integer. "
                f"Got {data} instead"
            )
        return (int(data),)
    try:
        data_tuple = tuple(data)
    except TypeError as e:
        msg = (
            f"Expected an integer or an iterable of integers. "
            f"Got {data} instead."
        )
        raise TypeError(msg) from e

    if not all(isinstance(v, INTEGRAL_TYPES) for v in data_tuple):
        msg = (
            f"Expected an iterable of integers. "
            f"Got {data} instead."
        )
        raise TypeError(msg)
    if not all(v > -1 for v in data_tuple):
        msg = (
            f"Expected all values to be non-negative. "
            f"Got {data} instead."
        )
        raise ValueError(msg)

    # cast NumPy scalars to plain python ints
    return tuple(int(x) for x in data_tuple)


def parse_fill_value(data: tx.Any) -> tx.Any:
    # todo: real validation
    return data


def parse_order(data: tx.Any) -> tz.MemoryOrder:
    if data in ("C", "F"):
        return tx.cast(tz.MemoryOrder, data)
    raise ValueError(f"Expected one of ('C', 'F'), got {data} instead.")


def parse_bool(data: tx.Any) -> bool:
    if isinstance(data, bool):
        return data
    raise ValueError(f"Expected bool, got {data} instead.")


def parse_int(data: tx.Any) -> int:
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    raise ValueError(f"Expected int, got {data} instead.")


def validate_rectilinear_kind(kind: tx.Optional[str]) -> None:
    """Validate the ``kind`` field of a rectilinear chunk grid configuration.

    The rectilinear spec requires ``kind: "inline"``.
    """
    if kind is None:
        raise ValueError(
            "Rectilinear chunk grid configuration requires a 'kind' field. "
            "Only 'inline' is currently supported."
        )
    if kind != "inline":
        raise ValueError(
            f"Unsupported rectilinear chunk grid kind: {kind!r}. "
            "Only 'inline' is currently supported."
        )


def validate_rectilinear_edges(
    chunk_shapes: tz.ChunksIsh, array_shape: tz.ShapeIsh
) -> None:
    """
    Validate that rectilinear chunk edges cover the array extent per dimension.

    Bare-int dimensions (regular step) always cover any extent, so they are
    skipped. Explicit edge lists must sum to at least the array extent.
    """
    for i, (dim_spec, extent) in enumerate(zip(chunk_shapes, array_shape, strict=True)):
        if isinstance(dim_spec, INTEGRAL_TYPES):
            continue
        edge_sum = sum(dim_spec)
        if edge_sum < extent:
            raise ValueError(
                f"Rectilinear chunk edges for dimension {i} sum to {edge_sum} "
                f"but array shape extent is {extent} (edge sum must be >= extent)"
            )
