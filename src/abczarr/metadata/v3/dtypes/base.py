__all__ = ["DType"]

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto import autofield, autofrozen
from abczarr._core.dtypes import asdtype

# locals
from ...base import register_subclass
from ..extensions import MustUnderstandExtension, TypedConfig


@autofrozen(extra_items=tz.FrozenJson)
class DTypeConfig(TypedConfig):
    """This class is the base for a v3 data type's own configuration
    parameters."""


@autofrozen
class DType(MustUnderstandExtension):
    """A Zarr v3 data type: a name and that type's own configuration.

    A core data type, such as ``float32``, has no configuration and
    is written as a bare name. An extension data type, such as a
    struct or a fixed-bit-width raw type, carries its parameters in
    `configuration`.
    """

    configuration: DTypeConfig
    """The data type's own parameters. Empty for a core data type."""

    def to_json(self) -> tx.Union[str, tz.JsonDict]:
        """Serialize this data type to its JSON representation.

        A core data type with no configuration is written as a bare
        name, such as ``"float32"``, as the Zarr v3 specification
        requires. An extension data type keeps the full object form.

        Returns
        -------
        str or dict
            The bare name for a core data type, or the JSON-compatible
            object form for an extension data type.
        """
        if not self.configuration.to_json():
            return self.name
        return super().to_json()

    @property
    def numpy(self) -> np.dtype:
        """The equivalent `numpy.dtype`."""
        return asdtype(self)

    def to_version(self, version: tz.ZarrVersion) -> tx.Any:
        """Convert this data type to another Zarr version.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        object
            This data type unchanged for version 3, or the equivalent
            v1 or v2 data type otherwise.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3.
        """
        if version == 3:
            return self
        if version == 2:
            from abczarr.metadata.v2.dtypes import DType as DTypeV2
            return DTypeV2(self.numpy)
        if version == 1:
            from abczarr.metadata.v1.dtypes import DType as DTypeV1
            return DTypeV1(self.numpy)
        raise ValueError(f"Unsupported version: {version}")



@autofrozen(extra_items=False)
class DTypeConfigImpl(DTypeConfig):
    """This class is the base for a data type configuration whose
    parameters are all declared."""


@autofrozen
class DTypeImpl(DType):
    """This class is the base for a data type whose configuration is
    declared, not open-ended."""

    configuration: DTypeConfigImpl


def _make_dtype_class(
    name: str,
    base: tx.Type[DType] = DType,
    module: str = __name__
) -> tx.Type[DType]:
    """Build and register a `DType` subclass for the core data type
    named `name`.

    The class name is derived from `name` by splitting on ``.`` and
    ``_`` and capitalizing each part, so ``"numpy.datetime64"``
    becomes ``NumpyDatetime64``.

    Parameters
    ----------
    name : str
        The Zarr v3 data type name the class is registered under.
    base : type
        The `DType` subclass the new class derives from.
    module : str
        The `__module__` recorded on the new class.

    Returns
    -------
    type
        The newly built and registered `DType` subclass.
    """
    class_name = "".join(map(str.capitalize, _splitall(name, sep=(".", "_"))))
    register = register_subclass(name=name)
    return register(autofrozen(type(
        class_name,
        (base,),
        {
            "__module__": module,
            "name": autofield(default=name, type=tx.Literal[name]),
        }
    )))


def _splitall(
    *s: str, sep: tx.Optional[tx.Tuple[str]] = None
) -> tx.Tuple[str, ...]:
    """Split each string in `s` on every separator in `sep`, in order.

    Parameters
    ----------
    *s : str
        The strings to split.
    sep : tuple of str or None
        The separators to split on, applied one after another. `None`
        or a single string is treated as a one-element tuple.

    Returns
    -------
    tuple of str
        The pieces left after every separator has been applied.
    """
    if sep is None or isinstance(sep, str):
        sep = (sep,)
    if len(sep) == 0:
        return s
    sep, *othersep = sep
    o = []
    for p in s:
        o.extend(_splitall(*p.split(sep), sep=othersep))
    return tuple(o)


def _make_dtype_classes(
    namespace: tx.MutableMapping,
    names: tx.Iterable[str] = (),
    ignore: tx.Sequence[str] = (),
    base: tx.Type[DType] = DTypeImpl,
) -> None:
    """Build and register a `DType` subclass for each name in `names`,
    installing every class into `namespace`.

    Parameters
    ----------
    namespace : mutable mapping
        The module namespace the new classes are installed into,
        keyed by class name.
    names : iterable of str
        The Zarr v3 data type names to build classes for.
    ignore : sequence of str
        Names to skip, typically ones a module defines by hand
        instead.
    base : type
        The `DType` subclass each new class derives from.

    Returns
    -------
    list of str
        The class names installed into `namespace`, suitable for
        extending a module's `__all__`.
    """
    if isinstance(ignore, str):
        ignore = (ignore,)

    cls_names = []
    for name in names:
        if name in ignore:
            continue
        cls = _make_dtype_class(
            name,
            base=base,
            module=namespace.get("__name__", __name__)
        )
        namespace[cls.__name__] = cls
        cls_names.append(cls.__name__)

    return cls_names
