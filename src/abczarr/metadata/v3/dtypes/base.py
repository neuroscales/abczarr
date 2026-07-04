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



@autofrozen(extra_items=tz.FrozenJSON)
class DTypeConfig(TypedConfig):
    ...


@autofrozen
class DType(MustUnderstandExtension):
    configuration: DTypeConfig

    @property
    def numpy(self) -> np.dtype:
        """
        Return the corresponding numpy dtype.
        """
        return asdtype(self)

    def to_version(self, version: tz.ZarrVersion) -> tx.Any:
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
    ...


@autofrozen
class DTypeImpl(DType):
    configuration: DTypeConfigImpl


def _make_dtype_class(
    name: str,
    base: tx.Type[DType] = DType,
    module: str = __name__
) -> tx.Type[DType]:
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


def _splitall(*s: str, sep: tx.Optional[tx.Tuple[str]] = None) -> tx.List[str]:
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
