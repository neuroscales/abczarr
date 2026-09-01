__all__ = ["OMEMetadata", "OME"]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field, fields

# core
from abczarr._core.metadata import FlexibleMetadata


@autodefine
class OMEMetadata(FlexibleMetadata):

    def to_version(self, version: str) -> tx.Self:
        """Convert this OME metadata to another OME-NGFF version.

        Each class has a same-named counterpart in the target version's
        package; this maps to it and rebuilds it field by field, recursing
        into nested OME objects and sequences of them.
        """
        return _to_version(self, version)


def _target_class(cls: type, version: str) -> type:
    # abczarr.ome.metadata.v0_4.images -> abczarr.ome.metadata.v0_5.images
    parts = cls.__module__.split(".")
    parts[parts.index("metadata") + 1] = "v" + version.replace(".", "_")
    try:
        obj: tx.Any = importlib.import_module(".".join(parts))
        for name in cls.__qualname__.split("."):
            obj = getattr(obj, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"{cls.__name__} does not exist in OME {version}"
        ) from e


def _to_version(value: tx.Any, version: str) -> tx.Any:
    if isinstance(value, OMEMetadata):
        newcls = _target_class(type(value), version)
        kwargs = {}
        for f in fields(newcls):
            if not f.init:
                continue
            if f.name == "version":
                kwargs["version"] = version
            elif hasattr(value, f.name):
                kwargs[f.name] = _to_version(getattr(value, f.name), version)
        return newcls(**kwargs)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_version(v, version) for v in value)
    return value


@autodefine
class OME(OMEMetadata):
    version: str = field(factory=False)
