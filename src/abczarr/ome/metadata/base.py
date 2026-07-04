__all__ = ["OMEMetadata", "OME"]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

# core
from abczarr._core.metadata import FlexibleMetadata
from abczarr._core.auto.attrs import autodefine, field, fields


@autodefine
class OMEMetadata(FlexibleMetadata):

    def to_version(self, version: str) -> tx.Self:

        # Find name of similar class in target version
        fullname = type(self).__qualname__
        abczarr, ome, schemas, v, module, *attrs = fullname.split(".")
        v = "v" + version.replace("_", ".")
        module = ".".join((abczarr, ome, schemas, v, module))

        try:
            newcls = importlib.import_module(module)
            for attr in attrs:
                newcls = getattr(newcls, attr)
        except (ModuleNotFoundError, AttributeError):
            raise ValueError(
                f"Object {type(self).__name__} does not exist in OME {version}"
            )

        kwargs = {}
        for f in fields(newcls):
            if f.init and hasattr(self, f.name):
                value = getattr(self, f.name)
                if isinstance(value, OMEMetadata):
                    value = value.to_version(version)
                kwargs[f.name] = value

        return newcls(**kwargs)


@autodefine
class OME(OMEMetadata):
    version: str = field(factory=False)
