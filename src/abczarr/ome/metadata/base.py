__all__ = ["OMEMetadata", "OME"]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field, fields

# core
from abczarr._core.metadata import FlexibleMetadata

#: OME-NGFF versions, oldest to newest, and the package that holds each.
_MODULES = {
    "0.1": "v0_1",
    "0.2": "v0_2",
    "0.3": "v0_3",
    "0.4": "v0_4",
    "0.5": "v0_5",
    "0.6.dev4": "v0_6dev4",
}
_VERSIONS = list(_MODULES)

#: A v0.3 axis is a bare name; v0.4 made it an object carrying a type.
_AXIS_TYPE = {
    "x": "space",
    "y": "space",
    "z": "space",
    "t": "time",
    "c": "channel",
}


@autodefine
class OMEMetadata(FlexibleMetadata):

    def to_version(self, version: str) -> tx.Self:
        """Convert this OME metadata to another OME-NGFF version.

        Conversion steps one version at a time between the source and the
        target. Most steps map each class to its same-named counterpart in
        the target version and rebuild it field by field, recursing into
        nested OME objects and sequences of them. A step that changed shape
        (v0.3 <-> v0.4 introduced typed axes and per-dataset transforms) has
        an explicit migration.
        """
        if version not in _MODULES:
            raise ValueError(f"Unknown OME version: {version!r}")
        source = _version_of(type(self))
        i, j = _VERSIONS.index(source), _VERSIONS.index(version)
        step = 1 if j >= i else -1
        obj: tx.Any = self
        for k in range(i, j, step):
            obj = _migrate(obj, _VERSIONS[k], _VERSIONS[k + step])
        return obj


def _version_of(cls: type) -> str:
    parts = cls.__module__.split(".")
    suffix = parts[parts.index("metadata") + 1]
    for version, module in _MODULES.items():
        if module == suffix:
            return version
    raise ValueError(f"{cls.__module__} is not a known OME version")


def _package(version: str) -> str:
    return "abczarr.ome.metadata." + _MODULES[version]


def _target_class(cls: type, version: str) -> type:
    parts = cls.__module__.split(".")
    parts[parts.index("metadata") + 1] = _MODULES[version]
    try:
        obj: tx.Any = importlib.import_module(".".join(parts))
        for name in cls.__qualname__.split("."):
            obj = getattr(obj, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"{cls.__name__} does not exist in OME {version}"
        ) from e


def _migrate(value: tx.Any, from_v: str, to_v: str) -> tx.Any:
    if isinstance(value, OMEMetadata):
        migration = _MIGRATIONS.get((from_v, to_v), {}).get(
            type(value).__qualname__
        )
        if migration is not None:
            return migration(value, to_v)
        newcls = _target_class(type(value), to_v)
        return _rebuild(value, newcls, to_v, from_v)
    if isinstance(value, (list, tuple)):
        return type(value)(_migrate(v, from_v, to_v) for v in value)
    return value


def _rebuild(source: tx.Any, newcls: type, to_v: str, from_v: str) -> tx.Any:
    kwargs = {}
    for f in fields(newcls):
        if not f.init:
            continue
        if f.name == "version":
            kwargs["version"] = to_v
        elif hasattr(source, f.name):
            kwargs[f.name] = _migrate(getattr(source, f.name), from_v, to_v)
    try:
        return newcls(**kwargs)
    except TypeError as e:
        # a required field the source version does not carry (e.g. axes,
        # added at v0.3, cannot be inferred from an older version alone)
        if "Required field" in str(e):
            raise ValueError(
                f"cannot convert {newcls.__qualname__} from OME {from_v} to "
                f"{to_v}: the target requires information OME {from_v} does "
                f"not carry"
            ) from e
        raise


# ----------------------------------------------------------------------
#   v0.3 <-> v0.4: typed axes and per-dataset coordinate transforms
# ----------------------------------------------------------------------


def _multiscale_3_to_4(ms: tx.Any, to_v: str) -> tx.Any:
    v4 = importlib.import_module(_package(to_v))
    axes = [
        v4.Axis.from_dict({"name": a, "type": _AXIS_TYPE.get(a, "space")})
        for a in ms.axes
    ]
    scale = [1.0] * len(axes)
    datasets = [
        v4.Dataset.from_dict(
            {
                "path": d.path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale}
                ],
            }
        )
        for d in ms.datasets
    ]
    return _carry(ms, v4.Multiscale, to_v, axes=axes, datasets=datasets)


def _multiscale_4_to_3(ms: tx.Any, to_v: str) -> tx.Any:
    v3 = importlib.import_module(_package(to_v))
    axes = [a.name for a in ms.axes]
    datasets = [v3.Dataset.from_dict({"path": d.path}) for d in ms.datasets]
    return _carry(ms, v3.Multiscale, to_v, axes=axes, datasets=datasets)


def _carry(
    source: tx.Any, newcls: type, to_v: str, **overrides: tx.Any
) -> tx.Any:
    """Build *newcls*, taking the given fields from *overrides* and the rest
    (that both versions share) straight from *source*."""
    kwargs = dict(overrides)
    for f in fields(newcls):
        if not f.init or f.name in kwargs:
            continue
        if f.name == "version":
            kwargs["version"] = to_v
        elif hasattr(source, f.name):
            kwargs[f.name] = getattr(source, f.name)
    return newcls(**kwargs)


_MIGRATIONS = {
    ("0.3", "0.4"): {"Multiscale": _multiscale_3_to_4},
    ("0.4", "0.3"): {"Multiscale": _multiscale_4_to_3},
}


@autodefine
class OME(OMEMetadata):
    version: str = field(factory=False)
