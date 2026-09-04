"""The version-independent OME-Zarr metadata model.

OME-Zarr (the NGFF spec) is a metadata convention for bioimaging data
stored in Zarr: multiscale image pyramids, high-content screening
plates, segmentation labels, and rendering settings, all described by
JSON attached to a Zarr group. This module defines the two classes
every version shares:
[OMEMetadata][abczarr.ome.metadata.base.OMEMetadata], the base of
every OME metadata class, and [OME][abczarr.ome.metadata.base.OME],
the version-tagged top-level container a group's metadata builds on.

Each supported NGFF version has its own package under
`abczarr.ome.metadata`: `v0_1` through `v0_5`, plus the 0.6
pre-release previews `v0_6dev1` through `v0_6dev4` and `v0_6rc0`.
Each package has its own typed classes for that version's shape.
[OMEMetadata.to_version][abczarr.ome.metadata.base.OMEMetadata.to_version]
converts an object built against one version to another.
"""

__all__ = ["OMEMetadata", "OME"]

# stdlib
import importlib
from collections import abc

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
    "0.6.dev1": "v0_6dev1",
    "0.6.dev2": "v0_6dev2",
    "0.6.dev3": "v0_6dev3",
    "0.6.dev4": "v0_6dev4",
    "0.6rc0": "v0_6rc0",
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
    """The base of every OME-Zarr metadata class.

    Every piece of OME-Zarr metadata is a subclass of this: a
    multiscale pyramid, an axis, a plate, a rendering setting, and so
    on. Each lives in the version package it belongs to
    (`abczarr.ome.metadata.v0_5.images`, for example). Build one with
    [from_dict][abczarr._core.metadata.Metadata.from_dict] from the
    JSON an OME-Zarr group carries, or with keyword arguments matching
    its fields. [to_dict][abczarr._core.metadata.Metadata.to_dict]
    serializes it back to that same shape, and any key it does not
    recognize survives the round trip unchanged.

    Use [to_version][abczarr.ome.metadata.base.OMEMetadata.to_version]
    to convert an object built against one NGFF version to another.
    """

    def to_version(self, version: str) -> tx.Self:
        """Convert this OME metadata to another OME-NGFF version.

        Works on any piece of OME metadata, not only the top-level
        container. A [Multiscale][abczarr.ome.metadata.v0_5.images.Multiscale]
        or an [Omero][abczarr.ome.metadata.v0_5.omero.Omero] converts
        just as well as an
        [OMEImage][abczarr.ome.metadata.v0_5.ome.OMEImage]. A field
        both versions carry is passed through unchanged. A field only
        the newer version has gets a reasonable default going forward,
        and is dropped going back.

        Raises
        ------
        ValueError
            If *version* names no known OME-NGFF version, or if
            converting to it would require information this object
            does not carry.

        !!! example
            ```pycon
            >>> from abczarr.ome.metadata import v0_4
            >>> m = v0_4.Multiscale.from_dict({
            ...     "version": "0.4",
            ...     "axes": [
            ...         {"name": "y", "type": "space"},
            ...         {"name": "x", "type": "space"},
            ...     ],
            ...     "datasets": [{
            ...         "path": "0",
            ...         "coordinateTransformations": [
            ...             {"type": "scale", "scale": [1.0, 1.0]}
            ...         ],
            ...     }],
            ... })
            >>> m5 = m.to_version("0.5")
            >>> type(m5).__module__
            'abczarr.ome.metadata.v0_5.images'
            >>> m5.to_version("0.4") == m
            True

            ```
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


def _version_package(data: abc.Mapping) -> tx.Any:
    """The version package a piece of top-level OME metadata belongs to, read
    from its ``version`` field."""
    version = data.get("version")
    if version is None:
        raise ValueError(
            "cannot tell which OME-NGFF version this metadata is: it has no "
            "'version' field. Build the version's own class instead (for "
            "example abczarr.ome.metadata.v0_5.OME), which knows its version."
        )
    if version not in _MODULES:
        raise ValueError(f"Unknown OME version: {version!r}")
    return importlib.import_module(_package(version))


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
    """The version-tagged, top-level OME-Zarr metadata for a group.

    Every OME-Zarr group carries one of these: an image and its
    multiscale pyramid, a collection of labels, a plate, or a well.
    Each has its own subclass in every version's package. `version`
    records which NGFF version the metadata is written against, and
    with it what the rest of its fields mean.

    Build the concrete class for your version instead of this one.
    See [v0_5.OME][abczarr.ome.metadata.v0_5.ome.OME] and its
    siblings, including [OMEImage][abczarr.ome.metadata.v0_5.ome.OMEImage]
    and [OMEPlate][abczarr.ome.metadata.v0_5.ome.OMEPlate].
    """

    version: str = field(factory=False)

    @classmethod
    def from_dict(cls, data: tx.Any) -> tx.Self:
        """Create an OME container from a JSON-serializable dict.

        Called on a version's own class -- ``v0_5.OME.from_dict`` -- this
        picks the right image / plate / well / label subclass for the data,
        as any OME class does.

        Called on this version-independent base, it first reads the ``version``
        field to decide which NGFF version the data belongs to, then hands off
        to that version's ``OME``. The base cannot make that choice on its own:
        every version's classes share this one, so it has no way to tell a
        v0.4 image from a v0.5 one. Metadata that carries no ``version`` is
        therefore ambiguous, and raises ``ValueError`` rather than guessing.

        Raises
        ------
        ValueError
            If called on the version-independent base with data that has no
            ``version`` field, or a ``version`` that names no known OME-NGFF
            version.
        """
        if cls is OME and isinstance(data, abc.Mapping):
            return _version_package(data).OME.from_dict(data)
        return super().from_dict(data)
