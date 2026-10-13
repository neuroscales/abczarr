"""The version-independent OME-Zarr metadata model.

OME-Zarr, the NGFF specification, is a metadata convention for
bioimaging data stored in Zarr. It describes multiscale image
pyramids, high-content screening plates, segmentation labels, and
rendering settings, all as JSON attached to a Zarr group. abczarr
models that metadata as typed classes under `abczarr.ome`. One
package exists per NGFF version, `v0_1` through `v0_5`, plus the 0.6
pre-release previews `v0_6dev1` through `v0_6dev4` and `v0_6rc0`.

The examples below target 0.5, the latest stable version. The
[Reference](#reference) documents every version.

## Describing a multiscale image

A multiscale image is a pyramid of resolution levels, each a Zarr
array, described by
[Multiscale][abczarr.ome.v0_5.images.Multiscale]. It is built from a
plain dict shaped like the JSON the spec defines:

```pycon
>>> from abczarr.ome import v0_5
>>> multiscale = v0_5.Multiscale.from_json({
...     "name": "nucleus-stain",
...     "type": "gaussian",
...     "axes": [
...         {"name": "c", "type": "channel"},
...         {"name": "y", "type": "space", "unit": "micrometer"},
...         {"name": "x", "type": "space", "unit": "micrometer"},
...     ],
...     "datasets": [
...         {
...             "path": "0",
...             "coordinateTransformations": [
...                 {"type": "scale", "scale": [1.0, 0.325, 0.325]}
...             ],
...         },
...         {
...             "path": "1",
...             "coordinateTransformations": [
...                 {"type": "scale", "scale": [1.0, 0.65, 0.65]}
...             ],
...         },
...     ],
... })
>>> [axis.name for axis in multiscale.axes]
['c', 'y', 'x']
>>> [dataset.path for dataset in multiscale.datasets]
['0', '1']

```

Each axis in `axes` becomes an
[Axis][abczarr.ome.v0_5.axes.Axis], matched by its `type` field. The
`y` and `x` axes in this example become
[SpaceAxis][abczarr.ome.v0_5.axes.SpaceAxis] objects, since each
carries `type="space"`. Each entry in `datasets` becomes a
[Dataset][abczarr.ome.v0_5.images.Dataset]. A dataset names an array
and carries the [Scale][abczarr.ome.v0_5.transformations.Scale]
that places it in physical space, one value per axis, in that axis's
unit.

A `Multiscale` describes the pyramid, not the whole group. Wrapping
it in [OMEImage][abczarr.ome.v0_5.ome.OMEImage] produces the
metadata an image group actually carries, optionally alongside
[Omero][abczarr.ome.v0_5.omero.Omero] rendering settings:

```python
from abczarr.ome import v0_5

image = v0_5.OMEImage(
    version="0.5",
    multiscales=[multiscale],
    omero=v0_5.Omero.from_json({
        "channels": [
            {
                "color": "00FF00",
                "window": {"start": 0, "end": 1500, "min": 0, "max": 65535},
            },
        ],
    }),
)
```

The same shape covers the other kinds of OME-Zarr group:
[ImageLabel][abczarr.ome.v0_5.labels.ImageLabel] for a
segmentation, and
[Plate][abczarr.ome.v0_5.plates.Plate] /
[Well][abczarr.ome.v0_5.wells.Well] for a high-content
screen, wrapped in
[OMEImageLabel][abczarr.ome.v0_5.ome.OMEImageLabel],
[OMEPlate][abczarr.ome.v0_5.ome.OMEPlate] and
[OMEWell][abczarr.ome.v0_5.ome.OMEWell].

## Reading and writing OME metadata on a group

OME-Zarr metadata lives in a group's user attributes, the same
`attrs` mapping any Zarr group exposes. NGFF 0.5 nests it all under
one `"ome"` key, so an object round-trips through
[to_json][abczarr._core.metadata.Metadata.to_json] and
[from_json][abczarr._core.metadata.Metadata.from_json] like this:

```python
group.attrs["ome"] = image.to_json()

loaded = v0_5.OMEImage.from_json(group.attrs["ome"])
loaded.multiscales[0].axes[0].name  # "c"
```

Earlier NGFF versions, 0.4 and before, write the same fields directly
at the top level of `attrs` instead of nesting them under `"ome"`:
`group.attrs.update(image.to_json())`.

## Converting between NGFF versions

[OMEMetadata.to_version][abczarr.ome.base.OMEMetadata.to_version]
converts a piece of OME metadata, built against one NGFF version, to
another. It works on the top-level container and on any nested piece
of metadata alike:

```pycon
>>> from abczarr.ome import v0_4
>>> old = v0_4.Multiscale.from_json({
...     "version": "0.4",
...     "axes": [
...         {"name": "y", "type": "space"},
...         {"name": "x", "type": "space"},
...     ],
...     "datasets": [
...         {
...             "path": "0",
...             "coordinateTransformations": [
...                 {"type": "scale", "scale": [1.0, 1.0]}
...             ],
...         }
...     ],
... })
>>> new = old.to_version("0.5")
>>> type(new).__module__
'abczarr.ome.v0_5.images'
>>> new.to_version("0.4") == old
True

```

Fields both versions share carry over unchanged. Converting forward, a
field only the newer version has gets a reasonable default. Axes, for
example, gained a `type` field in NGFF 0.4, defaulted from the axis's
name. Converting back, that field is dropped. Converting to a version
that would need information the source does not carry raises
`ValueError` rather than guessing.

!!! example
    ```pycon
    >>> from abczarr.ome import v0_2
    >>> untyped = v0_2.Multiscale.from_json({
    ...     "version": "0.2",
    ...     "name": "x",
    ...     "datasets": [{"path": "0"}],
    ... })
    >>> untyped.to_version("0.3")
    Traceback (most recent call last):
        ...
    ValueError: cannot convert Multiscale from OME 0.2 to 0.3:
    the target requires information OME 0.2 does not carry

    ```
"""

__all__ = ["OMEMetadata", "OME"]

# stdlib
import contextvars
import importlib
import warnings
from collections import abc

# dependencies
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, evolve, field, fields

# core
from abczarr._core.metadata import FlexibleMetadata
from abczarr._core.rfc2119 import MISSING
from abczarr.errors import UnsupportedConversion

# locals
from . import _datasets06, _transforms06

#: How a cross-version conversion treats information the target version
#: cannot hold.
#:
#: * ``"lossy"`` drops the information silently.
#: * ``"warn"`` drops the information and emits one warning naming what
#:   was dropped. This is the default for OME conversions.
#: * ``"strict"`` raises
#:   [UnsupportedConversion][abczarr.errors.UnsupportedConversion] instead
#:   of dropping anything.
#:
#: This type mirrors the Zarr metadata layer's own ``ConversionPolicy``.
#: It is defined again here, rather than imported, so that importing the
#: OME model does not also import the Zarr metadata package and, through
#: it, a storage backend.
ConversionPolicy = tx.Literal["lossy", "warn", "strict"]

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
    "0.6": "v0_6",
}
_VERSIONS = list(_MODULES)


def _is_stable(version: str) -> bool:
    """Whether `version` names a released version.

    A released version is written with digits and separators only. A
    pre-release marker such as ``dev``, ``rc``, ``alpha``, or ``beta``
    carries letters, following PEP 440. Any letter in `version` therefore
    marks it as not yet stable.
    """
    return not any(char.isalpha() for char in version)


def _version_key(version: str) -> "tx.Tuple[int, ...]":
    """Order a version by its numeric release segments.

    The segments are compared as integers, so ``0.10`` comes after ``0.9``
    rather than before it, as a string comparison would place it.
    """
    return tuple(int(part) for part in version.split("."))


#: The newest released, non-preview OME-NGFF version. This is ``"0.6"``
#: today, and the default version used when metadata is written without
#: one.
LATEST_STABLE = max(
    (v for v in _VERSIONS if _is_stable(v)), key=_version_key
)

#: The axis type a bare axis name implies. A v0.3 axis is written as a
#: bare name. v0.4 replaced it with an object that carries an explicit
#: type.
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
    on. Each lives in the version package it belongs to, such as
    `abczarr.ome.v0_5.images`. Build one with
    [from_json][abczarr._core.metadata.Metadata.from_json] from the
    JSON an OME-Zarr group carries, or with keyword arguments matching
    its fields. [to_json][abczarr._core.metadata.Metadata.to_json]
    serializes it back to that same shape, and any key it does not
    recognize survives the round trip unchanged.

    Use [to_version][abczarr.ome.base.OMEMetadata.to_version]
    to convert an object built against one NGFF version to another.
    """

    def to_version(
        self, version: str, policy: ConversionPolicy = "warn"
    ) -> tx.Self:
        """Convert this OME metadata to another OME-NGFF version.

        This method works on any piece of OME metadata, not only the
        top-level container. A
        [Multiscale][abczarr.ome.v0_5.images.Multiscale] or an
        [Omero][abczarr.ome.v0_5.omero.Omero] converts just as well as an
        [OMEImage][abczarr.ome.v0_5.ome.OMEImage]. A field both versions
        carry is passed through unchanged. Converting forward, a field
        only the newer version has gets a reasonable default. Converting
        backward, that field is dropped.

        Crossing the 0.5/0.6 boundary reshapes coordinate metadata.
        Converting forward, 0.5's per-multiscale `axes` and per-dataset
        scale and translation become 0.6's named `coordinateSystems` and
        general coordinate transformations. Converting backward, a 0.6
        transformation the stable model cannot express, such as an affine
        or a rotation, is treated according to `policy`.

        Between the 0.6 previews, a `mapAxis` or `byDimension`
        transformation switches between addressing axes by name and
        addressing them by index. Translating between axis names and axis
        indices needs the axes of the coordinate systems involved. A
        transformation converted on its own, apart from the multiscale
        that declares those coordinate systems, raises `ValueError` when
        that translation is needed. Later versions have no `mapAxis`
        counterpart for a 0.6.dev1 `mapAxis` that drops or repeats an input
        axis. Such a `mapAxis` becomes an equivalent `affine` that selects
        the same input coordinates. A `projectAxis`, which first appears in
        0.6rc0, becomes an equivalent `affine` in earlier versions in the
        same way. Neither `affine` turns back into the original
        transformation when converted again. A dataset in 0.6.dev2 cannot
        carry an `identity`. Converting a dataset to or through 0.6.dev2
        therefore replaces its `identity` with the equivalent `scale` of
        ones, which later versions keep as a `scale`.

        Up to 0.6.dev3, a transformation names its `input` and its `output`
        with strings. From 0.6.dev4 on, it uses objects. On a dataset, the
        `input` string becomes `{"path": ...}` and the `output` string
        becomes `{"name": ...}`. On a multiscale or a scene, both strings
        are coordinate system names and become `{"name": ...}`. Converting
        back replaces each object with its string when the target version
        requires a string. A 0.5 multiscale-level transformation maps the
        datasets' coordinate system onto a second coordinate system with
        the same axes. The name of that second system is the name of the
        first one followed by ``_transformed``.

        Parameters
        ----------
        version : str
            The target OME-NGFF version, such as ``"0.4"`` or
            ``"0.6rc0"``.
        policy : ConversionPolicy
            How to treat information the target version cannot hold:
            ``"lossy"`` drops it silently, ``"warn"`` (the default)
            drops it with one warning, and ``"strict"`` raises
            [UnsupportedConversion][abczarr.errors.UnsupportedConversion].

        !!! example
            ```pycon
            >>> from abczarr.ome import v0_4
            >>> m = v0_4.Multiscale.from_json({
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
            'abczarr.ome.v0_5.images'
            >>> m5.to_version("0.4") == m
            True

            ```

        Raises
        ------
        ValueError
            If `version` names no known OME-NGFF version, or if
            converting to it would require information this object
            does not carry. Within the 0.6 previews, this includes the
            axes of a coordinate system that is not declared alongside the
            transformation. A `byDimension` whose transformations do not
            write every output axis exactly once also raises this error.
            So does a `mapAxis` or `byDimension` that refers to an axis its
            coordinate system does not have. An index-list `mapAxis` whose
            length differs from the number of input axes also raises this
            error, because such a list is not a permutation of those axes.
            The error is also raised for a dataset whose coordinate
            transformations the target version does not allow. For example,
            a 0.6.dev1 dataset that lists several transformations cannot be
            converted to a later 0.6 version. A reference that the target
            version cannot spell also raises this error. Examples are a
            dataset output that carries a path, converted to a version
            that names the output with a string, and a string reference on
            a transformation converted on its own to 0.6.dev4 or later.
            The error is also raised when the target version requires a
            dataset reference that the source omits, and the enclosing
            multiscale does not declare exactly one coordinate system.
        UnsupportedConversion
            If `policy` is ``"strict"`` and a field or transformation
            cannot be represented in `version`.
        """
        if version not in _MODULES:
            raise ValueError(f"Unknown OME version: {version!r}")
        source = _version_of(type(self))
        i, j = _VERSIONS.index(source), _VERSIONS.index(version)
        step = 1 if j >= i else -1
        obj: tx.Any = self
        for k in range(i, j, step):
            obj = _migrate(obj, _VERSIONS[k], _VERSIONS[k + step], policy)
        return obj


def _version_of(cls: type) -> str:
    parts = cls.__module__.split(".")
    suffix = parts[parts.index("ome") + 1]
    for version, module in _MODULES.items():
        if module == suffix:
            return version
    raise ValueError(f"{cls.__module__} is not a known OME version")


def _package(version: str) -> str:
    return "abczarr.ome." + _MODULES[version]


def _version_package(data: tx.Mapping) -> tx.Any:
    """The version package a piece of top-level OME metadata belongs to, read
    from its ``version`` field."""
    version = data.get("version")
    if version is None:
        raise ValueError(
            "cannot tell which OME-NGFF version this metadata is: it has no "
            "'version' field. Build the version's own class instead (for "
            "example abczarr.ome.v0_5.OME), which knows its version."
        )
    if version not in _MODULES:
        raise ValueError(f"Unknown OME version: {version!r}")
    return importlib.import_module(_package(version))


def _target_class(cls: type, version: str) -> type:
    parts = cls.__module__.split(".")
    parts[parts.index("ome") + 1] = _MODULES[version]
    try:
        obj: tx.Any = importlib.import_module(".".join(parts))
        for name in cls.__qualname__.split("."):
            obj = getattr(obj, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"{cls.__name__} does not exist in OME {version}"
        ) from e


def _report_loss(policy: ConversionPolicy, field: str, version: str) -> None:
    """Apply a conversion policy to something the target version cannot hold.

    A migration calls this for each field or transformation it cannot
    carry over to OME `version`. It does nothing under ``"lossy"``, emits
    one warning under ``"warn"``, and raises
    [UnsupportedConversion][abczarr.errors.UnsupportedConversion] under
    ``"strict"``.

    Parameters
    ----------
    policy : ConversionPolicy
        How to treat the loss.
    field : str
        The name of what cannot be represented. This is a field name, or
        a transformation type such as ``"affine"``.
    version : str
        The OME-NGFF version being converted to.

    Raises
    ------
    UnsupportedConversion
        If `policy` is ``"strict"``.
    """
    if policy == "lossy":
        return
    if policy == "warn":
        warnings.warn(
            f"dropping {field!r}: not representable in OME {version}",
            stacklevel=3,
        )
        return
    if policy == "strict":
        raise UnsupportedConversion(field, version)
    raise ValueError(f"unknown conversion policy: {policy!r}")


def _migrate(
    value: tx.Any, from_v: str, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    if isinstance(value, OMEMetadata):
        # A migration registered for a base class (such as every 0.6
        # CoordinateTransformation) applies to all of its subclasses.
        table = _MIGRATIONS.get((from_v, to_v), {})
        for klass in type(value).__mro__:
            migration = table.get(klass.__qualname__)
            if migration is not None:
                return migration(value, to_v, policy)
        newcls = _target_class(type(value), to_v)
        return _rebuild(value, newcls, to_v, from_v, policy)
    if isinstance(value, (list, tuple)):
        return type(value)(_migrate(v, from_v, to_v, policy) for v in value)
    return value


def _rebuild(
    source: tx.Any,
    newcls: type,
    to_v: str,
    from_v: str,
    policy: ConversionPolicy,
    converted: tx.Optional[tx.Mapping[str, tx.Any]] = None,
) -> tx.Any:
    # `converted` holds fields the caller has already converted to `to_v`.
    # An object that declares coordinate systems (a 0.6 multiscale) makes
    # their axes available to the transformations nested inside it, which
    # need them to translate between axis names and axis indices.
    converted = converted or {}
    systems = getattr(source, "coordinateSystems", None)
    token = None
    if isinstance(systems, list):
        token = _SYSTEMS.set({**_systems_in_scope(), **_axis_names(systems)})
    try:
        kwargs = {}
        for f in fields(newcls):
            if not f.init:
                continue
            if f.name == "version":
                kwargs["version"] = to_v
            elif f.name in converted:
                kwargs[f.name] = converted[f.name]
            elif hasattr(source, f.name):
                kwargs[f.name] = _migrate(
                    getattr(source, f.name), from_v, to_v, policy
                )
    finally:
        if token is not None:
            _SYSTEMS.reset(token)
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


def _multiscale_3_to_4(
    ms: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    v4 = importlib.import_module(_package(to_v))
    axes = [
        v4.Axis.from_json({"name": a, "type": _AXIS_TYPE.get(a, "space")})
        for a in ms.axes
    ]
    scale = [1.0] * len(axes)
    datasets = [
        v4.Dataset.from_json(
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


def _multiscale_4_to_3(
    ms: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    v3 = importlib.import_module(_package(to_v))
    axes = [a.name for a in ms.axes]
    datasets = [v3.Dataset.from_json({"path": d.path}) for d in ms.datasets]
    return _carry(ms, v3.Multiscale, to_v, axes=axes, datasets=datasets)


def _carry(
    source: tx.Any, newcls: type, to_v: str, **overrides: tx.Any
) -> tx.Any:
    """Build `newcls`, taking the given fields from `overrides` and the rest
    (that both versions share) straight from `source`."""
    kwargs = dict(overrides)
    for f in fields(newcls):
        if not f.init or f.name in kwargs:
            continue
        if f.name == "version":
            kwargs["version"] = to_v
        elif hasattr(source, f.name):
            kwargs[f.name] = getattr(source, f.name)
    return newcls(**kwargs)


# ----------------------------------------------------------------------
#   v0.5 <-> v0.6 (RFC-5): axes/scale <-> coordinate systems & transforms
# ----------------------------------------------------------------------
#
# The 0.5 (stable) model puts the axes on the multiscale and a
# `[Scale]` / `[Scale, Translation]` on each dataset. The 0.6 (RFC-5)
# model drops `axes`, carries one or more named `coordinateSystems`, and
# gives each dataset a list of general coordinate transformations. By
# convention a dataset's transform maps the array, referenced by its path
# in `input`, onto a coordinate system named in `output`.
#
# The migration is registered at the 0.5 <-> 0.6.dev1 step. The 0.6.dev1
# schema spells both references as strings: `input` is the dataset path
# and `output` is the system name. `_dataset_06` rewrites them into the
# `{"path": ...}` and `{"name": ...}` objects of 0.6.dev4 and later as the
# object walks up the dev chain, and back into strings on the way down.
#
# A 0.5 multiscale-level transform is applied after the dataset
# transforms. In 0.6 every multiscale-level transform needs an `input` and
# an `output` system, so the 0.5 one maps the datasets' system onto a
# second system with the same axes, named by `_global_system_name`. Two
# 0.5 transforms (a scale and a translation) become one `sequence`,
# because a 0.6 list of transforms between the same two systems would
# offer alternatives rather than compose them. The reverse step composes
# the multiscale-level transforms again, so the round trip is exact.


def _multiscale_5_to_6(
    ms: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    """0.5 -> 0.6.dev1 (lossless).

    Synthesize one output coordinate system from the 0.5 axes, and rewrite
    each dataset's scale/translation into a 0.6 transform mapping the
    array onto that named system.
    """
    dev = importlib.import_module(_package(to_v))
    # The output system's name is the multiscale's own name if it has one,
    # else a plain default. It is independent of the multiscale `name`
    # field, which is carried across separately.
    system_name = ms.name if isinstance(ms.name, str) else "physical"
    axes = [a.to_json() for a in ms.axes]

    doc: tx.Dict[str, tx.Any] = {
        "coordinateSystems": [{"name": system_name, "axes": axes}],
        "datasets": [_dataset_5_to_6(d, system_name) for d in ms.datasets],
    }
    if _is_set(ms.coordinateTransformations):
        global_name = _global_system_name(system_name)
        doc["coordinateSystems"].append({"name": global_name, "axes": axes})
        transform = _transforms_5_to_6(ms.coordinateTransformations)
        transform.update(input=system_name, output=global_name)
        doc["coordinateTransformations"] = [transform]
    if isinstance(ms.name, str):
        doc["name"] = ms.name
    if isinstance(ms.type, str):
        doc["type"] = ms.type
    if _is_set(ms.metadata):
        doc["metadata"] = ms.metadata.to_json()
    return dev.images.Multiscale.from_json(doc)


def _global_system_name(system_name: str) -> str:
    """The name of the system a 0.5 multiscale-level transform maps onto."""
    return system_name + "_transformed"


def _dataset_5_to_6(d: tx.Any, system_name: str) -> tx.Dict[str, tx.Any]:
    """One 0.5 dataset -> one 0.6.dev1 dataset dict.

    A lone scale becomes a single 0.6 `scale`; a scale followed by a
    translation becomes a `sequence` of the two, matching the RFC-5 corpus.
    The references are the 0.6.dev1 strings: the dataset path and the
    system name.
    """
    transform = _transforms_5_to_6(d.coordinateTransformations)
    transform.update(input=d.path, output=system_name)
    return {"path": d.path, "coordinateTransformations": [transform]}


def _transforms_5_to_6(cts: tx.Sequence[tx.Any]) -> tx.Dict[str, tx.Any]:
    """A 0.5 scale, or scale and translation, as one 0.6 transform."""
    if len(cts) == 1:
        return _transform_5_to_6(cts[0])
    return {
        "type": "sequence",
        "transformations": [_transform_5_to_6(t) for t in cts],
    }


def _transform_5_to_6(t: tx.Any) -> tx.Dict[str, tx.Any]:
    """A 0.5 scale/translation -> the same 0.6 transform, values preserved."""
    if t.type == "scale":
        return {"type": "scale", "scale": list(t.scale)}
    return {"type": "translation", "translation": list(t.translation)}


def _multiscale_6_to_5(
    ms: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    """0.6.dev1 -> 0.5 (potentially lossy).

    The datasets' output coordinate system supplies the 0.5 axes. Each
    dataset's transform list is reduced to the `Scale` (+`Translation`) the
    stable model allows; anything it cannot express is routed through
    `policy`.
    """
    v5 = importlib.import_module(_package(to_v))
    system = _output_system(ms)
    naxes = len(system.axes)

    doc: tx.Dict[str, tx.Any] = {
        "axes": [a.to_json() for a in system.axes],
        "datasets": [
            {
                "path": d.path,
                "coordinateTransformations": _reduce_transforms(
                    d.coordinateTransformations, naxes, to_v, policy
                ),
            }
            for d in ms.datasets
        ],
    }
    if _is_set(ms.coordinateTransformations):
        doc["coordinateTransformations"] = _reduce_transforms(
            ms.coordinateTransformations, naxes, to_v, policy
        )
    if isinstance(ms.name, str):
        doc["name"] = ms.name
    if isinstance(ms.type, str):
        doc["type"] = ms.type
    if _is_set(ms.metadata):
        doc["metadata"] = ms.metadata.to_json()
    return v5.images.Multiscale.from_json(doc)


def _output_system(ms: tx.Any) -> tx.Any:
    """The coordinate system the datasets map their arrays onto.

    Read from the `output` reference of a dataset's transform; falls back to
    the multiscale's first coordinate system when no reference names one.
    """
    name = None
    for d in ms.datasets:
        for t in d.coordinateTransformations:
            name = _ref_name(getattr(t, "output", None))
            if name is not None:
                break
        if name is not None:
            break
    if name is not None:
        for system in ms.coordinateSystems:
            if system.name == name:
                return system
    return ms.coordinateSystems[0]


def _reduce_transforms(
    transforms: tx.Iterable,
    naxes: int,
    to_v: str,
    policy: ConversionPolicy,
) -> tx.List[tx.Dict[str, tx.Any]]:
    """Reduce 0.6 transforms to the 0.5 `Scale`(+`Translation`) form.

    Composes the scales and translations in the list (flattening a
    `sequence`) into one diagonal affine ``p -> scale * p + translation``.
    A transform the stable model cannot express is routed through `policy`
    and otherwise dropped. A dataset left with no representable scale falls
    back to an identity scale (all ones).
    """
    # (scale, translation) of the running composition, as a function of the
    # original input coordinates: applying a further `scale` s gives
    # s * (scale * p + translation); a further `translation` t adds t.
    scale = [1.0] * naxes
    translation = [0.0] * naxes

    def apply(t: tx.Any) -> None:
        ttype = getattr(t, "type", None)
        if ttype == "sequence":
            for inner in t.transformations:
                apply(inner)
        elif ttype == "identity":
            pass
        elif ttype == "scale" and isinstance(getattr(t, "scale", None), list):
            for i, s in enumerate(t.scale):
                scale[i] = scale[i] * s
                translation[i] = translation[i] * s
        elif ttype == "translation" and isinstance(
            getattr(t, "translation", None), list
        ):
            for i, offset in enumerate(t.translation):
                translation[i] = translation[i] + offset
        else:
            # affine, rotation, mapAxis, a path-referenced scale, ... --
            # nothing the stable model can carry.
            _report_loss(policy, ttype or "transformation", to_v)

    for t in transforms:
        apply(t)

    result = [{"type": "scale", "scale": scale}]
    if any(offset != 0.0 for offset in translation):
        result.append({"type": "translation", "translation": translation})
    return result


def _space_to_json(
    space: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Dict[str, tx.Any]:
    """Degrade a typed 0.6 `Space` to the bare-JSON reference older 0.6
    previews carry (the `Space` object was introduced at 0.6.dev4)."""
    return space.to_json()


def _is_set(value: tx.Any) -> bool:
    """Whether an optional/recommended field carries a real value."""
    return value is not MISSING and value is not None


# ----------------------------------------------------------------------
#   within 0.6: axis addressing of mapAxis / byDimension / projectAxis
# ----------------------------------------------------------------------
#
# The 0.6 previews disagree on how a transformation addresses single axes
# (by name or by index) and on which transformation types exist. The JSON-
# level rewrite lives in `_transforms06`; here it is hooked into the
# migration table for every step of the 0.6 chain. Translating names to
# indices needs the axes of the coordinate systems the transformation
# refers to. `_rebuild` publishes those in `_SYSTEMS` while it converts an
# object that declares them, so a transformation converted as part of its
# multiscale finds them.

#: Coordinate-system name -> axis names, for the systems in scope of the
#: object being converted.
_SYSTEMS: "contextvars.ContextVar[tx.Optional[tx.Mapping]]" = (
    contextvars.ContextVar("abczarr_ome_coordinate_systems", default=None)
)


def _systems_in_scope() -> tx.Mapping[str, tx.List[str]]:
    return _SYSTEMS.get() or {}


def _axis_names(systems: tx.Iterable) -> tx.Dict[str, tx.List[str]]:
    """Map each named coordinate system in `systems` to its axis names."""
    result = {}
    for system in systems:
        name = getattr(system, "name", None)
        axes = getattr(system, "axes", None)
        if not isinstance(name, str) or not isinstance(axes, list):
            continue
        names = [getattr(axis, "name", None) for axis in axes]
        if all(isinstance(n, str) for n in names):
            result[name] = names
    return result


# ----------------------------------------------------------------------
#   within 0.6: the form of the input and output references
# ----------------------------------------------------------------------
#
# Up to 0.6.dev3 a transformation names its `input` and `output` with
# strings, and from 0.6.dev4 on with `{"name": ...}` or `{"path": ...}`
# objects. What a string means depends on where the transformation sits:
#
# * on a dataset, `input` is the array path and `output` is a coordinate
#   system name. `_dataset_06` rewrites these references itself.
# * on a multiscale, both strings are coordinate system names. Going down,
#   a name-only object becomes its name. An object that also carries a
#   path is kept in 0.6.dev2 and 0.6.dev3, whose schemas accept it, and
#   cannot be converted to 0.6.dev1, whose schema requires a string.
# * on a scene, the 0.6.dev3 schema already requires objects that carry a
#   name, so a string found there is a coordinate system name, and an
#   object is kept as it is.
# * anywhere else, such as a transformation converted on its own, the
#   meaning of a string is unknown. Converting a string reference to an
#   object therefore raises a ValueError, and objects are kept as they are.

#: Where the transformation being converted sits: ``"dataset"``,
#: ``"multiscale"``, ``"scene"``, or ``None`` when it is converted on its
#: own.
_PLACE: "contextvars.ContextVar[tx.Optional[str]]" = contextvars.ContextVar(
    "abczarr_ome_transformation_place", default=None
)


def _object_refs(version: str) -> bool:
    """Whether `version` spells references as objects rather than strings."""
    return _datasets06.REFS[version] == _datasets06.OBJECT


def _ref_field(ref: tx.Any, key: str) -> tx.Optional[str]:
    """The `key` (``"name"`` or ``"path"``) of an object reference."""
    if isinstance(ref, abc.Mapping):
        value = ref.get(key)
    elif isinstance(ref, OMEMetadata):
        value = getattr(ref, key, None)
    else:
        value = None
    return value if isinstance(value, str) else None


def _ref_name(ref: tx.Any) -> tx.Optional[str]:
    """The coordinate system name a reference gives, if any."""
    return ref if isinstance(ref, str) else _ref_field(ref, "name")


def _convert_refs(
    doc: tx.Dict[str, tx.Any], from_v: str, to_v: str, place: tx.Optional[str]
) -> tx.Dict[str, tx.Any]:
    """Rewrite the references of a transformation that is not a dataset's."""
    if place == "dataset":
        return doc
    doc = dict(doc)
    for side in ("input", "output"):
        ref = doc.get(side)
        if isinstance(ref, str) and _object_refs(to_v):
            if place is None:
                raise ValueError(
                    f"cannot convert a {doc.get('type')} transformation "
                    f"from OME {from_v} to {to_v}. Its {side} {ref!r} must "
                    f"become an object, and whether the string names a "
                    f"coordinate system or an array path is known only for "
                    f"a transformation converted as part of its dataset, "
                    f"multiscale or scene."
                )
            doc[side] = {"name": ref}
        elif isinstance(ref, abc.Mapping) and not _object_refs(to_v):
            if place != "multiscale":
                continue
            if set(ref) == {"name"} and isinstance(ref["name"], str):
                doc[side] = ref["name"]
            elif to_v == "0.6.dev1":
                raise ValueError(
                    f"cannot convert a multiscale {doc.get('type')} "
                    f"transformation from OME {from_v} to {to_v}. Its "
                    f"{side} {dict(ref)!r} cannot be written as the "
                    f"coordinate system name that OME {to_v} requires."
                )
    return doc


def _with_place(
    place: str, rebuild: tx.Callable[[], tx.Any]
) -> tx.Any:
    token = _PLACE.set(place)
    try:
        return rebuild()
    finally:
        _PLACE.reset(token)


def _multiscale_06(
    ms: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    """Convert a 0.6 multiscale one step along the 0.6 chain."""
    from_v = _version_of(type(ms))
    newcls = _target_class(type(ms), to_v)
    return _with_place(
        "multiscale", lambda: _rebuild(ms, newcls, to_v, from_v, policy)
    )


def _scene_06(scene: tx.Any, to_v: str, policy: ConversionPolicy) -> tx.Any:
    """Convert a 0.6 scene one step along the 0.6 chain."""
    from_v = _version_of(type(scene))
    newcls = _target_class(type(scene), to_v)
    return _with_place(
        "scene", lambda: _rebuild(scene, newcls, to_v, from_v, policy)
    )


def _transformation_06(
    t: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    """Convert a 0.6 transformation one step along the 0.6 chain."""
    from_v = _version_of(type(t))
    doc = _transforms06.convert(
        t.to_json(),
        from_v,
        to_v,
        _systems_in_scope(),
        lambda field: _report_loss(policy, field, to_v),
    )
    doc = _convert_refs(doc, from_v, to_v, _PLACE.get())
    pkg = importlib.import_module(_package(to_v))
    return pkg.transformations.CoordinateTransformation.from_json(doc)


def _by_dimension_child_06(
    child: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    """Convert one wrapped `byDimension` entry (0.6.dev3 and later)."""
    newcls = _target_class(type(child), to_v)
    from_v = _version_of(type(child))
    doc = _transforms06.convert(
        {"type": "byDimension", "transformations": [child.to_json()]},
        from_v,
        to_v,
        _systems_in_scope(),
        lambda field: _report_loss(policy, field, to_v),
    )
    return newcls.from_json(doc["transformations"][0])


def _dataset_06(d: tx.Any, to_v: str, policy: ConversionPolicy) -> tx.Any:
    """Convert a 0.6 dataset one step along the 0.6 chain.

    The 0.6 previews disagree on which transformations a dataset may carry.
    0.6.dev1 allows any list of transformations. Later versions allow
    exactly one transformation, and only 0.6.dev2 forbids an identity. An
    identity is therefore replaced by the equivalent scale of ones on the
    way into 0.6.dev2. Any other dataset that the target version cannot
    hold raises a ValueError that names the dataset.

    The `input` and `output` references are rewritten into the form of the
    target version: the array path and the coordinate system name as
    strings up to 0.6.dev3, and as `{"path": ...}` and `{"name": ...}`
    objects from 0.6.dev4 on.
    """
    from_v = _version_of(type(d))
    cts = d.coordinateTransformations
    allowed = _datasets06.TYPES[to_v]
    if (
        allowed is not None
        and "identity" not in allowed
        and len(cts) == 1
        and cts[0].type == "identity"
    ):
        d = evolve(d, coordinateTransformations=[_identity_as_scale(d)])
    problem = _datasets06.check_dataset(
        d.coordinateTransformations, to_v, refs=False
    )
    if problem is not None:
        raise ValueError(
            f"cannot convert dataset {d.path!r} from OME {from_v} to "
            f"{to_v}. {problem}"
        )
    sources = d.coordinateTransformations
    targets = _with_place(
        "dataset", lambda: _migrate(sources, from_v, to_v, policy)
    )
    targets = type(sources)(
        _dataset_refs(d, source, target, to_v)
        for source, target in zip(sources, targets)
    )
    return _rebuild(
        d,
        _target_class(type(d), to_v),
        to_v,
        from_v,
        policy,
        converted={"coordinateTransformations": targets},
    )


def _dataset_refs(
    d: tx.Any, source: tx.Any, target: tx.Any, to_v: str
) -> tx.Any:
    """Give `target` the references of `source` in the form of `to_v`.

    `source` is a transformation of dataset `d`, and `target` is the same
    transformation already converted to `to_v`. A reference that `to_v`
    requires and `source` lacks is filled in. The input is the dataset path,
    and the output is the only coordinate system in scope.
    """
    from_v = _version_of(type(d))
    form = _datasets06.REFS[to_v]
    needed = form != _datasets06.SEQUENCE_STRING or target.type == "sequence"
    path = source.input if isinstance(source.input, str) else None
    path = path or _ref_field(source.input, "path")
    name = _ref_name(source.output)
    output_path = _ref_field(source.output, "path")
    if needed and path is None:
        path = d.path
    if needed and name is None:
        name = _only_system(d, from_v, to_v)
    refs: tx.Dict[str, tx.Any] = {}
    if form == _datasets06.OBJECT:
        if path is not None:
            refs["input"] = {"path": path}
        if name is not None:
            refs["output"] = {"name": name}
            if output_path is not None:
                refs["output"]["path"] = output_path
    else:
        if output_path is not None:
            raise ValueError(
                f"cannot convert dataset {d.path!r} from OME {from_v} to "
                f"{to_v}. Its output refers to the coordinate system "
                f"{name!r} at path {output_path!r}, and OME {to_v} names the "
                f"output with a single string."
            )
        if path is not None:
            refs["input"] = path
        if name is not None:
            refs["output"] = name
    return evolve(target, **refs) if refs else target


def _only_system(d: tx.Any, from_v: str, to_v: str) -> str:
    """The name of the only coordinate system in scope of dataset `d`."""
    names = list(_systems_in_scope())
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"cannot convert dataset {d.path!r} from OME {from_v} to {to_v}. "
        f"OME {to_v} requires the coordinate system that the dataset maps "
        f"onto, the dataset does not name it, and the enclosing multiscale "
        f"does not declare exactly one coordinate system."
    )


def _identity_as_scale(d: tx.Any) -> tx.Any:
    """The scale of ones equivalent to dataset `d`'s identity.

    The number of axes is read from the coordinate system the identity maps
    to, which must be declared by the enclosing multiscale.
    """
    (identity,) = d.coordinateTransformations
    from_v = _version_of(type(d))
    output = _ref_name(identity.output)
    axes = _systems_in_scope().get(output) if isinstance(output, str) else None
    if axes is None:
        raise ValueError(
            f"cannot convert dataset {d.path!r} from OME {from_v}. Its "
            f"identity transformation must become a scale, and the number "
            f"of axes of that scale is unknown. The coordinate system the "
            f"identity maps to is not declared alongside the dataset."
        )
    doc = dict(identity.to_json(), type="scale", scale=[1.0] * len(axes))
    pkg = importlib.import_module(_package(from_v))
    return pkg.transformations.CoordinateTransformation.from_json(doc)


_MIGRATIONS = {
    ("0.3", "0.4"): {"Multiscale": _multiscale_3_to_4},
    ("0.4", "0.3"): {"Multiscale": _multiscale_4_to_3},
    ("0.5", "0.6.dev1"): {"Multiscale": _multiscale_5_to_6},
    ("0.6.dev1", "0.5"): {"Multiscale": _multiscale_6_to_5},
    # The typed `Space` reference does not exist before 0.6.dev4; turn it
    # back into a bare-JSON reference when stepping down past that boundary.
    ("0.6.dev4", "0.6.dev3"): {"Space": _space_to_json},
}
for _a, _b in zip(_transforms06.CHAIN, _transforms06.CHAIN[1:]):
    for _step in ((_a, _b), (_b, _a)):
        _MIGRATIONS.setdefault(_step, {}).update(
            {
                "CoordinateTransformation": _transformation_06,
                "ByDimension.Transformation": _by_dimension_child_06,
                "Dataset": _dataset_06,
                "Multiscale": _multiscale_06,
                "Scene": _scene_06,
            }
        )
del _a, _b, _step


@autodefine
class OME(OMEMetadata):
    """The version-tagged, top-level OME-Zarr metadata for a group.

    Every OME-Zarr group carries one of these: an image and its
    multiscale pyramid, a collection of labels, a plate, or a well. Each
    kind has its own subclass in every version's package. The `version`
    field records which NGFF version the metadata is written against, and
    with it what the rest of the fields mean.

    Build the concrete class for the version in use instead of this one.
    See [v0_5.OME][abczarr.ome.v0_5.ome.OME] and its siblings, including
    [OMEImage][abczarr.ome.v0_5.ome.OMEImage] and
    [OMEPlate][abczarr.ome.v0_5.ome.OMEPlate].

    Parameters
    ----------
    version : str
        The OME-NGFF version the metadata is written against.
    """

    version: str = field(factory=False)

    @classmethod
    def from_json(cls, data: tx.Any) -> tx.Self:
        """Create an OME container from a JSON-serializable dict.

        Called on a version's own class, such as ``v0_5.OME.from_json``,
        this picks the right image, plate, well, or label subclass for the
        data, the same way any OME class does.

        Called on this version-independent base class, it first reads the
        ``version`` field to decide which NGFF version the data belongs
        to, then hands the data to that version's own ``OME`` class. The
        base class cannot make that choice on its own, because every
        version's classes share it and it has no way to tell a v0.4 image
        from a v0.5 one. Metadata that carries no ``version`` field is
        therefore ambiguous and raises ``ValueError`` rather than being
        guessed at.

        Raises
        ------
        ValueError
            If called on the version-independent base with data that has no
            ``version`` field, or a ``version`` that names no known OME-NGFF
            version.
        """
        if cls is OME and isinstance(data, abc.Mapping):
            return _version_package(data).OME.from_json(data)
        return super().from_json(data)
