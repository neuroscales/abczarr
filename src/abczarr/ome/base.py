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
    ValueError: cannot convert Multiscale from OME 0.2 to 0.3: the target requires information OME 0.2 does not carry

    ```
"""

__all__ = ["OMEMetadata", "OME"]

# stdlib
import importlib
import warnings
from collections import abc

# dependencies
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field, fields

# core
from abczarr._core.metadata import FlexibleMetadata
from abczarr._core.rfc2119 import MISSING
from abczarr.errors import UnsupportedConversion

#: How a cross-version conversion treats information the target version
#: cannot hold.
#:
#: * ``"lossy"`` -- drop it silently.
#: * ``"warn"`` (the default for OME conversions) -- drop it, but emit one
#:   warning naming what was dropped.
#: * ``"strict"`` -- raise
#:   [UnsupportedConversion][abczarr.errors.UnsupportedConversion] instead of
#:   dropping anything.
#:
#: This mirrors the Zarr metadata layer's ``ConversionPolicy``; it is
#: defined here rather than imported so the OME model does not pull the
#: Zarr metadata package (and, through it, a backend) in at import time.
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
}
_VERSIONS = list(_MODULES)


def _is_stable(version: str) -> bool:
    """Whether *version* names a released version.

    A released version is written with digits and separators only. A
    pre-release -- ``dev``, ``rc``, ``alpha``, ``beta`` and the like --
    carries letters, following PEP 440, so any letter marks it as not yet
    stable.
    """
    return not any(char.isalpha() for char in version)


def _version_key(version: str) -> "tx.Tuple[int, ...]":
    """Order a version by its numeric release segments.

    The segments are compared as integers, so ``0.10`` comes after ``0.9``
    rather than before it as a string comparison would have it.
    """
    return tuple(int(part) for part in version.split("."))


#: The newest released (non-preview) OME-NGFF version -- ``"0.5"`` today.
#: The convenient default when metadata is written without a version.
LATEST_STABLE = max(
    (v for v in _VERSIONS if _is_stable(v)), key=_version_key
)

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
    (`abczarr.ome.v0_5.images`, for example). Build one with
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

        Works on any piece of OME metadata, not only the top-level
        container. A [Multiscale][abczarr.ome.v0_5.images.Multiscale]
        or an [Omero][abczarr.ome.v0_5.omero.Omero] converts
        just as well as an
        [OMEImage][abczarr.ome.v0_5.ome.OMEImage]. A field
        both versions carry is passed through unchanged. A field only
        the newer version has gets a reasonable default going forward,
        and is dropped going back.

        Crossing the 0.5 <-> 0.6 boundary reshapes coordinate metadata:
        0.5's per-multiscale `axes` and per-dataset scale/translation
        become 0.6's named `coordinateSystems` and general coordinate
        transformations, and vice versa. Going back to 0.5, a 0.6
        transformation the stable model cannot express (an affine, a
        rotation, and so on) is treated according to *policy*.

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
            If *version* names no known OME-NGFF version, or if
            converting to it would require information this object
            does not carry.
        UnsupportedConversion
            If *policy* is ``"strict"`` and a field or transformation
            cannot be represented in *version*.
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
    """Apply a conversion policy to something the target version can't hold.

    Called by a migration for each field or transformation it cannot carry
    over to OME *version*. Does nothing under ``"lossy"``, emits one warning
    under ``"warn"``, and raises
    [UnsupportedConversion][abczarr.errors.UnsupportedConversion] under
    ``"strict"``.

    Parameters
    ----------
    policy : ConversionPolicy
        How to treat the loss.
    field : str
        What cannot be represented -- a field name, or a transformation
        type such as ``"affine"``.
    version : str
        The OME-NGFF version being converted to.

    Raises
    ------
    UnsupportedConversion
        If *policy* is ``"strict"``.
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
        migration = _MIGRATIONS.get((from_v, to_v), {}).get(
            type(value).__qualname__
        )
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
) -> tx.Any:
    kwargs = {}
    for f in fields(newcls):
        if not f.init:
            continue
        if f.name == "version":
            kwargs["version"] = to_v
        elif hasattr(source, f.name):
            kwargs[f.name] = _migrate(
                getattr(source, f.name), from_v, to_v, policy
            )
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


# ----------------------------------------------------------------------
#   v0.5 <-> v0.6 (RFC-5): axes/scale <-> coordinate systems & transforms
# ----------------------------------------------------------------------
#
# The 0.5 (stable) model puts the axes on the multiscale and a
# `[Scale]` / `[Scale, Translation]` on each dataset. The 0.6 (RFC-5)
# model drops `axes`, carries one or more named `coordinateSystems`, and
# gives each dataset a list of general coordinate transformations. By
# convention a dataset's transform maps the array's own (intrinsic)
# system -- referenced by `input={"path": <dataset.path>}` -- onto a
# named output system, `output={"name": <system>}`.
#
# The migration is registered at the 0.5 <-> 0.6.dev1 step. In 0.6.dev1 a
# transform's `input`/`output` are still free JSON (the typed `Space`
# object arrives at 0.6.dev4), so they are written here as bare dicts;
# the per-field converters coerce them to `Space` as the object walks up
# the dev chain. Going the other way, `_space_to_json` turns a `Space`
# back into a dict at the 0.6.dev4 -> 0.6.dev3 step, so by 0.6.dev1 the
# references are dicts again.


def _multiscale_5_to_6(
    ms: tx.Any, to_v: str, policy: ConversionPolicy
) -> tx.Any:
    """0.5 -> 0.6.dev1 (lossless).

    Synthesize one output coordinate system from the 0.5 axes, and rewrite
    each dataset's scale/translation into a 0.6 transform mapping the
    array's intrinsic system onto that named system.
    """
    dev = importlib.import_module(_package(to_v))
    # The output system's name is the multiscale's own name if it has one,
    # else a plain default. It is independent of the multiscale `name`
    # field, which is carried across separately.
    system_name = ms.name if isinstance(ms.name, str) else "physical"

    doc: tx.Dict[str, tx.Any] = {
        "coordinateSystems": [
            {"name": system_name, "axes": [a.to_json() for a in ms.axes]}
        ],
        "datasets": [_dataset_5_to_6(d, system_name) for d in ms.datasets],
    }
    # A multiscale-level transform applies globally, to no single pair of
    # systems, so it is carried across without an input/output reference
    # (both are optional in 0.6); the reverse step reads it back the same
    # way, keeping the round trip exact.
    if _is_set(ms.coordinateTransformations):
        doc["coordinateTransformations"] = [
            _transform_5_to_6(t) for t in ms.coordinateTransformations
        ]
    if isinstance(ms.name, str):
        doc["name"] = ms.name
    if isinstance(ms.type, str):
        doc["type"] = ms.type
    if _is_set(ms.metadata):
        doc["metadata"] = ms.metadata.to_json()
    return dev.images.Multiscale.from_json(doc)


def _dataset_5_to_6(d: tx.Any, system_name: str) -> tx.Dict[str, tx.Any]:
    """One 0.5 dataset -> one 0.6 dataset dict.

    A lone scale becomes a single 0.6 `scale`; a scale followed by a
    translation becomes a `sequence` of the two, matching the RFC-5 corpus.
    """
    refs = {"input": {"path": d.path}, "output": {"name": system_name}}
    cts = d.coordinateTransformations
    if len(cts) == 1:
        transform = dict(_transform_5_to_6(cts[0]), **refs)
    else:
        scale, translation = cts
        transform = {
            "type": "sequence",
            "transformations": [
                _transform_5_to_6(scale),
                _transform_5_to_6(translation),
            ],
            **refs,
        }
    return {"path": d.path, "coordinateTransformations": [transform]}


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
    *policy*.
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
            ref = getattr(t, "output", None)
            if isinstance(ref, abc.Mapping) and "name" in ref:
                name = ref["name"]
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
    A transform the stable model cannot express is routed through *policy*
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


_MIGRATIONS = {
    ("0.3", "0.4"): {"Multiscale": _multiscale_3_to_4},
    ("0.4", "0.3"): {"Multiscale": _multiscale_4_to_3},
    ("0.5", "0.6.dev1"): {"Multiscale": _multiscale_5_to_6},
    ("0.6.dev1", "0.5"): {"Multiscale": _multiscale_6_to_5},
    # The typed `Space` reference does not exist before 0.6.dev4; turn it
    # back into a bare-JSON reference when stepping down past that boundary.
    ("0.6.dev4", "0.6.dev3"): {"Space": _space_to_json},
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
    See [v0_5.OME][abczarr.ome.v0_5.ome.OME] and its
    siblings, including [OMEImage][abczarr.ome.v0_5.ome.OMEImage]
    and [OMEPlate][abczarr.ome.v0_5.ome.OMEPlate].
    """

    version: str = field(factory=False)

    @classmethod
    def from_json(cls, data: tx.Any) -> tx.Self:
        """Create an OME container from a JSON-serializable dict.

        Called on a version's own class -- ``v0_5.OME.from_json`` -- this
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
            return _version_package(data).OME.from_json(data)
        return super().from_json(data)
