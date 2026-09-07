"""Describe an OME-Zarr multiscale image to write.

An [ImageConfig][abczarr.ome.config.ImageConfig] is to OME-Zarr metadata
what an [ArrayConfig][abczarr.api.config.ArrayConfig] is to a Zarr array:
a high-level, human-facing description that lowers to the exact typed
metadata a group carries. Give it the axes, the voxel geometry, and how
the pyramid was built, and it produces OME-NGFF multiscales metadata for
any version you ask for.

Three coordinate spaces sit behind it::

    voxel (array index) --scale+translation--> intrinsic --transforms--> model

`scale` and `translation` place each resolution level's array indices
into a single *intrinsic* space shared by every level. `transforms` (or
the `voxel_to_world` shortcut) map that intrinsic space onto a *model* --
a world or anatomical frame. One config lowers to the stable 0.5 shape
and the 0.6 preview alike.
"""

__all__ = ["ImageConfig", "axis"]

# stdlib
from collections import abc as _abc

# dependencies
import numpy as np
import typing_extensions as tx

# core
from .._core.auto.attrs import define, evolve, field
from .base import (
    _AXIS_TYPE,
    _VERSIONS,
    LATEST_STABLE,
    OME,
    ConversionPolicy,
)
from .v0_6rc0.images import Dataset, Multiscale
from .v0_6rc0.ome import OMEImage
from .v0_6rc0.systems import Axis, CoordinateSystem
from .v0_6rc0.transformations import (
    Affine,
    CoordinateTransformation,
    Scale,
    Sequence,
    Space,
    Translation,
)

#: A single axis, as a user may spell it: a bare name, a ``(name, type)`` or
#: ``(name, type, unit)`` tuple, a JSON-style ``{"name", "type", "unit"}``
#: dict, or a ready-made 0.6 [Axis][...].
AxisSpec = tx.Union[str, tx.Mapping[str, tx.Any], tx.Sequence[tx.Any], Axis]

#: The axes of a config: a sequence of axis specs, or an ordered mapping from
#: axis name to a type string, a dict of the remaining fields, or `None`.
Axes = tx.Union[tx.Sequence[AxisSpec], tx.Mapping[str, tx.Any]]

#: One transform's key in the mapping form of `transforms`: a model system
#: name, or an ``(input, output)`` pair of system names.
TransformKey = tx.Union[str, tx.Tuple[str, str]]

#: The intrinsic-to-model transforms of a config: a sequence of transforms, or
#: a mapping from a [TransformKey][abczarr.ome.config.TransformKey] to one.
Transforms = tx.Union[tx.Sequence[tx.Any], tx.Mapping[TransformKey, tx.Any]]

#: A per-axis quantity (`scale`, `translation`, `factor`): one value for
#: every axis, a value per axis, or a mapping keyed by axis name or type.
PerAxis = tx.Union[float, int, tx.Sequence[float], tx.Mapping[str, float]]


def axis(
    name: str,
    type: tx.Optional[str] = None,  # noqa: A002 -- matches the OME field name
    unit: tx.Optional[str] = None,
) -> Axis:
    """Build a 0.6 [Axis][abczarr.ome.v0_6rc0.systems.Axis] from a name.

    The axis *type* is inferred from *name* when not given: ``x``, ``y``,
    and ``z`` are space, ``t`` is time, ``c`` is channel, and anything
    else is space. A known type gives back the matching axis subclass, a
    [SpaceAxis][abczarr.ome.v0_6rc0.systems.SpaceAxis] and so on.

    Parameters
    ----------
    name : str
        The axis name, such as ``"x"``.
    type : str, optional
        The axis type. Inferred from *name* when omitted.
    unit : str, optional
        The axis's physical unit, such as ``"micrometer"``.

    Returns
    -------
    Axis
        The typed axis.

    !!! example
        ```pycon
        >>> from abczarr.ome import axis
        >>> axis("x", unit="micrometer").type
        'space'
        >>> axis("t").type
        'time'

        ```
    """
    doc = {"name": name, "type": type or _AXIS_TYPE.get(name, "space")}
    if unit is not None:
        doc["unit"] = unit
    return Axis.from_json(doc)


@define
class ImageConfig:
    """A high-level description of an OME-Zarr multiscale image.

    Set the axes and the voxel geometry once, then lower to typed OME
    metadata for any version with
    [to_ome][abczarr.ome.config.ImageConfig.to_ome]. The same config
    produces the stable 0.5 shape and the 0.6 preview alike.

    Parameters
    ----------
    axes : sequence or mapping
        The array's axes, one per dimension. A sequence lists each axis as a
        name (``"x"``), a ``(name, type)`` or ``(name, type, unit)`` tuple, a
        ``{"name", "type", "unit"}`` dict, or a typed
        [Axis][abczarr.ome.v0_6rc0.systems.Axis]. A mapping keys each axis by
        its name, in order, and gives a type string, a dict of the remaining
        fields, or `None`. An axis name matches a v3 array's
        `dimension_names`. A missing type is inferred from the name: ``x``,
        ``y``, and ``z`` are space, ``t`` is time, ``c`` is channel, and any
        other name is space.
    scale : number, sequence, or mapping
        The voxel-to-intrinsic diagonal scale (physical size of one voxel).
        One value for all axes, a value per axis, or a mapping keyed by axis
        name or type (name wins). Defaults to 1.
    translation : number, sequence, or mapping
        The voxel-to-intrinsic translation, same shapes as `scale`.
        Defaults to 0.
    intrinsic_name : str
        The name of the intrinsic coordinate system every level maps into.
    model_name : str
        The default name of the model system the `transforms` map onto.
    transforms : sequence or mapping
        The transforms from the intrinsic system to a model system. A
        sequence lists each transform as a numpy affine matrix, which maps
        the intrinsic system to `model_name`, or as a typed OME
        transformation, which carries its own input and output systems. A
        mapping instead names the systems in its key. A string key is a model
        system name, and the transform maps `intrinsic_name` to it. An
        ``(input, output)`` tuple key names both systems. The key sets the
        input and output systems, overriding any that a typed transform of
        its own carries.
    voxel_to_world : array-like or CoordinateTransformation, optional
        A shortcut for images with a single voxel-to-world affine, in
        place of separate `scale`/`translation` and `transforms`. The
        composed voxel-to-world transform equals this matrix exactly.
        When `scale` is not given, it is derived from this matrix.
    name : str, optional
        The multiscale's name.
    ome_version : str
        The version to lower to: ``"stable"`` (the latest released version),
        ``"latest"`` (the newest, including previews), or an explicit
        version such as ``"0.6rc0"``.
    factor : number, sequence, or mapping
        The per-axis downsampling factor between levels, same shapes as
        `scale`. Defaults to 2.
    strategy : {"edge", "center", "window"} or int
        How a level's placement is worked out from the downsampling.
        ``"edge"`` and ``"center"`` need the level shapes; ``"window"``
        (or an int window size) needs only the factors.
    """

    axes: Axes = field()
    scale: tx.Optional[PerAxis] = field(default=None)
    translation: tx.Optional[PerAxis] = field(default=None)
    intrinsic_name: str = field(default="intrinsic")
    model_name: str = field(default="model")
    transforms: Transforms = field(factory=tuple)
    voxel_to_world: tx.Any = field(default=None)
    name: tx.Optional[str] = field(default=None)
    ome_version: str = field(default="stable")
    factor: PerAxis = field(default=2)
    strategy: tx.Union[str, int] = field(default="edge")

    # -- resolution ----------------------------------------------------

    def _axes(self) -> tx.List[Axis]:
        """The config axes as typed 0.6 [Axis][...] objects."""
        axes = self.axes
        if isinstance(axes, _abc.Mapping):
            return [_named_axis(name, spec) for name, spec in axes.items()]
        return [_as_axis(spec) for spec in axes]

    def _axis_info(self) -> tx.List[tx.Tuple[str, str]]:
        """``(name, type)`` for each axis, for keying a per-axis mapping."""
        return [(a.name, _axis_type(a)) for a in self._axes()]

    def resolved_version(self, version: tx.Optional[str] = None) -> str:
        """The concrete OME version this config lowers to.

        Resolves *version* (or `ome_version`, when *version* is `None`):
        ``"stable"`` is the latest released version, ``"latest"`` the
        newest including previews, and anything else is taken as an
        explicit version.

        Parameters
        ----------
        version : str, optional
            An override for `ome_version`.

        Returns
        -------
        str
            The resolved version string.
        """
        requested = self.ome_version if version is None else version
        if requested == "stable":
            return LATEST_STABLE
        if requested == "latest":
            return _VERSIONS[-1]
        if requested not in _VERSIONS:
            raise ValueError(f"Unknown OME version: {requested!r}")
        return requested

    # -- lowering ------------------------------------------------------

    def to_ome(
        self,
        *,
        version: tx.Optional[str] = None,
        policy: ConversionPolicy = "warn",
        level_shapes: tx.Optional[tx.Sequence[tx.Sequence[int]]] = None,
        level_paths: tx.Optional[tx.Sequence[str]] = None,
    ) -> OME:
        """Lower this config to typed OME-Zarr metadata.

        Produces the intrinsic and model coordinate systems, one dataset
        per resolution level mapping the array onto the intrinsic system,
        and the intrinsic-to-model transforms, for the resolved version.

        Parameters
        ----------
        version : str, optional
            An override for `ome_version` (see
            [resolved_version][abczarr.ome.config.ImageConfig.resolved_version]).
        policy : {"lossy", "warn", "strict"}
            How to treat information an older target version cannot hold.
            ``"warn"`` (the default) drops it with one warning;
            ``"strict"`` raises; ``"lossy"`` drops it silently.
        level_shapes : sequence of shape, optional
            The array shape of each resolution level, finest first. Required
            by the ``"edge"`` and ``"center"`` strategies for more than one
            level. When omitted, ``"edge"``/``"center"`` emit level 0 only.
        level_paths : sequence of str, optional
            The array path of each level. Defaults to ``"0"``, ``"1"``, and
            so on.

        Returns
        -------
        OME
            The typed metadata for the resolved version, ready to assign to
            a group's [ome][abczarr.abc.sync.ZarrNode.ome].

        Raises
        ------
        ValueError
            If a per-axis quantity does not match the axes, or a strategy is
            asked for more levels than it can build without shapes.
        """
        axes = self._axes()
        info = [(a.name, _axis_type(a)) for a in axes]
        naxes = len(axes)

        scale = _broadcast(self.scale, info, 1.0, "scale")
        translation = _broadcast(self.translation, info, 0.0, "translation")
        factor = _broadcast(self.factor, info, 2.0, "factor")

        # voxel_to_world can supply the scale, so resolve it against the
        # broadcast values before deciding whether scale was given.
        scale, transforms = self._transforms(
            scale, translation, naxes, scale_given=self.scale is not None
        )

        n_levels, paths = _levels(
            self.strategy, level_shapes, level_paths
        )
        datasets = [
            self._dataset(
                level, paths[level], scale, translation, factor,
                level_shapes, naxes,
            )
            for level in range(n_levels)
        ]

        system_names = [self.intrinsic_name]
        for name in _referenced_system_names(transforms):
            if name not in system_names:
                system_names.append(name)
        systems = [
            CoordinateSystem(name=name, axes=list(axes))
            for name in system_names
        ]

        extra = {}  # type: tx.Dict[str, tx.Any]
        if transforms:
            extra["coordinateTransformations"] = transforms
        if self.name is not None:
            extra["name"] = self.name
        multiscale = Multiscale(
            coordinateSystems=systems, datasets=datasets, **extra
        )
        ome = OMEImage(version="0.6rc0", multiscales=[multiscale])
        return ome.to_version(self.resolved_version(version), policy=policy)

    def apply(
        self, group: tx.Any, **kwargs: tx.Any
    ) -> OME:
        """Set a group's OME metadata from this config.

        Lowers the config with
        [to_ome][abczarr.ome.config.ImageConfig.to_ome] and assigns the
        result to *group*'s [ome][abczarr.abc.sync.ZarrNode.ome]. Keyword
        arguments are passed through to `to_ome`.

        Parameters
        ----------
        group : ZarrNode
            The group to write the metadata to.
        **kwargs
            Passed through to
            [to_ome][abczarr.ome.config.ImageConfig.to_ome].

        Returns
        -------
        OME
            The metadata that was written.
        """
        ome = self.to_ome(**kwargs)
        group.ome = ome
        return ome

    # -- internals -----------------------------------------------------

    def _transforms(
        self,
        scale: tx.Tuple[float, ...],
        translation: tx.Tuple[float, ...],
        naxes: int,
        scale_given: bool,
    ) -> tx.Tuple[tx.Tuple[float, ...], tx.List[CoordinateTransformation]]:
        """Resolve `transforms` and `voxel_to_world` to typed transforms.

        Returns the possibly-derived voxel scale and the list of transforms
        that map the intrinsic system onto a model system.
        """
        result = []  # type: tx.List[CoordinateTransformation]
        if self.voxel_to_world is not None:
            world = np.asarray(self.voxel_to_world, dtype=float)
            hom = _homogeneous(world, naxes)
            if not scale_given:
                scale = _column_norms(hom, naxes)
            v_inv = np.linalg.inv(_diag_affine(scale, translation, naxes))
            result.append(
                Affine(
                    type="affine",
                    affine=_ome_affine(hom @ v_inv, naxes),
                    input=Space(name=self.intrinsic_name),
                    output=Space(name=self.model_name),
                )
            )

        for input_name, output_name, data in self._transform_entries():
            result.append(
                self._one_transform(
                    data, input_name, output_name, scale, translation, naxes
                )
            )
        return scale, result

    def _transform_entries(
        self,
    ) -> tx.Iterator[tx.Tuple[tx.Optional[str], tx.Optional[str], tx.Any]]:
        """Each `transforms` entry as an ``(input, output, data)`` triple.

        A sequence yields ``(None, None, data)`` for every entry. A name is
        then taken from the transform's own references, and falls back to the
        intrinsic and model systems. A mapping yields the names its key
        supplies. A string key names the output system and takes the input
        from `intrinsic_name`. An ``(input, output)`` tuple key names both.
        A name a key supplies is used even when the transform carries one of
        its own.
        """
        transforms = self.transforms
        if isinstance(transforms, _abc.Mapping):
            for key, data in transforms.items():
                if isinstance(key, str):
                    yield self.intrinsic_name, key, data
                elif isinstance(key, tuple) and len(key) == 2:
                    yield key[0], key[1], data
                else:
                    raise ValueError(
                        f"a transforms key must be a model name or an "
                        f"(input, output) pair of names; got {key!r}"
                    )
        else:
            for data in transforms:
                yield None, None, data

    def _one_transform(
        self,
        data: tx.Any,
        input_name: tx.Optional[str],
        output_name: tx.Optional[str],
        scale: tx.Tuple[float, ...],
        translation: tx.Tuple[float, ...],
        naxes: int,
    ) -> CoordinateTransformation:
        """Turn one transform entry into a typed transform between systems."""
        if isinstance(data, CoordinateTransformation):
            return self._typed_transform(
                data, input_name, output_name, scale, translation, naxes
            )
        # A numpy affine matrix is read in intrinsic coordinates. The input is
        # the intrinsic system unless a mapping key names another.
        source = input_name if input_name is not None else self.intrinsic_name
        output = output_name if output_name is not None else self.model_name
        hom = _homogeneous(np.asarray(data, dtype=float), naxes)
        return Affine(
            type="affine",
            affine=_ome_affine(hom, naxes),
            input=Space(name=source),
            output=Space(name=output),
        )

    def _typed_transform(
        self,
        data: CoordinateTransformation,
        input_name: tx.Optional[str],
        output_name: tx.Optional[str],
        scale: tx.Tuple[float, ...],
        translation: tx.Tuple[float, ...],
        naxes: int,
    ) -> CoordinateTransformation:
        """Resolve a ready-made 0.6 transform's input and output systems.

        A name a mapping key supplies wins. Otherwise the transform's own
        reference is kept, falling back to the intrinsic and model systems.
        A transform whose own input is an array reference is read in voxel
        coordinates and composed with the voxel-to-intrinsic transform so
        that the result still starts from the intrinsic system.
        """
        own_input = getattr(data, "input", None)
        output = output_name
        if output is None:
            output = _space_name(getattr(data, "output", None))
        if output is None:
            output = self.model_name

        if input_name is None and _is_path_space(own_input):
            return self._from_voxel(data, output, scale, translation, naxes)

        source = input_name
        if source is None:
            source = _space_name(own_input)
        if source is None:
            source = self.intrinsic_name
        return _with_refs(data, Space(name=source), Space(name=output))

    def _from_voxel(
        self,
        data: CoordinateTransformation,
        output: str,
        scale: tx.Tuple[float, ...],
        translation: tx.Tuple[float, ...],
        naxes: int,
    ) -> CoordinateTransformation:
        """Re-express a voxel-input transform as one from the intrinsic system.

        The intrinsic-to-voxel transform is the inverse of the level-0
        voxel-to-intrinsic scale and translation. An inline affine is composed
        with it directly.
        """
        inline = getattr(data, "affine", None)
        if not (isinstance(data, Affine) and _is_matrix(inline)):
            raise ValueError(
                "a transform written in voxel coordinates (with an array-path "
                "input) must be an inline affine; express other transforms in "
                "intrinsic coordinates instead"
            )
        v_inv = np.linalg.inv(_diag_affine(scale, translation, naxes))
        hom = _homogeneous(np.asarray(inline, dtype=float), naxes)
        return Affine(
            type="affine",
            affine=_ome_affine(hom @ v_inv, naxes),
            input=Space(name=self.intrinsic_name),
            output=Space(name=output),
        )

    def _dataset(
        self,
        level: int,
        path: str,
        scale: tx.Tuple[float, ...],
        translation: tx.Tuple[float, ...],
        factor: tx.Tuple[float, ...],
        level_shapes: tx.Optional[tx.Sequence[tx.Sequence[int]]],
        naxes: int,
    ) -> Dataset:
        """The voxel-to-intrinsic transform for one resolution level."""
        s, t = _level_transform(
            level, scale, translation, factor, self.strategy,
            level_shapes, naxes,
        )
        refs = {
            "input": Space(path=path),
            "output": Space(name=self.intrinsic_name),
        }
        if any(offset != 0.0 for offset in t):
            transform = Sequence(
                type="sequence",
                transformations=[
                    Scale(type="scale", scale=list(s)),
                    Translation(type="translation", translation=list(t)),
                ],
                **refs,
            )  # type: CoordinateTransformation
        else:
            transform = Scale(type="scale", scale=list(s), **refs)
        return Dataset(path=path, coordinateTransformations=[transform])


# ----------------------------------------------------------------------
#   axis helpers
# ----------------------------------------------------------------------


def _as_axis(spec: AxisSpec) -> Axis:
    """One axis spec (name, tuple, dict, or typed axis) as a typed 0.6 axis."""
    if isinstance(spec, Axis):
        return spec
    if isinstance(spec, str):
        return axis(spec)
    if isinstance(spec, _abc.Mapping):
        if "name" not in spec:
            raise ValueError(
                f"an axis dict needs a 'name'; got {dict(spec)!r}"
            )
        return axis(spec["name"], spec.get("type"), spec.get("unit"))
    if isinstance(spec, _abc.Sequence):
        parts = list(spec)
        if not parts:
            raise ValueError("an axis spec tuple needs at least a name")
        name = parts[0]
        atype = parts[1] if len(parts) > 1 else None
        unit = parts[2] if len(parts) > 2 else None
        return axis(name, atype, unit)
    raise TypeError(
        f"an axis must be a name, a (name, type[, unit]) tuple, a dict, or an "
        f"Axis; got {spec!r}"
    )


def _named_axis(name: str, spec: tx.Any) -> Axis:
    """One axis from a ``name -> spec`` entry of the mapping form of `axes`.

    The value is a type string, a dict of the axis's remaining fields, or
    `None` to infer the type from the name.
    """
    if spec is None:
        return axis(name)
    if isinstance(spec, str):
        return axis(name, spec)
    if isinstance(spec, _abc.Mapping):
        return axis(name, spec.get("type"), spec.get("unit"))
    raise TypeError(
        f"an axis mapping value must be a type string, a dict, or None; got "
        f"{spec!r} for axis {name!r}"
    )


def _axis_type(a: Axis) -> str:
    """An axis's type, falling back to its inferred type by name."""
    atype = getattr(a, "type", None)
    if isinstance(atype, str):
        return atype
    return _AXIS_TYPE.get(a.name, "space")


# ----------------------------------------------------------------------
#   per-axis broadcasting
# ----------------------------------------------------------------------


def _broadcast(
    value: tx.Any,
    info: tx.Sequence[tx.Tuple[str, str]],
    default: float,
    what: str,
) -> tx.Tuple[float, ...]:
    """One value per axis, from a scalar, a sequence, or a name/type mapping.

    A mapping is looked up by axis name first, then by axis type, and falls
    back to *default*. A sequence must have one entry per axis. `None` is
    the default for every axis.
    """
    if value is None:
        return tuple(default for _ in info)
    if isinstance(value, _abc.Mapping):
        out = []
        for name, atype in info:
            if name in value:
                out.append(float(value[name]))
            elif atype in value:
                out.append(float(value[atype]))
            else:
                out.append(default)
        return tuple(out)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return tuple(float(value) for _ in info)
    seq = list(value)
    if len(seq) != len(info):
        raise ValueError(
            f"{what} has {len(seq)} values but there are {len(info)} axes"
        )
    return tuple(float(x) for x in seq)


# ----------------------------------------------------------------------
#   matrix helpers
# ----------------------------------------------------------------------


def _homogeneous(matrix: "np.ndarray", n: int) -> "np.ndarray":
    """An affine matrix in homogeneous ``(n+1, n+1)`` form.

    Accepts the three spellings a user may write: full homogeneous
    ``(n+1, n+1)``, linear-plus-translation ``(n, n+1)``, or a bare linear
    ``(n, n)`` block (no translation).
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2:
        raise ValueError("an affine transform must be a 2D matrix")
    rows, cols = a.shape
    hom = np.eye(n + 1)
    if (rows, cols) == (n + 1, n + 1):
        return a.copy()
    if (rows, cols) == (n, n + 1):
        hom[:n, :] = a
    elif (rows, cols) == (n, n):
        hom[:n, :n] = a
    else:
        raise ValueError(
            f"an affine for {n} axes must be {n + 1}x{n + 1}, {n}x{n + 1}, "
            f"or {n}x{n}; got {rows}x{cols}"
        )
    return hom


def _diag_affine(
    scale: tx.Sequence[float], translation: tx.Sequence[float], n: int
) -> "np.ndarray":
    """The homogeneous voxel-to-intrinsic affine ``diag(scale)`` + offset."""
    hom = np.eye(n + 1)
    for i in range(n):
        hom[i, i] = scale[i]
        hom[i, n] = translation[i]
    return hom


def _column_norms(hom: "np.ndarray", n: int) -> tx.Tuple[float, ...]:
    """Per-axis voxel size: the column norms of the linear block."""
    linear = hom[:n, :n]
    return tuple(
        float(np.linalg.norm(linear[:, j])) for j in range(n)
    )


def _ome_affine(hom: "np.ndarray", n: int) -> tx.List[tx.List[float]]:
    """The RFC-5 ``(n, n+1)`` affine list from a homogeneous matrix."""
    return [[float(x) for x in row] for row in hom[:n, :]]


def _with_refs(
    transform: CoordinateTransformation,
    input_space: Space,
    output_space: Space,
) -> CoordinateTransformation:
    """A copy of *transform* with its input and output systems set."""
    return evolve(transform, input=input_space, output=output_space)


def _is_matrix(value: tx.Any) -> bool:
    """Whether *value* is a 2D matrix an affine can be read from."""
    return isinstance(value, (list, tuple, np.ndarray))


def _space_name(value: tx.Any) -> tx.Optional[str]:
    """The system name a [Space][...] refers to, or `None`.

    A [Space][...] that references an array by path names no system, so the
    result is `None`.
    """
    if isinstance(value, Space):
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name
    return None


def _is_path_space(value: tx.Any) -> bool:
    """Whether *value* references an array by path.

    A path reference names the array's own voxel coordinate system rather than
    a named coordinate system.
    """
    return (
        isinstance(value, Space)
        and isinstance(getattr(value, "path", None), str)
        and _space_name(value) is None
    )


def _referenced_system_names(
    transforms: tx.Sequence[CoordinateTransformation],
) -> tx.List[str]:
    """The distinct named systems the transforms reference, in first-seen
    order."""
    names = []  # type: tx.List[str]
    for transform in transforms:
        for ref in (
            getattr(transform, "input", None),
            getattr(transform, "output", None),
        ):
            name = _space_name(ref)
            if name is not None and name not in names:
                names.append(name)
    return names


# ----------------------------------------------------------------------
#   per-level transform math (after nifti-zarr-py's ``_factor``)
# ----------------------------------------------------------------------


def _levels(
    strategy: tx.Union[str, int],
    level_shapes: tx.Optional[tx.Sequence[tx.Sequence[int]]],
    level_paths: tx.Optional[tx.Sequence[str]],
) -> tx.Tuple[int, tx.List[str]]:
    """How many levels to emit, and each level's path."""
    windowed = isinstance(strategy, int) or strategy == "window"
    if level_shapes is not None:
        n = len(level_shapes)
        if level_paths is not None and len(level_paths) != n:
            raise ValueError(
                "level_paths and level_shapes have different lengths"
            )
    elif level_paths is not None:
        n = len(level_paths)
        if not windowed and n != 1:
            raise ValueError(
                f"the {strategy!r} strategy needs level_shapes to build more "
                f"than one level"
            )
    else:
        n = 1
    paths = list(level_paths) if level_paths is not None else [
        str(i) for i in range(n)
    ]
    return n, paths


def _level_transform(
    level: int,
    scale: tx.Tuple[float, ...],
    translation: tx.Tuple[float, ...],
    factor: tx.Tuple[float, ...],
    strategy: tx.Union[str, int],
    level_shapes: tx.Optional[tx.Sequence[tx.Sequence[int]]],
    naxes: int,
) -> tx.Tuple[tx.Tuple[float, ...], tx.Tuple[float, ...]]:
    """The voxel-to-intrinsic scale and translation for one level.

    Follows the strategy formulas per axis, from base voxel size ``s0``,
    base translation ``t0`` and cumulative factor ``F``. An axis not
    downsampled (factor 1) keeps ``s0``/``t0``.
    """
    windowed = isinstance(strategy, int) or strategy == "window"
    out_s = []  # type: tx.List[float]
    out_t = []  # type: tx.List[float]
    for a in range(naxes):
        s0, t0 = scale[a], translation[a]
        step = float(strategy) if isinstance(strategy, int) else factor[a]
        if level == 0 or step == 1:
            out_s.append(s0)
            out_t.append(t0)
            continue
        if windowed:
            f = step ** level
            out_s.append(s0 * f)
            out_t.append(t0 + s0 * (f - 1) / 2)
        elif strategy in ("edge", "center"):
            n0 = float(level_shapes[0][a])
            nl = float(level_shapes[level][a])
            if strategy == "edge":
                ratio = n0 / nl
                out_s.append(s0 * ratio)
                out_t.append(t0 + s0 * (ratio - 1) / 2)
            else:  # center
                out_s.append(s0 * (n0 - 1) / (nl - 1) if nl > 1 else s0)
                out_t.append(t0)
        else:
            raise ValueError(f"unknown downsampling strategy: {strategy!r}")
    return tuple(out_s), tuple(out_t)
