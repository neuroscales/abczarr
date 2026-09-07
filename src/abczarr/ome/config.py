"""Describe an OME-Zarr multiscale image to write.

An [ImageConfig][abczarr.ome.config.ImageConfig] is to OME-Zarr metadata
what an [ArrayConfig][abczarr.api.config.ArrayConfig] is to a Zarr array:
high-level, human-facing attributes that lower to the exact typed metadata
a group carries. You give it the axes, the voxel size, and how the pyramid
was built; it resolves those to OME-NGFF multiscales metadata for any
version you ask for.

Three coordinate spaces line up behind it::

    voxel (array index) --scale+translation--> intrinsic --transforms--> model

`scale` and `translation` place each resolution level's array indices into a
single *intrinsic* space shared by every level. `transforms` (and the
convenience `voxel_to_world`) map that intrinsic space onto a *model* -- a
world or anatomical frame. The rich 0.6 (RFC-5) model is built internally,
then converted to the version you request, so one config lowers correctly to
the stable 0.5 shape or the 0.6 preview alike.
"""

__all__ = ["ImageConfig", "Transform", "axis"]

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

#: An axis, as a user may spell it: a bare name, a ``(name, type)`` or
#: ``(name, type, unit)`` tuple, or a ready-made 0.6 [Axis][...].
AxisSpec = tx.Union[str, tx.Sequence[tx.Any], Axis]

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
    and ``z`` are space, ``t`` is time, ``c`` is channel, and anything else
    is space. Constructing with a known type gives back the matching axis
    subclass (a [SpaceAxis][abczarr.ome.v0_6rc0.systems.SpaceAxis], and so
    on).

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
class Transform:
    """One intrinsic-to-model transform for an
    [ImageConfig][abczarr.ome.config.ImageConfig].

    Wrap a numpy affine matrix or a typed OME transformation to say which
    coordinate systems it maps between. The default maps the intrinsic
    system onto the config's model system.

    Parameters
    ----------
    data : array-like or CoordinateTransformation
        A 2D affine matrix (turned into an
        [Affine][abczarr.ome.v0_6rc0.transformations.Affine]) or a
        ready-made typed transformation.
    input : {"intrinsic", "voxel"}
        The system the transform starts from. ``"voxel"`` is reformulated
        to start from the intrinsic system by composing with the inverse of
        the voxel-to-intrinsic transform.
    output : str, optional
        The name of the model system the transform maps onto. `None` uses
        the config's `model_name`.
    """

    data: tx.Any = field()
    input: str = field(default="intrinsic")  # noqa: A003
    output: tx.Optional[str] = field(default=None)


@define
class ImageConfig:
    """A high-level description of an OME-Zarr multiscale image.

    Set the axes and the voxel geometry once; lower to typed OME metadata
    for any version with [to_ome][abczarr.ome.config.ImageConfig.to_ome].
    The rich 0.6 (RFC-5) model -- named coordinate systems and general
    transforms -- is built internally and converted down to the requested
    version, so a single config produces the stable 0.5 shape and the 0.6
    preview alike.

    Parameters
    ----------
    axes : sequence of axis specs
        One entry per array dimension: a name (``"x"``), a
        ``(name, type)`` or ``(name, type, unit)`` tuple, or a typed
        [Axis][abczarr.ome.v0_6rc0.systems.Axis]. Names are the OME axis
        names and line up with a v3 array's `dimension_names`. A missing
        type is inferred from the name (``x``/``y``/``z`` space, ``t``
        time, ``c`` channel, else space).
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
    transforms : sequence
        Intrinsic-to-model transforms. Each entry is a numpy affine matrix,
        a typed OME transformation, or a
        [Transform][abczarr.ome.config.Transform] wrapper.
    voxel_to_world : array-like or CoordinateTransformation, optional
        A convenience for the common single voxel-to-world affine. It is
        decomposed into the voxel-to-intrinsic scale/translation and an
        intrinsic-to-world transform so the composed voxel-to-world equals
        this matrix exactly. When `scale` is not given it is derived from
        this matrix's linear block.
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
        How a level's transform follows from downsampling. ``"edge"`` and
        ``"center"`` need the level shapes; ``"window"`` (or an int window
        size) follows from the factors alone.
    """

    axes: tx.Sequence[AxisSpec] = field()
    scale: tx.Optional[PerAxis] = field(default=None)
    translation: tx.Optional[PerAxis] = field(default=None)
    intrinsic_name: str = field(default="intrinsic")
    model_name: str = field(default="model")
    transforms: tx.Sequence[tx.Any] = field(factory=tuple)
    voxel_to_world: tx.Any = field(default=None)
    name: tx.Optional[str] = field(default=None)
    ome_version: str = field(default="stable")
    factor: PerAxis = field(default=2)
    strategy: tx.Union[str, int] = field(default="edge")

    # -- resolution ----------------------------------------------------

    def _axes(self) -> tx.List[Axis]:
        """The config axes as typed 0.6 [Axis][...] objects."""
        return [_as_axis(spec) for spec in self.axes]

    def _axis_info(self) -> tx.List[tx.Tuple[str, str]]:
        """``(name, type)`` for each axis, for keying a per-axis mapping."""
        return [(a.name, _axis_type(a)) for a in self._axes()]

    def resolved_version(self, version: tx.Optional[str] = None) -> str:
        """The concrete OME version this config lowers to.

        Resolves *version* (or, when `None`, `ome_version`): ``"stable"``
        becomes the latest released version, ``"latest"`` the newest
        including previews, and anything else is taken as an explicit
        version.

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

        Builds the 0.6 (RFC-5) model -- an intrinsic coordinate system, the
        model system(s), one dataset per resolution level mapping the array
        onto the intrinsic system, and the intrinsic-to-model transforms --
        then converts it to the resolved version under *policy*.

        Parameters
        ----------
        version : str, optional
            An override for `ome_version` (see
            [resolved_version][abczarr.ome.config.ImageConfig.resolved_version]).
        policy : {"lossy", "warn", "strict"}
            How to treat information the target version cannot hold, when it
            is older than 0.6. ``"warn"`` (the default) drops it with one
            warning; ``"strict"`` raises; ``"lossy"`` drops it silently.
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

        systems = [CoordinateSystem(name=self.intrinsic_name, axes=axes)]
        for out in _output_names(transforms, self.model_name):
            systems.append(CoordinateSystem(name=out, axes=list(axes)))

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

        A thin wrapper over
        [to_ome][abczarr.ome.config.ImageConfig.to_ome]: it lowers the
        config and assigns the result to *group*'s
        [ome][abczarr.abc.sync.ZarrNode.ome]. Keyword arguments are passed
        through to `to_ome`.

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

        Returns the (possibly derived) voxel scale and the list of
        intrinsic-to-model transforms.
        """
        entries = list(self.transforms)
        if self.voxel_to_world is not None:
            world = np.asarray(self.voxel_to_world, dtype=float)
            hom = _homogeneous(world, naxes)
            if not scale_given:
                scale = _column_norms(hom, naxes)
            v_inv = np.linalg.inv(_diag_affine(scale, translation, naxes))
            transform = Affine(
                type="affine",
                affine=_ome_affine(hom @ v_inv, naxes),
                input=Space(name=self.intrinsic_name),
                output=Space(name=self.model_name),
            )
            entries.append(transform)

        result = []
        for entry in entries:
            result.append(
                self._one_transform(entry, scale, translation, naxes)
            )
        return scale, result

    def _one_transform(
        self,
        entry: tx.Any,
        scale: tx.Tuple[float, ...],
        translation: tx.Tuple[float, ...],
        naxes: int,
    ) -> CoordinateTransformation:
        """Turn one `transforms` entry into a typed intrinsic-to-model
        transform."""
        if isinstance(entry, Transform):
            data, source, out = entry.data, entry.input, entry.output
        else:
            data, source, out = entry, "intrinsic", None
        out = self.model_name if out is None else out

        # a ready-made typed transform is used as is, its references filled
        # in only where it left them unset.
        if isinstance(data, CoordinateTransformation):
            return _refer(data, self.intrinsic_name, out)

        # a numpy affine matrix
        hom = _homogeneous(np.asarray(data, dtype=float), naxes)
        if source == "voxel":
            v_inv = np.linalg.inv(_diag_affine(scale, translation, naxes))
            hom = hom @ v_inv
        elif source != "intrinsic":
            raise ValueError(
                f"a transform's input must be 'intrinsic' or 'voxel', "
                f"not {source!r}"
            )
        return Affine(
            type="affine",
            affine=_ome_affine(hom, naxes),
            input=Space(name=self.intrinsic_name),
            output=Space(name=out),
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
    """One axis spec (name / tuple / typed axis) as a typed 0.6 axis."""
    if isinstance(spec, Axis):
        return spec
    if isinstance(spec, str):
        return axis(spec)
    if isinstance(spec, _abc.Sequence):
        parts = list(spec)
        if not parts:
            raise ValueError("an axis spec tuple needs at least a name")
        name = parts[0]
        atype = parts[1] if len(parts) > 1 else None
        unit = parts[2] if len(parts) > 2 else None
        return axis(name, atype, unit)
    raise TypeError(
        f"an axis must be a name, a (name, type[, unit]) tuple, or an Axis; "
        f"got {spec!r}"
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


def _refer(
    transform: CoordinateTransformation, intrinsic: str, output: str
) -> CoordinateTransformation:
    """A typed transform with its input/output filled in when unset."""
    changes = {}
    if not _is_space(getattr(transform, "input", None)):
        changes["input"] = Space(name=intrinsic)
    if not _is_space(getattr(transform, "output", None)):
        changes["output"] = Space(name=output)
    return evolve(transform, **changes) if changes else transform


def _is_space(value: tx.Any) -> bool:
    return isinstance(value, Space)


def _output_names(
    transforms: tx.Sequence[CoordinateTransformation], model_name: str
) -> tx.List[str]:
    """The distinct model-system names the transforms map onto, in order."""
    seen = []  # type: tx.List[str]
    for transform in transforms:
        out = getattr(transform, "output", None)
        name = out.name if _is_space(out) else None
        if not isinstance(name, str):
            name = model_name
        if name not in seen:
            seen.append(name)
    return seen


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
