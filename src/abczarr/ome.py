import typing_extensions as tx

from . import typing as tz
from .abc import ZarrGroup


def write_ome_metadata(
    omz: ZarrGroup,
    axes: tx.Sequence[tz.AxisName],
    space_scale: tz.OneOrMore[float] = 1.0,
    time_scale: float = 1.0,
    space_unit: str = "micrometer",
    time_unit: str = "second",
    name: str = "",
    pyramid_aligns: tx.OneOrMore[tx.Union[str, int]] = 2,
    levels: tx.Optional[int] = None,
    no_pool: tx.Optional[int] = None,
    multiscales_type: str = "",
    ome_version: tz.OMEVersion = "0.4"
) -> None:
    """
    Write OME metadata into Zarr.

    Parameters
    ----------
    omz : ZarrGroup
        Zarr group to write metadata
    axes : list[str]
        Name of each dimension, in Zarr order (t, c, z, y, x)
    space_scale : float | list[float]
        Finest-level voxel size, in Zarr order (z, y, x)
    time_scale : float
        Time scale
    space_unit : str
        Unit of spatial scale (assumed identical across dimensions)
    time_unit : str
        Unit of time scale
    name : str
        Name attribute
    pyramid_aligns : float | list[float] | {"center", "edge"}
        Whether the pyramid construction aligns the edges or the centers
        of the corner voxels. If a (list of) number, assume that a moving
        window of that size was used.
    levels : int
        Number of existing levels. Default: find out automatically.

    """
    # 1) Pull out all pyramid-level shapes
    shapes = []
    lvl = 0
    while True:
        if levels is not None and lvl > levels:
            break
        key = str(lvl)
        if key not in omz:
            break
        shapes.append(omz[key].shape)
        lvl += 1

    if not shapes:
        return  # nothing to do

    # 2) Map axes → types, count spatial vs. others
    axis_to_type = dict(x="space", y="space", z="space", t="time", c="channel")
    types = [axis_to_type[a] for a in axes]
    ndim = len(axes)
    sdim = types.count("space")
    bdim = ndim - sdim

    # 3) Normalize space_scale and pyramid_aligns to length==sdim
    def _normalize(
        val: tz.OneOrMore[float], length: int
    ) -> tx.Sequence[float]:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(val) < length:
            val = [val[0]] * (length - len(val)) + list(val)
        return val[-length:]
    space_scale = _normalize(space_scale, sdim)
    aligns = _normalize(pyramid_aligns, sdim)

    # 4) Precompute base shape
    shape0 = shapes[0]

    # 5) Build the multiscales dict
    space_unit = space_unit or "millimeter"
    time_unit = time_unit or "second"
    ms: dict = {
        "version": ome_version,
        "name": name,
        "type": multiscales_type or f"median window {'x'.join(['2']*sdim)}",
        "axes": [
            dict(
                name=a,
                type=t,
                **(
                    {"unit": space_unit} if t == "space" else
                    {"unit": time_unit} if t == "time" else
                    {}
                )
            )
            for a, t in zip(axes, types)
        ],
        "datasets": [],
    }

    # Helper to compute per-dimension scale/translation
    def _factor(
        a0: int, aN: int, align: tx.Union[int, str], n: int,
        scale: float, is_pool: bool
    ) -> tx.Tuple[float, float]:
        if is_pool:
            # no pooling along this axis
            return scale, 0.0
        if isinstance(align, str) and align.lower().startswith("e"):
            factor = (a0 / aN)
            trans = (factor - 1) * 0.5
        elif isinstance(align, str) and align.lower().startswith("c"):
            factor = ((a0 - 1) / (aN - 1))
            trans = 0.0
        else:
            # numeric align: repeated power
            factor = (align ** n)
            trans = (factor - 1) * 0.5
        return factor * scale, trans * scale

    prev_scale_axes = [None] * sdim
    prev_trans_axes = [None] * sdim
    # 7) Populate each pyramid level
    for n, shape in enumerate(shapes):
        # compute scale+translation arrays of length ndim
        scale = [1.0]*bdim + []
        translation = [0.0]*bdim + []
        for i in range(sdim):
            a0 = shape0[bdim+i]
            aN = shape[bdim+i]
            is_pool = (i == no_pool)
            if n > 0 and shapes[n-1][bdim + i] == aN:
                # no change from last level → re‐use
                s, tr = prev_scale_axes[i], prev_trans_axes[i]
            else:
                s, tr = _factor(a0, aN, aligns[i], n, space_scale[i], is_pool)
            scale.append(s)
            translation.append(tr)
            prev_scale_axes[i] = s
            prev_trans_axes[i] = tr

        ms["datasets"].append({
            "path": str(n),
            "coordinateTransformations": [
                {"type":"scale",       "scale": scale},
                {"type":"translation", "translation": translation},
            ]
        })

    # 8) Add global time‐scale transformation
    tscale = [time_scale if t=="time" else 1.0 for t in types]
    ms["coordinateTransformations"] = [{"type":"scale", "scale": tscale}]

    # 9) Write into Zarr attributes
    omz.attrs["multiscales"] = [ms]
    if ome_version == "0.5":
        omz.attrs["ome"] = {"version": ome_version, "multiscales": [ms]}
    elif ome_version not in {"0.4","0.5"}:
        raise ValueError(f"Unsupported ome_version {ome_version}")
