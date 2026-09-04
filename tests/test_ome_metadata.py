"""The OME-Zarr structured metadata layer builds and round-trips.

These pin that an OME metadata object can be constructed from a dict and
serialized back to an equal object -- exercising the RFC-2119 requirement
factories (which mark fields required / recommended / optional) and the
MISSING sentinel a required-but-unset field carries.
"""

import importlib

import pytest

from abczarr.ome.metadata import v0_4, v0_5

_MULTISCALE_V04 = {
    "version": "0.4",
    "name": "example",
    "axes": [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ],
    "datasets": [
        {
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0]}
            ],
        }
    ],
}

_MULTISCALE_V05 = {
    "name": "example",
    "axes": _MULTISCALE_V04["axes"],
    "datasets": _MULTISCALE_V04["datasets"],
}


@pytest.mark.parametrize(
    ("cls", "data"),
    [
        (v0_4.Multiscale, _MULTISCALE_V04),
        (v0_5.Multiscale, _MULTISCALE_V05),
    ],
)
def test_ome_multiscale_builds(cls: type, data: dict) -> None:
    m = cls.from_json(data)
    assert [a.name for a in m.axes] == ["y", "x"]
    assert m.datasets[0].path == "0"


@pytest.mark.parametrize(
    ("cls", "data"),
    [
        (v0_4.Multiscale, _MULTISCALE_V04),
        (v0_5.Multiscale, _MULTISCALE_V05),
    ],
)
def test_ome_multiscale_roundtrips_through_dict(
    cls: type, data: dict
) -> None:
    m = cls.from_json(data)
    assert cls.from_json(m.to_json()) == m


# --------------------------------------------------------------------------
# cross-version conversion between same-structure OME versions (v0.4 <-> v0.5)
# --------------------------------------------------------------------------


def test_multiscale_converts_v04_to_v05_and_back() -> None:
    m4 = v0_4.Multiscale.from_json(_MULTISCALE_V04)
    m5 = m4.to_version("0.5")
    assert type(m5).__module__.endswith("v0_5.images")
    assert [a.name for a in m5.axes] == ["y", "x"]
    assert m5.to_version("0.4") == m4


def test_omero_converts_v04_to_v05_and_back() -> None:
    data = {
        "channels": [
            {
                "color": "FF0000",
                "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                "active": True,
                "label": "c1",
            }
        ]
    }
    o4 = v0_4.Omero.from_json(data)
    assert o4.to_version("0.5").to_version("0.4") == o4


def test_conversion_to_absent_version_raises() -> None:
    m4 = v0_4.Multiscale.from_json(_MULTISCALE_V04)
    with pytest.raises(ValueError, match="Unknown OME version"):
        m4.to_version("9.9")


# --------------------------------------------------------------------------
# structural conversion: v0.3 <-> v0.4 (typed axes) and chaining
# --------------------------------------------------------------------------

_MULTISCALE_V03 = {
    "version": "0.3",
    "name": "example",
    "type": "gaussian",
    "axes": ["t", "c", "z", "y", "x"],
    "datasets": [{"path": "0"}, {"path": "1"}],
}


def test_multiscale_v03_to_v04_types_the_axes() -> None:
    from abczarr.ome.metadata import v0_3

    m3 = v0_3.Multiscale.from_json(_MULTISCALE_V03)
    m4 = m3.to_version("0.4")
    assert [(a.name, a.type) for a in m4.axes] == [
        ("t", "time"),
        ("c", "channel"),
        ("z", "space"),
        ("y", "space"),
        ("x", "space"),
    ]
    # v0.4 datasets require a coordinate transform; an identity scale is added
    assert m4.datasets[0].coordinateTransformations[0].scale == [1.0] * 5
    assert m4.to_version("0.3") == m3


@pytest.mark.parametrize("target", ["0.4", "0.5"])
def test_multiscale_v03_roundtrips_up_and_back(target: str) -> None:
    from abczarr.ome.metadata import v0_3

    m3 = v0_3.Multiscale.from_json(_MULTISCALE_V03)
    # chains v0.3 -> v0.4 (-> v0.5) and back
    assert m3.to_version(target).to_version("0.3") == m3


def test_down_conversion_drops_axes() -> None:
    from abczarr.ome.metadata import v0_3

    m3 = v0_3.Multiscale.from_json(_MULTISCALE_V03)
    m2 = m3.to_version("0.2")
    assert "axes" not in m2.to_json()


def test_underdetermined_up_conversion_raises_clearly() -> None:
    from abczarr.ome.metadata import v0_2

    m2 = v0_2.Multiscale.from_json(
        {
            "version": "0.2",
            "name": "x",
            "type": "g",
            "datasets": [{"path": "0"}],
        }
    )
    # v0.3 requires axes, which v0.2 does not carry
    with pytest.raises(ValueError, match="does not carry"):
        m2.to_version("0.3")


# --------------------------------------------------------------------------
# base.OME.from_json dispatches by the data's declared version, not by which
# version package happened to import last
# --------------------------------------------------------------------------

_PLATE = {
    "columns": [{"name": "1"}],
    "rows": [{"name": "A"}],
    "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
}


@pytest.mark.parametrize(
    ("version", "package"),
    [
        ("0.1", "v0_1"),
        ("0.2", "v0_2"),
        ("0.3", "v0_3"),
        ("0.4", "v0_4"),
        ("0.5", "v0_5"),
    ],
)
def test_base_ome_from_dict_routes_by_version(
    version: str, package: str
) -> None:
    # Every version package registers its OME subclasses into the shared
    # base.OME registry under the same non-version-specific keys, so the base
    # cannot tell them apart on its own. It used to dispatch to whichever
    # package imported last (v0_6dev4), silently and independent of the data's
    # version. It must now route by the declared version instead.
    from abczarr.ome.metadata import base

    data = {"version": version, "bioformats2raw_layout": 3, "plate": _PLATE}
    obj = base.OME.from_json(data)
    assert type(obj).__module__ == f"abczarr.ome.metadata.{package}.ome"


def test_base_ome_from_dict_rejects_versionless_metadata() -> None:
    # Under the old last-import-wins behavior this silently built an object
    # from whichever package imported last; ambiguous data must now raise.
    from abczarr.ome.metadata import base

    with pytest.raises(ValueError, match="version"):
        base.OME.from_json({"bioformats2raw_layout": 3, "plate": _PLATE})


def test_base_ome_from_dict_rejects_unknown_version() -> None:
    from abczarr.ome.metadata import base

    with pytest.raises(ValueError, match="Unknown OME version"):
        base.OME.from_json({"version": "9.9"})


def test_per_version_ome_from_dict_still_dispatches() -> None:
    # The per-version OME classes keep their own dispatch: routing through the
    # base must reach the same class a direct call would.
    from abczarr.ome.metadata import base, v0_5

    data = {"version": "0.5", "bioformats2raw_layout": 3, "plate": _PLATE}
    assert type(base.OME.from_json(data)) is type(v0_5.OME.from_json(data))


# --------------------------------------------------------------------------
# base.OME.from_json picks the carrier whose discriminator key is present in
# the document -- a plate document is an OMEPlate, a multiscales one an
# OMEImage, a bioformats2raw.layout one an OMEBioformats2Raw -- and a plain
# plate (no bioformats2raw.layout) is never mistaken for the bf2raw carrier.
# --------------------------------------------------------------------------

_WELL = {"images": [{"acquisition": 0, "path": "0"}]}


@pytest.mark.parametrize("version", ["0.1", "0.2", "0.3", "0.4", "0.5"])
def test_base_ome_dispatches_by_present_discriminator(version: str) -> None:
    from abczarr.ome.metadata import base

    ome = importlib.import_module(
        f"abczarr.ome.metadata.v0_{version.split('.')[1]}.ome"
    )

    def carrier(fields: dict) -> type:
        obj = base.OME.from_json({"version": version, **fields})
        # the carrier round-trips back to an equal object
        assert base.OME.from_json(obj.to_json()) == obj
        return type(obj)

    assert carrier({"plate": _PLATE}) is ome.OMEPlate
    assert carrier({"well": _WELL}) is ome.OMEWell
    assert carrier({"labels": ["cells", "nuclei"]}) is ome.OMELabels
    assert carrier({"series": ["0", "1"]}) is ome.OMESeries
    assert (
        carrier({"bioformats2raw_layout": 3, "plate": _PLATE})
        is ome.OMEBioformats2Raw
    )


@pytest.mark.parametrize("version", ["0.1", "0.2", "0.3", "0.4", "0.5"])
def test_a_plain_plate_is_not_a_bioformats2raw_layout(version: str) -> None:
    # Regression: OMEBioformats2Raw defaults its layout field to 3, so it used
    # to match any document under the registry's iteration order and hijack it.
    # It must only be chosen when the layout key is actually present.
    from abczarr.ome.metadata import base

    ome = importlib.import_module(
        f"abczarr.ome.metadata.v0_{version.split('.')[1]}.ome"
    )

    for fields in ({"plate": _PLATE}, {"well": _WELL}, {"labels": ["a"]}):
        obj = base.OME.from_json({"version": version, **fields})
        assert not isinstance(obj, ome.OMEBioformats2Raw)
