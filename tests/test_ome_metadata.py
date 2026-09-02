"""The OME-Zarr structured metadata layer builds and round-trips.

These pin that an OME metadata object can be constructed from a dict and
serialized back to an equal object -- exercising the RFC-2119 requirement
factories (which mark fields required / recommended / optional) and the
MISSING sentinel a required-but-unset field carries.
"""

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
    m = cls.from_dict(data)
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
    m = cls.from_dict(data)
    assert cls.from_dict(m.to_dict()) == m


# --------------------------------------------------------------------------
# cross-version conversion between same-structure OME versions (v0.4 <-> v0.5)
# --------------------------------------------------------------------------


def test_multiscale_converts_v04_to_v05_and_back() -> None:
    m4 = v0_4.Multiscale.from_dict(_MULTISCALE_V04)
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
    o4 = v0_4.Omero.from_dict(data)
    assert o4.to_version("0.5").to_version("0.4") == o4


def test_conversion_to_absent_version_raises() -> None:
    m4 = v0_4.Multiscale.from_dict(_MULTISCALE_V04)
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

    m3 = v0_3.Multiscale.from_dict(_MULTISCALE_V03)
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

    m3 = v0_3.Multiscale.from_dict(_MULTISCALE_V03)
    # chains v0.3 -> v0.4 (-> v0.5) and back
    assert m3.to_version(target).to_version("0.3") == m3


def test_down_conversion_drops_axes() -> None:
    from abczarr.ome.metadata import v0_3

    m3 = v0_3.Multiscale.from_dict(_MULTISCALE_V03)
    m2 = m3.to_version("0.2")
    assert "axes" not in m2.to_dict()


def test_underdetermined_up_conversion_raises_clearly() -> None:
    from abczarr.ome.metadata import v0_2

    m2 = v0_2.Multiscale.from_dict(
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
