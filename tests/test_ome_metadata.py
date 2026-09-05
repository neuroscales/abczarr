"""The OME-Zarr structured metadata layer builds and round-trips.

These pin that an OME metadata object can be constructed from a dict and
serialized back to an equal object -- exercising the RFC-2119 requirement
factories (which mark fields required / recommended / optional) and the
MISSING sentinel a required-but-unset field carries.
"""

import importlib

import pytest

from abczarr.ome import v0_4, v0_5

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


def test_multiscale_metadata_args_string_is_not_shredded() -> None:
    """A bare-string ``metadata.args`` stays a string, not a list of chars.

    The upstream OME corpus writes ``multiscales.metadata.args`` as a
    bare string as well as a list, so the field is free-form JSON. A
    sequence-typed field would have iterated the string, coercing
    ``"[true]"`` into ``['[', 't', 'r', 'u', 'e', ']']``.
    """
    data = {
        "axes": [
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1]}
                ],
            }
        ],
        "metadata": {"method": "x", "args": "[true]"},
    }
    m = v0_4.Multiscale.from_json(data)
    assert m.metadata.args == "[true]"
    assert m.metadata.args != ["[", "t", "r", "u", "e", "]"]
    # A genuine list of JSON values still passes through element-wise.
    m2 = v0_4.Multiscale.from_json({**data, "metadata": {"args": [True, 3]}})
    assert m2.metadata.args == [True, 3]


# --------------------------------------------------------------------------
# An unset optional ``omero`` is OMITTED, not emitted as ``"omero": null``
# (regression for issue #116; the stable chain shared the same bug). The NGFF
# ``image`` schema types ``omero`` as an object, so a serialized
# ``"omero": null`` fails validation; an absent ``omero`` conforms.
# --------------------------------------------------------------------------


def test_stable_unset_omero_is_omitted_and_validates() -> None:
    from abczarr.ome import schemas

    o = v0_5.OME.from_json(
        {"version": "0.5", "multiscales": [_MULTISCALE_V05]}
    )
    assert type(o).__name__ == "OMEImage"
    out = o.to_json()
    assert "omero" not in out
    schemas.validate({"ome": out}, "0.5", "image")


def test_stable_present_omero_round_trips() -> None:
    from abczarr.ome import schemas

    omero = {
        "channels": [
            {
                "color": "FF0000",
                "window": {"start": 0.0, "min": 0.0, "end": 255.0,
                           "max": 255.0},
            }
        ]
    }
    o = v0_5.OME.from_json(
        {"version": "0.5", "multiscales": [_MULTISCALE_V05], "omero": omero}
    )
    out = o.to_json()
    assert out.get("omero") == omero
    assert v0_5.OME.from_json(out) == o
    schemas.validate({"ome": out}, "0.5", "image")


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
    from abczarr.ome import v0_3

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
    from abczarr.ome import v0_3

    m3 = v0_3.Multiscale.from_json(_MULTISCALE_V03)
    # chains v0.3 -> v0.4 (-> v0.5) and back
    assert m3.to_version(target).to_version("0.3") == m3


def test_down_conversion_drops_axes() -> None:
    from abczarr.ome import v0_3

    m3 = v0_3.Multiscale.from_json(_MULTISCALE_V03)
    m2 = m3.to_version("0.2")
    assert "axes" not in m2.to_json()


def test_underdetermined_up_conversion_raises_clearly() -> None:
    from abczarr.ome import v0_2

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
    from abczarr.ome import base

    data = {"version": version, "bioformats2raw.layout": 3, "plate": _PLATE}
    obj = base.OME.from_json(data)
    assert type(obj).__module__ == f"abczarr.ome.{package}.ome"


def test_base_ome_from_dict_rejects_versionless_metadata() -> None:
    # Under the old last-import-wins behavior this silently built an object
    # from whichever package imported last; ambiguous data must now raise.
    from abczarr.ome import base

    with pytest.raises(ValueError, match="version"):
        base.OME.from_json({"bioformats2raw.layout": 3, "plate": _PLATE})


def test_base_ome_from_dict_rejects_unknown_version() -> None:
    from abczarr.ome import base

    with pytest.raises(ValueError, match="Unknown OME version"):
        base.OME.from_json({"version": "9.9"})


def test_per_version_ome_from_dict_still_dispatches() -> None:
    # The per-version OME classes keep their own dispatch: routing through the
    # base must reach the same class a direct call would.
    from abczarr.ome import base, v0_5

    data = {"version": "0.5", "bioformats2raw.layout": 3, "plate": _PLATE}
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
    from abczarr.ome import base

    ome = importlib.import_module(
        f"abczarr.ome.v0_{version.split('.')[1]}.ome"
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
        carrier({"bioformats2raw.layout": 3, "plate": _PLATE})
        is ome.OMEBioformats2Raw
    )


@pytest.mark.parametrize("version", ["0.1", "0.2", "0.3", "0.4", "0.5"])
def test_a_plain_plate_is_not_a_bioformats2raw_layout(version: str) -> None:
    # Regression: OMEBioformats2Raw defaults its layout field to 3, so it used
    # to match any document under the registry's iteration order and hijack it.
    # It must only be chosen when the layout key is actually present.
    from abczarr.ome import base

    ome = importlib.import_module(
        f"abczarr.ome.v0_{version.split('.')[1]}.ome"
    )

    for fields in ({"plate": _PLATE}, {"well": _WELL}, {"labels": ["a"]}):
        obj = base.OME.from_json({"version": version, **fields})
        assert not isinstance(obj, ome.OMEBioformats2Raw)


# --------------------------------------------------------------------------
# JSON-key aliases: an NGFF key that is not a Python identifier
# (``bioformats2raw.layout``, ``image-label``, ``label-value``) reaches its
# typed field and serializes back under that same key -- never an underscore
# twin in ``extra_items`` and never a doubled key on the way out.
# --------------------------------------------------------------------------


def test_bioformats2raw_layout_alias_roundtrips_cleanly() -> None:
    o = v0_5.OME.from_json(
        {"version": "0.5", "bioformats2raw.layout": 3, "plate": _PLATE}
    )
    # dispatched to the bf2raw carrier and populated the typed field
    assert type(o).__name__ == "OMEBioformats2Raw"
    assert o.bioformats2raw_layout == 3
    # the aliased key was consumed -- it did not ride in extra_items
    assert not o.extra_items
    j = o.to_json()
    # emitted only under the spec key, with no spurious underscore twin
    assert set(j) == {"version", "bioformats2raw.layout", "plate"}
    assert j["bioformats2raw.layout"] == 3
    assert "bioformats2raw_layout" not in j
    assert v0_5.OME.from_json(j) == o


def test_label_value_alias_roundtrips_cleanly() -> None:
    label = {
        "colors": [{"label-value": 1, "rgba": [0, 128, 0, 255]}],
    }
    il = v0_5.labels.ImageLabel.from_json(label)
    color = il.colors[0]
    assert color.label_value == 1
    assert not color.extra_items
    # exact serialized shape of a fully-determined color: only the spec keys
    assert color.to_json() == {"label-value": 1, "rgba": [0, 128, 0, 255]}
    assert "label_value" not in color.to_json()
    assert v0_5.labels.ImageLabel.from_json(il.to_json()) == il


def test_image_label_document_dispatches_to_image_label_carrier() -> None:
    # An OME document carrying ``image-label`` (the spec key, a hyphen) now
    # dispatches to OMEImageLabel and populates its ``image_label`` field --
    # before the alias, the singular discriminator key named no field and the
    # document fell through to OMEImage.
    doc = {
        "version": "0.5",
        "multiscales": [_MULTISCALE_V05],
        "image-label": {"colors": [{"label-value": 1, "rgba": [0, 0, 0, 0]}]},
    }
    o = v0_5.OME.from_json(doc)
    assert type(o).__name__ == "OMEImageLabel"
    assert o.image_label.colors[0].label_value == 1
    assert not o.extra_items
    j = o.to_json()
    assert "image-label" in j
    assert "image_label" not in j
    assert "image_labels" not in j
    assert v0_5.OME.from_json(j) == o


# --------------------------------------------------------------------------
# ImageLabel matches the NGFF spec: ``properties`` is an array and ``source``
# carries only ``image`` (regression for the stable chain -- issue #97). The
# schema defines ``properties`` as ``type: array`` (a list of per-label
# property objects) and ``source`` as ``{"image": str}`` with no
# ``label-value``.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.2", "0.4"])
def test_image_label_source_is_image_only(version: str) -> None:
    labels = importlib.import_module(
        f"abczarr.ome.v0_{version.split('.')[1]}.labels"
    )
    data = {"version": version, "source": {"image": "../../"}}
    # ``source`` is ``{"image": str}`` -- building it must not demand a
    # ``label-value`` that the spec does not define on ``source``.
    il = labels.ImageLabel.from_json(data)
    assert il.source.image == "../../"
    assert not il.source.extra_items
    assert il.source.to_json() == {"image": "../../"}


@pytest.mark.parametrize("version", ["0.2", "0.4"])
def test_image_label_properties_is_a_list(version: str) -> None:
    labels = importlib.import_module(
        f"abczarr.ome.v0_{version.split('.')[1]}.labels"
    )
    data = {
        "version": version,
        "properties": [{"label-value": 1}, {"label-value": 2}],
    }
    # ``properties`` is a list of per-label property objects, one per label
    # value -- not a single object.
    il = labels.ImageLabel.from_json(data)
    assert [p.label_value for p in il.properties] == [1, 2]
    assert il.to_json()["properties"] == [
        {"label-value": 1},
        {"label-value": 2},
    ]
    assert labels.ImageLabel.from_json(il.to_json()) == il
