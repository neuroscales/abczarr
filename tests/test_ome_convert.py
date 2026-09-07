"""Cross-version OME conversion, including the 0.5 <-> 0.6 (RFC-5) boundary.

0.5 puts the axes on the multiscale and a scale (+ translation) on each
dataset; 0.6 drops ``axes``, carries named ``coordinateSystems``, and gives
each dataset general coordinate transformations mapping the array's intrinsic
system (``input={"path": ...}``) onto a named output system. These tests pin
the boundary conversion both ways and the loss policy that governs a 0.6
transform the stable model cannot express.
"""

import warnings

import pytest

from abczarr.errors import UnsupportedConversion
from abczarr.ome import v0_5, v0_6rc0

_AXES = [
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def _multiscale_05(datasets: list, **extra: object) -> object:
    return v0_5.Multiscale.from_json(
        {"name": "example", "axes": _AXES, "datasets": datasets, **extra}
    )


# --- the boundary round trip -----------------------------------------------


def test_scale_only_roundtrips_through_06_exactly() -> None:
    m = _multiscale_05(
        [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [2.0, 4.0]}]}]
    )
    assert m.to_version("0.6rc0").to_version("0.5") == m


def test_scale_and_translation_roundtrips_through_06_exactly() -> None:
    m = _multiscale_05(
        [
            {"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 2.0]},
                {"type": "translation", "translation": [0.0, 5.0]}]},
            {"path": "1", "coordinateTransformations": [
                {"type": "scale", "scale": [2.0, 4.0]}]},
        ]
    )
    assert m.to_version("0.6rc0").to_version("0.5") == m


def test_container_roundtrips_through_06_exactly() -> None:
    ome = v0_5.OME.from_json(
        {
            "version": "0.5",
            "multiscales": [
                {
                    "name": "example",
                    "axes": _AXES,
                    "datasets": [{"path": "0", "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]}]}],
                }
            ],
        }
    )
    converted = ome.to_version("0.6rc0")
    assert type(converted).__module__ == "abczarr.ome.v0_6rc0.ome"
    assert converted.version == "0.6rc0"
    assert converted.to_version("0.5") == ome


# --- the forward shape (RFC-5 conventions) ---------------------------------


def test_05_to_06_builds_one_named_output_system_from_the_axes() -> None:
    m = _multiscale_05(
        [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0]}]}]
    )
    r = m.to_version("0.6rc0")
    # the axes move onto a single coordinate system, named after the
    # multiscale, with their types and units preserved
    assert len(r.coordinateSystems) == 1
    system = r.coordinateSystems[0]
    assert system.name == "example"
    assert [(a.name, a.type) for a in system.axes] == [
        ("y", "space"), ("x", "space")
    ]
    assert [a.unit for a in system.axes] == ["micrometer", "micrometer"]


def test_05_to_06_maps_the_array_via_path_onto_the_named_system() -> None:
    m = _multiscale_05(
        [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [3.0, 3.0]}]}]
    )
    transform = m.to_version("0.6rc0").datasets[0].coordinateTransformations[0]
    # input references the array (intrinsic system) by its path; output
    # names the coordinate system the array maps onto
    assert transform.input.path == "0"
    assert transform.output.name == "example"
    assert transform.scale == [3.0, 3.0]


def test_05_to_06_uses_a_sequence_for_scale_then_translation() -> None:
    m = _multiscale_05(
        [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 2.0]},
            {"type": "translation", "translation": [0.0, 5.0]}]}]
    )
    transform = m.to_version("0.6rc0").datasets[0].coordinateTransformations[0]
    assert transform.type == "sequence"
    assert transform.input.path == "0"
    assert transform.output.name == "example"
    kinds = [t.type for t in transform.transformations]
    assert kinds == ["scale", "translation"]


def test_05_multiscale_without_a_name_uses_a_default_system_name() -> None:
    m = v0_5.Multiscale.from_json(
        {"axes": _AXES, "datasets": [{"path": "0",
         "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]
         }]}
    )
    r = m.to_version("0.6rc0")
    assert r.coordinateSystems[0].name == "physical"
    # the (absent) multiscale name is not invented on the way over
    from abczarr._core.rfc2119 import MISSING

    assert r.name is MISSING


# --- the reduction and its loss policy -------------------------------------


def _affine_multiscale_06() -> object:
    return v0_6rc0.Multiscale.from_json(
        {
            "coordinateSystems": [{"name": "physical", "axes": _AXES}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "affine",
                            "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            "input": {"path": "0"},
                            "output": {"name": "physical"},
                        }
                    ],
                }
            ],
        }
    )


def test_affine_is_dropped_silently_under_lossy() -> None:
    m = _affine_multiscale_06()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        reduced = m.to_version("0.5", policy="lossy")
    # a dataset with nothing representable falls back to an identity scale
    (scale,) = reduced.datasets[0].coordinateTransformations
    assert scale.type == "scale"
    assert scale.scale == [1.0, 1.0]


def test_affine_warns_once_under_warn() -> None:
    m = _affine_multiscale_06()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.to_version("0.5", policy="warn")
    assert len(caught) == 1
    assert "affine" in str(caught[0].message)


def test_affine_raises_under_strict() -> None:
    m = _affine_multiscale_06()
    with pytest.raises(UnsupportedConversion):
        m.to_version("0.5", policy="strict")


def test_warn_is_the_default_policy() -> None:
    m = _affine_multiscale_06()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.to_version("0.5")
    assert len(caught) == 1


# --- axes / coordinate-system correspondence -------------------------------


def test_06_to_05_takes_axes_from_the_datasets_output_system() -> None:
    # two systems; the datasets map their arrays onto "physical", so its
    # axes -- not the other system's -- become the 0.5 axes
    m = v0_6rc0.Multiscale.from_json(
        {
            "coordinateSystems": [
                {"name": "anatomical", "axes": [
                    {"name": "a", "type": "space"}]},
                {"name": "physical", "axes": _AXES},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [1.0, 1.0],
                            "input": {"path": "0"},
                            "output": {"name": "physical"},
                        }
                    ],
                }
            ],
        }
    )
    reduced = m.to_version("0.5")
    assert [a.name for a in reduced.axes] == ["y", "x"]


def test_06_to_05_preserves_a_scale_and_translation_sequence() -> None:
    m = v0_6rc0.Multiscale.from_json(
        {
            "coordinateSystems": [{"name": "physical", "axes": _AXES}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "sequence",
                            "input": {"path": "0"},
                            "output": {"name": "physical"},
                            "transformations": [
                                {"type": "scale", "scale": [2.0, 2.0]},
                                {"type": "translation",
                                 "translation": [1.0, 3.0]},
                            ],
                        }
                    ],
                }
            ],
        }
    )
    reduced = m.to_version("0.5")
    scale, translation = reduced.datasets[0].coordinateTransformations
    assert scale.scale == [2.0, 2.0]
    assert translation.translation == [1.0, 3.0]
