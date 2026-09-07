"""The typed ``node.ome`` accessor: read, write, and delete.

The [ome][abczarr.abc.sync.ZarrNode.ome] property reads a group's
OME-Zarr metadata into a typed [OME][abczarr.ome.base.OME] object and
writes one back, choosing the envelope the version calls for -- the
``"ome"`` attribute for 0.5 and later, the bare attribute keys for 0.4
and earlier. These tests exercise a round trip through a real local
group (no backend needed) for each envelope.
"""

import pathlib

import pytest

from abczarr.abc.sync import PathGroup
from abczarr.metadata.base import GroupMetadataV3
from abczarr.ome import v0_4, v0_5, v0_6rc0

# --- example metadata, one per envelope ------------------------------------

_AXES = [
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]
_SCALE = {"type": "scale", "scale": [1.0, 1.0]}

_OME_04 = {
    "version": "0.4",
    "multiscales": [
        {
            "version": "0.4",
            "name": "example",
            "axes": _AXES,
            "datasets": [{"path": "0", "coordinateTransformations": [_SCALE]}],
        }
    ],
}

_OME_05 = {
    "version": "0.5",
    "multiscales": [
        {
            "name": "example",
            "axes": _AXES,
            "datasets": [{"path": "0", "coordinateTransformations": [_SCALE]}],
        }
    ],
}

_OME_06 = {
    "version": "0.6rc0",
    "multiscales": [
        {
            "name": "example",
            "coordinateSystems": [{"name": "example", "axes": _AXES}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [1.0, 1.0],
                            "input": {"path": "0"},
                            "output": {"name": "example"},
                        }
                    ],
                }
            ],
        }
    ],
}


def _group(tmp_path: pathlib.Path) -> PathGroup:
    """A fresh, empty Zarr v3 group backed by a local store."""
    root = pathlib.Path(tmp_path) / "image.zarr"
    root.mkdir()
    GroupMetadataV3(attributes={}).to_file(root)
    return PathGroup(str(root))


def _reopen(group: PathGroup) -> PathGroup:
    """A fresh handle on the same store, reading persisted state afresh."""
    return PathGroup(group.store_path)


@pytest.mark.parametrize(
    ("module", "data"),
    [(v0_4, _OME_04), (v0_5, _OME_05), (v0_6rc0, _OME_06)],
)
def test_ome_roundtrips_through_a_node(
    tmp_path: pathlib.Path, module: object, data: dict
) -> None:
    group = _group(tmp_path)
    ome = module.OME.from_json(data)
    group.ome = ome
    assert _reopen(group).ome == ome


def test_read_ome_is_none_without_metadata(tmp_path: pathlib.Path) -> None:
    assert _group(tmp_path).ome is None


@pytest.mark.parametrize(
    ("module", "data", "version"),
    [
        (v0_4, _OME_04, "0.4"),
        (v0_5, _OME_05, "0.5"),
        (v0_6rc0, _OME_06, "0.6rc0"),
    ],
)
def test_ome_version_reports_the_declared_version(
    tmp_path: pathlib.Path, module: object, data: dict, version: str
) -> None:
    from abczarr.ome.node import ome_version

    group = _group(tmp_path)
    assert ome_version(group) is None
    group.ome = module.OME.from_json(data)
    assert ome_version(_reopen(group)) == version


def test_the_05_envelope_stores_under_the_ome_key(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.ome = v0_5.OME.from_json(_OME_05)
    attrs = dict(_reopen(group).attrs)
    assert "ome" in attrs
    assert attrs["ome"]["version"] == "0.5"


def test_the_04_envelope_stores_bare_without_a_top_level_version(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.ome = v0_4.OME.from_json(_OME_04)
    attrs = dict(_reopen(group).attrs)
    assert "ome" not in attrs
    assert "multiscales" in attrs
    # <= 0.4 records the version on the multiscale, not at the top.
    assert "version" not in attrs
    assert attrs["multiscales"][0]["version"] == "0.4"


def test_write_ome_keeps_unrelated_attrs_for_the_05_envelope(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.attrs["unrelated"] = "keep me"
    group.ome = v0_5.OME.from_json(_OME_05)
    reopened = _reopen(group)
    assert reopened.attrs["unrelated"] == "keep me"
    assert reopened.ome == v0_5.OME.from_json(_OME_05)


def test_setting_ome_replaces_a_different_envelope(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.ome = v0_5.OME.from_json(_OME_05)
    group.ome = v0_4.OME.from_json(_OME_04)
    attrs = dict(_reopen(group).attrs)
    # the stale 0.5 wrapper is gone, not left beside the bare 0.4 payload
    assert "ome" not in attrs
    assert _reopen(group).ome == v0_4.OME.from_json(_OME_04)


def test_del_ome_removes_the_metadata(tmp_path: pathlib.Path) -> None:
    group = _group(tmp_path)
    group.ome = v0_6rc0.OME.from_json(_OME_06)
    del group.ome
    reopened = _reopen(group)
    assert reopened.ome is None
    assert dict(reopened.attrs) == {}


def test_setting_ome_from_a_plain_dict(tmp_path: pathlib.Path) -> None:
    group = _group(tmp_path)
    group.ome = _OME_05
    assert _reopen(group).ome == v0_5.OME.from_json(_OME_05)
