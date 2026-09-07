"""The typed ``node.ome`` accessor: read, write, and delete.

The [ome][abczarr.abc.sync.ZarrNode.ome] property reads a group's
OME-Zarr metadata into a typed [OME][abczarr.ome.base.OME] object and
writes one back, choosing the envelope the version calls for -- the
``"ome"`` attribute for 0.5 and later, the bare attribute keys for 0.4
and earlier. These tests exercise a round trip through a real local
group (no backend needed) for each envelope.
"""

import asyncio
import pathlib

import pytest

from abczarr.abc.sync import PathGroup
from abczarr.metadata.base import GroupMetadataV3
from abczarr.ome import v0_4, v0_5, v0_6rc0
from abczarr.ome.base import LATEST_STABLE

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


# --- update_ome: shallow merge --------------------------------------------


def test_update_ome_replaces_only_the_keys_given(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.ome = v0_5.OME.from_json(_OME_05)
    group.update_ome({"omero": {"channels": []}})
    merged = _reopen(group).ome.to_json()
    # the key passed is added; the untouched multiscales survive
    assert "omero" in merged
    assert merged["multiscales"] == _OME_05["multiscales"]
    assert merged["version"] == "0.5"


def test_update_ome_replaces_a_whole_top_level_key(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.ome = v0_5.OME.from_json(_OME_05)
    replacement = {
        "multiscales": [
            {
                "axes": [{"name": "x", "type": "space"}],
                "datasets": [{"path": "9", "coordinateTransformations": [
                    {"type": "scale", "scale": [2.0]}]}],
            }
        ]
    }
    group.update_ome(replacement)
    reopened = _reopen(group).ome
    # the whole multiscales key is replaced, not deep-merged
    assert reopened.multiscales[0].datasets[0].path == "9"


def test_update_ome_defaults_version_when_the_node_had_none(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    group.attrs["unrelated"] = "keep me"
    group.update_ome(
        {"multiscales": _OME_05["multiscales"]}  # no version anywhere
    )
    reopened = _reopen(group)
    assert reopened.ome.version == LATEST_STABLE == "0.5"
    assert reopened.attrs["unrelated"] == "keep me"


# --- the async twin: read is a sync property, writes are awaited -----------
#
# pytest-asyncio is not a dependency, so each coroutine is driven with
# ``asyncio.run`` from a plain synchronous test -- the same harness
# ``tests/test_async_nodes.py`` uses. The async path group runs over a local
# store, so these need no backend.


def _async_group(tmp_path: pathlib.Path) -> object:
    return _group(tmp_path).as_async()


def _async_reopen(node: object) -> object:
    return PathGroup(node.store_path).as_async()


def test_async_ome_read_is_a_synchronous_property() -> None:
    # mirrors ``attrs``: reading OME on the async node never blocks
    from abczarr.abc.asynchronous import AsyncZarrNode

    prop = AsyncZarrNode.ome
    assert isinstance(prop, property)
    assert not asyncio.iscoroutinefunction(prop.fget)


@pytest.mark.parametrize(
    ("module", "data"),
    [(v0_4, _OME_04), (v0_5, _OME_05), (v0_6rc0, _OME_06)],
)
def test_async_set_ome_roundtrips_through_the_sync_property(
    tmp_path: pathlib.Path, module: object, data: dict
) -> None:
    node = _async_group(tmp_path)
    ome = module.OME.from_json(data)
    assert node.ome is None
    asyncio.run(node.set_ome(ome))
    assert _async_reopen(node).ome == ome


def test_async_update_ome_shallow_merges(tmp_path: pathlib.Path) -> None:
    node = _async_group(tmp_path)
    asyncio.run(node.set_ome(v0_5.OME.from_json(_OME_05)))
    asyncio.run(node.update_ome({"omero": {"channels": []}}))
    merged = _async_reopen(node).ome.to_json()
    assert "omero" in merged
    assert merged["multiscales"] == _OME_05["multiscales"]


def test_async_update_ome_defaults_version(tmp_path: pathlib.Path) -> None:
    node = _async_group(tmp_path)
    asyncio.run(node.update_ome({"multiscales": _OME_05["multiscales"]}))
    assert _async_reopen(node).ome.version == LATEST_STABLE


def test_async_del_ome_removes_the_metadata(tmp_path: pathlib.Path) -> None:
    node = _async_group(tmp_path)
    asyncio.run(node.set_ome(v0_6rc0.OME.from_json(_OME_06)))
    asyncio.run(node.del_ome())
    reopened = _async_reopen(node)
    assert reopened.ome is None
    assert dict(reopened.attrs) == {}
