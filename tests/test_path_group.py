"""PathGroup: walking a Zarr hierarchy from the store alone.

PathGroup is the backend-independent group. It lists members, reads its own
metadata, and navigates into subgroups using only abczarr's path and metadata
layers; a driver supplies just how to open a child array. These tests use a
stub array opener, so they run with no backend installed.
"""

import json
import pathlib

import pytest

from abczarr.abc.sync import PathGroup
from abczarr.errors import UnsupportedZarrOperation
from abczarr.metadata.base import (
    ArrayMetadataV1,
    ArrayMetadataV2,
    GroupMetadataV2,
    GroupMetadataV3,
    NodeMetadata,
    NodeMetadataV3,
    _node_type_at,
)


class _StubArray:
    """A stand-in for a driver's opened array."""

    def __init__(self, store_path: object) -> None:
        self.store_path = store_path


class _Group(PathGroup):
    """A PathGroup that opens child arrays as markers, for testing the
    backend-independent surface with no real backend."""

    def _open_array(self, store_path: object) -> _StubArray:
        return _StubArray(store_path)


def _write_array(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [4],
                "data_type": "int32",
            }
        )
    )


def _hierarchy(tmp_path: pathlib.Path) -> str:
    """A v3 store: a root group with an array ``img`` and a subgroup ``sub``
    that itself holds an array ``inner``. Returns the root as a string, so the
    group builds a store-aware path (with a recursive ``rmdir``)."""
    root = pathlib.Path(tmp_path) / "data.zarr"
    root.mkdir()
    GroupMetadataV3(attributes={}).to_file(root)
    _write_array(root / "img")
    sub = root / "sub"
    sub.mkdir()
    GroupMetadataV3(attributes={}).to_file(sub)
    _write_array(sub / "inner")
    return str(root)


# --------------------------------------------------------------------------
# listing and navigation
# --------------------------------------------------------------------------


def test_keys_lists_the_members(tmp_path: pathlib.Path) -> None:
    group = _Group(_hierarchy(tmp_path))
    assert sorted(group.keys()) == ["img", "sub"]


def test_iter_matches_keys(tmp_path: pathlib.Path) -> None:
    group = _Group(_hierarchy(tmp_path))
    assert sorted(group) == ["img", "sub"]


def test_contains_a_member(tmp_path: pathlib.Path) -> None:
    group = _Group(_hierarchy(tmp_path))
    assert "img" in group
    assert "sub" in group
    assert "missing" not in group


def test_indexing_an_array_uses_the_open_hook(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    assert isinstance(group["img"], _StubArray)


def test_indexing_a_subgroup_gives_the_same_kind(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    sub = group["sub"]
    assert isinstance(sub, _Group)
    assert sorted(sub.keys()) == ["inner"]


def test_indexing_a_missing_member_raises_keyerror(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    with pytest.raises(KeyError):
        group["nope"]


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def test_metadata_reads_the_group_metadata(tmp_path: pathlib.Path) -> None:
    group = _Group(_hierarchy(tmp_path))
    assert group.zarr_version == 3
    assert group.attrs == {}
    assert group.metadata.node_type == "group"


# --------------------------------------------------------------------------
# create and delete
# --------------------------------------------------------------------------


def test_create_group_writes_metadata_and_appears(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    made = group.create_group("new")
    assert isinstance(made, _Group)
    assert "new" in group
    assert _node_type_at(group.store_path / "new") == "group"


def test_create_group_refuses_an_existing_member(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    with pytest.raises(FileExistsError):
        group.create_group("img")


def test_delitem_removes_a_member_and_its_tree(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    del group["sub"]
    assert "sub" not in group


def test_setitem_is_unsupported(tmp_path: pathlib.Path) -> None:
    group = _Group(_hierarchy(tmp_path))
    with pytest.raises(UnsupportedZarrOperation):
        group["x"] = None


# --------------------------------------------------------------------------
# the backend hooks default to unsupported
# --------------------------------------------------------------------------


def test_bare_path_group_cannot_open_an_array(
    tmp_path: pathlib.Path,
) -> None:
    group = PathGroup(_hierarchy(tmp_path))
    with pytest.raises(UnsupportedZarrOperation):
        group["img"]


def test_bare_path_group_cannot_create_an_array(
    tmp_path: pathlib.Path,
) -> None:
    group = PathGroup(_hierarchy(tmp_path))
    with pytest.raises(UnsupportedZarrOperation):
        group.create_array("a", (2,), "int32")


# --------------------------------------------------------------------------
# a group only sees children of its own version
# --------------------------------------------------------------------------


def _write_v2_group(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))


def test_a_child_of_another_version_is_not_a_member(
    tmp_path: pathlib.Path,
) -> None:
    root = _hierarchy(tmp_path)  # a v3 group
    _write_v2_group(pathlib.Path(root) / "legacy")
    group = _Group(root)
    assert "legacy" not in group
    assert "legacy" not in list(group.keys())
    with pytest.raises(KeyError):
        group["legacy"]


def test_a_v3_node_without_a_node_type_is_not_detected(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.metadata.base import _node_at

    directory = pathlib.Path(tmp_path) / "x"
    directory.mkdir()
    (directory / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "shape": [4]})
    )
    assert _node_at(directory) is None


def test_create_array_writes_metadata_and_opens_via_the_hook(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.metadata.base import _node_at

    group = _Group(_hierarchy(tmp_path))
    made = group.create_array(
        "fresh", shape=(4,), dtype="int32", chunks=(4,)
    )
    assert isinstance(made, _StubArray)
    # the array metadata was written, in the group's own version
    assert _node_at(group.store_path / "fresh") == ("array", 3)


# --------------------------------------------------------------------------
# attrs are a live, write-through mapping (inherited from ZarrNode)
# --------------------------------------------------------------------------


def test_attrs_persist_on_mutation(tmp_path: pathlib.Path) -> None:
    group = _Group(_hierarchy(tmp_path))
    group.attrs["author"] = "me"
    group.attrs["levels"] = 3
    # a freshly opened group reads them back from the store
    reopened = _Group(group.store_path)
    assert dict(reopened.attrs) == {"author": "me", "levels": 3}
    del group.attrs["levels"]
    assert "levels" not in _Group(group.store_path).attrs


# --------------------------------------------------------------------------
# v1 / v2 metadata load through from_file (regression: it used to call a
# non-existent from_files and always raise AttributeError)
# --------------------------------------------------------------------------


def _write_v2_array(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [10],
                "chunks": [5],
                "dtype": "<f8",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": [],
            }
        )
    )
    (path / ".zattrs").write_text(json.dumps({"note": "v2 array"}))


def _write_v1_array(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # v1 stores its metadata in "meta" and user attributes in "attrs"
    (path / "meta").write_text(
        json.dumps(
            {
                "zarr_format": 1,
                "shape": [10],
                "chunks": [5],
                "dtype": "<f8",
                "compression": None,
                "compression_opts": None,
                "fill_value": 0,
                "order": "C",
            }
        )
    )
    (path / "attrs").write_text(json.dumps({"note": "v1 array"}))


def test_v2_group_loads_through_path_group(tmp_path: pathlib.Path) -> None:
    root = pathlib.Path(tmp_path) / "v2.zarr"
    GroupMetadataV2(attributes={"kind": "root"}).to_file(root)
    _write_v2_array(root / "img")

    group = _Group(str(root))
    assert group.zarr_version == 2
    assert group.attrs == {"kind": "root"}
    assert isinstance(group.metadata, GroupMetadataV2)
    assert group.metadata.node_type == "group"
    assert sorted(group.keys()) == ["img"]


def test_v2_array_metadata_from_file(tmp_path: pathlib.Path) -> None:
    node = pathlib.Path(tmp_path) / "arr"
    _write_v2_array(node)
    meta = NodeMetadata.from_file(node)
    assert isinstance(meta, ArrayMetadataV2)
    assert meta.zarr_format == 2
    assert meta.attributes == {"note": "v2 array"}


def test_v1_array_metadata_from_file(tmp_path: pathlib.Path) -> None:
    node = pathlib.Path(tmp_path) / "arr"
    _write_v1_array(node)
    meta = NodeMetadata.from_file(node)
    assert isinstance(meta, ArrayMetadataV1)
    assert meta.zarr_format == 1
    assert meta.attributes == {"note": "v1 array"}


def test_v3_from_file_without_zarr_json_raises(tmp_path: pathlib.Path) -> None:
    node = pathlib.Path(tmp_path) / "arr"
    node.mkdir()
    with pytest.raises(FileNotFoundError, match="zarr.json"):
        NodeMetadataV3.from_file(node)
