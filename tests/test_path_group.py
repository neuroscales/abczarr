"""PathGroup: walking a Zarr hierarchy from the store alone.

PathGroup is the backend-independent group. It lists members, reads its own
metadata, and navigates into subgroups using only abczarr's path and metadata
layers; a driver supplies just how to open a child array. These tests use a
stub array opener, so they run with no backend installed.
"""

import json
import pathlib

import pytest

from abczarr.abc.errors import UnsupportedZarrOperation
from abczarr.abc.group import PathGroup
from abczarr.metadata.base import GroupMetadataV3, node_type_at


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
    assert node_type_at(group.store_path / "new") == "group"


def test_create_group_refuses_an_existing_member(
    tmp_path: pathlib.Path,
) -> None:
    group = _Group(_hierarchy(tmp_path))
    with pytest.raises(UnsupportedZarrOperation):
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
    from abczarr.metadata.base import node_at

    directory = pathlib.Path(tmp_path) / "x"
    directory.mkdir()
    (directory / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "shape": [4]})
    )
    assert node_at(directory) is None


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
