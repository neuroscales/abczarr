"""The abstract node contract: native escape hatch, capability query, error.

These exercise the surface every driver targets, using a tiny in-repo fake
node -- so they run with no zarr / tensorstore backend installed.
"""

import pathlib

import pytest
from bagof.paths import Path

from abczarr import UnsupportedZarrOperation
from abczarr import _errors as errors
from abczarr.abc import sync
from abczarr.abc.sync import KNOWN_CAPABILITIES, Support, ZarrNode


class _FakeNode(ZarrNode):
    """A minimal concrete node, standing in for a driver."""

    _CAPABILITIES = {
        "sharding": Support.NATIVE,
        "async": Support.SYNTHESIZED,
    }

    @property
    def metadata(self) -> None:
        return None

    @property
    def attrs(self) -> dict:
        return {}

    @property
    def zarr_version(self) -> int:
        return 3


# --------------------------------------------------------------------------
# native escape hatch
# --------------------------------------------------------------------------


def test_native_defaults_to_none() -> None:
    assert _FakeNode("/store").native is None


def test_native_returns_backend_object() -> None:
    n = _FakeNode("/store")
    sentinel = object()
    n._native = sentinel
    assert n.native is sentinel


# --------------------------------------------------------------------------
# store_path normalization -- mirrors Store.__init__
# --------------------------------------------------------------------------


def test_a_pathlike_store_path_is_wrapped() -> None:
    # a node's location need not be a str: a pathlib.Path (any os.PathLike)
    # is normalized to a bagof.paths Path, so it goes through bagof.paths
    # rather than reaching driver code raw -- exactly as Store.__init__ does
    from_str = _FakeNode("/store")
    from_path = _FakeNode(pathlib.Path("/store"))
    assert isinstance(from_str.store_path, Path)
    assert isinstance(from_path.store_path, Path)
    assert str(from_path.store_path) == str(from_str.store_path)


# --------------------------------------------------------------------------
# capability query -- tri-state, answered from the instance
# --------------------------------------------------------------------------


def test_supports_reads_declared_capabilities() -> None:
    n = _FakeNode("/store")
    assert n.supports("sharding") is True
    assert n.supports("async") is True
    assert n.supports("consolidated_metadata") is False


def test_support_reports_native_versus_synthesized() -> None:
    n = _FakeNode("/store")
    assert n.capability("sharding") is Support.NATIVE
    assert n.capability("async") is Support.SYNTHESIZED
    assert n.capability("consolidated_metadata") is Support.NONE


def test_supports_native_only_excludes_synthesized() -> None:
    n = _FakeNode("/store")
    assert n.supports("sharding", native=True) is True
    assert n.supports("async", native=True) is False  # synthesized, not native


def test_supports_unknown_capability_is_false_not_error() -> None:
    assert _FakeNode("/store").supports("teleportation") is False


def test_base_node_declares_no_capabilities() -> None:
    assert ZarrNode._CAPABILITIES == {}


def test_declared_capabilities_are_known() -> None:
    # a driver should only advertise names from the shared vocabulary
    assert set(_FakeNode._CAPABILITIES) <= KNOWN_CAPABILITIES


# --------------------------------------------------------------------------
# UnsupportedZarrOperation
# --------------------------------------------------------------------------


def test_unsupported_operation_names_operation_and_driver() -> None:
    err = UnsupportedZarrOperation("consolidate", driver="zarr-python")
    assert err.operation == "consolidate"
    assert err.driver == "zarr-python"
    assert "consolidate" in str(err)
    assert "zarr-python" in str(err)


def test_unsupported_operation_without_driver() -> None:
    err = UnsupportedZarrOperation("consolidate")
    assert err.driver is None
    assert "consolidate" in str(err)


def test_unsupported_operation_is_a_notimplementederror() -> None:
    # existing ``except NotImplementedError`` still catches it
    with pytest.raises(NotImplementedError):
        raise UnsupportedZarrOperation("walk", driver="memory")
    assert errors.UnsupportedZarrOperation is UnsupportedZarrOperation


# --------------------------------------------------------------------------
# the old flat layer is gone -- abc/ is the single source of truth
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dead", ["abczarr._abc", "abczarr._ome"])
def test_dead_modules_removed(dead: str) -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(dead)


def test_node_module_exports_capabilities() -> None:
    assert "KNOWN_CAPABILITIES" in sync.__all__
