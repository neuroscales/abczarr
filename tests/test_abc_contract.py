"""The abstract node contract: native escape hatch, capability query, error.

These exercise the surface every driver targets, using a tiny in-repo fake
node -- so they run with no zarr / tensorstore backend installed.
"""

import pytest

from abczarr import UnsupportedZarrOperation
from abczarr.abc import errors, node
from abczarr.abc.node import KNOWN_CAPABILITIES, ZarrNode


class _FakeNode(ZarrNode):
    """A minimal concrete node, standing in for a driver."""

    _CAPABILITIES = frozenset({"sharding", "async"})

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
# capability query -- answered from the class, no live store
# --------------------------------------------------------------------------


def test_supports_reads_declared_capabilities() -> None:
    assert _FakeNode.supports("sharding") is True
    assert _FakeNode.supports("async") is True
    assert _FakeNode.supports("consolidated_metadata") is False


def test_supports_unknown_capability_is_false_not_error() -> None:
    assert _FakeNode.supports("teleportation") is False


def test_base_node_declares_no_capabilities() -> None:
    assert ZarrNode._CAPABILITIES == frozenset()


def test_declared_capabilities_are_known() -> None:
    # a driver should only advertise names from the shared vocabulary
    assert _FakeNode._CAPABILITIES <= KNOWN_CAPABILITIES


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
    assert "KNOWN_CAPABILITIES" in node.__all__
