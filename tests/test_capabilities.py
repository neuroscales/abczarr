"""The shared capability model: the tri-state, the query, and feature keys."""

import pytest

from abczarr.abc.capabilities import (
    KNOWN_CAPABILITIES,
    Support,
    SupportsCapabilities,
    feature_key,
)

# --------------------------------------------------------------------------
# Support tri-state
# --------------------------------------------------------------------------


def test_support_truthiness() -> None:
    assert bool(Support.NATIVE) is True
    assert bool(Support.SYNTHESIZED) is True
    assert bool(Support.NONE) is False


# --------------------------------------------------------------------------
# the query mixin
# --------------------------------------------------------------------------


class _Thing(SupportsCapabilities):
    _CAPABILITIES = {
        "listing": Support.NATIVE,
        "partial_read": Support.SYNTHESIZED,
    }


def test_support_returns_declared_state() -> None:
    t = _Thing()
    assert t.support("listing") is Support.NATIVE
    assert t.support("partial_read") is Support.SYNTHESIZED
    assert t.support("teleportation") is Support.NONE


def test_supports_collapses_to_bool() -> None:
    t = _Thing()
    assert t.supports("listing") is True
    assert t.supports("partial_read") is True
    assert t.supports("teleportation") is False


def test_supports_native_only() -> None:
    t = _Thing()
    assert t.supports("listing", native=True) is True
    assert t.supports("partial_read", native=True) is False


def test_default_declares_nothing() -> None:
    assert SupportsCapabilities().support("listing") is Support.NONE


# --------------------------------------------------------------------------
# feature keys
# --------------------------------------------------------------------------


def test_feature_key_builds_namespaced_string() -> None:
    assert feature_key("v3", "codec", "zstd") == "v3:codec:zstd"
    assert feature_key("v2", "filter", "delta") == "v2:filter:delta"


def test_feature_key_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="feature version"):
        feature_key("v4", "codec", "zstd")


def test_feature_key_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="feature kind"):
        feature_key("v3", "sprocket", "zstd")


def test_known_capabilities_are_coarse_names() -> None:
    # the coarse vocabulary, not the fine feature keys
    assert "listing" in KNOWN_CAPABILITIES
    assert "sharding" in KNOWN_CAPABILITIES
    assert "transactions" in KNOWN_CAPABILITIES
