"""Import smoke tests.

These guard against a broken core submodule taking down the whole package
at import time -- the failure mode that motivated this file, where a
renamed class in ``_core.path`` left ``abc.path`` (and therefore every
``import abczarr...``) raising, so even unrelated metadata/OME tests could
not be collected.

They install no optional backend, so they also assert that importing the
package never eagerly imports a driver (zarr-python, tensorstore, ...).
"""

import importlib

import pytest


def test_import_package() -> None:
    """The top-level package imports with no optional backend installed."""
    import abczarr  # noqa: F401


def test_public_names_resolve() -> None:
    """Every name advertised in ``__all__`` is actually reachable."""
    import abczarr

    for name in abczarr.__all__:
        assert hasattr(abczarr, name), name


@pytest.mark.parametrize(
    "module",
    [
        "abczarr.abc",
        "abczarr.abc.node",
        "abczarr.abc.array",
        "abczarr.abc.group",
        "abczarr.abc.store",
        "abczarr.abc.path",
        "abczarr.api",
        "abczarr.config",
        "abczarr.registry",
        "abczarr.metadata",
        "abczarr.ome",
        "abczarr.schemas",
        "abczarr._core.path",
    ],
)
def test_submodule_imports(module: str) -> None:
    """Each core submodule imports on its own."""
    importlib.import_module(module)


def test_store_path_classes_exist() -> None:
    """The store/path lattice that broke stays wired to ``_core.path``."""
    from abczarr.abc import path as spath

    for name in spath.__all__:
        assert hasattr(spath, name), name
