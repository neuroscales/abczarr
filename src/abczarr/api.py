"""Factory module for creating/opening Zarr Nodes with different drivers."""

# dependencies
import typing_extensions as tx

# locals
from ._core import typing as tz
from .abc import ZarrArray, ZarrGroup, ZarrNode
from .config import ZarrConfig
from .registry import get_driver


def open(
    path: tz.PathLike,
    mode: tz.AccessMode = "a",
    zarr_version: tz.Optional[tz.ZarrVersion] = None,
    driver: tx.Optional[tz.AnyDriver] = None,
    **kwargs
) -> ZarrArray:
    """
    Open a Zarr node (array or group) based on the specified driver.

    Parameters
    ----------
    path : PathLike
        The path to the Zarr array or group.
    mode : AccessMode, optional
        The access mode ('r', 'r+', 'w', 'a').
    zarr_version : Optional[ZarrVersion], optional
        The Zarr version to use. If None, and the mode is not `w`,
        the version is guessed from the store. Otherwise, the most
        recent version is used.
    driver : AnyDriver | None
        The driver to use for opening the Zarr node.
        If None, the first available driver is used.

    Returns
    -------
    ZarrNode
        The opened Zarr node.
    """
    return _open(
        path, mode,
        zarr_version=zarr_version,
        node_type="node",
        driver=driver,
        **kwargs
    )


def open_array(
    path: tz.PathLike,
    mode: tz.AccessMode = "a",
    zarr_version: tz.Optional[tz.ZarrVersion] = None,
    driver: tx.Optional[tz.AnyDriver] = None,
    **kwargs
) -> ZarrArray:
    """
    Open a Zarr array based on the specified driver.

    Parameters
    ----------
    path : PathLike
        The path to the Zarr array.
    mode : AccessMode, optional
        The access mode ('r', 'r+', 'w', 'a').
    zarr_version : Optional[ZarrVersion], optional
        The Zarr version to use. If None, and the mode is not `w`,
        the version is guessed from the store. Otherwise, the most
        recent version is used.
    driver : AnyDriver | None
        The driver to use for opening the Zarr node.
        If None, the first available driver is used.

    Returns
    -------
    ZarrArray
        The opened Zarr array.
    """
    return _open(
        path, mode,
        zarr_version=zarr_version,
        node_type="array",
        driver=driver,
        **kwargs
    )


def open_group(
    path: tz.PathLike,
    mode: tz.AccessMode = "a",
    zarr_version: tz.Optional[tz.ZarrVersion] = None,
    driver: tx.Optional[tz.AnyDriver] = None,
    **kwargs
) -> ZarrGroup:
    """
    Open a Zarr group based on the specified driver.

    Parameters
    ----------
    path : PathLike
        The path to the Zarr group.
    mode : AccessMode, optional
        The access mode ('r', 'r+', 'w', 'a').
    zarr_version : Optional[ZarrVersion], optional
        The Zarr version to use. If None, and the mode is not `w`,
        the version is guessed from the store. Otherwise, the most
        recent version is used.
    driver : AnyDriver | None
        The driver to use for opening the Zarr node.
        If None, the first available driver is used.

    Returns
    -------
    ZarrGroup
        The opened Zarr group.
    """
    return _open(
        path, mode,
        zarr_version=zarr_version,
        node_type="group",
        driver=driver,
        **kwargs
    )


def from_config(out: tz.PathLike, zarr_config: ZarrConfig) -> ZarrGroup:
    """Create a ZarrGroup from a ZarrConfig."""
    group_cls = get_driver(zarr_config.driver, "group")
    return group_cls.from_config(out, zarr_config)


def _open(
    path: tz.PathLike,
    mode: tz.AccessMode = "a",
    zarr_version: tz.Optional[tz.ZarrVersion] = None,
    node_type: tx.Literal["node", "array", "group"] = "node",
    driver: tx.Optional[tz.AnyDriver] = None,
    **kwargs
) -> ZarrNode:
    node_cls = get_driver(driver, node_type)
    return node_cls.open(path, mode, zarr_version=zarr_version, **kwargs)
