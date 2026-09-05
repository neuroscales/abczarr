"""The atomic metadata write is local-filesystem only.

`_atomic_write` writes a temporary file next to the target and renames it
over the target, which needs a local filesystem. A remote store is refused
with a clear, named error rather than the opaque one the temp-file machinery
would raise on a non-local path.
"""

import json
import pathlib

import pytest

from abczarr.errors import UnsupportedZarrOperation
from abczarr.metadata.base import _atomic_write


class _RemotePath:
    """A path-like just remote enough to trip the locality guard.

    Only the ``protocol`` attribute is read before the guard fires, so no
    real remote backend (universal-pathlib / cloudpathlib) is needed to
    exercise the refusal.
    """

    protocol = "s3"

    def __fspath__(self) -> str:
        return "s3://bucket/grp/zarr.json"


def test_plain_pathlib_path_is_written(tmp_path: pathlib.Path) -> None:
    # a stdlib path has no ``protocol`` and counts as local
    target = tmp_path / "zarr.json"
    _atomic_write(target, {"zarr_format": 3})
    assert json.loads(target.read_text()) == {"zarr_format": 3}


def test_remote_path_raises_a_clear_error() -> None:
    with pytest.raises(UnsupportedZarrOperation) as excinfo:
        _atomic_write(_RemotePath(), {"zarr_format": 3})
    message = str(excinfo.value)
    assert "local filesystem" in message
    assert "atomic" in message
