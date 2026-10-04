"""Import-cost regression tests.

``import abczarr`` should load only what the package needs up front. The
OME version packages, the OME config and pyramid helpers, and abczarr's own
use of dask are all loaded on first use instead.

Each check runs in a fresh interpreter, because other tests in the same
session import these modules and would hide an eager import.
"""

import json
import subprocess
import sys
import textwrap

import pytest
import typing_extensions as tx

_OME_LAZY = [
    "config",
    "pyramid",
    "schemas",
    "v0_1",
    "v0_2",
    "v0_3",
    "v0_4",
    "v0_5",
    "v0_6",
    "v0_6dev1",
    "v0_6dev2",
    "v0_6dev3",
    "v0_6dev4",
    "v0_6rc0",
]


def _run(code: str) -> tx.Any:
    """Run `code` in a fresh interpreter and decode the JSON it prints last."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_loads_no_ome_version_package() -> None:
    """``import abczarr`` imports none of the lazily loaded OME modules."""
    loaded = _run(
        """
        import json, sys
        import abczarr
        print(json.dumps(sorted(m for m in sys.modules
                                if m.startswith("abczarr.ome."))))
        """
    )
    eager = [m for m in loaded if m.split(".")[2] in _OME_LAZY]
    assert not eager, eager


def test_ome_version_loads_on_access() -> None:
    """Accessing one OME version imports that version only."""
    result = _run(
        """
        import json, sys
        import abczarr
        listed = "v0_4" in dir(abczarr.ome)
        cls = abczarr.ome.v0_4.OME.__module__
        from abczarr.ome import v0_5
        loaded = sorted(
            m.split(".")[2] for m in sys.modules
            if m.startswith("abczarr.ome.v0_") and m.count(".") == 2
        )
        print(json.dumps([listed, cls, v0_5.__name__, loaded]))
        """
    )
    assert result == [
        True,
        "abczarr.ome.v0_4.ome",
        "abczarr.ome.v0_5",
        ["v0_4", "v0_5"],
    ]


def test_ome_version_detection_loads_its_package() -> None:
    """Parsing versioned OME metadata imports the matching package."""
    module = _run(
        """
        import json
        from abczarr.ome import OME
        ome = OME.from_json({
            "version": "0.4",
            "multiscales": [{
                "version": "0.4",
                "axes": [{"name": "x", "type": "space"}],
                "datasets": [{
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0]}
                    ],
                }],
            }],
        })
        print(json.dumps(type(ome).__module__))
        """
    )
    assert module == "abczarr.ome.v0_4.ome"


def test_ome_lazy_names_match_their_modules() -> None:
    """The names ``abczarr.ome`` re-exports lazily are exactly the ones its
    submodules export, and all of them resolve."""
    import abczarr.ome as ome

    expected = {}
    for module in ("config", "pyramid"):
        for name in getattr(ome, module).__all__:
            expected[name] = module
    assert ome._LAZY_ATTRIBUTES == expected
    for name in ome.__all__:
        assert hasattr(ome, name), name
        assert name in dir(ome), name


def test_import_does_not_need_dask() -> None:
    """abczarr imports, pyramid modules included, with dask unavailable."""
    ok = _run(
        """
        import json, sys
        sys.modules["dask"] = None  # any `import dask...` now fails
        import abczarr
        import abczarr._core.pyramid
        import abczarr.ome.pyramid
        print(json.dumps(True))
        """
    )
    assert ok is True


def test_import_loads_no_dask_beyond_dependencies() -> None:
    """``import abczarr`` loads dask only when a dependency already does."""
    loaded = _run(
        """
        import json, sys
        import bagof.converters
        print(json.dumps("dask" in sys.modules))
        """
    )
    if loaded:
        pytest.skip("bagof.converters imports dask itself")
    loaded = _run(
        """
        import json, sys
        import abczarr
        print(json.dumps(sorted(m for m in sys.modules
                                if m == "dask" or m.startswith("dask."))))
        """
    )
    assert loaded == []
