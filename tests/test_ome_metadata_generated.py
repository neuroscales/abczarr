"""The generated OME metadata tree stays in sync with its template.

``v0_2``..``v0_5`` under ``abczarr.ome.metadata`` are generated from the
hand-written ``v0_1`` template by ``tools/gen_ome_metadata.py`` applying a
forward delta table. This runs the tool's ``--check`` mode and fails if any
committed file has drifted from what the template plus deltas would produce
(same classes, fields, requirement levels, ``register_subclass`` keys and
``__all__``).

``v0_6dev4`` is a hand-written preview and is not part of the generated tree;
the tool excludes it, and so does this test.

The generator uses ``ast.unparse`` and so needs Python >= 3.9; the check is
skipped on 3.8. The generated *outputs* remain valid on 3.8 -- that is covered
by the rest of the suite importing and exercising them.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = (
    Path(__file__).resolve().parent.parent / "tools" / "gen_ome_metadata.py"
)


def _load_tool() -> object:
    spec = importlib.util.spec_from_file_location("gen_ome_metadata", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="the generator needs ast.unparse (Python >= 3.9)",
)
@pytest.mark.skipif(
    not _TOOL.exists(),
    reason="generator tool not present (installed package, not a checkout)",
)
def test_generated_ome_tree_is_in_sync() -> None:
    tool = _load_tool()
    problems = tool.check()
    assert not problems, "OME metadata tree has drifted:\n\n" + "\n\n".join(
        problems
    )
