"""Shared test fixtures."""

import pytest
import typing_extensions as tx

from abczarr.ome import schemas

SystemsAndTransformsValidator = tx.Callable[[dict, str], None]


def _validate_systems_and_transforms(doc: dict, version: str) -> None:
    """Validate a ``{coordinateSystems, coordinateTransformations}`` document.

    dev1/dev2 ship a single ``coordinate_systems_and_transforms`` schema for
    exactly this shape. Later versions moved it into ``scene`` (where a
    transform's ``input``/``output`` may name-reference a declared coordinate
    system), so the document is validated as a scene body -- the official
    in-context route. ``version`` is the official NGFF string (e.g.
    ``"0.6rc0"``, ``"0.6.dev1"``).
    """
    docs = schemas.documents(version)
    if "coordinate_systems_and_transforms" in docs:
        schemas.validate(doc, version, "coordinate_systems_and_transforms")
    elif "scene" in docs:
        wrapped = {"ome": {"version": version, "scene": doc}}
        schemas.validate(wrapped, version, "scene")
    else:
        schemas.validate(
            doc["coordinateSystems"], version, "coordinate_systems"
        )
        schemas.validate(
            doc["coordinateTransformations"], version,
            "coordinate_transformations",
        )


@pytest.fixture
def validate_systems_and_transforms() -> SystemsAndTransformsValidator:
    """A callable ``(doc, version) -> None`` that validates a standalone
    coordinate systems + transformations document against the official schema.
    """
    return _validate_systems_and_transforms
