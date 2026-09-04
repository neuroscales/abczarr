"""Offline JSON-schema validation of OME-Zarr metadata.

`abczarr.ome.schemas` validates an OME-NGFF document against the official
vendored schemas, offline. See
[the validation module][abczarr.ome.schemas] for the API:
[get_validator][abczarr.ome.schemas.get_validator],
[validate][abczarr.ome.schemas.validate], and
[documents][abczarr.ome.schemas.documents].
"""

__all__ = [
    "VERSIONS",
    "documents",
    "get_validator",
    "validate",
]

from ._validation import VERSIONS, documents, get_validator, validate
