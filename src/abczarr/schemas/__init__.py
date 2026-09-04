"""Offline JSON-schema validation of Zarr array/group metadata.

`abczarr.schemas` validates a Zarr `zarr.json`/`.zarray`/`.zgroup` document
against the version's JSON schema, offline. See
[the validation module][abczarr.schemas] for the API:
[get_validator][abczarr.schemas.get_validator],
[validate][abczarr.schemas.validate], and
[documents][abczarr.schemas.documents].
"""

__all__ = [
    "VERSIONS",
    "documents",
    "get_validator",
    "validate",
]

from ._validation import VERSIONS, documents, get_validator, validate
