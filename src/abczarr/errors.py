"""The exceptions abczarr raises.

Each exception names the reason an operation failed, not only the
fact that it failed. Depending on the exception, this includes the
driver that lacks a capability, the field with no representation in a
target Zarr version, or the schema rule a document violated. Every
exception defined here is also reachable from the top level of the
`abczarr` package. `abczarr.UnsupportedZarrOperation` and
`abczarr.errors.UnsupportedZarrOperation` name the same class.
"""

__all__ = [
    "UnsupportedZarrOperation",
    "UnsupportedConversion",
    "TransactionConflict",
    "SchemaValidationError",
]

import typing_extensions as tx


class TransactionConflict(RuntimeError):
    """A transaction could not commit because the store moved on.

    Something else changed the store while the transaction was open.
    The cause may be another writer, a versioned backend that
    advanced, or an atomic commit the backend refused. The
    transaction's writes were not applied. The operation must be
    retried against the current state of the store.
    """


class UnsupportedZarrOperation(NotImplementedError):
    """An operation this driver can neither perform nor build itself.

    The message names the operation and, when known, the driver. It
    points at what happened rather than at an opaque backend error.

    This class subclasses `NotImplementedError`, so an existing
    `except NotImplementedError` still catches it.

    !!! example
        ```pycon
        >>> try:
        ...     raise UnsupportedZarrOperation(
        ...         "atomic transaction", driver="PathBasedStore"
        ...     )
        ... except UnsupportedZarrOperation as e:
        ...     print(e)
        the 'PathBasedStore' driver does not support 'atomic transaction'
        ```
    """

    def __init__(
        self, operation: str, driver: tx.Optional[str] = None
    ) -> None:
        if driver:
            message = f"the {driver!r} driver does not support {operation!r}"
        else:
            message = f"unsupported zarr operation: {operation!r}"
        super().__init__(message)
        self.operation = operation
        self.driver = driver


class UnsupportedConversion(ValueError):
    """A field has no representation in the target Zarr version.

    `to_version` raises this exception when it is asked to convert
    under the ``"strict"`` policy and a field cannot be carried over.
    The message names the field and the version it could not be
    represented in. An optional `hint` is appended when there is a
    concrete way to make the conversion succeed, for example an
    optional dependency that would supply the missing dtype.
    """

    def __init__(
        self, field: str, version: int, hint: tx.Optional[str] = None
    ) -> None:
        message = f"cannot represent {field!r} in Zarr v{version}"
        if hint:
            message = f"{message}; {hint}"
        super().__init__(message)
        self.field = field
        self.version = version
        self.hint = hint


class SchemaValidationError(ValueError):
    """A metadata document did not conform to its JSON schema.

    This exception is raised when a document is validated against an
    OME-NGFF or Zarr JSON schema and fails. The message names the
    schema, by version and document kind, and describes the first
    violation. `path` locates the offending value within the
    document.

    This class subclasses `ValueError`.
    """

    def __init__(
        self,
        message: str,
        schema: tx.Optional[str] = None,
        path: tx.Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.schema = schema
        self.path = path
