"""Errors raised by the abstract zarr surface."""

__all__ = [
    "UnsupportedZarrOperation",
    "TransactionConflict",
]

import typing_extensions as tx


class TransactionConflict(RuntimeError):
    """A transaction could not be committed because the store changed under it.

    A concurrent writer moved the store on (a versioned backend such as
    Icechunk, or an atomic commit a backend refused), so the transaction's
    view is stale. The operation did not apply; a caller can retry it against
    the current state.
    """


class UnsupportedZarrOperation(NotImplementedError):
    """A zarr operation a driver can neither perform nor synthesize.

    Raised when a member of the uniform surface cannot be delegated to the
    backend nor built from more primitive operations. The message names the
    operation and, when known, the driver -- so it points at what happened,
    never at an internal helper.

    Subclasses :class:`NotImplementedError`, so an existing
    ``except NotImplementedError`` still catches it.
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
