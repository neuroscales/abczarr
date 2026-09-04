"""Errors raised by the abczarr surface."""

__all__ = [
    "UnsupportedZarrOperation",
    "UnsupportedConversion",
    "TransactionConflict",
]

import typing_extensions as tx

# `UnsupportedConversion` is defined in `_core` so the metadata layer can
# import it at module top (see abczarr._core.errors); re-exported here to
# keep its public `abczarr.abc.errors` path.
from abczarr._core.errors import UnsupportedConversion


class TransactionConflict(RuntimeError):
    """A transaction could not commit because the store moved on.

    Something else changed the store while the transaction was open
    -- another writer, a versioned backend that advanced, or an
    atomic commit the backend refused. The transaction's writes were
    not applied; retry the operation against the current state.
    """


class UnsupportedZarrOperation(NotImplementedError):
    """An operation this driver can neither perform nor build itself.

    The message names the operation and, when known, the driver, so
    it points at what happened rather than at an opaque backend
    error.

    Subclasses `NotImplementedError`, so an existing
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
