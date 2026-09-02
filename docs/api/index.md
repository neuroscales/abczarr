# API reference

The public API, grouped by what it does.

- **Opening** — [`open`][abczarr.api.open] and its array/group variants, the
  entry point that picks a backend for a dataset and returns it wrapped.
- **Nodes** — [`ZarrArray`][abczarr.abc.array.ZarrArray] and
  [`ZarrGroup`][abczarr.abc.group.ZarrGroup], the typed arrays and groups
  you read and write, and the [node base][abczarr.abc.node.ZarrNode] they
  share.
- **Stores** — the key-to-bytes [`Store`][abczarr.abc.store.Store] beneath
  every node, its [transactions][abczarr.abc.transactions.Transaction], and
  the [store paths][abczarr.abc.path.StorePath] they address.
- **Capabilities** — the [capability query][abczarr.abc.capabilities.Support]
  for asking what a backend supports, and the
  [errors][abczarr.abc.errors.UnsupportedZarrOperation] raised when it does
  not.
- **Drivers** — how a [driver][abczarr.drivers.base.Driver] is chosen for an
  array.
- **Metadata** — the typed, versioned
  [metadata model][abczarr.metadata.base.ArrayMetadata] and conversion
  between Zarr formats.
