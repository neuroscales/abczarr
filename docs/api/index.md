# Reference

The public API, grouped by what it does. New to abczarr? Start with the
[tutorial](../tutorial.md) instead.

- **API**: [`open`][abczarr.api.open] and its array/group variants, the
  entry point that picks a backend for a dataset and returns it wrapped.
- **ABC**: the abstract layer every backend implements. The
  [`ZarrArray`][abczarr.abc.sync.ZarrArray] and
  [`ZarrGroup`][abczarr.abc.sync.ZarrGroup] nodes and the
  [node base][abczarr.abc.sync.ZarrNode] they share, plus the key-to-bytes
  [`Store`][abczarr.abc.store.Store] beneath every node, its
  [transactions][abczarr.abc.transactions.Transaction], and the
  [store paths][abczarr.abc.path.StorePath] they address.
- **Capabilities**: the [capability query][abczarr.abc.capabilities.Support]
  for asking what a backend supports, and the
  [errors][abczarr.abc.errors.UnsupportedZarrOperation] raised when it does
  not.
- **Drivers**: how a [driver][abczarr.drivers.base.Driver] is chosen for an
  array, including the [registry][abczarr.api.registry] that picks one.
- **Metadata**: the typed, versioned
  [metadata model][abczarr.metadata.base.ArrayMetadata] and conversion
  between Zarr formats.
- **OME**: the typed [OME-Zarr metadata][abczarr.ome.metadata.base] model,
  for bioimaging data stored in Zarr.
