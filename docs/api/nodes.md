# Nodes

Every object abczarr hands back, an array or a group, is a *node*.
[ZarrNode][abczarr.abc.sync.ZarrNode] is the common base: it carries
the store path, the Zarr format version, the metadata, and the user
attributes, and it is where the capability query lives.

[ZarrArray][abczarr.abc.sync.ZarrArray] is the n-dimensional node.
Read and write it like a NumPy array, with NumPy-style selections:

```python
array[0, :10]
array[...] = data
```

[ZarrGroup][abczarr.abc.sync.ZarrGroup] is the container node. It
holds other nodes by name, arrays and groups alike, and behaves like a
mapping: `group["images"]`, `group.keys()`, `"images" in group`.

Every node has a coroutine twin. Open one with `asynchronous=True` -- which
returns a coroutine you **await**, opening asynchronously -- or convert a node
you already hold with [as_async][abczarr.abc.sync.ZarrArray.as_async]; go back
with [as_sync][abczarr.abc.asynchronous.AsyncZarrNode.as_sync]. The async array
reads and writes through **methods**, not `[]` -- an assignment expression
cannot be awaited:

```python
array = await abczarr.open("data.zarr", mode="a", asynchronous=True)
block = await array.getitem((slice(0, 64), slice(0, 64)))
await array.setitem((slice(0, 64), slice(0, 64)), block * 2)
```

A backend that is natively async (tensorstore, zarr-python) awaits its own
futures; one that is not runs its blocking ops in a bounded thread pool.
`array.supports("async", native=True)` says which you got.

## `abczarr.abc.sync`

::: abczarr.abc.sync

## `abczarr.abc.asynchronous`

::: abczarr.abc.asynchronous
