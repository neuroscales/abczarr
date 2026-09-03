# Nodes

Every object abczarr hands back, an array or a group, is a *node*.
[ZarrNode][abczarr.abc.node.ZarrNode] is the common base: it carries
the store path, the Zarr format version, the metadata, and the user
attributes, and it is where the capability query lives.

[ZarrArray][abczarr.abc.array.ZarrArray] is the n-dimensional node.
Read and write it like a NumPy array, with NumPy-style selections:

```python
array[0, :10]
array[...] = data
```

[ZarrGroup][abczarr.abc.group.ZarrGroup] is the container node. It
holds other nodes by name, arrays and groups alike, and behaves like a
mapping: `group["images"]`, `group.keys()`, `"images" in group`.

Every node has a coroutine twin. Open one with `asynchronous=True`, or
convert a node you already hold with
[as_async][abczarr.abc.array.ZarrArray.as_async]; go back with
[as_sync][abczarr.abc.asyncnode.AsyncZarrArray.as_sync]. The async array
reads and writes through **methods**, not `[]` -- an assignment expression
cannot be awaited:

```python
array = abczarr.open("data.zarr", mode="a", asynchronous=True)
block = await array.getitem((slice(0, 64), slice(0, 64)))
await array.setitem((slice(0, 64), slice(0, 64)), block * 2)
```

A backend that is natively async (tensorstore, zarr-python) awaits its own
futures; one that is not runs its blocking ops in a bounded thread pool.
`array.supports("async", native=True)` says which you got.

## `abczarr.abc.node`

::: abczarr.abc.node

## `abczarr.abc.array`

::: abczarr.abc.array

## `abczarr.abc.group`

::: abczarr.abc.group

## `abczarr.abc.asyncnode`

::: abczarr.abc.asyncnode
