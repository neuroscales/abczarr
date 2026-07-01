# stdlib
import asyncio
from itertools import starmap

# dependencies
import typing_extensions as tx

# typing
T = tx.TypeVar("T", bound=tuple[tx.Any, ...])
V = tx.TypeVar("V")


async def concurrent_map(
    items: tx.Iterable[T],
    func: tx.Callable[..., tx.Awaitable[V]],
    limit: int | None = None,
) -> list[V]:
    if limit is None:
        return await asyncio.gather(*list(starmap(func, items)))

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item: tx.Tuple[tx.Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[
            asyncio.ensure_future(run(item)) for item in items
        ])
