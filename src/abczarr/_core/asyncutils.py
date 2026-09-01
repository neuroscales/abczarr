# stdlib
import asyncio
from functools import partial
from itertools import starmap

# dependencies
import typing_extensions as tx

# typing
T = tx.TypeVar("T", bound=tx.Tuple[tx.Any, ...])
V = tx.TypeVar("V")


async def concurrent_map(
    items: tx.Iterable[T],
    func: tx.Callable[..., tx.Awaitable[V]],
    limit: tx.Optional[int] = None,
) -> tx.List[V]:
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


def get_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_event_loop()


async def run_in_loop(
    loop: asyncio.AbstractEventLoop,
    func: tx.Callable[..., V], *args: tx.Any, **kwargs: tx.Any
) -> V:
    return await loop.run_in_executor(None, func, *args, **kwargs)


async def run_sync(
    func: tx.Callable[..., V], *args: tx.Any, **kwargs: tx.Any
) -> V:
    """Run a synchronous function in an asynchronous context."""
    loop = asyncio.get_event_loop()
    return await run_in_loop(loop, func, *args, **kwargs)


def ensure_coroutine(
    fn: tx.Callable[..., tx.Any]
) -> tx.Callable[..., tx.Awaitable[tx.Any]]:
    """Ensure that a function is a coroutine function."""
    if asyncio.iscoroutinefunction(fn):
        return fn
    return partial(run_sync, fn)
