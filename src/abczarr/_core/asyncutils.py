"""Small async helpers: run a blocking call in a bounded pool, and fan a
batch of coroutines out with a concurrency cap.

abczarr synthesizes the async-from-sync direction only (a synchronous
backend run in a worker thread). The thread pool here is a dedicated,
bounded one -- not asyncio's default executor -- so a burst of chunk I/O
cannot starve the interpreter's shared pool, and the fan-out helper carries
a default concurrency limit so a wide ``gather`` does not open thousands of
threads or backend connections at once.
"""

__all__ = [
    "DEFAULT_CONCURRENCY",
    "run_sync",
    "concurrent_map",
    "ensure_coroutine",
]

# stdlib
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# dependencies
import typing_extensions as tx

# typing
T = tx.TypeVar("T", bound=tx.Tuple[tx.Any, ...])
V = tx.TypeVar("V")

#: The default fan-out width, matching the thread pool's size, so a batch
#: run through :func:`concurrent_map` never wants more workers than there
#: are threads to serve it.
DEFAULT_CONCURRENCY = min(32, (os.cpu_count() or 1) + 4)

#: The dedicated executor, built on first use. Kept apart from asyncio's
#: default (``None``) pool so abczarr's blocking chunk I/O never contends
#: with whatever else a caller's event loop offloads.
_executor = None  # type: tx.Optional[ThreadPoolExecutor]


def _thread_pool() -> ThreadPoolExecutor:
    """The shared bounded thread pool, created on first use."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=DEFAULT_CONCURRENCY,
            thread_name_prefix="abczarr-async",
        )
    return _executor


async def run_sync(
    func: tx.Callable[..., V], *args: tx.Any, **kwargs: tx.Any
) -> V:
    """Run the blocking *func* in the dedicated thread pool and await it.

    A cancelled ``await`` cannot interrupt the running thread, so a write
    already handed to the backend may still land even when the awaiting task
    is cancelled.
    """
    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)
    return await loop.run_in_executor(_thread_pool(), call)


async def concurrent_map(
    items: tx.Iterable[T],
    func: tx.Callable[..., tx.Awaitable[V]],
    limit: tx.Optional[int] = DEFAULT_CONCURRENCY,
) -> tx.List[V]:
    """Await *func* over each of *items*, at most *limit* at a time.

    Each item is a tuple of positional arguments for *func*. Results come
    back in the order of *items*. A *limit* of ``None`` runs them all at
    once (unbounded); the default caps the fan-out so a wide batch does not
    open more connections or threads than the pool can serve.
    """
    if limit is None:
        return await asyncio.gather(*[func(*item) for item in items])

    sem = asyncio.Semaphore(limit)

    async def run(item: T) -> V:
        async with sem:
            return await func(*item)

    return await asyncio.gather(*[run(item) for item in items])


def ensure_coroutine(
    fn: tx.Callable[..., tx.Any]
) -> tx.Callable[..., tx.Awaitable[tx.Any]]:
    """Adapt *fn* to a coroutine function.

    A coroutine function is returned unchanged; a plain callable is wrapped
    so calling it runs it in the thread pool and returns an awaitable.
    """
    if asyncio.iscoroutinefunction(fn):
        return fn
    return partial(run_sync, fn)
