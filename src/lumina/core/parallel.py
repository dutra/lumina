"""Per-draw work of the output stage in a bounded pool of threads.

The output products evaluate every reporting draw independently (image solves, lens maps on grids).  Most of a draw's
time is spent inside JAX and NumPy calls that release the GIL, so a few threads overlap the draws.  Results come back
in draw order and are the same numbers as a plain loop: every draw seeds its own random numbers, and nothing a draw
computes depends on another draw.
"""
from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, TypeVar

from .utils import progress

T = TypeVar("T")
R = TypeVar("R")

_inside = threading.local()


def map_draws(function: Callable[[T], R], items: Iterable[T], *, workers: int = 1, desc: str = "", first: int = 0) -> list[R]:
    """``[function(item) for item in items]``, with up to ``workers`` threads.

    Parameters
    ----------
    function : callable
        Work of one item; must not depend on the other items.
    items : iterable
        The items, e.g. draw indices.
    workers : int, optional
        Threads (at least 1).  With 1, a single item, or when called from inside another ``map_draws``, it is a plain
        loop in the calling thread.  Default 1.
    desc : str, optional
        Label of the progress bar, which only the calling thread updates.
    first : int, optional
        Position of the item evaluated alone before the pool starts, so that the caches it fills (compiled
        functions, tables) are built once and not by several threads at the same time.  Default 0.

    Returns
    -------
    list
        The results, in the order of ``items``.

    Raises
    ------
    Exception
        The exception of the earliest item that failed; the items not yet started are cancelled.
    """
    items = list(items)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    results: list[Any] = [None] * len(items)
    bar = progress(total=len(items), desc=desc)
    try:
        if workers == 1 or len(items) <= 1 or getattr(_inside, "active", False):
            for i, item in enumerate(items):
                results[i] = function(item)
                bar.update()
            return results
        first = int(first) % len(items)
        results[first] = function(items[first])
        bar.update()

        def run(item):
            _inside.active = True
            try:
                return function(item)
            finally:
                _inside.active = False

        rest = [i for i in range(len(items)) if i != first]
        failed: dict[int, BaseException] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(rest)), thread_name_prefix="lumina-draw") as pool:
            pending = {pool.submit(run, items[i]): i for i in rest}
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        i = pending.pop(future)
                        if future.cancelled():
                            continue
                        error = future.exception()
                        if error is not None:
                            failed[i] = error
                            for other in pending:
                                other.cancel()
                        else:
                            results[i] = future.result()
                        bar.update()
            except BaseException:
                # An interrupt in the calling thread (Ctrl-C): drop the items not yet started, so the stage stops
                # after the draws already running, as the serial loop stops after its current one.
                for other in pending:
                    other.cancel()
                raise
        if failed:
            raise failed[min(failed)]
        return results
    finally:
        bar.close()
