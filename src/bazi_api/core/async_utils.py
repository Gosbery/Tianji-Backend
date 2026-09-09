from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_sync[**P, T](function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        # Cancelling an await does not stop its thread; drain it before releasing resources.
        while True:
            try:
                await asyncio.shield(operation)
                break
            except asyncio.CancelledError:
                if operation.done():
                    break
            except Exception:
                break
        raise
