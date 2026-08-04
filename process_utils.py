from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any


async def terminate_process_tree(process: Any, timeout: float = 10.0) -> None:
    """Terminate one owned process and its descendants without matching by name."""
    if not process or process.poll() is not None:
        return
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=creationflags,
        )
        await killer.wait()
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await asyncio.to_thread(process.wait)
        return
    process.terminate()
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await asyncio.to_thread(process.wait)
