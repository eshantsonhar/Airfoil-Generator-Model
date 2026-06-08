"""
Runtime management: watchdogs, timeouts, and runtime snapshot helpers.
"""

from airfoil_discovery.runtime.snapshot import refresh_runtime_snapshot
from airfoil_discovery.runtime.watchdog import (
    ProcessWatchdog,
    SystemWatchdog,
    TimeoutError,
    WatchdogEvent,
    WatchdogResult,
    WatchdogStatus,
    WatchdogTimer,
    get_system_watchdog,
    run_with_timeout,
)

__all__ = [
    "WatchdogStatus",
    "WatchdogEvent",
    "WatchdogResult",
    "WatchdogTimer",
    "ProcessWatchdog",
    "SystemWatchdog",
    "TimeoutError",
    "get_system_watchdog",
    "run_with_timeout",
    "refresh_runtime_snapshot",
]
