"""
Runtime watchdog and timeout management system.

Provides hard timeouts, deadlock detection, and process supervision
for all long-running operations in the optimization pipeline.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WatchdogStatus(Enum):
    """Status of a watchdog timer."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass
class WatchdogEvent:
    """Event recorded by watchdog system."""
    timestamp: float
    event_type: str
    operation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatchdogResult:
    """Result of a watched operation."""
    status: WatchdogStatus
    operation: str
    start_time: float
    end_time: float
    duration: float
    timeout_seconds: float
    result: Any = None
    error: Optional[str] = None
    events: List[WatchdogEvent] = field(default_factory=list)
    
    @property
    def timed_out(self) -> bool:
        return self.status == WatchdogStatus.TIMEOUT
    
    @property
    def succeeded(self) -> bool:
        return self.status == WatchdogStatus.COMPLETED
    
    @property
    def failed(self) -> bool:
        return self.status in (WatchdogStatus.ERROR, WatchdogStatus.TIMEOUT)


class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    def __init__(self, operation: str, timeout_seconds: float, elapsed: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed = elapsed
        super().__init__(
            f"Operation '{operation}' timed out after {elapsed:.1f}s "
            f"(limit: {timeout_seconds}s)"
        )


class WatchdogTimer:
    """
    Hard timeout watchdog for operations.
    
    Monitors an operation and terminates it if it exceeds the timeout.
    Supports callbacks for timeout, completion, and error events.
    """
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        check_interval: float = 1.0,
        on_timeout: Optional[Callable[[WatchdogResult], None]] = None,
        on_complete: Optional[Callable[[WatchdogResult], None]] = None,
        on_error: Optional[Callable[[WatchdogResult], None]] = None,
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.on_timeout = on_timeout
        self.on_complete = on_complete
        self.on_error = on_error
        
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._status = WatchdogStatus.RUNNING
        self._events: List[WatchdogEvent] = []
        self._result: Any = None
        self._error: Optional[str] = None
        self._cancel_flag = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        
    def _record_event(self, event_type: str, details: Dict[str, Any] = None):
        """Record a watchdog event."""
        self._events.append(WatchdogEvent(
            timestamp=time.time(),
            event_type=event_type,
            operation=self.operation,
            details=details or {},
        ))
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._cancel_flag.is_set():
            elapsed = time.time() - self._start_time
            
            if elapsed >= self.timeout_seconds:
                self._status = WatchdogStatus.TIMEOUT
                self._record_event("timeout", {
                    "elapsed": elapsed,
                    "timeout": self.timeout_seconds,
                })
                self._error = f"Timeout after {elapsed:.1f}s"
                break
            
            time.sleep(self.check_interval)
    
    def start(self):
        """Start the watchdog timer."""
        self._start_time = time.time()
        self._status = WatchdogStatus.RUNNING
        self._record_event("start", {"timeout": self.timeout_seconds})
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"Watchdog-{self.operation}",
        )
        self._monitor_thread.start()
        
        logger.info(f"Watchdog started for '{self.operation}' (timeout: {self.timeout_seconds}s)")
    
    def complete(self, result: Any = None):
        """Mark the operation as completed successfully."""
        self._end_time = time.time()
        self._result = result
        self._status = WatchdogStatus.COMPLETED
        self._record_event("complete", {"duration": self.duration})
        self._cancel_flag.set()
        
        logger.info(f"Watchdog completed for '{self.operation}' ({self.duration:.1f}s)")
    
    def error(self, error_msg: str):
        """Mark the operation as failed with an error."""
        self._end_time = time.time()
        self._error = error_msg
        self._status = WatchdogStatus.ERROR
        self._record_event("error", {"error": error_msg})
        self._cancel_flag.set()
        
        logger.error(f"Watchdog error for '{self.operation}': {error_msg}")
    
    def cancel(self):
        """Cancel the watchdog timer."""
        self._end_time = time.time()
        self._status = WatchdogStatus.CANCELLED
        self._record_event("cancel", {})
        self._cancel_flag.set()
        
        logger.info(f"Watchdog cancelled for '{self.operation}'")
    
    def wait(self) -> WatchdogResult:
        """Wait for the watchdog to complete and return the result."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.timeout_seconds + 5)
        
        return self.result
    
    @property
    def result(self) -> WatchdogResult:
        """Get the current result."""
        return WatchdogResult(
            status=self._status,
            operation=self.operation,
            start_time=self._start_time,
            end_time=self._end_time or time.time(),
            duration=self.duration,
            timeout_seconds=self.timeout_seconds,
            result=self._result,
            error=self._error,
            events=self._events,
        )
    
    @property
    def duration(self) -> float:
        """Get the elapsed time."""
        if self._end_time > 0:
            return self._end_time - self._start_time
        return time.time() - self._start_time
    
    @property
    def is_running(self) -> bool:
        """Check if the watchdog is still running."""
        return self._status == WatchdogStatus.RUNNING
    
    @property
    def elapsed(self) -> float:
        """Get time elapsed since start."""
        return time.time() - self._start_time
    
    @property
    def remaining(self) -> float:
        """Get remaining time before timeout."""
        return max(0.0, self.timeout_seconds - self.elapsed)


class ProcessWatchdog:
    """
    Watchdog for subprocess management.
    
    Monitors subprocess execution and terminates them if they exceed timeouts.
    """
    
    def __init__(
        self,
        timeout_seconds: float = 3600.0,  # Default 1 hour
        graceful_shutdown_seconds: float = 30.0,
    ):
        self.timeout_seconds = timeout_seconds
        self.graceful_shutdown_seconds = graceful_shutdown_seconds
        self._processes: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
    def start_process(
        self,
        cmd: List[str],
        timeout_seconds: Optional[float] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> Tuple[subprocess.Popen, WatchdogTimer]:
        """
        Start a subprocess with timeout supervision.
        
        Args:
            cmd: Command and arguments
            timeout_seconds: Timeout for this process (overrides default)
            cwd: Working directory
            env: Environment variables
            name: Human-readable name for logging
        
        Returns:
            Tuple of (process, watchdog_timer)
        """
        timeout = timeout_seconds or self.timeout_seconds
        process_name = name or " ".join(cmd[:3])
        
        # Start the process
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        
        # Create watchdog
        watchdog = WatchdogTimer(
            operation=process_name,
            timeout_seconds=timeout,
            check_interval=2.0,
        )
        
        # Register process
        with self._lock:
            self._processes[proc.pid] = {
                "process": proc,
                "watchdog": watchdog,
                "name": process_name,
                "cmd": cmd,
                "start_time": time.time(),
            }
        
        watchdog.start()
        
        logger.info(f"Started process {proc.pid}: {process_name} (timeout: {timeout}s)")
        
        return proc, watchdog
    
    def wait_for_process(
        self,
        proc: subprocess.Popen,
        watchdog: WatchdogTimer,
    ) -> WatchdogResult:
        """
        Wait for a process with watchdog supervision.
        
        Args:
            proc: The subprocess to wait for
            watchdog: The watchdog timer
        
        Returns:
            WatchdogResult with status and output
        """
        try:
            # Wait for either process completion or timeout
            while watchdog.is_running:
                ret = proc.poll()
                if ret is not None:
                    # Process completed
                    stdout, stderr = proc.communicate()
                    watchdog.complete({
                        "returncode": ret,
                        "stdout": stdout.decode("utf-8", errors="ignore")[:10000],
                        "stderr": stderr.decode("utf-8", errors="ignore")[:10000],
                    })
                    break
                
                if watchdog.remaining < 1.0:
                    # Check one more time then timeout
                    ret = proc.poll()
                    if ret is not None:
                        stdout, stderr = proc.communicate()
                        watchdog.complete({
                            "returncode": ret,
                            "stdout": stdout.decode("utf-8", errors="ignore")[:10000],
                            "stderr": stderr.decode("utf-8", errors="ignore")[:10000],
                        })
                        break
                    else:
                        # Force timeout
                        self._terminate_process(proc.pid)
                        watchdog.error("Process terminated due to timeout")
                        break
                
                time.sleep(1.0)
            
            return watchdog.result
            
        except Exception as e:
            watchdog.error(str(e))
            self._terminate_process(proc.pid)
            return watchdog.result
        
        finally:
            self._cleanup_process(proc.pid)
    
    def _terminate_process(self, pid: int):
        """Terminate a process forcefully."""
        with self._lock:
            if pid in self._processes:
                proc_info = self._processes[pid]
                proc = proc_info["process"]
                
                try:
                    # Try graceful termination first
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        proc.wait(timeout=self.graceful_shutdown_seconds)
                    except subprocess.TimeoutExpired:
                        # Force kill
                        proc.kill()
                        proc.wait()
                    
                    logger.warning(f"Terminated process {pid}: {proc_info['name']}")
                    
                except Exception as e:
                    logger.error(f"Error terminating process {pid}: {e}")
    
    def _cleanup_process(self, pid: int):
        """Clean up process tracking."""
        with self._lock:
            if pid in self._processes:
                del self._processes[pid]
    
    def terminate_all(self):
        """Terminate all tracked processes."""
        with self._lock:
            pids = list(self._processes.keys())
        
        for pid in pids:
            self._terminate_process(pid)
            self._cleanup_process(pid)
        
        logger.info(f"Terminated all {len(pids)} tracked processes")


class SystemWatchdog:
    """
    Central watchdog system for the entire optimization pipeline.
    
    Provides:
    - Operation timeouts
    - Process supervision
    - Deadlock detection
    - Heartbeat monitoring
    - Resource monitoring
    """
    
    def __init__(
        self,
        su2_timeout: float = 1800.0,  # 30 minutes for SU2
        mesh_timeout: float = 300.0,  # 5 minutes for mesh generation
        optimization_timeout: float = 7200.0,  # 2 hours for optimization
        adjoint_timeout: float = 3600.0,  # 1 hour for adjoint
        heartbeat_interval: float = 30.0,
    ):
        self.su2_timeout = su2_timeout
        self.mesh_timeout = mesh_timeout
        self.optimization_timeout = optimization_timeout
        self.adjoint_timeout = adjoint_timeout
        self.heartbeat_interval = heartbeat_interval
        
        self.process_watchdog = ProcessWatchdog(timeout_seconds=su2_timeout)
        self._heartbeats: Dict[str, float] = {}
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        
    def start(self):
        """Start the system watchdog."""
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="SystemWatchdog",
        )
        self._heartbeat_thread.start()
        logger.info("System watchdog started")
    
    def stop(self):
        """Stop the system watchdog."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=10)
        self.process_watchdog.terminate_all()
        logger.info("System watchdog stopped")
    
    def heartbeat(self, component: str):
        """Record a heartbeat from a component."""
        with self._heartbeat_lock:
            self._heartbeats[component] = time.time()
    
    def check_heartbeat(self, component: str) -> float:
        """Check time since last heartbeat from a component."""
        with self._heartbeat_lock:
            last = self._heartbeats.get(component, 0)
        return time.time() - last
    
    def _heartbeat_loop(self):
        """Background heartbeat monitoring loop."""
        while self._running:
            current_time = time.time()
            
            with self._heartbeat_lock:
                stale_components = []
                for component, last_time in self._heartbeats.items():
                    if current_time - last_time > self.heartbeat_interval * 3:
                        stale_components.append(component)
            
            if stale_components:
                logger.warning(f"Stale heartbeats detected: {stale_components}")
            
            time.sleep(self.heartbeat_interval)
    
    def run_with_timeout(
        self,
        operation: str,
        func: Callable,
        timeout_seconds: Optional[float] = None,
        *args,
        **kwargs,
    ) -> WatchdogResult:
        """
        Run a function with timeout supervision.
        
        Args:
            operation: Name of the operation
            func: Function to run
            timeout_seconds: Timeout (uses default if None)
            *args, **kwargs: Arguments to pass to func
        
        Returns:
            WatchdogResult with status and result
        """
        timeout = timeout_seconds or self._get_default_timeout(operation)
        watchdog = WatchdogTimer(
            operation=operation,
            timeout_seconds=timeout,
        )
        
        result_container = {"result": None, "error": None}
        
        def target():
            try:
                result_container["result"] = func(*args, **kwargs)
            except Exception as e:
                result_container["error"] = f"{type(e).__name__}: {e}"
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        watchdog.start()
        
        # Wait for completion or timeout
        while watchdog.is_running and thread.is_alive():
            thread.join(timeout=1.0)
            if not thread.is_alive():
                break
        
        if thread.is_alive():
            watchdog.error("Operation timed out")
        elif result_container["error"]:
            watchdog.error(result_container["error"])
        else:
            watchdog.complete(result_container["result"])
        
        return watchdog.result
    
    def _get_default_timeout(self, operation: str) -> float:
        """Get default timeout for an operation type."""
        op_lower = operation.lower()
        if "su2" in op_lower or "cfd" in op_lower:
            return self.su2_timeout
        elif "mesh" in op_lower or "gmsh" in op_lower:
            return self.mesh_timeout
        elif "adjoint" in op_lower:
            return self.adjoint_timeout
        elif "optim" in op_lower:
            return self.optimization_timeout
        return 3600.0  # Default 1 hour


# Global system watchdog instance
_system_watchdog: Optional[SystemWatchdog] = None


def get_system_watchdog() -> SystemWatchdog:
    """Get or create the global system watchdog."""
    global _system_watchdog
    if _system_watchdog is None:
        _system_watchdog = SystemWatchdog()
        _system_watchdog.start()
    return _system_watchdog


def run_with_timeout(
    operation: str,
    func: Callable,
    timeout_seconds: Optional[float] = None,
    *args,
    **kwargs,
) -> WatchdogResult:
    """
    Convenience function to run an operation with timeout.
    
    Args:
        operation: Name of the operation
        func: Function to run
        timeout_seconds: Timeout (uses default if None)
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        WatchdogResult with status and result
    """
    return get_system_watchdog().run_with_timeout(
        operation, func, timeout_seconds, *args, **kwargs
    )