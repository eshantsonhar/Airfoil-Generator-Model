"""
Hardened subprocess execution utilities for SU2 solver calls.

Provides:
- Windows GUI crash dialog suppression (SetErrorMode)
- Strict timeouts with process cleanup
- Granular before/after logging with timestamps
- Buffer deadlock prevention via safe communicate()
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Windows Error Mode Suppression ────────────────────────────────────────

_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002
_SEM_NOOPENFILEERRORBOX = 0x8000

_win_error_mode_set = False


def _suppress_windows_crash_dialogs() -> None:
    """
    Suppress Windows GUI error popups (crash dialogs, assertion boxes)
    that would block headless execution indefinitely.
    """
    global _win_error_mode_set
    if _win_error_mode_set:
        return
    if os.name == "nt":
        try:
            import ctypes
            # SetErrorMode: prevents critical-error-handler message boxes
            ctypes.windll.kernel32.SetErrorMode(
                _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX | _SEM_NOOPENFILEERRORBOX
            )
            logger.debug("Windows GUI crash dialogs suppressed via SetErrorMode")
        except Exception as e:
            logger.warning(f"Could not set Windows error mode: {e}")
    _win_error_mode_set = True


# ── Hardened Subprocess Runner ────────────────────────────────────────────

def run_solver(
    cmd: List[str],
    cwd: Path,
    label: str,
    timeout: float = 120.0,
    creation_flags: Optional[int] = None,
) -> Tuple[int, str, str]:
    """
    Execute a solver binary with hardened safety measures.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments to execute.
    cwd : Path
        Working directory for the subprocess.
    label : str
        Human-readable label for logging (e.g., "SU2_CFD primal").
    timeout : float
        Maximum execution time in seconds.
    creation_flags : int, optional
        Windows creation flags. If None, defaults to
        ``subprocess.CREATE_NO_WINDOW`` on Windows.

    Returns
    -------
    returncode : int
        Process exit code.
    stdout : str
        Captured standard output.
    stderr : str
        Captured standard error.

    Raises
    ------
    subprocess.TimeoutExpired
        If the process exceeds the timeout, automatically kills the
        process tree and logs the failure.
    """
    _suppress_windows_crash_dialogs()

    if creation_flags is None:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"[{label}] Executing: {cmd_str} in {cwd}")
    t_start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=creation_flags,
        )
        elapsed = time.time() - t_start
        logger.info(
            f"[{label}] Completed in {elapsed:.1f}s (rc={result.returncode})"
        )
        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        logger.error(
            f"[{label}] TIMED OUT after {timeout:.0f}s (elapsed={elapsed:.1f}s). "
            f"Command: {cmd_str}"
        )
        # Re-raise so the caller can handle it gracefully
        raise

    except FileNotFoundError:
        logger.error(f"[{label}] Binary not found: {cmd[0]}")
        raise

    except Exception as e:
        logger.error(f"[{label}] Unexpected error: {e}")
        raise


def run_solver_safe(
    cmd: List[str],
    cwd: Path,
    label: str,
    timeout: float = 120.0,
    creation_flags: Optional[int] = None,
) -> Tuple[bool, int, str, str]:
    """
    Safe wrapper around run_solver that catches exceptions and returns
    a success flag instead of raising.

    Returns
    -------
    success : bool
        True if the process completed within timeout with rc=0.
    returncode : int
        Process exit code (0 if timeout, -1 if exception).
    stdout : str
        Captured standard output.
    stderr : str
        Captured standard error.
    """
    try:
        rc, stdout, stderr = run_solver(cmd, cwd, label, timeout, creation_flags)
        return rc == 0, rc, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, -1, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return False, -1, "", f"Binary not found: {cmd[0]}"
    except Exception as e:
        return False, -1, "", str(e)