"""
CFD Execution State Machine with comprehensive supervision.

Implements a robust state machine for CFD execution that:
- Tracks execution state through well-defined transitions
- Provides heartbeat monitoring
- Detects divergence and corruption
- Handles subprocess supervision
- Archives partial results on failure
- Prevents silent failures

States:
- CREATED: Initial state, case not yet started
- VALIDATING: Pre-execution validation in progress
- MESHING: Mesh generation in progress
- RUNNING: CFD solver running
- MONITORING: Actively monitoring solver output
- CONVERGED: CFD converged successfully
- DIVERGED: CFD diverged
- TIMEOUT: Execution exceeded timeout
- INVALID: Invalid configuration or input
- TERMINATED: User-terminated
- ARCHIVED: Results archived and case closed

Illegal state transitions are rejected with detailed error logging.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


class CFDExecutionState(Enum):
    """States in the CFD execution lifecycle."""
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    MESHING = "MESHING"
    RUNNING = "RUNNING"
    MONITORING = "MONITORING"
    CONVERGED = "CONVERGED"
    DIVERGED = "DIVERGED"
    TIMEOUT = "TIMEOUT"
    INVALID = "INVALID"
    TERMINATED = "TERMINATED"
    ARCHIVED = "ARCHIVED"


# Define legal state transitions
LEGAL_TRANSITIONS = {
    CFDExecutionState.CREATED: {
        CFDExecutionState.VALIDATING,
        CFDExecutionState.INVALID,
        CFDExecutionState.TERMINATED,
    },
    CFDExecutionState.VALIDATING: {
        CFDExecutionState.MESHING,
        CFDExecutionState.INVALID,
        CFDExecutionState.TERMINATED,
    },
    CFDExecutionState.MESHING: {
        CFDExecutionState.RUNNING,
        CFDExecutionState.DIVERGED,
        CFDExecutionState.TIMEOUT,
        CFDExecutionState.INVALID,
        CFDExecutionState.TERMINATED,
    },
    CFDExecutionState.RUNNING: {
        CFDExecutionState.MONITORING,
        CFDExecutionState.CONVERGED,
        CFDExecutionState.DIVERGED,
        CFDExecutionState.TIMEOUT,
        CFDExecutionState.TERMINATED,
    },
    CFDExecutionState.MONITORING: {
        CFDExecutionState.CONVERGED,
        CFDExecutionState.DIVERGED,
        CFDExecutionState.TIMEOUT,
        CFDExecutionState.TERMINATED,
    },
    CFDExecutionState.CONVERGED: {
        CFDExecutionState.ARCHIVED,
    },
    CFDExecutionState.DIVERGED: {
        CFDExecutionState.ARCHIVED,
    },
    CFDExecutionState.TIMEOUT: {
        CFDExecutionState.ARCHIVED,
    },
    CFDExecutionState.INVALID: {
        CFDExecutionState.ARCHIVED,
    },
    CFDExecutionState.TERMINATED: {
        CFDExecutionState.ARCHIVED,
    },
    CFDExecutionState.ARCHIVED: set(),  # Terminal state
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: CFDExecutionState
    to_state: CFDExecutionState
    timestamp: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatEvent:
    """A heartbeat event from the solver."""
    timestamp: float
    iteration: int
    residual: float
    cl: float
    cd: float
    cfl: Optional[float] = None
    continuity: Optional[float] = None


@dataclass
class CFDExecutionSnapshot:
    """Snapshot of CFD execution state for recovery."""
    state: CFDExecutionState
    case_id: str
    start_time: float
    last_heartbeat: float
    current_iteration: int
    residuals: List[float]
    cl_history: List[float]
    cd_history: List[float]
    mesh_quality: Optional[Dict[str, float]]
    error_message: Optional[str]
    subprocess_pid: Optional[int]


class ConvergenceDetector:
    """
    Detects convergence, divergence, and false convergence in CFD solutions.
    """
    
    def __init__(
        self,
        residual_threshold: float = 1e-6,
        divergence_threshold: float = 1e6,
        stagnation_iterations: int = 100,
        oscillation_threshold: float = 0.01,
        min_iterations: int = 50,
    ):
        self.residual_threshold = residual_threshold
        self.divergence_threshold = divergence_threshold
        self.stagnation_iterations = stagnation_iterations
        self.oscillation_threshold = oscillation_threshold
        self.min_iterations = min_iterations
    
    def analyze(self, residuals: List[float], cl_history: List[float], cd_history: List[float]) -> Tuple[str, str]:
        """
        Analyze convergence status.
        
        Returns:
            (status, reason) tuple
        """
        if len(residuals) < self.min_iterations:
            return "running", "Insufficient iterations"
        
        residuals = np.array(residuals)
        current_residual = residuals[-1]
        
        # Check for divergence
        if current_residual > self.divergence_threshold:
            return "diverged", f"Residual {current_residual:.2e} exceeds divergence threshold"
        
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            return "diverged", "NaN or Inf in residuals"
        
        # Check for convergence
        if current_residual < self.residual_threshold:
            # Verify force convergence
            if len(cl_history) > 10 and len(cd_history) > 10:
                cl_std = np.std(cl_history[-10:])
                cd_std = np.std(cd_history[-10:])
                if cl_std < 0.001 and cd_std < 0.001:
                    return "converged", f"Residual {current_residual:.2e} below threshold, forces stable"
                else:
                    return "running", f"Residual converged but forces oscillating (Cl_std={cl_std:.4f}, Cd_std={cd_std:.4f})"
            return "converged", f"Residual {current_residual:.2e} below threshold"
        
        # Check for stagnation
        if len(residuals) > self.stagnation_iterations:
            recent = residuals[-self.stagnation_iterations:]
            relative_change = (np.max(recent) - np.min(recent)) / (np.abs(recent[0]) + 1e-15)
            if relative_change < 0.01:
                return "diverged", f"Residual stagnation detected (relative change {relative_change:.4f})"
        
        # Check for oscillation
        if len(residuals) > 20:
            recent = residuals[-20:]
            sign_changes = np.sum(np.diff(np.sign(np.diff(recent))) != 0)
            if sign_changes > 15:
                return "diverged", f"Oscillatory convergence (sign changes: {sign_changes})"
        
        return "running", f"Current residual: {current_residual:.2e}"


class CFDExecutionStateMachine:
    """
    State machine for CFD execution with comprehensive supervision.
    
    This class manages the entire lifecycle of a CFD execution:
    1. Validation of inputs
    2. Mesh generation
    3. Solver execution with real-time monitoring
    4. Convergence/divergence detection
    5. Result archival
    
    All state transitions are validated and logged.
    """
    
    def __init__(
        self,
        case_id: str,
        work_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.case_id = case_id
        self.work_dir = work_dir
        self.config = config or {}
        
        # State
        self._state = CFDExecutionState.CREATED
        self._transitions: List[StateTransition] = []
        self._start_time = 0.0
        self._last_heartbeat = 0.0
        self._current_iteration = 0
        self._residuals: List[float] = []
        self._cl_history: List[float] = []
        self._cd_history: List[float] = []
        self._error_message: Optional[str] = None
        self._subprocess: Optional[subprocess.Popen] = None
        self._subprocess_pid: Optional[int] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Convergence detection
        self.convergence_detector = ConvergenceDetector(
            residual_threshold=self.config.get("residual_threshold", 1e-6),
            divergence_threshold=self.config.get("divergence_threshold", 1e6),
            stagnation_iterations=self.config.get("stagnation_iterations", 100),
        )
        
        # Timeout configuration
        self.timeout_seconds = self.config.get("timeout_seconds", 3600)
        self.heartbeat_interval = self.config.get("heartbeat_interval", 5.0)
        self.heartbeat_timeout = self.config.get("heartbeat_timeout", 60.0)
        
        # Mesh quality tracking
        self.mesh_quality: Optional[Dict[str, float]] = None
        
        # Callbacks
        self.on_state_change: Optional[Callable[[CFDExecutionState, CFDExecutionState, str], None]] = None
        
        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot file for crash recovery
        self.snapshot_file = self.work_dir / "execution_snapshot.json"
        
        self._log_transition(CFDExecutionState.CREATED, "State machine initialized")
    
    @property
    def state(self) -> CFDExecutionState:
        """Current state."""
        return self._state
    
    @property
    def is_terminal(self) -> bool:
        """Whether the state machine is in a terminal state."""
        return self._state in {
            CFDExecutionState.CONVERGED,
            CFDExecutionState.DIVERGED,
            CFDExecutionState.TIMEOUT,
            CFDExecutionState.INVALID,
            CFDExecutionState.TERMINATED,
            CFDExecutionState.ARCHIVED,
        }
    
    @property
    def elapsed_time(self) -> float:
        """Time elapsed since start."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time
    
    @property
    def remaining_time(self) -> float:
        """Estimated remaining time."""
        return max(0.0, self.timeout_seconds - self.elapsed_time)
    
    def _log_transition(self, to_state: CFDExecutionState, reason: str, metadata: Dict[str, Any] = None):
        """Log a state transition."""
        from_state = self._state
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=time.time(),
            reason=reason,
            metadata=metadata or {},
        )
        self._transitions.append(transition)
        
        logger.info(f"[{self.case_id}] {from_state.value} -> {to_state.value}: {reason}")
        
        if self.on_state_change:
            self.on_state_change(from_state, to_state, reason)
    
    def transition_to(self, to_state: CFDExecutionState, reason: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            to_state: Target state
            reason: Reason for transition
            metadata: Additional metadata
        
        Returns:
            True if transition was successful, False if illegal
        """
        if to_state not in LEGAL_TRANSITIONS.get(self._state, set()):
            error_msg = (
                f"Illegal state transition: {self._state.value} -> {to_state.value}. "
                f"Legal transitions: {[s.value for s in LEGAL_TRANSITIONS.get(self._state, set())]}"
            )
            logger.error(f"[{self.case_id}] {error_msg}")
            self._error_message = error_msg
            return False
        
        self._log_transition(to_state, reason, metadata)
        self._state = to_state
        
        # Save snapshot on every transition
        self._save_snapshot()
        
        return True
    
    def validate_inputs(self, geometry_file: Path, config_file: Path) -> bool:
        """
        Validate inputs before execution.
        
        Args:
            geometry_file: Path to geometry file
            config_file: Path to SU2 config file
        
        Returns:
            True if validation passed
        """
        if not self.transition_to(CFDExecutionState.VALIDATING, "Starting input validation"):
            return False
        
        errors = []
        
        # Check geometry file
        if not geometry_file.exists():
            errors.append(f"Geometry file not found: {geometry_file}")
        elif geometry_file.stat().st_size == 0:
            errors.append(f"Geometry file is empty: {geometry_file}")
        
        # Check config file
        if not config_file.exists():
            errors.append(f"Config file not found: {config_file}")
        elif config_file.stat().st_size == 0:
            errors.append(f"Config file is empty: {config_file}")
        
        # Validate geometry coordinates
        if geometry_file.exists() and geometry_file.stat().st_size > 0:
            try:
                coords = np.loadtxt(geometry_file, skiprows=1)
                if coords.ndim != 2 or coords.shape[1] < 2:
                    errors.append(f"Invalid geometry format: expected (N, 2) or (N, 3), got {coords.shape}")
                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                    errors.append("Geometry contains NaN or Inf values")
            except Exception as e:
                errors.append(f"Failed to parse geometry file: {e}")
        
        # Validate config file syntax
        if config_file.exists() and config_file.stat().st_size > 0:
            try:
                content = config_file.read_text()
                # Check for required SU2 keywords
                required_keywords = ["MESH_FILENAME", "SOLVER"]
                for keyword in required_keywords:
                    if keyword not in content:
                        errors.append(f"Missing required config keyword: {keyword}")
            except Exception as e:
                errors.append(f"Failed to read config file: {e}")
        
        if errors:
            self._error_message = "; ".join(errors)
            self.transition_to(CFDExecutionState.INVALID, f"Validation failed: {self._error_message}")
            return False
        
        return self.transition_to(CFDExecutionState.MESHING, "Input validation passed")
    
    def run_meshing(
        self,
        gmsh_bin: str,
        geo_file: Path,
        mesh_file: Path,
        timeout: float = 300.0,
    ) -> bool:
        """
        Run mesh generation.
        
        Args:
            gmsh_bin: Path to GMSH binary
            config_file: Path to SU2 config file
            timeout: Mesh generation timeout
        
        Returns:
            True if meshing succeeded
        """
        if self._state != CFDExecutionState.MESHING:
            logger.error(f"[{self.case_id}] Cannot start meshing from state {self._state.value}")
            return False
        
        if not Path(gmsh_bin).exists():
            self._error_message = f"GMSH binary not found: {gmsh_bin}"
            self.transition_to(CFDExecutionState.INVALID, self._error_message)
            return False
        
        if not geo_file.exists():
            self._error_message = f"Geometry file not found: {geo_file}"
            self.transition_to(CFDExecutionState.INVALID, self._error_message)
            return False
        
        try:
            cmd = [
                gmsh_bin,
                str(geo_file),
                "-2",
                "-format", "su2",
                "-o", str(mesh_file),
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(geo_file.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            elapsed = time.time() - start_time
            
            # Save logs
            (self.work_dir / "gmsh_stdout.log").write_text(result.stdout)
            (self.work_dir / "gmsh_stderr.log").write_text(result.stderr)
            
            if result.returncode != 0:
                self._error_message = f"GMSH failed with return code {result.returncode}: {result.stderr[:500]}"
                self.transition_to(CFDExecutionState.DIVERGED, f"Mesh generation failed after {elapsed:.1f}s")
                return False
            
            if not mesh_file.exists():
                self._error_message = f"Mesh file not created: {mesh_file}"
                self.transition_to(CFDExecutionState.DIVERGED, self._error_message)
                return False
            
            # Validate mesh
            mesh_validation = self._validate_mesh(mesh_file)
            self.mesh_quality = mesh_validation
            
            if not mesh_validation.get("valid", False):
                self._error_message = f"Mesh validation failed: {mesh_validation.get('errors', [])}"
                self.transition_to(CFDExecutionState.DIVERGED, f"Invalid mesh: {self._error_message}")
                return False
            
            logger.info(f"[{self.case_id}] Mesh generated successfully in {elapsed:.1f}s "
                       f"({mesh_validation.get('nodes', 0)} nodes, {mesh_validation.get('elements', 0)} elements)")
            
            return self.transition_to(CFDExecutionState.RUNNING, f"Meshing completed, ready to run solver")
            
        except subprocess.TimeoutExpired:
            self._error_message = f"Mesh generation timed out after {timeout}s"
            self.transition_to(CFDExecutionState.TIMEOUT, self._error_message)
            return False
        except Exception as e:
            self._error_message = f"Mesh generation error: {e}"
            self.transition_to(CFDExecutionState.DIVERGED, self._error_message)
            return False
    
    def _validate_mesh(self, mesh_file: Path) -> Dict[str, Any]:
        """Validate mesh quality."""
        result = {"valid": False}
        
        try:
            # Read mesh file and extract basic statistics
            with open(mesh_file, 'r') as f:
                content = f.read()
            
            # Count nodes and elements (SU2 mesh format)
            lines = content.split('\n')
            n_points = 0
            n_elements = 0
            
            for i, line in enumerate(lines):
                if line.strip() == "NPOI=":
                    if i + 1 < len(lines):
                        n_points = int(lines[i + 1].strip())
                elif line.strip() in ["NELEM=", "NBOU="]:
                    if i + 1 < len(lines):
                        n_elements += int(lines[i + 1].strip())
            
            result["nodes"] = n_points
            result["elements"] = n_elements
            
            if n_points < 100:
                result["errors"] = [f"Too few nodes: {n_points}"]
                return result
            
            if n_elements < 50:
                result["errors"] = [f"Too few elements: {n_elements}"]
                return result
            
            result["valid"] = True
            
        except Exception as e:
            result["errors"] = [str(e)]
        
        return result
    
    def run_solver(
        self,
        su2_bin: str,
        config_file: Path,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Run SU2 solver with real-time monitoring.
        
        Args:
            su2_bin: Path to SU2 CFD binary
            config_file: Path to SU2 config file
            timeout: Solver timeout (overrides default)
        
        Returns:
            True if solver converged
        """
        if self._state not in {CFDExecutionState.RUNNING, CFDExecutionState.MONITORING}:
            logger.error(f"[{self.case_id}] Cannot start solver from state {self._state.value}")
            return False
        
        if not Path(su2_bin).exists():
            self._error_message = f"SU2 binary not found: {su2_bin}"
            self.transition_to(CFDExecutionState.INVALID, self._error_message)
            return False
        
        if not config_file.exists():
            self._error_message = f"Config file not found: {config_file}"
            self.transition_to(CFDExecutionState.INVALID, self._error_message)
            return False
        
        solver_timeout = timeout or self.timeout_seconds
        
        try:
            # Start solver process
            cmd = [su2_bin, str(config_file)]
            
            self._subprocess = subprocess.Popen(
                cmd,
                cwd=str(config_file.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            
            self._subprocess_pid = self._subprocess.pid
            self._start_time = time.time()
            
            logger.info(f"[{self.case_id}] Started SU2 solver (PID: {self._subprocess_pid})")
            
            # Start monitoring thread
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_solver,
                args=(solver_timeout,),
                daemon=True,
            )
            self._monitor_thread.start()
            
            # Wait for solver to complete
            stdout, stderr = self._subprocess.communicate(timeout=solver_timeout + 10)
            
            # Save logs
            (self.work_dir / "su2_stdout.log").write_text(stdout)
            (self.work_dir / "su2_stderr.log").write_text(stderr)
            
            # Wait for monitor thread
            self._stop_monitoring.set()
            if self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=10)
            
            # Determine final state
            if self._state in {CFDExecutionState.CONVERGED, CFDExecutionState.DIVERGED, CFDExecutionState.TIMEOUT}:
                # State was set by monitor
                pass
            elif self._subprocess.returncode != 0:
                self._error_message = f"SU2 crashed with return code {self._subprocess.returncode}: {stderr[:500]}"
                self.transition_to(CFDExecutionState.DIVERGED, self._error_message)
            else:
                # Check if results are valid
                if self._validate_results():
                    self.transition_to(CFDExecutionState.CONVERGED, "Solver completed successfully")
                else:
                    self._error_message = "Solver completed but results are invalid"
                    self.transition_to(CFDExecutionState.DIVERGED, self._error_message)
            
            return self._state == CFDExecutionState.CONVERGED
            
        except subprocess.TimeoutExpired:
            self._terminate_solver()
            self._error_message = f"Solver timed out after {solver_timeout}s"
            self.transition_to(CFDExecutionState.TIMEOUT, self._error_message)
            return False
        except Exception as e:
            self._error_message = f"Solver error: {e}"
            self.transition_to(CFDExecutionState.DIVERGED, self._error_message)
            return False
        finally:
            self._subprocess = None
            self._subprocess_pid = None
    
    def _monitor_solver(self, timeout: float):
        """Monitor solver execution in real-time."""
        history_file = self.work_dir / "history.csv"
        residual_file = self.work_dir / "residuals.log"
        last_heartbeat = time.time()
        
        while not self._stop_monitoring.is_set():
            # Check timeout
            if self.elapsed_time > timeout:
                self._error_message = f"Solver timed out after {timeout}s"
                self.transition_to(CFDExecutionState.TIMEOUT, self._error_message)
                break
            
            # Check if process is still running
            if self._subprocess and self._subprocess.poll() is not None:
                # Process completed
                break
            
            # Try to read history file for heartbeat
            if history_file.exists():
                try:
                    self._parse_history_file(history_file)
                    last_heartbeat = time.time()
                except Exception as e:
                    logger.warning(f"[{self.case_id}] Could not read solver heartbeat "
                                   f"from {history_file}: {e}")
            
            # Check for heartbeat timeout
            if time.time() - last_heartbeat > self.heartbeat_timeout:
                self._error_message = f"No solver heartbeat for {self.heartbeat_timeout}s"
                self.transition_to(CFDExecutionState.TIMEOUT, self._error_message)
                break
            
            # Check convergence
            if len(self._residuals) > 10:
                status, reason = self.convergence_detector.analyze(
                    self._residuals, self._cl_history, self._cd_history
                )
                
                if status == "converged":
                    self.transition_to(CFDExecutionState.CONVERGED, reason)
                    break
                elif status == "diverged":
                    self._error_message = reason
                    self.transition_to(CFDExecutionState.DIVERGED, reason)
                    break
            
            time.sleep(self.heartbeat_interval)
    
    def _parse_history_file(self, history_file: Path):
        """Parse SU2 history file for monitoring."""
        try:
            import pandas as pd
            df = pd.read_csv(history_file)
            
            if len(df) > 0:
                last_row = df.iloc[-1]
                
                # Extract iteration
                if 'ITER' in df.columns:
                    self._current_iteration = int(last_row['ITER'])
                
                # Extract residuals
                if 'RES_RMS' in df.columns:
                    self._residuals.append(float(last_row['RES_RMS']))
                
                # Extract forces
                if 'CL' in df.columns:
                    self._cl_history.append(float(last_row['CL']))
                if 'CD' in df.columns:
                    self._cd_history.append(float(last_row['CD']))
                
                self._last_heartbeat = time.time()
                
        except Exception as e:
            logger.debug(f"[{self.case_id}] Failed to parse history file: {e}")
    
    def _validate_results(self) -> bool:
        """Validate solver results."""
        # Check that we have results
        if not self._cl_history or not self._cd_history:
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(self._cl_history)) or np.any(np.isnan(self._cd_history)):
            return False
        if np.any(np.isinf(self._cl_history)) or np.any(np.isinf(self._cd_history)):
            return False
        
        # Check for reasonable values
        final_cl = self._cl_history[-1]
        final_cd = self._cd_history[-1]
        
        if abs(final_cl) > 10 or abs(final_cd) > 10:
            return False
        
        if final_cd < 0:
            return False
        
        return True
    
    def _terminate_solver(self):
        """Terminate the solver process."""
        if self._subprocess_pid:
            try:
                if os.name == "nt":
                    # Windows: use taskkill for process tree termination
                    killed = subprocess.run(
                        ["taskkill", "/PID", str(self._subprocess_pid), "/T", "/F"],
                        capture_output=True, text=True,
                    )
                    if killed.returncode != 0:
                        logger.error(f"[{self.case_id}] taskkill for PID {self._subprocess_pid} "
                                     f"returned rc={killed.returncode}: "
                                     f"{(killed.stderr or '').strip()[:300]}")
                else:
                    # Unix: send SIGTERM, then SIGKILL
                    try:
                        os.kill(self._subprocess_pid, signal.SIGTERM)
                        time.sleep(2)
                        os.kill(self._subprocess_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except Exception as e:
                logger.error(f"[{self.case_id}] Failed to terminate process {self._subprocess_pid}: {e}")
        
        self._subprocess = None
        self._subprocess_pid = None
    
    def terminate(self, reason: str = "User requested termination") -> bool:
        """Terminate the execution."""
        if self.is_terminal:
            return False
        
        self._terminate_solver()
        self._stop_monitoring.set()
        
        return self.transition_to(CFDExecutionState.TERMINATED, reason)
    
    def archive(self) -> bool:
        """Archive the execution results."""
        if self._state not in {
            CFDExecutionState.CONVERGED,
            CFDExecutionState.DIVERGED,
            CFDExecutionState.TIMEOUT,
            CFDExecutionState.INVALID,
            CFDExecutionState.TERMINATED,
        }:
            logger.error(f"[{self.case_id}] Cannot archive from state {self._state.value}")
            return False
        
        # Create archive directory
        archive_dir = self.work_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Save execution report
        report = {
            "case_id": self.case_id,
            "final_state": self._state.value,
            "start_time": self._start_time,
            "elapsed_time": self.elapsed_time,
            "total_iterations": self._current_iteration,
            "final_residual": self._residuals[-1] if self._residuals else None,
            "final_cl": self._cl_history[-1] if self._cl_history else None,
            "final_cd": self._cd_history[-1] if self._cd_history else None,
            "mesh_quality": self.mesh_quality,
            "error_message": self._error_message,
            "transitions": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "timestamp": t.timestamp,
                    "reason": t.reason,
                }
                for t in self._transitions
            ],
            "residual_history": self._residuals[-100:],  # Last 100 values
            "cl_history": self._cl_history[-100:],
            "cd_history": self._cd_history[-100:],
        }
        
        report_file = archive_dir / "execution_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        
        # Save snapshot
        self._save_snapshot()
        
        return self.transition_to(CFDExecutionState.ARCHIVED, "Results archived")
    
    def _save_snapshot(self):
        """Save execution snapshot for crash recovery."""
        snapshot = CFDExecutionSnapshot(
            state=self._state,
            case_id=self.case_id,
            start_time=self._start_time,
            last_heartbeat=self._last_heartbeat,
            current_iteration=self._current_iteration,
            residuals=self._residuals[-100:],
            cl_history=self._cl_history[-100:],
            cd_history=self._cd_history[-100:],
            mesh_quality=self.mesh_quality,
            error_message=self._error_message,
            subprocess_pid=self._subprocess_pid,
        )
        
        snapshot_dict = {
            "state": snapshot.state.value,
            "case_id": snapshot.case_id,
            "start_time": snapshot.start_time,
            "last_heartbeat": snapshot.last_heartbeat,
            "current_iteration": snapshot.current_iteration,
            "residuals": snapshot.residuals,
            "cl_history": snapshot.cl_history,
            "cd_history": snapshot.cd_history,
            "mesh_quality": snapshot.mesh_quality,
            "error_message": snapshot.error_message,
            "subprocess_pid": snapshot.subprocess_pid,
            "timestamp": time.time(),
        }
        
        try:
            self.snapshot_file.write_text(json.dumps(snapshot_dict, indent=2))
        except Exception as e:
            logger.error(f"[{self.case_id}] Failed to save snapshot: {e}")
    
    def load_snapshot(self) -> bool:
        """Load execution snapshot for recovery."""
        if not self.snapshot_file.exists():
            return False
        
        try:
            snapshot_dict = json.loads(self.snapshot_file.read_text())
            
            self._state = CFDExecutionState(snapshot_dict["state"])
            self._start_time = snapshot_dict.get("start_time", 0)
            self._last_heartbeat = snapshot_dict.get("last_heartbeat", 0)
            self._current_iteration = snapshot_dict.get("current_iteration", 0)
            self._residuals = snapshot_dict.get("residuals", [])
            self._cl_history = snapshot_dict.get("cl_history", [])
            self._cd_history = snapshot_dict.get("cd_history", [])
            self.mesh_quality = snapshot_dict.get("mesh_quality")
            self._error_message = snapshot_dict.get("error_message")
            
            logger.info(f"[{self.case_id}] Loaded snapshot: state={self._state.value}, iter={self._current_iteration}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.case_id}] Failed to load snapshot: {e}")
            return False
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "case_id": self.case_id,
            "state": self._state.value,
            "is_terminal": self.is_terminal,
            "elapsed_time": self.elapsed_time,
            "remaining_time": self.remaining_time,
            "current_iteration": self._current_iteration,
            "residuals": {
                "count": len(self._residuals),
                "final": self._residuals[-1] if self._residuals else None,
                "min": min(self._residuals) if self._residuals else None,
                "max": max(self._residuals) if self._residuals else None,
            },
            "forces": {
                "final_cl": self._cl_history[-1] if self._cl_history else None,
                "final_cd": self._cd_history[-1] if self._cd_history else None,
                "cl_range": (min(self._cl_history), max(self._cl_history)) if self._cl_history else None,
                "cd_range": (min(self._cd_history), max(self._cd_history)) if self._cd_history else None,
            },
            "mesh_quality": self.mesh_quality,
            "error_message": self._error_message,
            "transitions": len(self._transitions),
            "subprocess_pid": self._subprocess_pid,
        }


class CFDExecutionManager:
    """
    Manages multiple CFD executions with centralized supervision.
    """
    
    def __init__(self, base_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.base_dir = base_dir
        self.config = config or {}
        self.executions: Dict[str, CFDExecutionStateMachine] = {}
        self._lock = threading.Lock()
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_execution(
        self,
        case_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> CFDExecutionStateMachine:
        """Create a new CFD execution."""
        with self._lock:
            work_dir = self.base_dir / case_id
            execution = CFDExecutionStateMachine(case_id, work_dir, config or self.config)
            self.executions[case_id] = execution
            return execution
    
    def get_execution(self, case_id: str) -> Optional[CFDExecutionStateMachine]:
        """Get an existing execution."""
        return self.executions.get(case_id)
    
    def terminate_all(self, reason: str = "Manager shutdown") -> int:
        """Terminate all active executions."""
        count = 0
        with self._lock:
            for execution in self.executions.values():
                if not execution.is_terminal:
                    execution.terminate(reason)
                    count += 1
        return count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        summary = {
            "total": len(self.executions),
            "states": {},
            "executions": {},
        }
        
        for case_id, execution in self.executions.items():
            state = execution.state.value
            summary["states"][state] = summary["states"].get(state, 0) + 1
            summary["executions"][case_id] = execution.get_diagnostics()
        
        return summary