"""
Runtime serialization for reproducibility.

Implements serialization of runtime state for exact reproducibility
of optimization runs.
"""

from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class RuntimeState:
    """Complete runtime state for reproducibility."""
    
    # Timestamp
    timestamp: str
    
    # Seed information
    master_seed: int
    seed_state: Dict[str, Any]
    
    # Configuration hashes
    config_hash: str
    mesh_hash: Optional[str] = None
    
    # Solver information
    solver_version: Optional[str] = None
    solver_binary_hash: Optional[str] = None
    
    # Optimization state
    iteration: int = 0
    design_vector: Optional[list] = None
    objective_value: Optional[float] = None
    
    # Verification state
    verification_results: Optional[Dict[str, Any]] = None
    
    # Environment
    python_version: Optional[str] = None
    numpy_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Path):
        """
        Save runtime state to file.
        
        Args:
            filepath: Path to save state
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'RuntimeState':
        """
        Load runtime state from file.
        
        Args:
            filepath: Path to load state from
        
        Returns:
            RuntimeState object
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class RuntimeSerializer:
    """
    Serializes runtime state for reproducibility.
    
    Provides methods to save and load complete runtime state
    for exact reproducibility of optimization runs.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize runtime serializer.
        
        Args:
            output_dir: Directory for saving runtime states
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def serialize(
        self,
        master_seed: int,
        seed_state: Dict[str, Any],
        config_hash: str,
        iteration: int = 0,
        design_vector: Optional[list] = None,
        objective_value: Optional[float] = None,
        verification_results: Optional[Dict[str, Any]] = None,
        mesh_hash: Optional[str] = None,
        solver_version: Optional[str] = None,
        solver_binary_hash: Optional[str] = None,
    ) -> RuntimeState:
        """
        Serialize current runtime state.
        
        Args:
            master_seed: Master seed
            seed_state: Seed state dictionary
            config_hash: Configuration hash
            iteration: Current iteration
            design_vector: Current design vector
            objective_value: Current objective value
            verification_results: Verification results
            mesh_hash: Mesh hash
            solver_version: Solver version string
            solver_binary_hash: Solver binary hash
        
        Returns:
            RuntimeState object
        """
        import sys
        import numpy as np
        
        state = RuntimeState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            master_seed=master_seed,
            seed_state=seed_state,
            config_hash=config_hash,
            mesh_hash=mesh_hash,
            solver_version=solver_version,
            solver_binary_hash=solver_binary_hash,
            iteration=iteration,
            design_vector=design_vector,
            objective_value=objective_value,
            verification_results=verification_results,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            numpy_version=np.__version__,
        )
        
        return state
    
    def save_state(self, state: RuntimeState, iteration: int):
        """
        Save runtime state to file.
        
        Args:
            state: RuntimeState to save
            iteration: Iteration number for filename
        """
        filename = f"runtime_state_iter_{iteration:04d}.pkl"
        filepath = self.output_dir / filename
        state.save(filepath)
    
    def load_state(self, iteration: int) -> Optional[RuntimeState]:
        """
        Load runtime state from file.
        
        Args:
            iteration: Iteration number to load
        
        Returns:
            RuntimeState if exists, None otherwise
        """
        filename = f"runtime_state_iter_{iteration:04d}.pkl"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            return RuntimeState.load(filepath)
        
        return None
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "run_metadata.json"):
        """
        Save run metadata as JSON.
        
        Args:
            metadata: Metadata dictionary
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_metadata(self, filename: str = "run_metadata.json") -> Optional[Dict[str, Any]]:
        """
        Load run metadata from JSON.
        
        Args:
            filename: Metadata filename
        
        Returns:
            Metadata dictionary if exists, None otherwise
        """
        filepath = self.output_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
