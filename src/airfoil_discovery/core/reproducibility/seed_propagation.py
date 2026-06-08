"""
Master seed propagation for deterministic reproducibility.

Implements master seed management and propagation to ensure
deterministic behavior across all random number generators.
"""

from __future__ import annotations

import random
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SeedState:
    """State of random number generators."""
    
    master_seed: int
    python_seed: Optional[int] = None
    numpy_seed: Optional[int] = None
    seed_history: Dict[str, int] = field(default_factory=dict)


class MasterSeedManager:
    """
    Manages master seed propagation for reproducibility.
    
    Ensures that all random number generators use seeds derived
    from a single master seed for exact reproducibility.
    """
    
    def __init__(self, master_seed: Optional[int] = None):
        """
        Initialize master seed manager.
        
        Args:
            master_seed: Master seed (None for random)
        """
        if master_seed is None:
            master_seed = random.randint(0, 2**31 - 1)
        
        self.master_seed = master_seed
        self.seed_counter = 0
        self.seed_state = SeedState(master_seed=master_seed)
        self.seed_registry: Dict[str, int] = {}
    
    def get_seed(self, component: str) -> int:
        """
        Get a deterministic seed for a component.
        
        Args:
            component: Component name (e.g., 'geometry', 'optimization')
        
        Returns:
            Deterministic seed for the component
        """
        if component in self.seed_registry:
            return self.seed_registry[component]
        
        # Generate deterministic seed from master seed and component name
        seed = hash(f"{self.master_seed}_{component}_{self.seed_counter}") % (2**31 - 1)
        self.seed_counter += 1
        self.seed_registry[component] = seed
        
        return seed
    
    def set_all_seeds(self):
        """Set seeds for all random number generators."""
        # Python random
        python_seed = self.get_seed('python_random')
        random.seed(python_seed)
        self.seed_state.python_seed = python_seed
        
        # NumPy
        numpy_seed = self.get_seed('numpy_random')
        np.random.seed(numpy_seed)
        self.seed_state.numpy_seed = numpy_seed
    
    def reset_seeds(self):
        """Reset all seeds to initial state."""
        self.seed_counter = 0
        self.seed_registry.clear()
        self.set_all_seeds()
    
    def get_state(self) -> SeedState:
        """
        Get current seed state.
        
        Returns:
            SeedState with current seed information
        """
        self.seed_state.seed_history = self.seed_registry.copy()
        return self.seed_state
    
    def restore_state(self, state: SeedState):
        """
        Restore seed state from saved state.
        
        Args:
            state: SeedState to restore
        """
        self.master_seed = state.master_seed
        self.seed_counter = len(state.seed_history)
        self.seed_registry = state.seed_history.copy()
        self.seed_state = state
        self.set_all_seeds()
