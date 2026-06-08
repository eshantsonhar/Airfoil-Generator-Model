import numpy as np
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from airfoil_discovery.cfd.su2 import SU2Status

@dataclass
class CFDResult:
    state: 'CFDState'
    cd: float
    cl: float
    gradient_objective: np.ndarray
    mesh_level: int

class CFDState(Enum):
    CONVERGED = 1
    PARTIAL = 2
    DIVERGED = 3
    ADJOINT_INVALID = 4

class FidelityController:
    """
    Manages multi-fidelity hierarchy and consistency audits.
    """
    def __init__(self, threshold: float = 0.01):
        self.fidelity_threshold = threshold
        
    def audit(self, result_l1: CFDResult, result_l2: CFDResult) -> bool:
        """Consistency audit between L1 and L2."""
        diff = abs(result_l2.cd - result_l1.cd)
        return diff < self.fidelity_threshold
    
    def check_gradients(self, grad_l1: np.ndarray, grad_l2: np.ndarray) -> float:
        """Cosine similarity of gradients."""
        norm1 = np.linalg.norm(grad_l1)
        norm2 = np.linalg.norm(grad_l2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(grad_l1, grad_l2) / (norm1 * norm2)

class FidelityIntegrator:
    """
    Connects FidelityController to the actual CFD execution pipe.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def execute_with_audit(self, x: np.ndarray, level: str) -> CFDResult:
        eval_res = self.evaluator.run_evaluation(x, Path("./temp_case"), level)

        # State mapping
        if eval_res.status == SU2Status.OK:
            state = CFDState.CONVERGED
        elif eval_res.status == SU2Status.ADJOINT_INVALID:
            state = CFDState.ADJOINT_INVALID
        else:
            state = CFDState.DIVERGED

        return CFDResult(
            state=state,
            cd=eval_res.cd,
            cl=eval_res.cl,
            gradient_objective=eval_res.adjoint.grad_cd if eval_res.adjoint else np.zeros_like(x),
            mesh_level=1 if level == "L1" else 2
        )

