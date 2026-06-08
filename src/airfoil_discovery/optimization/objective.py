import numpy as np
from typing import Dict, Any

class ConstrainedObjective:
    """
    Standard industrial ASO objective packaging.
    Formulates the problem as J = Cd / cd_ref subject to Cl = target and thickness >= min.
    """
    def __init__(self, target_cl: float = 0.6, min_thickness: float = 0.12):
        self.target_cl = target_cl
        self.min_thickness = min_thickness

    def package(self, 
                cd: float, 
                cl: float, 
                thickness: float, 
                grad_cd: np.ndarray, 
                grad_cl: np.ndarray, 
                grad_thickness: np.ndarray) -> Dict[str, Any]:
        """
        Translates physical values and sensitivities into optimizer format.
        Objective: Cd
        Constraints: [target_cl - cl, min_thickness - thickness] <= 0
        """
        # Objective
        f = cd
        df = grad_cd
        
        # Constraints (g <= 0)
        g = np.array([
            self.target_cl - cl,
            self.min_thickness - thickness
        ])
        
        # Constraint Gradients
        dg = np.vstack([
            -grad_cl,
            -grad_thickness
        ])
        
        return {
            "f": f,
            "df": df,
            "g": g,
            "dg": dg
        }
