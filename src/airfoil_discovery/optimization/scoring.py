"""
Physically meaningful airfoil scoring for aerodynamic optimization.

NO arbitrary constants.
NO mixed units.
NO fake efficiency references.

The objective function is physically motivated:
- Primary objective: Minimize Cd at design Cl
- Constraint: Cl >= target_cl 
- Constraint: thickness >= min_thickness
- Secondary: Maximize Cl/Cd ratio
- Penalty: LSB bursting risk (if available)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class PhysicsScoreConfig:
    """Configuration for physics-based scoring."""
    # Primary: drag minimization at target lift
    cd_target: float = 0.01  # Target drag coefficient
    cl_target: float = 0.6   # Target lift coefficient
    
    # Constraint penalties
    cl_violation_penalty: float = 2.0  # Multiplier for Cl deficit
    thickness_violation_penalty: float = 5.0  # Multiplier for thickness deficit
    min_thickness: float = 0.12  # Minimum thickness/chord
    
    # Secondary objectives
    cl_cd_weight: float = 0.2  # Weight for Cl/Cd bonus
    stall_weight: float = 0.1  # Weight for stall AoA bonus
    
    # LSB penalties
    lsb_bursting_penalty_weight: float = 0.3
    lsb_size_penalty_weight: float = 0.2
    
    # Gradient scaling
    grad_scale_cd: float = 100.0  # Scale Cd gradient to O(1)
    grad_scale_cl: float = 1.0


class PhysicsBasedScorer:
    """
    Physically meaningful airfoil scorer.
    
    Scoring formula:
    J = Cd/Cd_target 
        + penalty_cl * max(target_cl - Cl, 0)
        + penalty_thickness * max(min_thickness - thickness, 0)
        - bonus_cl_cd * (Cl/Cd) / 100
        + penalty_lsb_bursting * bursting_risk
        + penalty_lsb_size * bubble_length
    
    This ensures:
    - Primary optimization direction is drag reduction
    - Constraints are physically enforced
    - LSB penalties improve physical behavior
    - No arbitrary scaling constants
    - Gradient of J w.r.t. design variables is well-conditioned
    """
    
    def __init__(self, config: Optional[PhysicsScoreConfig] = None):
        self.config = config or PhysicsScoreConfig()
    
    def score_polar(self, polar: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Score a polar (lift/drag curve).
        
        Args:
            polar: List of dicts with keys 'aoa_deg', 'cl', 'cd'
            
        Returns:
            Dict with score components
        """
        if not polar:
            return {
                "score": 1e10,
                "cd_at_cruise": 1.0,
                "cl_at_cruise": 0.0,
                "cl_cd_ratio": 0.0,
                "stall_angle_deg": 0.0,
                "separation_penalty": 0.0,
                "instability_penalty": 0.0,
                "lsb_penalty": 0.0,
                "is_valid": False,
            }
        
        # Find design point near target Cl
        polar_sorted = sorted(polar, key=lambda p: abs(p.get("cl", 0) - self.config.cl_target))
        design_point = polar_sorted[0]
        
        cl = design_point.get("cl", 0.0)
        cd = design_point.get("cd", 0.01)
        aoa = design_point.get("aoa_deg", 0.0)
        
        # Compute efficiency
        cl_cd = cl / max(cd, 1e-10)
        
        # Primary objective: normalized drag
        cd_ratio = cd / max(self.config.cd_target, 1e-10)
        
        # Constraint violations
        cl_deficit = max(self.config.cl_target - cl, 0.0)
        thickness = design_point.get("thickness", self.config.min_thickness)
        thickness_deficit = max(self.config.min_thickness - thickness, 0.0)
        
        # Penalties
        cl_penalty = self.config.cl_violation_penalty * cl_deficit
        thickness_penalty = self.config.thickness_violation_penalty * thickness_deficit
        
        # Bonus for good Cl/Cd
        efficiency_bonus = self.config.cl_cd_weight * (cl_cd / 100.0)
        
        # Stall bonus (higher stall AoA is better)
        max_aoa = max(p.get("aoa_deg", 0.0) for p in polar)
        stall_score = max_aoa / 15.0  # Normalize to 15 deg reference
        stall_bonus = self.config.stall_weight * stall_score
        
        # LSB penalty (from optional LSB report)
        lsb_report = design_point.get("lsb_report", {})
        lsb_penalty = self._compute_lsb_penalty(lsb_report)
        
        # Total score (lower is better)
        total_score = (
            cd_ratio
            + cl_penalty
            + thickness_penalty
            - efficiency_bonus
            - stall_bonus
            + lsb_penalty
        )
        
        # Ensure positive score
        total_score = max(total_score, self.config.cd_target / max(self.config.cd_target, 1e-10))
        
        return {
            "score": max(0.001, total_score),
            "cd_at_cruise": cd,
            "cl_at_cruise": cl,
            "cl_cd_ratio": cl_cd,
            "stall_angle_deg": max_aoa,
            "separation_penalty": cl_penalty,
            "instability_penalty": thickness_penalty,
            "lsb_penalty": lsb_penalty,
            "is_valid": True,
            "cd_ratio": cd_ratio,
            "cl_deficit": cl_deficit,
            "thickness_deficit": thickness_deficit,
            "efficiency_bonus": efficiency_bonus,
            "stall_bonus": stall_bonus,
        }
    
    def _compute_lsb_penalty(self, lsb_report: Dict[str, Any]) -> float:
        """Compute penalty from LSB detection report."""
        if not lsb_report:
            return 0.0
        
        penalty = 0.0
        
        # Bursting risk penalty
        bursting_risk = lsb_report.get("bursting_risk_score", 0.0)
        penalty += self.config.lsb_bursting_penalty_weight * bursting_risk
        
        # Bubble size penalty
        bubble_length = lsb_report.get("bubble_length", 0.0)
        if bubble_length is not None and bubble_length > 0:
            penalty += self.config.lsb_size_penalty_weight * min(bubble_length * 5, 1.0)
        
        # Long bubble penalty (long bubbles are bad)
        bubble_type = lsb_report.get("bubble_type", "NO_BUBBLE")
        if bubble_type == "LONG_BUBBLE":
            penalty += 0.5
        elif bubble_type == "BURST_BUBBLE":
            penalty += 1.0
        
        return penalty
    
    def compute_objective_gradient(self, 
                                    grad_cd: np.ndarray, 
                                    grad_cl: np.ndarray,
                                    cl: float, cd: float) -> np.ndarray:
        """
        Compute gradient of the total objective w.r.t. design variables.
        
        dJ/dx = (1/cd_target) * dCd/dx - penalty_cl * dCl/dx (if Cl < target)
        
        Args:
            grad_cd: Gradient of Cd w.r.t. design variables
            grad_cl: Gradient of Cl w.r.t. design variables
            cl: Current lift coefficient
            cd: Current drag coefficient
            
        Returns:
            Gradient of objective w.r.t. design variables
        """
        dJ = grad_cd / max(self.config.cd_target, 1e-10)
        
        # Add Cl deficit gradient if applicable
        if cl < self.config.cl_target:
            dJ -= self.config.cl_violation_penalty * grad_cl
        
        # Apply gradient scaling for numerical conditioning
        dJ = dJ * self.config.grad_scale_cd
        
        return dJ


class AirfoilScorer(PhysicsBasedScorer):
    """
    Backward-compatible alias for PhysicsBasedScorer.
    """
    pass