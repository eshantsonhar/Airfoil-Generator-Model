"""
Low-Fidelity Smoke Test Mode.

Provides the `--smoke-test` global flag capability that overrides
SU2 configuration to run a lightning-fast, end-to-end verification:

  - Primal iterations: 20 steps (instead of 2000-3000)
  - Adjoint iterations: 10 steps (instead of 300-500)
  - Convergence thresholds: loose (1e-3)
  - Mesh deformation: 50 iterations (instead of 500)

The smoke test executes a complete cycle:
  1. Primal CFD evaluation (RANS + transition)
  2. Adjoint sensitivity solve
  3. CST gradient projection
  4. Mesh deformation
  5. One optimizer step

This validates the entire technical chain in under 60 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SmokeTestOverrides:
    """Configuration overrides for smoke test mode."""
    n_iter_primal: int = 20
    n_iter_adjoint: int = 10
    cfl_primal: float = 5.0
    cfl_adjoint: float = 5.0
    convergence_tolerance: float = 1e-3
    n_iter_mesh_deform: int = 50
    max_iterations: int = 2          # only do 2 optimizer steps
    move_limit: float = 0.01         # small move limit for safety
    timeout_primal: float = 120.0    # 2 min max per primal
    timeout_adjoint: float = 60.0    # 1 min max per adjoint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primal_iters": self.n_iter_primal,
            "adjoint_iters": self.n_iter_adjoint,
            "cfl_primal": self.cfl_primal,
            "cfl_adjoint": self.cfl_adjoint,
            "convergence_tol": self.convergence_tolerance,
            "max_optimizer_iters": self.max_iterations,
        }


def get_smoke_overrides() -> SmokeTestOverrides:
    """Return the default smoke test overrides."""
    return SmokeTestOverrides()


def is_smoke_mode(args: Any) -> bool:
    """Check if smoke test flag is set in argparse namespace."""
    return getattr(args, "smoke_test", False)


def apply_smoke_overrides(
    config_kwargs: Dict[str, Any],
    overrides: Optional[SmokeTestOverrides] = None,
) -> Dict[str, Any]:
    """
    Apply smoke test overrides to a keyword arguments dictionary.

    Modifies config_kwargs in-place and returns it.

    Parameters
    ----------
    config_kwargs : dict
        Keyword arguments for PDEOptimizer, run_primal_and_adjoint, etc.
    overrides : SmokeTestOverrides, optional
        If None, uses defaults.

    Returns
    -------
    config_kwargs : dict
        Modified keyword arguments with reduced iteration counts.
    """
    if overrides is None:
        overrides = get_smoke_overrides()

    # Map optimizer kwargs
    if "n_iter_primal" in config_kwargs:
        config_kwargs["n_iter_primal"] = overrides.n_iter_primal
    if "n_iter_adjoint" in config_kwargs:
        config_kwargs["n_iter_adjoint"] = overrides.n_iter_adjoint
    if "cfl_primal" in config_kwargs:
        config_kwargs["cfl_primal"] = overrides.cfl_primal
    if "cfl_adjoint" in config_kwargs:
        config_kwargs["cfl_adjoint"] = overrides.cfl_adjoint
    if "convergence_tolerance" in config_kwargs:
        config_kwargs["convergence_tolerance"] = overrides.convergence_tolerance
    if "max_iterations" in config_kwargs:
        config_kwargs["max_iterations"] = overrides.max_iterations
    if "move_limit" in config_kwargs:
        config_kwargs["move_limit"] = overrides.move_limit

    # Map timeout kwargs for run_primal_and_adjoint
    if "timeout_primal" in config_kwargs:
        config_kwargs["timeout_primal"] = overrides.timeout_primal
    if "timeout_adjoint" in config_kwargs:
        config_kwargs["timeout_adjoint"] = overrides.timeout_adjoint

    # Mesh deformation
    if "n_iter_def" in config_kwargs:
        config_kwargs["n_iter_def"] = overrides.n_iter_mesh_deform

    return config_kwargs


def smoke_test_message() -> str:
    """
    Return a prominent warning message for the user.
    """
    return (
        "\n" + "!" * 70 + "\n"
        "!!! SMOKE TEST MODE ACTIVE !!!\n"
        "!!! Reduced iterations: primal=20, adjoint=10, max_opt=2\n"
        "!!! Results are NOT physically meaningful.\n"
        "!!! This only validates the execution pipeline.\n"
        + "!" * 70 + "\n"
    )