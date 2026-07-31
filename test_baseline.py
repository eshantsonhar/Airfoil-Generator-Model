#!/usr/bin/env python3
"""
Test baseline CFD run with fixed mesh and synchronized config.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from airfoil_discovery.aso.config_primal import write_primal_config

# Generate config with fixed mesh
mesh_path = "data/mesh_fixed.su2"
config_path = Path("data/config_baseline_test.cfg")

write_primal_config(
    output_path=config_path,
    mesh_filename=mesh_path,
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,
    n_iter=500,
    cfl_initial=1.5,
    cfl_final=50.0,
    cfl_adapt=True,
    muscl=False,
    slope_limiter_flow="NONE",
    slope_limiter_turb="NONE",
    transition_model=True,
    turbulence_intensity=0.001,
    turb_viscosity_ratio=5.0,
)

print(f"Config generated at {config_path}")
print(f"Mesh: {mesh_path}")
