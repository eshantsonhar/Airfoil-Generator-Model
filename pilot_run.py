import os
import sys
import time
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, 'src')

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Runner, SU2Evaluator
from airfoil_discovery.aso.mesh_deform import deform_mesh
from airfoil_discovery.schemas import CandidateDesign, CSTParameters

print("=== PILOT RUN START ===")
settings = load_settings('config/default.yaml')

# We need to make sure binaries exist
su2_def_bin = settings.solver.su2_def_bin
su2_cfd_bin = settings.solver.su2_cfd_bin
gmsh_bin = settings.solver.gmsh_bin

print(f"SU2_DEF: {su2_def_bin}")
print(f"SU2_CFD: {su2_cfd_bin}")
print(f"GMSH: {gmsh_bin}")

work_dir = Path("data/pilot_run")
work_dir.mkdir(parents=True, exist_ok=True)

evaluator = SU2Evaluator(settings)
dv_baseline = np.zeros(12)
dv_baseline[0:4] = 0.13 
dv_baseline[4:8] = -0.13
dv_baseline[8] = 0.004
dv_new = np.zeros(12)
dv_new[0:4] = 0.14
dv_new[4:8] = -0.14
dv_new[8] = 0.004

print("\n--- Running Baseline Evaluation (GMSH + CFD) ---")
t0 = time.time()
res = evaluator.run_evaluation(dv_baseline, work_dir, mesh_level="L0", aoa=4.0, design_id="baseline")
print(f"Baseline finished in {time.time()-t0:.2f}s. Status: {res.status}")
print(f"Failure Stage: {res.failure_stage}")
print(f"Failure Reason: {res.failure_reason}")
print(f"Failure Detail: {res.failure_detail}")

# Check what files we got
mesh_path = work_dir / "airfoil.su2"
if not mesh_path.exists():
    print("Baseline mesh not found!")
    sys.exit(1)

print("\n--- Running Mesh Deformation (SU2_DEF) ---")
t1 = time.time()
deformed_mesh = deform_mesh(
    su2_def_bin=su2_def_bin,
    original_mesh_path=mesh_path,
    dv_old=dv_baseline,
    dv_new=dv_new,
    work_dir=work_dir / "def",
    marker="airfoil",
    n_iter_def=100
)
print(f"Deformation finished in {time.time()-t1:.2f}s. Output: {deformed_mesh}")

if deformed_mesh and deformed_mesh.exists():
    print("\n--- Running CFD on Deformed Mesh ---")
    t2 = time.time()
    # To run CFD on deformed mesh, we can manually copy it to a new dir and call run_su2_primal
    eval_dir = work_dir / "eval_deformed"
    eval_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(deformed_mesh, eval_dir / "airfoil.su2")
    
    # We write config manually
    from airfoil_discovery.cfd.su2_config import build_stage1_config
    params = CSTParameters(upper=dv_new[:4], lower=dv_new[4:8], trailing_edge_thickness=0.004)
    candidate = CandidateDesign(params=params, reynolds=settings.flow.reynolds_min)
    config_path = eval_dir / "config_primal.cfg"
    evaluator.runner._write_su2_config(candidate, eval_dir / "airfoil.su2", config_path, aoa=4.0, mesh_level="L0")
    
    evaluator.runner._run_su2_primal(config_path, eval_dir)
    history_path = eval_dir / "history.csv"
    cl, cd = evaluator.runner._read_results(history_path)
    print(f"CFD on deformed finished in {time.time()-t2:.2f}s. CL={cl:.4f}, CD={cd:.4f}")
else:
    print("Mesh deformation failed, skipping CFD on deformed mesh.")

print("=== PILOT RUN COMPLETE ===")
