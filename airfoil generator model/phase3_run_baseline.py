"""Phase 3: Validate mesh and run baseline primal CFD"""
import sys, subprocess, os
from pathlib import Path
import numpy as np

MESH_PATH = Path("data/cache/final_test/airfoil.su2")
SU2_CFD = "bin/SU2_CFD.exe"
CASE_DIR = Path("data/cache/final_test")

assert MESH_PATH.exists(), f"Mesh not found at {MESH_PATH.resolve()}"

# 1. Validate mesh (standard SU2 format: NDIME, NPOIN, nodes, NELEM, elems, NMARK)
print("VALIDATING MESH...")
lines = MESH_PATH.read_text().splitlines()

ne_block = next(i for i, l in enumerate(lines) if l.strip().startswith("NELEM"))
npoin = int(lines[1].split("=")[1].strip())
nelem = int(lines[ne_block].split("=")[1].strip())
nmarker_line = next(i for i in range(ne_block + 1 + nelem, len(lines)) if lines[i].strip().startswith("NMARK"))
nmarker = int(lines[nmarker_line].strip().split("=")[1])

node_start = ne_block + 1
pts = np.array([l.strip().split()[:2] for l in lines[node_start:ne_block + 1 + nelem]], dtype=float)

idx = nmarker_line + 1
markers = {}
for m in range(nmarker):
    name = lines[idx].strip()
    n_e = int(lines[idx + 1].strip())
    markers[name] = n_e
    idx += 2 + n_e

x_min, x_max = pts[:,0].min(), pts[:,0].max()
print(f"Nodes: {npoin}, Elems: {nelem}, Markers: {nmarker}")
for k,v in markers.items(): print(f"  {k}: {v}")
print(f"Chord: {x_max-x_min:.4f}m, X: [{x_min:.1f},{x_max:.1f}]")
assert npoin > 10000
assert markers.get('airfoil',0) > 300

# 2. Generate config
print("\nGENERATING CONFIG...")
sys.path.insert(0, "src")
from airfoil_discovery.aso.config_primal import write_primal_config
cfg = CASE_DIR / "config_primal.cfg"
write_primal_config(output_path=cfg, mesh_filename="airfoil.su2", aoa_deg=4.0,
    reynolds=1e5, mach=0.1, n_iter=5000, cfl_initial=0.5, cfl_final=3.0,
    transition_model=True, turbulence_intensity=0.001, turb_viscosity_ratio=5.0)
print(f"Config: {cfg}")

# 3. Run CFD
print(f"\nRUNNING: {SU2_CFD} {cfg.name}")
flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
result = subprocess.run([SU2_CFD, cfg.name], cwd=str(CASE_DIR),
    capture_output=True, text=True, timeout=3600, creationflags=flags)
(CASE_DIR/"su2_stdout.log").write_text(result.stdout)
(CASE_DIR/"su2_stderr.log").write_text(result.stderr)
print(f"Exit code: {result.returncode}")

# 4. Parse results
hist = CASE_DIR / "history.csv"
if hist.exists():
    hl = hist.read_text().splitlines()
    hdr = [h.strip().strip('"') for h in hl[0].split(",")]
    last = [v.strip() for v in hl[-1].split(",")]
    if len(last) == len(hdr):
        d = dict(zip(hdr, last))
        cl = float(d.get("CL","0")); cd = float(d.get("CD","0"))
        rms = {k: float(d[k]) for k in d if "RMS" in k.upper() or "rms" in k.lower()}
        print(f"\nRESULTS: CL={cl:.6f}, CD={cd:.6f}, L/D={cl/cd:.2f}")
        for k,v in rms.items(): print(f"  {k}: {v:.2e}")
        print(f"Final residual: {list(rms.values())[0]:.2e}" if rms else "No residuals")
else:
    print("No history.csv found")

print("\nPHASE 3 COMPLETE")
