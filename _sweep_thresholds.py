import numpy as np, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from pathlib import Path
from airfoil_discovery.verification.convergence import (
    ResidualConvergenceAnalyzer, IterativeConvergenceMonitor
)

h = Path('data/failures/iter_001_aoa_+02p0/history.csv')
text = h.read_text(encoding='utf-8')
lines = text.splitlines()
headers = [it.strip().strip('"') for it in lines[0].split(',')]
traces = {h2: [] for h2 in headers}
for line in lines[1:]:
    if not line.strip() or line.strip() == ',': continue
    lvs = [it.strip() for it in line.split(',')]
    for i, h2 in enumerate(headers):
        if i < len(lvs):
            try: traces[h2].append(float(lvs[i]))
            except: pass

rmsP = np.array(traces.get('rms[P]', []), dtype=float)
cl = np.array(traces.get('CL', []), dtype=float)
cd = np.array(traces.get('CD', []), dtype=float)

print(f"Lengths: rms={len(rmsP)}, CL={len(cl)}, CD={len(cd)}")
print(f"Final abs(rms[P])={abs(rmsP[-1]):.3f} abs(rms[0])={abs(rmsP[0]):.3f}")

# Test different thresholds
print("\n=== Residual threshold sweep ===")
for thr in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    analyzer = ResidualConvergenceAnalyzer(residual_threshold=thr)
    m = analyzer.analyze(rmsP.tolist())
    print(f"  thr={thr}: below={m.below_threshold}, final_mag={abs(m.final_residual):.3f}")

# Test forces with different sw
print("\n=== Force stabilization sweep (mean-vs-mean) ===")
for sw in [5, 8, 10, 12, 15, 20, 25, 30]:
    if sw > len(cl): continue
    fm = IterativeConvergenceMonitor(
        force_stabilization_threshold=0.005,
        stabilization_window=sw)
    f = fm.analyze_forces(cl.tolist(), cd.tolist())
    print(f"  sw={sw}: stab={f.forces_stabilized}, osc={f.force_oscillation_acceptable}, "
          f"cl_change={abs(f.final_cl-np.mean(cl[-sw:]))/max(abs(np.mean(cl[-sw:])),1e-15):.4f}, "
          f"cd_change={abs(f.final_cd-np.mean(cd[-sw:]))/max(abs(np.mean(cd[-sw:])),1e-15):.4f}")
