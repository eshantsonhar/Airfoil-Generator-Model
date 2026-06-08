import sys
sys.path.insert(0, ".")
import numpy as np
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
cl = np.array(traces.get('CL', []))
cd = np.array(traces.get('CD', []))

print("Testing different stabilization windows for mean-vs-last change check:\n")
for sw in [5, 10, 15, 20, 25, 30]:
    if len(cl) < sw: break
    rc = cl[-sw:]
    cc = cd[-sw:]
    rc_mean = float(np.mean(rc))
    cc_mean = float(np.mean(cc))
    cl_chg = abs(rc[-1] - rc_mean) / (abs(rc_mean) + 1e-15)
    cd_chg = abs(cc[-1] - cc_mean) / (abs(cc_mean) + 1e-15)
    cl_dual = abs(rc[-1] - rc[0]) / (abs(rc[0]) + 1e-15)
    cd_dual = abs(cc[-1] - cc[0]) / (abs(cc[0]) + 1e-15)
    ok = (cl_chg < 0.005 and cd_chg < 0.005)
    oka = (cl_dual < 0.005 and cd_dual < 0.005)
    print(f"  sw={sw:2d}: mean cl_chg={cl_chg:.4f} cd_chg={cd_chg:.4f} -> {'PASS' if ok else 'FAIL'}  | first_l cl_chg={cl_dual:.4f} cd_chg={cd_dual:.4f} -> {'PASS' if oka else 'FAIL'}")

# Show what the last 10 iterations' CL and CD values look like
print("\nLast 10 iterations:")
for i in range(20, 30):
    print(f"  iter {i:2d}: CL={cl[i]:.6f}, CD={cd[i]:.6f}")
print(f"\nMean of last 10: CL={np.mean(cl[20:]):.6f}, CD={np.mean(cd[20:]):.6f}")
print(f"Change from mean: CL_chg={abs(cl[29]-np.mean(cl[20:]))/abs(np.mean(cl[20:])+1e-15):.4f}, CD_chg={abs(cd[29]-np.mean(cd[20:]))/abs(np.mean(cd[20:])+1e-15):.4f}")
