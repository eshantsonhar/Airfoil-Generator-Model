from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.config import load_settings


def resolve_executable(value: str) -> tuple[bool, str]:
    if not value:
        return False, "not configured"
    candidate = Path(value)
    if candidate.exists():
        return True, str(candidate)
    resolved = shutil.which(value)
    if resolved:
        return True, resolved
    return False, value


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "config" / "default.yaml")
    effective_cores = settings.solver.n_cores if settings.solver.n_cores > 0 else max(1, os.cpu_count() or 1)
    checks = {
        "Python": (True, sys.executable),
        "Gmsh": resolve_executable(settings.solver.gmsh_bin),
        "SU2_CFD": resolve_executable(settings.solver.su2_cfd_bin),
        "MPIEXEC": resolve_executable(settings.solver.mpiexec_bin) if settings.solver.use_mpi else (True, "disabled"),
        "Work root": (settings.paths.work_root.exists(), str(settings.paths.work_root)),
        "Database dir": (settings.paths.database_path.parent.exists(), str(settings.paths.database_path.parent)),
        "Plots dir": (settings.paths.plots_dir.exists(), str(settings.paths.plots_dir)),
    }

    failed = False
    for name, (ok, detail) in checks.items():
        status = "OK" if ok else "MISSING"
        print(f"{name:12} {status:8} {detail}")
        failed = failed or not ok
    print(f"{'CPU cores':12} {'INFO':8} requested={settings.solver.n_cores} effective={effective_cores}")
    print(f"{'MPI':12} {'INFO':8} enabled={settings.solver.use_mpi} ranks/case={settings.solver.mpi_ranks_per_case}")
    print(f"{'OMP':12} {'INFO':8} threads/rank={settings.solver.omp_threads_per_rank or 'auto'}")
    print(f"{'GPU':12} {'INFO':8} prefer_gpu={settings.solver.prefer_gpu} backend={settings.solver.gpu_backend}")

    if settings.storage.provider == "supabase":
        print(f"Supabase URL {'OK' if settings.storage.supabase_url else 'MISSING'}")
        print(f"Supabase bucket {'OK' if settings.storage.supabase_bucket else 'MISSING'}")
    elif settings.storage.provider == "firebase":
        print(f"Firebase bucket {'OK' if settings.storage.firebase_bucket else 'MISSING'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
