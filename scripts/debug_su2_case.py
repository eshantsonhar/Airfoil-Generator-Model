from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airfoil_discovery.config import load_settings
from airfoil_discovery.cfd.su2 import SU2Runner
from airfoil_discovery.schemas import CSTParameters, CandidateDesign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and debug a single SU2 case.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--run-dir", default="data/debug_case")
    parser.add_argument("--aoa", type=float, default=4.0)
    parser.add_argument("--reynolds", type=float, default=20000.0)
    parser.add_argument("--serial", action="store_true", help="Run SU2 serially instead of via MPI.")
    return parser.parse_args()


def run_cmd(command: list[str], cwd: Path, label: str) -> tuple[int, str, str]:
    print(f"\n=== {label} ===")
    print("CWD:", cwd)
    print("CMD:", " ".join(command))
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    print(f"EXIT: {proc.returncode}")
    print("--- STDOUT ---")
    print(proc.stdout)
    print("--- STDERR ---")
    print(proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    if args.serial:
        settings.solver.use_mpi = False

    run_dir = PROJECT_ROOT / args.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = SU2Runner(settings)
    params = CSTParameters(
        upper=np.array([0.17933770871077237, 0.05130698471822659, 0.34051670543061097, 0.10406137837150395]),
        lower=np.array([-0.1929602781744858, 0.05213286159233149, -0.09608367605263071, 0.038860098546347754]),
        trailing_edge_thickness=0.006757454502482736,
    )
    candidate = CandidateDesign(params=params, reynolds=args.reynolds)

    coord_path = run_dir / "airfoil.dat"
    mesh_path = run_dir / "airfoil.su2"
    geo_path = run_dir / "airfoil.geo"
    case_dir = run_dir / f"aoa_{args.aoa:+05.1f}".replace(".", "p")
    case_dir.mkdir(exist_ok=True)
    cfg_path = case_dir / "config.cfg"

    runner._write_airfoil_dat(candidate, coord_path)
    runner._write_gmsh_geo(candidate, geo_path)
    run_cmd([settings.solver.gmsh_bin, geo_path.name, "-2", "-format", "su2", "-o", mesh_path.name], run_dir, "GMSH")
    runner._write_su2_config(candidate, mesh_path, cfg_path, args.aoa)

    print("\n=== CONFIG ===")
    print(cfg_path.read_text(encoding="utf-8"))

    serial_cmd = [settings.solver.su2_cfd_bin, cfg_path.name]
    mpi_cmd = [settings.solver.mpiexec_bin, "-n", str(settings.solver.mpi_ranks_per_case), settings.solver.su2_cfd_bin, cfg_path.name]

    os.environ["SU2_USE_MPI"] = "false" if args.serial else os.environ.get("SU2_USE_MPI", "true")
    run_cmd([settings.solver.su2_cfd_bin, "-d", cfg_path.name], case_dir, "SU2 DRYRUN SERIAL")
    run_cmd(serial_cmd, case_dir, "SU2 SERIAL")
    if not args.serial:
        run_cmd(mpi_cmd, case_dir, "SU2 MPI")

    history = case_dir / "history.csv"
    if history.exists():
        print("\n=== HISTORY HEAD ===")
        print("\n".join(history.read_text(encoding="utf-8").splitlines()[:10]))
    else:
        print("\nNo history.csv found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
