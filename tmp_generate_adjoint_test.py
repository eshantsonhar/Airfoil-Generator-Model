from pathlib import Path
import sys
repo = Path.cwd()
sys.path.insert(0, str(repo))
from src.airfoil_discovery.aso.config_adjoint import write_adjoint_config

out_dir = repo / "tmp_su2_adjoint_test"
out_dir.mkdir(exist_ok=True)
mesh_src = repo / "data" / "cache" / "final_test" / "airfoil_perfect.su2"
restart_src = repo / "aso_results" / "cfd_cases" / "eval_1782577562" / "restart_flow.dat"
mesh_dst = out_dir / mesh_src.name
restart_dst = out_dir / restart_src.name
for src, dst in [(mesh_src, mesh_dst), (restart_src, restart_dst)]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.write_bytes(src.read_bytes())

config_path = out_dir / "config_adjoint.cfg"
write_adjoint_config(
    output_path=config_path,
    mesh_filename=mesh_dst.name,
    primal_config_filename=str(repo / "test_dry_run" / "config_adjoint_test.cfg"),
    objective="DRAG",
    n_iter=10,
    cfl_adjoint=1.0,
    restart_filename=restart_dst.name,
)
print("WROTE", config_path)
print(config_path.read_text())
