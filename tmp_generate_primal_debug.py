from pathlib import Path
from src.airfoil_discovery.aso.config_primal import write_primal_config

root = Path(__file__).resolve().parent
work_dir = root / 'tmp_primal_debug'
work_dir.mkdir(exist_ok=True)
mesh_src = root / 'data' / 'cache' / 'final_test' / 'airfoil_perfect.su2'
mesh_dst = work_dir / mesh_src.name
if mesh_src.exists():
    import shutil
    shutil.copy2(mesh_src, mesh_dst)
else:
    raise FileNotFoundError(f'Mesh not found: {mesh_src}')

config_path = work_dir / 'config_primal.cfg'
write_primal_config(
    output_path=config_path,
    mesh_filename=mesh_dst.name,
    aoa_deg=4.0,
    reynolds=1e5,
    mach=0.1,
    n_iter=100,
    cfl_initial=1.0,
    cfl_final=2.0,
    transition_model=True,
    turbulence_intensity=0.001,
    turb_viscosity_ratio=5.0,
)
print(f'Wrote {config_path}')
print('Mesh copied to', mesh_dst)
