import sys
import re
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Monkeypatch primal config generator
import airfoil_discovery.aso.config_primal as cp
original_generate_primal = cp.generate_primal_config

def patched_generate_primal(*args, **kwargs):
    reynolds = 1e5
    if len(args) > 2:
        reynolds = args[2]
    elif 'reynolds' in kwargs:
        reynolds = kwargs['reynolds']
        
    config_text = original_generate_primal(*args, **kwargs)
    
    extra_lines = [
        "VISCOSITY_MODEL= CONSTANT_VISCOSITY",
        f"MU_CONSTANT= {1.225 / reynolds:.8e}",
        "INC_DENSITY_INIT= 1.225",
    ]
    
    config_text = config_text.replace(
        "% ------------ Solver ------------",
        "% ------------ Solver ------------\n" + "\n".join(extra_lines)
    )
    config_text = config_text.replace("CONV_NUM_METHOD_TURB= ROE_TURB", "")
    config_text = config_text.replace("VENKATKRISHNAN_WANG_LIMITER_COEFF= 0.05", "VENKAT_LIMITER_COEFF= 0.05")
    
    # Fix CFL_ADAPT_PARAM: change (cfl_initial, cfl_final, 1.5, 100.0) to (0.5, 1.5, cfl_initial, cfl_final)
    config_text = re.sub(
        r"CFL_ADAPT_PARAM=\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*1.5\s*,\s*100.0\s*\)",
        r"CFL_ADAPT_PARAM= ( 0.5, 1.5, \1, \2 )",
        config_text
    )
    
    lines = [line for line in config_text.splitlines() if not line.strip().startswith("OUTPUT_DIR=")]
    return "\n".join(lines)

cp.generate_primal_config = patched_generate_primal

# Monkeypatch adjoint config generator
import airfoil_discovery.aso.config_adjoint as ca
original_generate_adjoint = ca.generate_adjoint_config

def patched_generate_adjoint(*args, **kwargs):
    primal_config_path = None
    if len(args) > 1:
        primal_config_path = args[1]
    elif 'primal_config_filename' in kwargs:
        primal_config_path = kwargs['primal_config_filename']
        
    mu_value = "1.225e-5"
    if primal_config_path and Path(primal_config_path).exists():
        for line in Path(primal_config_path).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MU_CONSTANT="):
                mu_value = line.split("=")[1].strip()
                break
                
    config_text = original_generate_adjoint(*args, **kwargs)
    
    extra_lines = [
        "VISCOSITY_MODEL= CONSTANT_VISCOSITY",
        f"MU_CONSTANT= {mu_value}",
        "INC_DENSITY_INIT= 1.225",
    ]
    
    config_text = config_text.replace(
        "% ------------ Solver ------------",
        "% ------------ Solver ------------\n" + "\n".join(extra_lines)
    )
    config_text = config_text.replace("CONV_NUM_METHOD_TURB= ROE_TURB", "")
    config_text = config_text.replace("VENKATKRISHNAN_WANG_LIMITER_COEFF= 0.05", "VENKAT_LIMITER_COEFF= 0.05")
    
    lines = [line for line in config_text.splitlines() if not line.strip().startswith("OUTPUT_DIR=")]
    return "\n".join(lines)

ca.generate_adjoint_config = patched_generate_adjoint

# Now import the main script and run it
import scripts.run_aso_pde_optimization as run_aso
if __name__ == "__main__":
    run_aso.main()
