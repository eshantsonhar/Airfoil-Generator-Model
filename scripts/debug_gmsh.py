import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.config import load_settings
from airfoil_discovery.schemas import CSTParameters
import numpy as np

settings = load_settings("config/default.yaml")
config = settings.geometry
airfoil = CSTAirfoil(config)
params = CSTParameters(
    upper=np.array([0.1, 0.1, 0.1, 0.1]),
    lower=np.array([-0.1, -0.1, -0.1, -0.1]),
    trailing_edge_thickness=0.0
)
coords = airfoil.full_coordinates(params)
print(f"Number of points: {len(coords)}")
p1 = coords[0]
pN = coords[-1]
dist = np.linalg.norm(p1 - pN)
print(f"Distance between coords[0] and coords[-1]: {dist}")

with open("debug_airfoil.geo", "w") as f:
    f.write('SetFactory("OpenCASCADE");\n')
    for i, (x, y) in enumerate(coords, 1):
        f.write(f"Point({i}) = {{{x}, {y}, 0, 0.0025}};\n")
    # if dist is 0, we shouldn't append 1 maybe?
    f.write(f"Spline(1) = {{{','.join(str(i) for i in range(1, len(coords)+1))},1}};\n")
    f.write("Curve Loop(1) = {1};\n")

import subprocess
import shutil

gmsh_path = shutil.which("gmsh")
if gmsh_path:
    print(f"Running gmsh: {gmsh_path}")
    result = subprocess.run([gmsh_path, "debug_airfoil.geo", "-2"], capture_output=True, text=True)
    if result.returncode != 0:
        print("GMSH FAILED")
        print(result.stderr)
    else:
        print("GMSH SUCCESS")
else:
    # gmsh path is hardcoded in the default.yaml. Let's load config
    from airfoil_discovery.config import load_settings
    settings = load_settings("config/default.yaml")
    gmsh_path = settings.solver.gmsh_bin
    print(f"Running gmsh from config: {gmsh_path}")
    result = subprocess.run([gmsh_path, "debug_airfoil.geo", "-2"], capture_output=True, text=True)
    if result.returncode != 0:
        print("GMSH FAILED")
        print(result.stderr)
    else:
        print("GMSH SUCCESS")
