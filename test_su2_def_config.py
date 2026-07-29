import sys
sys.path.insert(0, 'src')

from airfoil_discovery.aso.mesh_deform import generate_su2_def_config

config_text = generate_su2_def_config(
    mesh_input="mesh.su2",
    mesh_output="mesh_deformed.su2",
    marker="airfoil"
)

print("Generated SU2_DEF config:")
print(config_text)

# Check for SOLVER setting
lines = config_text.split('\n')
for i, line in enumerate(lines):
    if 'SOLVER=' in line:
        print(f"\nLine {i+1}: {line}")
        break