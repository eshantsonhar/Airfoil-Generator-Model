import sys
sys.path.insert(0, '.')
import numpy as np
from pathlib import Path

from airfoil_discovery.cfd.su2 import SU2Evaluator
from airfoil_discovery.config import load_settings

settings = load_settings('config/default.yaml')
evaluator = SU2Evaluator(settings)

report = evaluator._check_convergence(
    residual_history=[-2.842, -1.018, -1.392, -1.705, -2.001, -2.305,
                      -2.576, -2.785, -3.002, -3.258, -3.530, -3.802,
                      -4.060, -4.272, -4.441, -4.602, -4.766, -4.896,
                      -5.008, -5.134, -5.266, -5.407, -5.533, -5.645,
                      -5.749, -5.842, -5.926, -6.002, -6.072, -6.138],
    cl_history=[0.025255, 0.038462, 0.052945, 0.068899, 0.085962, 0.103828,
                0.122252, 0.141008, 0.159752, 0.178033, 0.195325, 0.211062,
                0.224743, 0.235512, 0.243075, 0.248389, 0.252115, 0.254717,
                0.256525, 0.257786, 0.258667, 0.259284, 0.259716, 0.260018,
                0.260229, 0.260377, 0.260480, 0.260551, 0.260601, 0.260635],
    cd_history=[0.007725, 0.017862, 0.026409, 0.033930, 0.040844, 0.047341,
                0.053435, 0.059061, 0.064131, 0.068600, 0.072464, 0.075724,
                0.078390, 0.080389, 0.081744, 0.082670, 0.083308, 0.083749,
                0.084055, 0.084268, 0.084417, 0.084522, 0.084595, 0.084647,
                0.084684, 0.084709, 0.084728, 0.084741, 0.084750, 0.084757],
)
print(f"residual_converged: {report['residual_converged']}")
print(f"forces_stabilized: {report['forces_stabilized']}")
print(f"is_valid: {report['is_valid']}")
print(f"residual: {report['residual']:.4f}")
print(f"failure_reasons: {report['failure_reasons']}")
