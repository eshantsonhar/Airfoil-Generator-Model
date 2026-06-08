import numpy as np

class ReferenceScaler:
    """Scales objective and sensitivities by reference values to maintain O(1) magnitudes."""
    def __init__(self, cd_ref: float = 0.012):
        self.cd_ref = cd_ref
        
    def scale_cd(self, cd: float) -> float:
        return cd / self.cd_ref
    
    def scale_grad_cd(self, grad_cd: np.ndarray) -> np.ndarray:
        return grad_cd / self.cd_ref

class VariableNormalizer:
    """Normalizes design variables to [0, 1] range."""
    def __init__(self, x_min: np.ndarray, x_max: np.ndarray):
        self.x_min = x_min
        self.x_max = x_max
        
    def to_normalized(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_min) / (self.x_max - self.x_min)
    
    def from_normalized(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * (self.x_max - self.x_min) + self.x_min

class Preconditioner:
    """Applies Laplacian smoothing to the design variable space to suppress CST artifacts."""
    def __init__(self, lambda_smooth: float = 0.01):
        self.lambda_smooth = lambda_smooth
        
    def smooth_gradients(self, grad: np.ndarray) -> np.ndarray:
        # Simplified Laplacian filter
        smoothed = np.copy(grad)
        for i in range(1, len(grad)-1):
            smoothed[i] = grad[i] + self.lambda_smooth * (grad[i-1] - 2*grad[i] + grad[i+1])
        return smoothed
