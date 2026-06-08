import numpy as np
from typing import Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class BayesianSurrogate:
    """
    Uncertainty-aware surrogate model using Gaussian Processes.
    """
    def __init__(self):
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        self.gp.fit(X, y)
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns mean and standard deviation."""
        return self.gp.predict(X, return_std=True)
    
    def acquisition(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Lower Confidence Bound acquisition function."""
        mu, sigma = self.predict(X)
        return mu - beta * sigma

