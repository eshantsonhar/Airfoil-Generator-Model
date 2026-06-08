"""
Airfoil shape manifold for geometric governance.

Implements a statistical manifold of realistic airfoil shapes based on
known airfoil databases (UIUC, Selig, etc.). This manifold is used to
reject geometries that are too far from physically realistic airfoil shapes.

The manifold provides:
- PCA-based latent space representation
- Distance-to-manifold metric
- Outlier detection
- Reconstruction error analysis
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
import warnings

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Manifold features will be limited.")


class ManifoldStatus(Enum):
    """Status of manifold model."""
    NOT_INITIALIZED = "NOT_INITIALIZED"
    TRAINING = "TRAINING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass
class ManifoldQueryResult:
    """Result of querying the airfoil manifold."""
    
    # Distance metrics
    manifold_distance: float
    reconstruction_error: float
    outlier_score: float
    
    # Latent space metrics
    latent_coordinates: np.ndarray
    latent_distance_from_origin: float
    
    # Nearest neighbors (if available)
    nearest_airfoil_names: List[str]
    nearest_distances: List[float]
    
    # Assessment
    is_realistic: bool
    confidence: float
    
    # Diagnostics
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "manifold_distance": self.manifold_distance,
            "reconstruction_error": self.reconstruction_error,
            "outlier_score": self.outlier_score,
            "latent_distance_from_origin": self.latent_distance_from_origin,
            "is_realistic": self.is_realistic,
            "confidence": self.confidence,
            "nearest_airfoils": self.nearest_airfoil_names[:3] if self.nearest_airfoil_names else [],
            "warnings": self.warnings,
        }


@dataclass
class ManifoldConfig:
    """Configuration for airfoil manifold."""
    
    # PCA components
    n_components: int = 20
    explained_variance_threshold: float = 0.95
    
    # Outlier detection
    outlier_method: str = "lof"  # Local Outlier Factor
    outlier_contamination: float = 0.05
    outlier_threshold: float = -1.5  # LOF score threshold
    
    # Distance thresholds
    manifold_distance_threshold: float = 3.0
    reconstruction_error_threshold: float = 0.01
    
    # Airfoil normalization
    n_sample_points: int = 200  # Points per surface
    normalize_chord: bool = True
    align_trailing_edge: bool = True
    
    # Data sources
    airfoil_database_path: str = "data/airfoil_database"
    
    # Training parameters
    min_training_samples: int = 50
    training_split: float = 0.8


class AirfoilManifold:
    """
    Statistical manifold of realistic airfoil shapes.
    
    This class builds a PCA-based latent space representation of known
    airfoil shapes. New airfoils can be projected into this space and
    their distance from the manifold can be computed.
    
    The manifold is trained on a database of known airfoils (UIUC, Selig,
    low-Re literature sections) and provides a quantitative measure of
    how "realistic" a new airfoil shape is.
    """
    
    def __init__(self, config: Optional[ManifoldConfig] = None):
        """
        Initialize airfoil manifold.
        
        Args:
            config: Manifold configuration. Uses defaults if None.
        """
        self.config = config or ManifoldConfig()
        self.status = ManifoldStatus.NOT_INITIALIZED
        
        # PCA model and scaler
        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._lof: Optional[LocalOutlierFactor] = None
        
        # Training data
        self._training_data: Optional[np.ndarray] = None
        self._airfoil_names: List[str] = []
        
        # Statistics
        self._mean_shape: Optional[np.ndarray] = None
        self._std_shape: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        
    @property
    def is_ready(self) -> bool:
        """Check if manifold is ready for queries."""
        return self.status == ManifoldStatus.READY
    
    @property
    def n_training_samples(self) -> int:
        """Get number of training samples."""
        if self._training_data is None:
            return 0
        return self._training_data.shape[0]
    
    def normalize_airfoil(
        self,
        x_upper: np.ndarray,
        y_upper: np.ndarray,
        x_lower: np.ndarray,
        y_lower: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize airfoil coordinates to standard form.
        
        Args:
            x_upper: Upper surface x-coordinates
            y_upper: Upper surface y-coordinates
            x_lower: Lower surface x-coordinates
            y_lower: Lower surface y-coordinates
        
        Returns:
            Normalized coordinates (x_u, y_u, x_l, y_l)
        """
        # Normalize chord to [0, 1]
        x_max = max(x_upper[-1], x_lower[-1])
        if x_max > 0:
            x_upper = x_upper / x_max
            x_lower = x_lower / x_max
        
        # Align trailing edge (average TE coordinates)
        if self.config.align_trailing_edge:
            te_x = (x_upper[-1] + x_lower[-1]) / 2
            te_y = (y_upper[-1] + y_lower[-1]) / 2
            
            # Shift so TE is at (1, 0)
            x_upper = x_upper - te_x + 1.0
            x_lower = x_lower - te_x + 1.0
            y_upper = y_upper - te_y
            y_lower = y_lower - te_y
        
        # Resample to standard number of points
        n = self.config.n_sample_points
        t_standard = np.linspace(0, 1, n)
        
        # Interpolate upper surface
        t_upper = np.linspace(0, 1, len(x_upper))
        y_upper_interp = np.interp(t_standard, t_upper, y_upper)
        x_upper_interp = np.interp(t_standard, t_upper, x_upper)
        
        # Interpolate lower surface (reverse order for consistent parameterization)
        t_lower = np.linspace(0, 1, len(x_lower))
        y_lower_interp = np.interp(t_standard, t_lower, y_lower)
        x_lower_interp = np.interp(t_standard, t_lower, x_lower)
        
        return x_upper_interp, y_upper_interp, x_lower_interp, y_lower_interp
    
    def airfoil_to_feature_vector(
        self,
        x_upper: np.ndarray,
        y_upper: np.ndarray,
        x_lower: np.ndarray,
        y_lower: np.ndarray,
    ) -> np.ndarray:
        """
        Convert airfoil coordinates to feature vector.
        
        Args:
            x_upper: Upper surface x-coordinates
            y_upper: Upper surface y-coordinates
            x_lower: Lower surface x-coordinates
            y_lower: Lower surface y-coordinates
        
        Returns:
            Feature vector for PCA
        """
        # Normalize
        x_u, y_u, x_l, y_l = self.normalize_airfoil(x_upper, y_upper, x_lower, y_lower)
        
        # Feature vector: concatenate upper and lower surface y-coordinates
        # (x-coordinates are standardized after normalization)
        features = np.concatenate([y_u, y_l])
        
        return features
    
    def feature_vector_to_airfoil(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct airfoil from feature vector.
        
        Args:
            features: Feature vector
        
        Returns:
            (x_upper, y_upper, x_lower, y_lower)
        """
        n = self.config.n_sample_points
        
        # Split feature vector
        y_upper = features[:n]
        y_lower = features[n:]
        
        # Standard x-coordinates
        x_upper = np.linspace(0, 1, n)
        x_lower = np.linspace(0, 1, n)
        
        return x_upper, y_upper, x_lower, y_lower
    
    def add_airfoil(
        self,
        name: str,
        x_upper: np.ndarray,
        y_upper: np.ndarray,
        x_lower: np.ndarray,
        y_lower: np.ndarray,
    ) -> bool:
        """
        Add an airfoil to the training set.
        
        Args:
            name: Airfoil name/identifier
            x_upper: Upper surface x-coordinates
            y_upper: Upper surface y-coordinates
            x_lower: Lower surface x-coordinates
            y_lower: Lower surface y-coordinates
        
        Returns:
            True if successfully added
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available. Cannot add airfoils to manifold.")
            return False
        
        try:
            features = self.airfoil_to_feature_vector(x_upper, y_upper, x_lower, y_lower)
            
            if self._training_data is None:
                self._training_data = features.reshape(1, -1)
            else:
                self._training_data = np.vstack([self._training_data, features.reshape(1, -1)])
            
            self._airfoil_names.append(name)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to add airfoil {name}: {e}")
            return False
    
    def train(self, force_retrain: bool = False) -> bool:
        """
        Train the manifold model on collected airfoils.
        
        Args:
            force_retrain: Force retraining even if already trained
        
        Returns:
            True if training successful
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available. Cannot train manifold.")
            self.status = ManifoldStatus.FAILED
            return False
        
        if self._training_data is None or self._training_data.shape[0] < self.config.min_training_samples:
            warnings.warn(
                f"Insufficient training samples ({self.n_training_samples} < {self.config.min_training_samples})"
            )
            self.status = ManifoldStatus.DEGRADED
            return False
        
        if self.status == ManifoldStatus.READY and not force_retrain:
            return True
        
        self.status = ManifoldStatus.TRAINING
        
        try:
            # Standardize features
            self._scaler = StandardScaler()
            scaled_data = self._scaler.fit_transform(self._training_data)
            
            # Compute mean and std shapes
            self._mean_shape = np.mean(self._training_data, axis=0)
            self._std_shape = np.std(self._training_data, axis=0)
            
            # Determine number of components
            n_components = min(
                self.config.n_components,
                self._training_data.shape[0] - 1,
                self._training_data.shape[1]
            )
            
            # Fit PCA
            self._pca = PCA(n_components=n_components)
            self._pca.fit(scaled_data)
            
            self._explained_variance = self._pca.explained_variance_ratio_
            
            # Check explained variance
            total_variance = np.sum(self._explained_variance)
            if total_variance < self.config.explained_variance_threshold:
                warnings.warn(
                    f"Explained variance ({total_variance:.3f}) below threshold "
                    f"({self.config.explained_variance_threshold})"
                )
            
            # Fit outlier detector
            if self.config.outlier_method == "lof":
                self._lof = LocalOutlierFactor(
                    n_neighbors=min(20, self._training_data.shape[0] - 1),
                    contamination=self.config.outlier_contamination,
                    novelty=True,
                )
                self._lof.fit(scaled_data)
            
            self.status = ManifoldStatus.READY
            return True
            
        except Exception as e:
            warnings.warn(f"Manifold training failed: {e}")
            self.status = ManifoldStatus.FAILED
            return False
    
    def query(
        self,
        x_upper: np.ndarray,
        y_upper: np.ndarray,
        x_lower: np.ndarray,
        y_lower: np.ndarray,
    ) -> ManifoldQueryResult:
        """
        Query the manifold with a new airfoil.
        
        Args:
            x_upper: Upper surface x-coordinates
            y_upper: Upper surface y-coordinates
            x_lower: Lower surface x-coordinates
            y_lower: Lower surface y-coordinates
        
        Returns:
            ManifoldQueryResult with distance metrics and assessment
        """
        warnings_list = []
        
        if not self.is_ready:
            return ManifoldQueryResult(
                manifold_distance=float('inf'),
                reconstruction_error=float('inf'),
                outlier_score=float('inf'),
                latent_coordinates=np.array([]),
                latent_distance_from_origin=float('inf'),
                nearest_airfoil_names=[],
                nearest_distances=[],
                is_realistic=False,
                confidence=0.0,
                warnings=["Manifold not trained or ready"],
            )
        
        try:
            # Convert to feature vector
            features = self.airfoil_to_feature_vector(x_upper, y_upper, x_lower, y_lower)
            
            # Scale features
            features_scaled = self._scaler.transform(features.reshape(1, -1))
            
            # Transform to latent space
            latent = self._pca.transform(features_scaled)
            
            # Reconstruct
            reconstructed_scaled = self._pca.inverse_transform(latent)
            reconstructed = self._scaler.inverse_transform(reconstructed_scaled)
            
            # Compute reconstruction error
            reconstruction_error = float(np.sqrt(np.mean((features - reconstructed[0])**2)))
            
            # Compute manifold distance (Mahalanobis-like in latent space)
            # Use weighted Euclidean distance based on explained variance
            if self._explained_variance is not None and len(self._explained_variance) > 0:
                weights = 1.0 / (self._explained_variance + 1e-10)
                weighted_latent = latent[0] * np.sqrt(weights)
                manifold_distance = float(np.sqrt(np.sum(weighted_latent**2)))
            else:
                manifold_distance = float(np.linalg.norm(latent[0]))
            
            # Compute outlier score
            if self._lof is not None:
                outlier_score = float(self._lof.score_samples(features_scaled)[0])
            else:
                outlier_score = 0.0
            
            # Latent distance from origin
            latent_distance_from_origin = float(np.linalg.norm(latent[0]))
            
            # Find nearest neighbors in training set
            nearest_names = []
            nearest_dists = []
            
            if self._training_data is not None:
                # Compute distances to all training samples in latent space
                training_latent = self._pca.transform(
                    self._scaler.transform(self._training_data)
                )
                distances = np.linalg.norm(training_latent - latent, axis=1)
                
                # Get top 5 nearest
                n_nearest = min(5, len(distances))
                nearest_indices = np.argsort(distances)[:n_nearest]
                
                nearest_names = [self._airfoil_names[i] for i in nearest_indices]
                nearest_dists = [float(distances[i]) for i in nearest_indices]
            
            # Assess realism
            is_realistic = (
                manifold_distance < self.config.manifold_distance_threshold and
                reconstruction_error < self.config.reconstruction_error_threshold and
                outlier_score > self.config.outlier_threshold
            )
            
            # Compute confidence
            confidence = 1.0
            
            # Reduce confidence based on various factors
            if manifold_distance > self.config.manifold_distance_threshold * 0.7:
                confidence -= 0.2
                warnings_list.append("Manifold distance approaching threshold")
            
            if reconstruction_error > self.config.reconstruction_error_threshold * 0.7:
                confidence -= 0.2
                warnings_list.append("Reconstruction error approaching threshold")
            
            if outlier_score < self.config.outlier_threshold * 0.8:
                confidence -= 0.15
                warnings_list.append("Outlier score near threshold")
            
            confidence = max(0.0, min(1.0, confidence))
            
            return ManifoldQueryResult(
                manifold_distance=manifold_distance,
                reconstruction_error=reconstruction_error,
                outlier_score=outlier_score,
                latent_coordinates=latent[0],
                latent_distance_from_origin=latent_distance_from_origin,
                nearest_airfoil_names=nearest_names,
                nearest_distances=nearest_dists,
                is_realistic=is_realistic,
                confidence=confidence,
                warnings=warnings_list,
            )
            
        except Exception as e:
            warnings.warn(f"Manifold query failed: {e}")
            return ManifoldQueryResult(
                manifold_distance=float('inf'),
                reconstruction_error=float('inf'),
                outlier_score=float('inf'),
                latent_coordinates=np.array([]),
                latent_distance_from_origin=float('inf'),
                nearest_airfoil_names=[],
                nearest_distances=[],
                is_realistic=False,
                confidence=0.0,
                warnings=[f"Query failed: {e}"],
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get manifold statistics.
        
        Returns:
            Dictionary with manifold statistics
        """
        if not self.is_ready:
            return {
                "status": self.status.value,
                "n_training_samples": self.n_training_samples,
                "message": "Manifold not ready",
            }
        
        stats = {
            "status": self.status.value,
            "n_training_samples": self.n_training_samples,
            "n_components": self._pca.n_components_,
            "explained_variance_ratio": float(np.sum(self._explained_variance)),
            "per_component_variance": self._explained_variance.tolist() if self._explained_variance is not None else [],
        }
        
        # Add cumulative variance
        if self._explained_variance is not None:
            cum_variance = np.cumsum(self._explained_variance)
            stats["cumulative_variance"] = cum_variance.tolist()
            
            # Components for 95% variance
            n_95 = int(np.argmax(cum_variance >= 0.95) + 1)
            stats["components_for_95_variance"] = n_95
        
        return stats
    
    def save(self, path: Union[str, Path]) -> bool:
        """
        Save manifold model to disk.
        
        Args:
            path: Path to save model
        
        Returns:
            True if saved successfully
        """
        if not self.is_ready:
            return False
        
        try:
            import pickle
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "pca": self._pca,
                "scaler": self._scaler,
                "lof": self._lof,
                "training_data": self._training_data,
                "airfoil_names": self._airfoil_names,
                "mean_shape": self._mean_shape,
                "std_shape": self._std_shape,
                "explained_variance": self._explained_variance,
                "config": self.config,
            }
            
            with open(path / "manifold.pkl", "wb") as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to save manifold: {e}")
            return False
    
    def load(self, path: Union[str, Path]) -> bool:
        """
        Load manifold model from disk.
        
        Args:
            path: Path to load model from
        
        Returns:
            True if loaded successfully
        """
        try:
            import pickle
            path = Path(path)
            
            with open(path / "manifold.pkl", "rb") as f:
                data = pickle.load(f)
            
            self._pca = data.get("pca")
            self._scaler = data.get("scaler")
            self._lof = data.get("lof")
            self._training_data = data.get("training_data")
            self._airfoil_names = data.get("airfoil_names", [])
            self._mean_shape = data.get("mean_shape")
            self._std_shape = data.get("std_shape")
            self._explained_variance = data.get("explained_variance")
            
            if self._pca is not None and self._scaler is not None:
                self.status = ManifoldStatus.READY
                return True
            else:
                self.status = ManifoldStatus.DEGRADED
                return False
            
        except Exception as e:
            warnings.warn(f"Failed to load manifold: {e}")
            self.status = ManifoldStatus.FAILED
            return False


def create_default_manifold() -> AirfoilManifold:
    """
    Create a default manifold with some basic low-Re airfoils.
    
    This provides a starting point for the manifold when no training
    data is available. The user should add more airfoils for better
    coverage.
    
    Returns:
        AirfoilManifold with basic training data
    """
    manifold = AirfoilManifold()
    
    # Add some representative low-Re airfoils (placeholder coordinates)
    # In practice, these would be loaded from actual airfoil databases
    
    # NACA 4-digit series (low-Re variants)
    naca_4412_x = np.linspace(0, 1, 100)
    naca_4412_y_upper = 0.12 * np.sqrt(naca_4412_x) * (1 - naca_4412_x)  # Simplified
    naca_4412_y_lower = -0.12 * np.sqrt(naca_4412_x) * (1 - naca_4412_x)  # Simplified
    
    # Add a few basic shapes
    for thickness in [0.06, 0.08, 0.10, 0.12, 0.15]:
        y_upper = thickness * np.sqrt(naca_4412_x) * (1 - naca_4412_x) * 5
        y_lower = -y_upper
        manifold.add_airfoil(
            f"NACA_00{int(thickness*100):02d}",
            naca_4412_x, y_upper, naca_4412_x, y_lower
        )
    
    return manifold