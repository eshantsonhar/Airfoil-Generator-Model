"""
Research telemetry system for real-time metrics tracking.

Tracks all critical metrics during optimization:
- Residuals
- Force histories
- Transition movement
- Bubble metrics
- FD gradient error
- KKT metrics
- Gain ratio
- Mesh quality
- Intermittency evolution
- Numerical dissipation
- CFL evolution
- Solver stiffness
- Oscillation spectra
- Trust-region behavior
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .metrics_store import MetricsStore, TelemetryRecord


@dataclass
class TelemetryMetric:
    """Single telemetry metric."""
    
    name: str
    value: float
    unit: Optional[str] = None
    category: str = "general"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TelemetrySnapshot:
    """Snapshot of all telemetry at a point in time."""
    
    timestamp: str
    iteration: Optional[int]
    run_id: Optional[str]
    
    # Convergence metrics
    residual: Optional[float] = None
    max_residual: Optional[float] = None
    convergence_rate: Optional[float] = None
    
    # Force metrics
    cl: Optional[float] = None
    cd: Optional[float] = None
    cl_std: Optional[float] = None
    cd_std: Optional[float] = None
    
    # Transition metrics
    transition_onset: Optional[float] = None
    transition_completion: Optional[float] = None
    intermittency_mean: Optional[float] = None
    
    # LSB metrics
    lsb_detected: bool = False
    bubble_length: Optional[float] = None
    bubble_height: Optional[float] = None
    
    # Gradient metrics
    gradient_norm: Optional[float] = None
    fd_error: Optional[float] = None
    cosine_similarity: Optional[float] = None
    
    # KKT metrics
    stationarity: Optional[float] = None
    complementarity: Optional[float] = None
    
    # Mesh metrics
    max_y_plus: Optional[float] = None
    mesh_quality: Optional[float] = None
    
    # Dissipation metrics
    numerical_dissipation: Optional[float] = None
    
    # Trust region metrics
    trust_region_radius: Optional[float] = None
    
    # Optimizer metrics
    gain_ratio: Optional[float] = None
    
    # Additional metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "run_id": self.run_id,
            "residual": self.residual,
            "max_residual": self.max_residual,
            "convergence_rate": self.convergence_rate,
            "cl": self.cl,
            "cd": self.cd,
            "cl_std": self.cl_std,
            "cd_std": self.cd_std,
            "transition_onset": self.transition_onset,
            "transition_completion": self.transition_completion,
            "intermittency_mean": self.intermittency_mean,
            "lsb_detected": self.lsb_detected,
            "bubble_length": self.bubble_length,
            "bubble_height": self.bubble_height,
            "gradient_norm": self.gradient_norm,
            "fd_error": self.fd_error,
            "cosine_similarity": self.cosine_similarity,
            "stationarity": self.stationarity,
            "complementarity": self.complementarity,
            "max_y_plus": self.max_y_plus,
            "mesh_quality": self.mesh_quality,
            "numerical_dissipation": self.numerical_dissipation,
            "trust_region_radius": self.trust_region_radius,
            "gain_ratio": self.gain_ratio,
        }
        data.update(self.additional_metrics)
        return data


class ResearchTelemetry:
    """
    Research telemetry system for real-time metrics tracking.
    
    Tracks all critical metrics during optimization and provides:
    - Real-time monitoring
    - Historical analysis
    - Export functionality
    - Query capabilities
    """
    
    def __init__(self, db_path: Optional[Path] = None, run_id: Optional[str] = None):
        """
        Initialize research telemetry.
        
        Args:
            db_path: Path to metrics database
            run_id: Run identifier
        """
        if db_path is None:
            db_path = Path("data/telemetry/metrics.db")
        
        self.store = MetricsStore(db_path)
        self.run_id = run_id
        self.snapshots: List[TelemetrySnapshot] = []
    
    def record_snapshot(self, snapshot: TelemetrySnapshot):
        """
        Record a telemetry snapshot.
        
        Args:
            snapshot: Telemetry snapshot to record
        """
        self.snapshots.append(snapshot)
        
        # Store each metric individually
        data = snapshot.to_dict()
        
        for metric_name, value in data.items():
            if value is not None and isinstance(value, (int, float)):
                self.store.store_metric(
                    metric_name=metric_name,
                    metric_value=float(value),
                    iteration=snapshot.iteration,
                    run_id=snapshot.run_id or self.run_id,
                    timestamp=snapshot.timestamp,
                )
    
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a single metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            iteration: Iteration number
            metadata: Additional metadata
        """
        self.store.store_metric(
            metric_name=metric_name,
            metric_value=metric_value,
            iteration=iteration,
            run_id=self.run_id,
            metadata=metadata,
        )
    
    def create_snapshot(
        self,
        iteration: Optional[int] = None,
        **metrics,
    ) -> TelemetrySnapshot:
        """
        Create a telemetry snapshot from metrics.
        
        Args:
            iteration: Iteration number
            **metrics: Metric key-value pairs
        
        Returns:
            TelemetrySnapshot
        """
        snapshot = TelemetrySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            iteration=iteration,
            run_id=self.run_id,
            **metrics
        )
        
        return snapshot
    
    def get_metric_history(
        self,
        metric_name: str,
        run_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            run_id: Run identifier
        
        Returns:
            List of (timestamp, value) tuples
        """
        records = self.store.query_metrics(metric_name=metric_name, run_id=run_id or self.run_id)
        return [(r.timestamp, r.metric_value) for r in records]
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current values of all metrics.
        
        Returns:
            Dictionary mapping metric names to current values
        """
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        data = latest.to_dict()
        
        # Filter out None values
        return {k: v for k, v in data.items() if v is not None and isinstance(v, (int, float))}
    
    def export_metrics(
        self,
        output_dir: Path,
        format: str = "csv",
    ):
        """
        Export all metrics to files.
        
        Args:
            output_dir: Output directory
            format: Export format ('csv' or 'json')
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metric_names = self.store.get_metric_names()
        
        for metric_name in metric_names:
            if format == "csv":
                output_path = output_dir / f"{metric_name}.csv"
                self.store.export_to_csv(output_path, metric_name=metric_name, run_id=self.run_id)
            elif format == "json":
                output_path = output_dir / f"{metric_name}.json"
                self.store.export_to_json(output_path, metric_name=metric_name, run_id=self.run_id)
    
    def plot_metric(
        self,
        metric_name: str,
        output_path: Optional[Path] = None,
    ):
        """
        Plot a metric over time.
        
        Args:
            metric_name: Name of the metric to plot
            output_path: Output path for plot (optional)
        """
        import matplotlib.pyplot as plt
        
        history = self.get_metric_history(metric_name)
        
        if not history:
            print(f"No data available for metric: {metric_name}")
            return
        
        timestamps, values = zip(*history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, values, marker='o', linestyle='-')
        plt.xlabel('Time')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary mapping metric names to statistics
        """
        metric_names = self.store.get_metric_names()
        
        summary = {}
        for metric_name in metric_names:
            stats = self.store.get_statistics(metric_name, run_id=self.run_id)
            if stats:
                summary[metric_name] = stats
        
        return summary
