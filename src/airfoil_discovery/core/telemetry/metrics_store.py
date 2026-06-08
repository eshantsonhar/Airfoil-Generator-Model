"""
Metrics store for telemetry data.

Provides efficient storage, querying, and export of time-series
telemetry metrics for research analysis.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class TelemetryRecord:
    """Single telemetry record."""
    
    timestamp: str
    metric_name: str
    metric_value: float
    iteration: Optional[int] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsStore:
    """
    Stores and queries telemetry metrics.
    
    Uses SQLite for efficient storage and querying of time-series data.
    Provides export functionality for analysis and plotting.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize metrics store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    iteration INTEGER,
                    run_id TEXT,
                    metadata TEXT,
                    UNIQUE(timestamp, metric_name, iteration, run_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_name 
                ON telemetry(metric_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_iteration 
                ON telemetry(iteration)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_id 
                ON telemetry(run_id)
            """)
            
            conn.commit()
    
    def store_metric(
        self,
        metric_name: str,
        metric_value: float,
        iteration: Optional[int] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        """
        Store a telemetry metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            iteration: Iteration number
            run_id: Run identifier
            metadata: Additional metadata
            timestamp: Timestamp (ISO format)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO telemetry 
                (timestamp, metric_name, metric_value, iteration, run_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, metric_name, metric_value, iteration, run_id, metadata_json))
            conn.commit()
    
    def store_metrics_batch(
        self,
        records: List[TelemetryRecord],
    ):
        """
        Store multiple telemetry metrics in batch.
        
        Args:
            records: List of telemetry records
        """
        with sqlite3.connect(self.db_path) as conn:
            for record in records:
                metadata_json = json.dumps(record.metadata) if record.metadata else None
                conn.execute("""
                    INSERT OR REPLACE INTO telemetry 
                    (timestamp, metric_name, metric_value, iteration, run_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp,
                    record.metric_name,
                    record.metric_value,
                    record.iteration,
                    record.run_id,
                    metadata_json,
                ))
            conn.commit()
    
    def query_metrics(
        self,
        metric_name: Optional[str] = None,
        run_id: Optional[str] = None,
        iteration_min: Optional[int] = None,
        iteration_max: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[TelemetryRecord]:
        """
        Query telemetry metrics.
        
        Args:
            metric_name: Filter by metric name
            run_id: Filter by run ID
            iteration_min: Minimum iteration
            iteration_max: Maximum iteration
            limit: Maximum number of records
        
        Returns:
            List of telemetry records
        """
        query = "SELECT timestamp, metric_name, metric_value, iteration, run_id, metadata FROM telemetry WHERE 1=1"
        params = []
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        
        if iteration_min is not None:
            query += " AND iteration >= ?"
            params.append(iteration_min)
        
        if iteration_max is not None:
            query += " AND iteration <= ?"
            params.append(iteration_max)
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            records = []
            
            for row in cursor.fetchall():
                metadata = json.loads(row[5]) if row[5] else None
                records.append(TelemetryRecord(
                    timestamp=row[0],
                    metric_name=row[1],
                    metric_value=row[2],
                    iteration=row[3],
                    run_id=row[4],
                    metadata=metadata,
                ))
        
        return records
    
    def get_metric_names(self) -> List[str]:
        """Get all unique metric names."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT metric_name FROM telemetry ORDER BY metric_name")
            return [row[0] for row in cursor.fetchall()]
    
    def get_run_ids(self) -> List[str]:
        """Get all unique run IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT run_id FROM telemetry WHERE run_id IS NOT NULL ORDER BY run_id")
            return [row[0] for row in cursor.fetchall()]
    
    def export_to_csv(
        self,
        output_path: Path,
        metric_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Export metrics to CSV file.
        
        Args:
            output_path: Output CSV file path
            metric_name: Filter by metric name
            run_id: Filter by run ID
        """
        import csv
        
        records = self.query_metrics(metric_name=metric_name, run_id=run_id)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric_name', 'metric_value', 'iteration', 'run_id'])
            
            for record in records:
                writer.writerow([
                    record.timestamp,
                    record.metric_name,
                    record.metric_value,
                    record.iteration,
                    record.run_id,
                ])
    
    def export_to_json(
        self,
        output_path: Path,
        metric_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Export metrics to JSON file.
        
        Args:
            output_path: Output JSON file path
            metric_name: Filter by metric name
            run_id: Filter by run ID
        """
        records = self.query_metrics(metric_name=metric_name, run_id=run_id)
        
        data = [record.to_dict() for record in records]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def clear_run(self, run_id: str):
        """
        Clear all metrics for a specific run.
        
        Args:
            run_id: Run identifier to clear
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM telemetry WHERE run_id = ?", (run_id,))
            conn.commit()
    
    def get_statistics(
        self,
        metric_name: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Metric name
            run_id: Filter by run ID
        
        Returns:
            Dictionary with statistics (mean, std, min, max, count)
        """
        records = self.query_metrics(metric_name=metric_name, run_id=run_id)
        
        if not records:
            return {}
        
        values = [r.metric_value for r in records]
        
        import numpy as np
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
