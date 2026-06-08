"""
Environment snapshot for reproducibility.

Captures runtime environment information including Python version,
package versions, system information, and solver versions.
"""

from __future__ import annotations

import platform
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EnvironmentSnapshot:
    """Snapshot of runtime environment."""
    
    # System information
    system: str
    machine: str
    processor: str
    python_version: str
    
    # Package versions
    numpy_version: Optional[str] = None
    scipy_version: Optional[str] = None
    pandas_version: Optional[str] = None
    
    # SU2 information
    su2_version: Optional[str] = None
    su2_cfd_hash: Optional[str] = None
    
    # GMSH information
    gmsh_version: Optional[str] = None
    gmsh_hash: Optional[str] = None
    
    # Timestamp
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system': self.system,
            'machine': self.machine,
            'processor': self.processor,
            'python_version': self.python_version,
            'numpy_version': self.numpy_version,
            'scipy_version': self.scipy_version,
            'pandas_version': self.pandas_version,
            'su2_version': self.su2_version,
            'su2_cfd_hash': self.su2_cfd_hash,
            'gmsh_version': self.gmsh_version,
            'gmsh_hash': self.gmsh_hash,
            'timestamp': self.timestamp,
        }
    
    def save(self, filepath: Path):
        """
        Save environment snapshot to file.
        
        Args:
            filepath: Path to save snapshot
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'EnvironmentSnapshot':
        """
        Load environment snapshot from file.
        
        Args:
            filepath: Path to load snapshot from
        
        Returns:
            EnvironmentSnapshot object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(**data)


def capture_environment(
    su2_cfd_bin: Optional[Path] = None,
    gmsh_bin: Optional[Path] = None,
) -> EnvironmentSnapshot:
    """
    Capture current runtime environment.
    
    Args:
        su2_cfd_bin: Path to SU2_CFD binary
        gmsh_bin: Path to GMSH binary
    
    Returns:
        EnvironmentSnapshot with current environment
    """
    from datetime import datetime, timezone
    
    # Get package versions
    numpy_version = None
    scipy_version = None
    pandas_version = None
    
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        pass
    
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        pass
    
    try:
        import pandas
        pandas_version = pandas.__version__
    except ImportError:
        pass
    
    # Get SU2 information
    su2_version = None
    su2_cfd_hash = None
    
    if su2_cfd_bin and su2_cfd_bin.exists():
        from .hashing import BinaryFingerprinter
        su2_cfd_hash = BinaryFingerprinter.fingerprint_binary(su2_cfd_bin)
        # Version would need to be extracted from binary output
    
    # Get GMSH information
    gmsh_version = None
    gmsh_hash = None
    
    if gmsh_bin and gmsh_bin.exists():
        from .hashing import BinaryFingerprinter
        gmsh_hash = BinaryFingerprinter.fingerprint_binary(gmsh_bin)
        # Version would need to be extracted from binary output
    
    snapshot = EnvironmentSnapshot(
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        numpy_version=numpy_version,
        scipy_version=scipy_version,
        pandas_version=pandas_version,
        su2_version=su2_version,
        su2_cfd_hash=su2_cfd_hash,
        gmsh_version=gmsh_version,
        gmsh_hash=gmsh_hash,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    return snapshot
