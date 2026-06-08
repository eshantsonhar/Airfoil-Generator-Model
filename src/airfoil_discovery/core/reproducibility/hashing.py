"""
Hashing and fingerprinting for reproducibility.

Implements config hashing, mesh hashing, and binary fingerprinting
to ensure exact reproducibility of CFD simulations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np


class ConfigHasher:
    """
    Computes cryptographic hashes of configuration files.
    
    Ensures that configuration changes are tracked and reproducible.
    """
    
    @staticmethod
    def hash_dict(config: Dict[str, Any]) -> str:
        """
        Hash a configuration dictionary.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        # Convert to JSON with sorted keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_file(config_file: Path) -> str:
        """
        Hash a configuration file.
        
        Args:
            config_file: Path to configuration file
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        content = config_file.read_text(encoding='utf-8')
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_yaml(config_file: Path) -> str:
        """
        Hash a YAML configuration file.
        
        Args:
            config_file: Path to YAML file
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        import yaml
        
        if not config_file.exists():
            raise FileNotFoundError(f"YAML file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return ConfigHasher.hash_dict(config)


class MeshHasher:
    """
    Computes hashes of mesh files for reproducibility.
    
    Ensures that mesh changes are tracked and reproducible.
    """
    
    @staticmethod
    def hash_su2_mesh(mesh_file: Path) -> str:
        """
        Hash a SU2 mesh file.
        
        Args:
            mesh_file: Path to SU2 mesh file
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        # Read mesh file content
        content = mesh_file.read_text(encoding='utf-8')
        
        # Normalize content for consistent hashing
        # Remove comments and whitespace
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('%'):
                lines.append(line)
        
        normalized = '\n'.join(lines)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_mesh_stats(mesh_stats: Dict[str, Any]) -> str:
        """
        Hash mesh statistics.
        
        Args:
            mesh_stats: Dictionary with mesh statistics
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        return ConfigHasher.hash_dict(mesh_stats)


class BinaryFingerprinter:
    """
    Creates fingerprints of binary files for version tracking.
    
    Used for tracking solver binaries and ensuring reproducibility.
    """
    
    @staticmethod
    def fingerprint_binary(binary_file: Path) -> str:
        """
        Create fingerprint of a binary file.
        
        Args:
            binary_file: Path to binary file
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        if not binary_file.exists():
            raise FileNotFoundError(f"Binary file not found: {binary_file}")
        
        sha256 = hashlib.sha256()
        with open(binary_file, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    @staticmethod
    def fingerprint_directory(directory: Path, pattern: str = "*") -> str:
        """
        Create combined fingerprint of all files in a directory.
        
        Args:
            directory: Directory to fingerprint
            pattern: File pattern to match
        
        Returns:
            SHA256 hash as hexadecimal string
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        sha256 = hashlib.sha256()
        
        # Sort files for deterministic hashing
        files = sorted(directory.glob(pattern))
        
        for file_path in files:
            if file_path.is_file():
                sha256.update(file_path.name.encode('utf-8'))
                sha256.update(BinaryFingerprinter.fingerprint_binary(file_path).encode('utf-8'))
        
        return sha256.hexdigest()
