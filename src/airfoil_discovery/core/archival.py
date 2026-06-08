"""
Output archival system for full reproducibility.

Archives geometry, meshes, solver settings, convergence histories,
force histories, Cp distributions, intermittency fields, transition
locations, gradients, optimizer states, trust-region states, mesh
metrics, diagnostics, and runtime metadata. Everything must be
reproducible.
"""

from __future__ import annotations

import shutil
import json
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib


@dataclass
class ArchiveManifest:
    """Manifest of archived contents."""
    
    # Archive information
    archive_id: str
    timestamp: str
    run_id: str
    iteration: Optional[int] = None
    
    # Contents
    geometry_files: List[str] = field(default_factory=list)
    mesh_files: List[str] = field(default_factory=list)
    solver_configs: List[str] = field(default_factory=list)
    convergence_data: List[str] = field(default_factory=list)
    force_data: List[str] = field(default_factory=list)
    field_data: List[str] = field(default_factory=list)
    gradient_data: List[str] = field(default_factory=list)
    optimizer_state: List[str] = field(default_factory=list)
    verification_data: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    metadata: List[str] = field(default_factory=list)
    
    # Hashes
    content_hashes: Dict[str, str] = field(default_factory=dict)
    
    # Archive metadata
    archive_size_bytes: int = 0
    compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "archive_id": self.archive_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "geometry_files": self.geometry_files,
            "mesh_files": self.mesh_files,
            "solver_configs": self.solver_configs,
            "convergence_data": self.convergence_data,
            "force_data": self.force_data,
            "field_data": self.field_data,
            "gradient_data": self.gradient_data,
            "optimizer_state": self.optimizer_state,
            "verification_data": self.verification_data,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
            "content_hashes": self.content_hashes,
            "archive_size_bytes": self.archive_size_bytes,
            "compression_ratio": self.compression_ratio,
        }


class OutputArchiver:
    """
    Archives all outputs for full reproducibility.
    
    Ensures that every result is reproducible by archiving:
    - Geometry
    - Meshes
    - Solver settings
    - Convergence histories
    - Force histories
    - Cp distributions
    - Intermittency fields
    - Transition locations
    - Gradients
    - Optimizer states
    - Trust-region states
    - Mesh metrics
    - Diagnostics
    - Runtime metadata
    """
    
    def __init__(self, archive_dir: Path, compress: bool = True):
        """
        Initialize output archiver.
        
        Args:
            archive_dir: Directory for archives
            compress: Whether to compress archives
        """
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
    
    def _compute_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def create_archive(
        self,
        run_id: str,
        iteration: Optional[int] = None,
        source_dir: Optional[Path] = None,
        files_to_archive: Optional[Dict[str, List[Path]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Path, ArchiveManifest]:
        """
        Create an archive of all outputs.
        
        Args:
            run_id: Run identifier
            iteration: Iteration number
            source_dir: Source directory to archive
            files_to_archive: Dictionary mapping categories to file lists
            metadata: Additional metadata to include
        
        Returns:
            (archive_path, manifest)
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_id = f"{run_id}_{timestamp}"
        if iteration is not None:
            archive_id += f"_iter{iteration:04d}"
        
        # Create temporary directory for archive contents
        temp_dir = self.archive_dir / f"temp_{archive_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = ArchiveManifest(
            archive_id=archive_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            iteration=iteration,
        )
        
        # Archive files by category
        if files_to_archive:
            for category, file_list in files_to_archive.items():
                category_dir = temp_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)
                
                for file_path in file_list:
                    if file_path.exists():
                        dest_path = category_dir / file_path.name
                        shutil.copy2(file_path, dest_path)
                        
                        # Add to manifest
                        if category == "geometry":
                            manifest.geometry_files.append(file_path.name)
                        elif category == "mesh":
                            manifest.mesh_files.append(file_path.name)
                        elif category == "solver_configs":
                            manifest.solver_configs.append(file_path.name)
                        elif category == "convergence":
                            manifest.convergence_data.append(file_path.name)
                        elif category == "force":
                            manifest.force_data.append(file_path.name)
                        elif category == "field":
                            manifest.field_data.append(file_path.name)
                        elif category == "gradient":
                            manifest.gradient_data.append(file_path.name)
                        elif category == "optimizer":
                            manifest.optimizer_state.append(file_path.name)
                        elif category == "verification":
                            manifest.verification_data.append(file_path.name)
                        elif category == "diagnostics":
                            manifest.diagnostics.append(file_path.name)
                        
                        # Compute hash
                        manifest.content_hashes[file_path.name] = self._compute_hash(file_path)
        
        # Archive entire source directory if provided
        if source_dir and source_dir.exists():
            source_copy_dir = temp_dir / "cfd_case"
            shutil.copytree(source_dir, source_copy_dir, dirs_exist_ok=True)
        
        # Add metadata
        if metadata:
            metadata_file = temp_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            manifest.metadata.append("metadata.json")
        
        # Create archive
        if self.compress:
            archive_path = self.archive_dir / f"{archive_id}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)
            
            # Get archive size
            manifest.archive_size_bytes = archive_path.stat().st_size
            
            # Estimate compression ratio (simplified)
            original_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
            if original_size > 0:
                manifest.compression_ratio = original_size / manifest.archive_size_bytes
        else:
            archive_path = self.archive_dir / archive_id
            shutil.move(temp_dir, archive_path)
            
            # Get archive size
            manifest.archive_size_bytes = sum(
                f.stat().st_size for f in archive_path.rglob('*') if f.is_file()
            )
        
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Save manifest
        manifest_file = self.archive_dir / f"{archive_id}_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        return archive_path, manifest
    
    def restore_archive(
        self,
        archive_id: str,
        destination_dir: Path,
    ) -> Path:
        """
        Restore an archive to a destination directory.
        
        Args:
            archive_id: Archive identifier
            destination_dir: Destination directory
        
        Returns:
            Path to restored directory
        """
        if self.compress:
            archive_path = self.archive_dir / f"{archive_id}.zip"
        else:
            archive_path = self.archive_dir / archive_id
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        if self.compress:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(destination_dir)
        else:
            shutil.copytree(archive_path, destination_dir, dirs_exist_ok=True)
        
        return destination_dir
    
    def list_archives(self) -> List[Dict[str, Any]]:
        """
        List all available archives.
        
        Returns:
            List of archive information
        """
        archives = []
        
        for manifest_file in self.archive_dir.glob("*_manifest.json"):
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            archives.append(manifest)
        
        return sorted(archives, key=lambda x: x['timestamp'], reverse=True)
    
    def get_archive_manifest(self, archive_id: str) -> Optional[ArchiveManifest]:
        """
        Get manifest for a specific archive.
        
        Args:
            archive_id: Archive identifier
        
        Returns:
            ArchiveManifest if exists, None otherwise
        """
        manifest_file = self.archive_dir / f"{archive_id}_manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ArchiveManifest(**data)
        
        return None
    
    def cleanup_old_archives(self, keep_count: int = 10):
        """
        Clean up old archives, keeping only the most recent.
        
        Args:
            keep_count: Number of archives to keep
        """
        archives = self.list_archives()
        
        if len(archives) > keep_count:
            # Remove oldest archives
            for archive_info in archives[keep_count:]:
                archive_id = archive_info['archive_id']
                
                # Remove archive file
                if self.compress:
                    archive_file = self.archive_dir / f"{archive_id}.zip"
                else:
                    archive_file = self.archive_dir / archive_id
                
                if archive_file.exists():
                    if archive_file.is_dir():
                        shutil.rmtree(archive_file)
                    else:
                        archive_file.unlink()
                
                # Remove manifest
                manifest_file = self.archive_dir / f"{archive_id}_manifest.json"
                if manifest_file.exists():
                    manifest_file.unlink()
