"""
Pre-Flight Verification Engine.

Runs mandatory validation checks before any CFD execution:
  1. Environment/Binary validation (SU2_CFD, SU2_DEF existence, version)
  2. Mesh integrity (SU2 format valid, 2D, markers present)
  3. Directory permissions, write access, disk space
  4. Baseline design geometry validity
  5. Configuration file generation test
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreflightReport:
    """Result of pre-flight verification checks."""
    all_checks_passed: bool = False
    binary_check_passed: bool = False
    mesh_check_passed: bool = False
    directory_check_passed: bool = False
    geometry_check_passed: bool = False
    config_check_passed: bool = False
    disk_space_gb: float = 0.0
    required_disk_space_gb: float = 5.0
    su2_version: str = ""
    su2_def_version: str = ""
    mesh_info: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ── Binary Checks ─────────────────────────────────────────────────────────────

def check_su2_binary(bin_path: str, label: str) -> Tuple[bool, str]:
    """
    Verify an SU2 binary exists, is executable, and returns version info.

    Returns
    -------
    (ok, version_or_error) : tuple
    """
    path = Path(bin_path)
    if not path.exists():
        return False, f"Binary not found: {bin_path}"
    if not path.is_file():
        return False, f"Not a file: {bin_path}"
    if not os.access(str(path), os.X_OK):
        return False, f"Binary not executable: {bin_path}"

    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True, text=True, timeout=30,
            creationflags=creation_flags,
        )
        version = (result.stdout or result.stderr or "").strip()[:200]
        if result.returncode == 0 or result.returncode == 1:
            return True, version or f"{label} executed (rc={result.returncode})"
        else:
            return False, f"{label} returned rc={result.returncode}: {version}"
    except FileNotFoundError:
        return False, f"Binary not found at: {bin_path}"
    except subprocess.TimeoutExpired:
        return False, f"{label} timed out after 30s (may not be an executable)"
    except Exception as e:
        return False, f"{label} check failed: {e}"


# ── Mesh Checks ────────────────────────────────────────────────────────────────

def check_su2_mesh(mesh_path: Path) -> Tuple[bool, str]:
    """
    Validate an SU2 format mesh file.

    Checks:
      - File exists and is readable
      - Contains NDIME=2 (2D mesh)
      - Contains at least one marker boundary
      - Has reasonable file size (> 1KB)
      - Has correct read permissions

    Returns
    -------
    (ok, info_message) : tuple
    """
    if not mesh_path.exists():
        return False, f"Mesh file not found: {mesh_path}"
    if not mesh_path.is_file():
        return False, f"Not a file: {mesh_path}"

    try:
        size_kb = mesh_path.stat().st_size / 1024
    except OSError as e:
        return False, f"Cannot stat mesh file: {e}"

    if size_kb < 1:
        return False, f"Mesh file too small: {size_kb:.1f} KB (expected > 1 KB)"

    try:
        text = mesh_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read mesh file: {e}"

    lines = text.splitlines()
    if len(lines) < 10:
        return False, f"Mesh file has only {len(lines)} lines (corrupted?)"

    # Check NDIME
    ndime_found = False
    for line in lines[:50]:
        stripped = line.strip()
        if stripped.upper().startswith("NDIME"):
            try:
                ndime = int(stripped.split("=")[1].strip())
                if ndime == 2:
                    ndime_found = True
                else:
                    return False, f"NDIME={ndime}, expected NDIME=2 for 2D mesh"
            except (IndexError, ValueError):
                pass

    if not ndime_found:
        # Check if it might be a .su2 mesh without explicit NDIME
        if "NPOIN" not in text.upper() and "NELEM" not in text.upper():
            return False, "Cannot find NDIME, NPOIN, or NELEM - not a valid SU2 mesh"

    # Check for boundary markers
    marker_count = 0
    for line in lines:
        if line.strip().upper().startswith("NMARK"):
            marker_count += 1

    if marker_count == 0:
        return False, "No boundary markers (NMARK) found in mesh"

    # Build info string
    npoin = 0
    nelem = 0
    for line in lines:
        s = line.strip().upper()
        if s.startswith("NPOIN"):
            try:
                npoin = int(s.split("=")[1].strip())
            except (IndexError, ValueError):
                pass
        if s.startswith("NELEM"):
            try:
                nelem = int(s.split("=")[1].strip())
            except (IndexError, ValueError):
                pass

    info = f"{npoin} nodes, {nelem} elements, {marker_count} markers, {size_kb:.1f} KB"
    return True, info


# ── Directory Checks ──────────────────────────────────────────────────────────

def check_output_directory(output_path: Path, required_gb: float = 5.0) -> Tuple[bool, float, str]:
    """
    Verify output directory is writable and has sufficient disk space.

    Returns
    -------
    (ok, free_gb, message) : tuple
    """
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, 0.0, f"Cannot create output directory {output_path}: {e}"

    # Test write permission by creating a temp file
    test_file = output_path / ".preflight_write_test"
    try:
        test_file.write_text("preflight-check", encoding="utf-8")
        test_file.unlink()
    except Exception as e:
        return False, 0.0, f"Output directory not writable: {e}"

    # Check disk space
    try:
        if hasattr(shutil, "disk_usage"):
            usage = shutil.disk_usage(output_path)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < required_gb:
                return (False, free_gb,
                        f"Insufficient disk space: {free_gb:.1f} GB free, "
                        f"requires {required_gb:.1f} GB")
            return True, free_gb, f"{free_gb:.1f} GB free"
        else:
            return True, 0.0, "disk_usage not available on this platform"
    except Exception as e:
        return True, 0.0, f"Disk space check failed (non-critical): {e}"


# ── Geometry Check ────────────────────────────────────────────────────────────

def check_baseline_design(
    dv: Optional[np.ndarray],
    bounds: Optional["CSTBounds"] = None,
) -> Tuple[bool, str]:
    """
    Validate the baseline CST design vector produces a valid airfoil.
    """
    if dv is None:
        return False, "No design vector provided"

    if len(dv) != 12:
        return False, f"Design vector has {len(dv)} elements, expected 12"

    if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
        return False, "Design vector contains NaN or Inf values"

    try:
        from .cst import check_geometry_validity
        valid, reason = check_geometry_validity(dv, bounds=bounds)
        if not valid:
            return False, f"Invalid baseline geometry: {reason}"
        return True, "Geometry valid"
    except Exception as e:
        return False, f"Geometry check error: {e}"


# ── Config Generation Check ───────────────────────────────────────────────────

def check_config_generation(mesh_filename: str = "mesh.su2") -> Tuple[bool, str]:
    """
    Verify that SU2 configuration files can be generated without errors.
    """
    try:
        from .config_primal import generate_primal_config
        from .config_adjoint import generate_adjoint_config

        primal = generate_primal_config(mesh_filename=mesh_filename, aoa_deg=4.0, reynolds=1e5)
        required_keys = ["SOLVER", "KIND_TURB_MODEL", "KIND_TRANS_MODEL",
                         "MUSCL_FLOW", "CONV_NUM_METHOD_FLOW", "MESH_FILENAME"]
        for key in required_keys:
            if key not in primal:
                return False, f"Primal config missing key: {key}"

        return True, "Config generation verified"
    except Exception as e:
        return False, f"Config generation check failed: {e}"


# ── Main Preflight Runner ─────────────────────────────────────────────────────

def run_preflight_checks(
    su2_cfd_bin: str,
    mesh_path: Path,
    output_dir: Path,
    dv: Optional[np.ndarray] = None,
    bounds: Optional["CSTBounds"] = None,
    su2_def_bin: Optional[str] = None,
    required_disk_gb: float = 5.0,
    verbose: bool = True,
) -> PreflightReport:
    """
    Run all pre-flight checks and return a structured report.

    Parameters
    ----------
    su2_cfd_bin : str
        Path to SU2_CFD executable.
    mesh_path : Path
        Path to baseline mesh file.
    output_dir : Path
        Output directory for optimization results.
    dv : np.ndarray, optional
        Baseline design vector (12 CST coefficients).
    bounds : CSTBounds, optional
        Design variable bounds.
    su2_def_bin : str, optional
        Path to SU2_DEF executable.
    required_disk_gb : float
        Minimum required disk space in GB.
    verbose : bool
        Print check results to stdout.

    Returns
    -------
    PreflightReport
    """
    report = PreflightReport(required_disk_space_gb=required_disk_gb)
    all_ok = True

    if verbose:
        print("\n" + "=" * 60)
        print("PRE-FLIGHT VERIFICATION")
        print("=" * 60)

    # 1. Binary checks
    if verbose:
        print("\n[1/5] Verifying SU2 binaries...")
    ok, info = check_su2_binary(su2_cfd_bin, "SU2_CFD")
    report.binary_check_passed = ok
    report.su2_version = info
    if not ok:
        all_ok = False
        report.errors.append(f"SU2_CFD: {info}")
        if verbose:
            print(f"  [FAIL] SU2_CFD: {info}")
    else:
        if verbose:
            print(f"  [PASS] SU2_CFD: {info[:80]}")

    if su2_def_bin:
        ok_def, info_def = check_su2_binary(su2_def_bin, "SU2_DEF")
        report.su2_def_version = info_def
        if not ok_def:
            all_ok = False
            report.errors.append(f"SU2_DEF: {info_def}")
            if verbose:
                print(f"  [FAIL] SU2_DEF: {info_def[:80]}")
        else:
            if verbose:
                print(f"  [PASS] SU2_DEF: {info_def[:80]}")
    else:
        if verbose:
            print(f"  [SKIP] SU2_DEF not provided (mesh deformation disabled)")

    # 2. Mesh check
    if verbose:
        print("\n[2/5] Validating mesh...")
    ok, info = check_su2_mesh(mesh_path)
    report.mesh_check_passed = ok
    report.mesh_info = info
    if not ok:
        all_ok = False
        report.errors.append(f"Mesh: {info}")
        if verbose:
            print(f"  [FAIL] {info}")
    else:
        if verbose:
            print(f"  [PASS] Mesh: {info}")

    # 3. Directory check
    if verbose:
        print("\n[3/5] Checking output directory...")
    ok, free_gb, info = check_output_directory(output_dir, required_gb=required_disk_gb)
    report.directory_check_passed = ok
    report.disk_space_gb = free_gb
    if not ok:
        all_ok = False
        report.errors.append(f"Directory: {info}")
        if verbose:
            print(f"  [FAIL] {info}")
    else:
        if verbose:
            print(f"  [PASS] Output dir: {info}")

    # 4. Geometry check
    if verbose:
        print("\n[4/5] Checking baseline geometry...")
    ok, info = check_baseline_design(dv, bounds=bounds)
    report.geometry_check_passed = ok
    if not ok:
        all_ok = False
        report.errors.append(f"Geometry: {info}")
        if verbose:
            print(f"  [FAIL] {info}")
    else:
        if verbose:
            print(f"  [PASS] {info}")

    # 5. Config generation check
    if verbose:
        print("\n[5/5] Testing config generation...")
    ok, info = check_config_generation(mesh_filename=mesh_path.name)
    report.config_check_passed = ok
    if not ok:
        all_ok = False
        report.errors.append(f"Config: {info}")
        if verbose:
            print(f"  [FAIL] {info}")
    else:
        if verbose:
            print(f"  [PASS] {info}")

    report.all_checks_passed = all_ok

    if verbose:
        print(f"\n{'=' * 60}")
        if all_ok:
            print("PRE-FLIGHT: ALL CHECKS PASSED")
        else:
            print(f"PRE-FLIGHT: {len(report.errors)} FAILURES DETECTED")
            for err in report.errors:
                print(f"  - {err}")
        print(f"{'=' * 60}\n")

    return report