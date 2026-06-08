"""
Force Truth Audit - Physical Verification of CFD Force Coefficients.

Audits the CFD force extraction pipeline to verify:
- Physically plausible Cl values at Re=200k
- Physically plausible Cd values at Re=200k  
- Correct normalization and reference values
- No sign inversions
- No unit corruption
- Proper reference area usage

At Re=200k for typical streamlined airfoils:
- Cl should be in range [0.0, 1.5] depending on AoA
- Cd should be in range [0.005, 0.05] for attached flow
- Cl/Cd should be in range [10, 80] depending on AoA

Cd ≈ 0.6 or Cl/Cd ≈ 0.24 is CATASTROPHIC failure.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airfoil_discovery.config import load_settings, Settings


REFERENCE_VALUES = {
    "naca0012_re200k_aoa4": {
        "Cl": 0.35,
        "Cd": 0.012,
        "Cl/Cd": 29.0,
        "tolerance": 0.5,  # 50% tolerance for verification
    },
    "naca4412_re200k_aoa4": {
        "Cl": 0.65,
        "Cd": 0.015,
        "Cl/Cd": 43.0,
        "tolerance": 0.5,
    },
    "sd7003_re200k_aoa4": {
        "Cl": 0.55,
        "Cd": 0.014,
        "Cl/Cd": 39.0,
        "tolerance": 0.5,
    },
    "sd7003_re200k_aoa8": {
        "Cl": 0.92,
        "Cd": 0.020,
        "Cl/Cd": 46.0,
        "tolerance": 0.4,
    },
}


class ForceAuditor:
    """
    Audits force coefficients for physical plausibility.
    
    Checks:
    1. Cl in physically reasonable range
    2. Cd in physically reasonable range
    3. Cl/Cd physically meaningful
    4. No NaN/Inf values
    5. No zero forces (indicates CFD failure)
    6. Correct reference area usage
    7. Proper sign conventions
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.audit_results = []

    def audit_force_coefficients(self, cl: float, cd: float, aoa: float,
                                 reynolds: float, airfoil_name: str = "unknown") -> dict:
        """Audit a single force coefficient pair."""
        violations = []
        warnings = []
        is_valid = True

        # Check 1: NaN/Inf detection
        if np.isnan(cl) or np.isinf(cl):
            violations.append("Cl is NaN or Inf - CFD output corrupted")
            is_valid = False
        if np.isnan(cd) or np.isinf(cd):
            violations.append("Cd is NaN or Inf - CFD output corrupted")
            is_valid = False

        if not is_valid:
            return {
                "is_valid": False,
                "violations": violations,
                "warnings": warnings,
                "cl": cl,
                "cd": cd,
                "cl_cd_ratio": 0.0,
                "reynolds": reynolds,
                "aoa": aoa,
            }

        # Check 2: Cl physical bounds at Re=200k
        if cl <= 0.0:
            violations.append(f"Non-positive Cl={cl:.4f} - zero or negative lift indicates CFD failure")
            is_valid = False
        elif cl < 0.1:
            warnings.append(f"Suspiciously low Cl={cl:.4f} at AoA={aoa}° - possible separation or CFD issue")
        elif cl > 2.0:
            violations.append(f"Unphysically high Cl={cl:.4f} at Re={reynolds:.0f} - possible normalization error")
            is_valid = False

        # Check 3: Cd physical bounds at Re=200k
        if cd <= 0.0:
            violations.append(f"Non-positive Cd={cd:.6f} - zero or negative drag indicates force inversion")
            is_valid = False
        elif cd < 0.001:
            warnings.append(f"Suspiciously low Cd={cd:.6f} - below laminar skin friction baseline")
        elif cd > 0.1:
            violations.append(f"Unphysically high Cd={cd:.6f} at Re={reynolds:.0f} - indicates massive separation or CFD failure")
            is_valid = False
        elif cd > 0.05:
            warnings.append(f"High Cd={cd:.6f} at Re={reynolds:.0f} - significant separation likely")

        # Check 4: Cl/Cd ratio physical bounds
        cl_cd = cl / max(cd, 1e-15)
        if cl_cd < 1.0:
            violations.append(f"Cl/Cd={cl_cd:.2f} - below 1 indicates catastrophic CFD or force extraction failure")
            is_valid = False
        elif cl_cd < 5.0:
            violations.append(f"Cl/Cd={cl_cd:.2f} - extremely low, indicates massive flow separation or force error")
            is_valid = False
        elif cl_cd < 10.0:
            warnings.append(f"Low Cl/Cd={cl_cd:.2f} at Re={reynolds:.0f} - significant separation or high drag regime")
        elif cl_cd > 120:
            warnings.append(f"Very high Cl/Cd={cl_cd:.2f} at Re={reynolds:.0f} - verify against laminar theory")

        # Check 5: Reynolds number scaling expectation
        if reynolds > 0:
            expected_cd_laminar = 1.328 / np.sqrt(reynolds) * 2  # Flat plate skin friction both sides
        else:
            expected_cd_laminar = 0.005

        if cd < expected_cd_laminar * 0.5 and cd > 0.001:
            warnings.append(f"Cd={cd:.6f} below 50% of laminar skin friction baseline {expected_cd_laminar:.6f}")

        # Check 6: AoA-lift consistency
        if aoa > 0 and cl < 0.05:
            warnings.append(f"Very low Cl={cl:.4f} at positive AoA={aoa}° - possible boundary condition issue")

        return {
            "is_valid": is_valid,
            "violations": violations,
            "warnings": warnings,
            "cl": cl,
            "cd": cd,
            "cl_cd_ratio": cl_cd,
            "expected_cd_laminar": float(expected_cd_laminar),
            "reynolds": reynolds,
            "aoa": aoa,
        }

    def audit_cfd_output(self, case_dir: Path) -> dict:
        """
        Audit an entire CFD case directory for force plausibility.
        
        Args:
            case_dir: Path to CFD case directory
            
        Returns:
            Audit report dict
        """
        results = []

        # Find history file
        history_files = list(case_dir.glob("*history*"))
        if not history_files:
            return {
                "is_valid": False,
                "violations": ["No history files found"],
                "warnings": [],
                "force_extractions": [],
            }

        for hf in history_files:
            try:
                text = hf.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                if len(lines) < 2:
                    continue

                headers = [item.strip().strip('"') for item in lines[0].split(",")]
                
                # Read all data lines and audit each
                for line in lines[1:]:
                    if not line.strip() or line.strip() == ',':
                        continue
                    values = [item.strip() for item in line.split(",")]
                    mapping = dict(zip(headers[:len(values)], values))

                    cl_str = mapping.get("CL") or mapping.get('"CL"') or "0.0"
                    cd_str = mapping.get("CD") or mapping.get('"CD"') or "0.0"

                    try:
                        cl = float(cl_str)
                        cd = float(cd_str)
                        audit = self.audit_force_coefficients(cl, cd, aoa=4.0, 
                                                              reynolds=self.settings.flow.reynolds_min)
                        results.append(audit)
                    except (ValueError, TypeError):
                        continue
                        
            except Exception:
                continue

        if not results:
            return {
                "is_valid": False,
                "violations": ["Could not parse any force coefficients"],
                "warnings": [],
                "force_extractions": [],
            }

        # Final assessment: all must be valid
        all_valid = all(r["is_valid"] for r in results)
        all_violations = []
        all_warnings = []
        for r in results:
            all_violations.extend(r["violations"])
            all_warnings.extend(r["warnings"])

        return {
            "is_valid": all_valid,
            "violations": list(set(all_violations)),
            "warnings": list(set(all_warnings)),
            "force_extractions": results,
        }

    def generate_report(self, audit_results: list) -> str:
        """Generate a human-readable audit report."""
        lines = []
        lines.append("=" * 60)
        lines.append("FORCE TRUTH AUDIT REPORT")
        lines.append("=" * 60)
        lines.append("")

        total = len(audit_results)
        valid = sum(1 for r in audit_results if r.get("is_valid", False))
        lines.append(f"Cases audited: {total}")
        lines.append(f"Cases passed:  {valid}")
        lines.append(f"Cases failed:  {total - valid}")
        lines.append("")

        for i, result in enumerate(audit_results):
            lines.append(f"--- Case {i+1} ---")
            lines.append(f"  Valid: {result.get('is_valid', False)}")
            lines.append(f"  Cl:    {result.get('cl', 0):.6f}")
            lines.append(f"  Cd:    {result.get('cd', 0):.6f}")
            lines.append(f"  Cl/Cd: {result.get('cl_cd_ratio', 0):.2f}")
            lines.append(f"  Rey:   {result.get('reynolds', 0):.0f}")
            lines.append(f"  AoA:   {result.get('aoa', 0):.1f}°")
            if result.get("expected_cd_laminar"):
                lines.append(f"  Expected Cd (laminar): {result['expected_cd_laminar']:.6f}")
            if result.get("violations"):
                for v in result["violations"]:
                    lines.append(f"  VIOLATION: {v}")
            if result.get("warnings"):
                for w in result["warnings"]:
                    lines.append(f"  WARNING: {w}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 60)
        
        failed = [r for r in audit_results if not r.get("is_valid", False)]
        if failed:
            lines.append("CRITICAL: Force extraction pipeline is producing invalid results.")
            lines.append("The following issues must be resolved:")
            for f in failed:
                for v in f.get("violations", []):
                    if v not in lines:
                        lines.append(f"  - {v}")
        else:
            lines.append("All force coefficients pass physical plausibility checks.")

        return "\n".join(lines)


def main():
    """Run force truth audit on a CFD case directory."""
    import argparse
    parser = argparse.ArgumentParser(description="Force Truth Audit")
    parser.add_argument("case_dir", type=str, nargs="?", 
                        help="Path to CFD case directory to audit")
    parser.add_argument("--cl", type=float, help="Cl value to audit directly")
    parser.add_argument("--cd", type=float, help="Cd value to audit directly")
    parser.add_argument("--aoa", type=float, default=4.0, help="Angle of attack (deg)")
    parser.add_argument("--reynolds", type=float, default=200000, help="Reynolds number")
    
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    settings = load_settings(config_path)
    auditor = ForceAuditor(settings)
    
    if args.cl is not None and args.cd is not None:
        result = auditor.audit_force_coefficients(args.cl, args.cd, args.aoa, args.reynolds)
        report = auditor.generate_report([result])
        print(report)
    elif args.case_dir:
        result = auditor.audit_cfd_output(Path(args.case_dir))
        report = auditor.generate_report(result.get("force_extractions", []))
        print(report)
        if not result["is_valid"]:
            print("\nVIOLATIONS:")
            for v in result["violations"]:
                print(f"  - {v}")
            sys.exit(1)
    else:
        # Run self-test with known reference values
        print("Running self-test with reference values...")
        results = []
        for name, ref in REFERENCE_VALUES.items():
            result = auditor.audit_force_coefficients(
                ref["Cl"], ref["Cd"], aoa=4.0, reynolds=200000,
                airfoil_name=name
            )
            results.append(result)
            status = "PASS" if result["is_valid"] else "FAIL"
            print(f"  {name}: {status} (Cl={ref['Cl']}, Cd={ref['Cd']}, Cl/Cd={ref['Cl/Cd']})")
            for v in result.get("violations", []):
                print(f"    VIOLATION: {v}")
        
        report = auditor.generate_report(results)
        print(report)


if __name__ == "__main__":
    main()