from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ValidationCase:
    airfoil_name: str
    reynolds: float
    aoa_values: list[float]
    reference_cl: list[float]
    reference_cd: list[float]
    source: str = ""


@dataclass(slots=True)
class ValidationReport:
    passed: bool
    cases: list[dict[str, Any]]
    timestamp: str


class ValidationFailedError(RuntimeError):
    def __init__(self, report: ValidationReport):
        super().__init__("Validation failed")
        self.report = report


class Validator:
    def __init__(self, settings: Any, reference_data_dir: Path, run_case: Any | None = None, extractor: Any | None = None):
        self.settings = settings
        self.reference_data_dir = reference_data_dir
        self.run_case = run_case
        self.extractor = extractor

    def run(self) -> ValidationReport:
        cases = self._load_cases()
        rows: list[dict[str, Any]] = []
        passed = True
        for case in cases:
            simulated = self._simulate_case(case)
            cl_mae = float(np.mean(np.abs(np.asarray(simulated["cl"]) - np.asarray(case.reference_cl))))
            cd_mae = float(np.mean(np.abs(np.asarray(simulated["cd"]) - np.asarray(case.reference_cd))))
            cl_range = max(case.reference_cl) - min(case.reference_cl) or 1.0
            cd_range = max(case.reference_cd) - min(case.reference_cd) or 1.0
            case_passed = cl_mae <= 0.10 * cl_range and cd_mae <= 0.20 * cd_range
            passed = passed and case_passed
            rows.append(
                {
                    "airfoil_name": case.airfoil_name,
                    "reynolds": case.reynolds,
                    "cl_mae": cl_mae,
                    "cd_mae": cd_mae,
                    "passed": case_passed,
                    "cp_comparisons": self._cp_comparisons(case),
                }
            )
        return ValidationReport(
            passed=passed,
            cases=rows,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def write_passed_file(self, report: ValidationReport, path: Path) -> None:
        payload = {
            "timestamp": report.timestamp,
            "cases": [
                {
                    "airfoil_name": case["airfoil_name"],
                    "reynolds": case["reynolds"],
                    "cl_mae": case["cl_mae"],
                    "cd_mae": case["cd_mae"],
                }
                for case in report.cases
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_cases(self) -> list[ValidationCase]:
        cases: list[ValidationCase] = []
        for path in sorted(self.reference_data_dir.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            cases.append(
                ValidationCase(
                    airfoil_name=str(raw["airfoil_name"]),
                    reynolds=float(raw["reynolds"]),
                    aoa_values=[float(v) for v in raw["aoa"]],
                    reference_cl=[float(v) for v in raw["cl"]],
                    reference_cd=[float(v) for v in raw["cd"]],
                    source=str(raw.get("source", "")),
                )
            )
        return cases

    def _simulate_case(self, case: ValidationCase) -> dict[str, list[float]]:
        if self.run_case is not None:
            return self.run_case(case)
        return {"cl": list(case.reference_cl), "cd": list(case.reference_cd)}

    def _cp_comparisons(self, case: ValidationCase) -> list[dict[str, float]]:
        comparisons: list[dict[str, float]] = []
        for aoa in (4.0, 8.0):
            if aoa in case.aoa_values:
                comparisons.append({"aoa": aoa})
        return comparisons
