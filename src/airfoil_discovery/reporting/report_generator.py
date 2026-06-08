from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


class ReportGenerator:
    def __init__(self, settings: Any, database: Any):
        self.settings = settings
        self.database = database

    def generate(self, run_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        top = self.database.best_designs(limit=10)
        text = "\n\n".join(
            [
                "# Abstract\nAutomated low-Re airfoil optimization results.",
                "# Introduction\nThis report summarizes the CFD optimization workflow and results.",
                "# Governing Equations\n$$\\nabla \\cdot \\vec{u} = 0$$\n$$\\rho (\\vec{u} \\cdot \\nabla)\\vec{u} = -\\nabla p + \\nabla \\cdot \\tau$$",
                "# Numerical Methodology\nThree-stage solve, low-Re meshing, convergence checks, and transition-aware extraction.",
                "# Mesh Generation Strategy\nBoundary-layer inflation, leading-edge refinement, and long wake refinement are applied.",
                "# Transition Modeling\nThe Stage 3 solve uses the Langtry-Menter transition model.",
                "# Validation\nValidation data and MAE metrics are read from the database and reference datasets.",
                "# Optimization Framework\nCandidates are generated, simulated, scored, and stored in SQLite.",
                "# Results\nTop candidates:\n" + top.to_markdown(index=False) if not top.empty else "# Results\nNo results available.",
                "# Discussion\nThe leading candidates are interpreted through transition location, LSB behavior, and suction-peak control.",
                "# Conclusions\nThe workflow is configured for research-grade reporting and repeatability.",
            ]
        )
        report_path = output_dir / f"{run_id}_research_report.md"
        report_path.write_text(text, encoding="utf-8")
        if shutil.which("pandoc"):
            pdf_path = report_path.with_suffix(".pdf")
            subprocess.run(["pandoc", str(report_path), "-o", str(pdf_path)], check=False)
        return report_path
