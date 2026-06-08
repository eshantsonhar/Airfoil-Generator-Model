# R-based Analytics Pipeline

This directory contains R scripts for uncertainty quantification and research analytics.

## Requirements

Install required R packages:

```r
install.packages(c("tidyverse", "lme4", "broom", "data.table", "ggplot2", "patchwork", "RSQLite"))
```

## Usage

### Run uncertainty analysis

```bash
Rscript uncertainty_analysis.R data/telemetry/metrics.db data/analytics
```

### Python wrapper

```python
from pathlib import Path
import subprocess

db_path = Path("data/telemetry/metrics.db")
output_dir = Path("data/analytics")

subprocess.run([
    "Rscript",
    "scripts/analytics/uncertainty_analysis.R",
    str(db_path),
    str(output_dir)
])
```

## Analytics Capabilities

- **Uncertainty quantification**: Bootstrap confidence intervals for all metrics
- **Repeated-run variance**: Statistical analysis of multiple optimization runs
- **Sensitivity analysis**: Variance decomposition to identify influential parameters
- **Mixed-effects modeling**: Account for repeated measures and hierarchical structure
- **Regression analysis**: Linear and non-linear trend analysis
- **Optimization trend analysis**: Track convergence and optimization progress
- **Reproducibility summaries**: Quantify run-to-run variability

## Output

The pipeline generates:

- Publication-ready figures (PNG, 300 DPI)
- Uncertainty envelopes
- Convergence plots
- Mesh-sensitivity plots
- Optimization-history analytics
- RDS files with analysis results

## Integration with Python

The R pipeline can be called from Python using subprocess or rpy2:

```python
import subprocess
from pathlib import Path

def run_r_analytics(db_path: Path, output_dir: Path):
    """Run R-based uncertainty analysis."""
    subprocess.run([
        "Rscript",
        "scripts/analytics/uncertainty_analysis.R",
        str(db_path),
        str(output_dir)
    ], check=True)
```
