# Airfoil Discovery System

Production-oriented Python system for discovering high-performance, manufacturable 2D airfoils in the Reynolds number range `1e4` to `5e4` using CST geometry, SU2 CFD, surrogate modeling, Bayesian candidate selection, SQLite tracking, and remote archival of CFD folders.

## Folder Structure

```text
.
|-- config/
|   `-- default.yaml
|-- data/
|   |-- cache/
|   |-- database/
|   |-- plots/
|   `-- remote_manifest/
|-- scripts/
|   `-- run_optimization.py
|-- src/
|   `-- airfoil_discovery/
|       |-- cfd/
|       |-- geometry/
|       |-- ml/
|       |-- optimization/
|       |-- storage/
|       |-- visualization/
|       |-- config.py
|       |-- pipeline.py
|       `-- schemas.py
|-- tests/
|   `-- test_geometry.py
`-- pyproject.toml
```

## Setup

# Airfoil Discovery System

This project is a production-oriented Python system designed for discovering high-performance, manufacturable 2D airfoils in the low-Reynolds-number regime (1e4 to 5e4). It integrates CST geometry parametrization, SU2 CFD simulations, machine learning surrogate models, and Bayesian optimization to iterate through airfoil designs efficiently.

## Core Architecture

The codebase is structured as a modular Python package located in `src/airfoil_discovery/`.

### Modules
- `geometry/`: Implements CST (Class Shape Transformation) parametrization to generate and validate airfoil geometries.
- `cfd/`: Manages the computational fluid dynamics pipeline, including airfoil export, Gmsh-based meshing, SU2 execution, and polar extraction.
- `ml/`: Contains surrogate models (e.g., XGBoost) used to predict `Cl` and `Cd` for new candidates, reducing the need for expensive CFD runs.
- `optimization/`: Handles the scoring of airfoils and candidate selection using Bayesian-style optimization techniques.
- `storage/`: Manages a SQLite database for tracking simulation metadata and supports remote archival of full CFD case folders.
- `ui/`: Provides a FastAPI-based web dashboard for monitoring and managing optimization runs.
- `visualization/`: Generates plots for optimization progress and airfoil shapes.

## Key Features
- **Efficiency:** Uses surrogate models to accelerate discovery and minimize full CFD runs.
- **Scalability:** Supports MPI-parallelized CFD simulations.
- **Workflow:** Automated pipeline from geometry generation to performance evaluation and archiving.
- **Dashboard:** Interactive UI for running and managing optimizations.

## Getting Started

### Prerequisites
1. Install Python 3.10+ and create a virtual environment.
2. Install package dependencies with `pip install -e .[ml,meshing,ui]`.
3. Install SU2 and expose `SU2_CFD` on `PATH`, or set `SU2_CFD_BIN`.
4. Install `gmsh` and expose `gmsh` on `PATH`, or set `GMSH_BIN`.

### Setup
1. Copy `.env.example` to `.env` and fill remote-storage credentials if needed.

### Running the System
- **CLI Optimization:** Use `scripts/run_optimization.py --config config/default.yaml`
- **Web UI:** Execute `launch_ui.bat` (Windows) or `python scripts/run_ui.py` to start the dashboard.

## Development & Maintenance
The project structure is optimized for maintainability, keeping core logic within `src/` while providing scripts for automation and testing.

### SU2 Guidance

Recommended solver choices for this workflow:

- `INC_RANS`
- `SST k-omega`
- transition model enabled when your SU2 build supports it
- angle-of-attack sweep from `-2 deg` to `15 deg`
- Reynolds number between `10000` and `50000`

The pipeline stores only compact scalar polar data locally. Raw CFD folders are zipped, uploaded, and then deleted locally after successful archival.

## Example Run

```bash
python scripts/run_optimization.py --config config/default.yaml --iterations 8 --batch-size 6
```

## Local Setup Commands

On this machine, the project is set up to run from the local virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_check.py
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\run_optimization.py --config config/default.yaml --iterations 2 --batch-size 2
```

The current `.env` file is used automatically by the config loader. Set `SU2_CFD_BIN` there to either the executable name on `PATH` or the full path to your `SU2_CFD.exe`.

If you want MPI acceleration, also set:

```env
SU2_USE_MPI=true
MPIEXEC_BIN=mpiexec
SU2_MPI_RANKS=4
```

`n_cores` in `config/default.yaml` is treated as the total CPU budget for CFD. When MPI is enabled, the runner computes:

```text
parallel CFD cases = floor(n_cores / SU2_MPI_RANKS)
```

So with `n_cores: 16` and `SU2_MPI_RANKS=4`, the pipeline will run up to 4 CFD cases at once, each using 4 MPI ranks.

## Web Dashboard

The easiest way to start the UI on this machine is to double-click `launch_ui.bat` in the project root. It will:

- create `.venv` if needed
- install the project with the UI extras
- copy `.env.example` to `.env` if `.env` does not exist yet
- start the dashboard on `http://127.0.0.1:8000`
- open the default browser once the server is ready

If you want to start it manually, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_ui.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

The dashboard now starts with conservative defaults intended to avoid overloading the machine:

- `2` iterations
- batch size `1`
- MPI disabled by default
- CPU budget capped to a safe recommended value from detected system cores

If you intentionally want a heavier run, you can still raise the advanced compute settings from the UI.

## Stored Features

Each CFD operating point contributes only:

- CST coefficients
- trailing-edge thickness
- Reynolds number
- angle of attack
- `Cl`
- `Cd`
- `Cl/Cd`

Aggregated design-level score terms are also stored for ranking and optimization. Full field data is not retained locally.

## Main Modules

- `geometry/`: CST generation, geometric validation, and prior filtering
- `cfd/`: airfoil export, Gmsh meshing, SU2 execution, and polar extraction
- `storage/`: SQLite cache and remote archive upload/retrieval
- `ml/`: surrogate models for `Cl` and `Cd`
- `optimization/`: scoring and Bayesian-style candidate selection
- `visualization/`: optimization progress and airfoil plots
