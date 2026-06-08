from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    name: str
    random_seed: int = 42


class PathsConfig(BaseModel):
    work_root: Path
    database_path: Path
    plots_dir: Path
    remote_manifest_dir: Path


class GeometryConfig(BaseModel):
    samples_per_surface: int
    upper_bounds: tuple[float, float]
    lower_bounds: tuple[float, float]
    te_thickness_bounds: tuple[float, float]
    thickness_bounds: tuple[float, float]
    camber_bounds: tuple[float, float]
    max_curvature_spike: float
    min_le_radius: float
    smoothness_penalty_scale: float
    prior_threshold: float


class FlowConfig(BaseModel):
    reynolds_min: float
    reynolds_max: float
    mach: float
    aoa_start: float
    aoa_end: float
    aoa_step: float
    reference_area: float
    reference_length: float

    @property
    def aoa_values(self) -> list[float]:
        values = []
        aoa = self.aoa_start
        while aoa <= self.aoa_end + 1e-9:
            values.append(round(aoa, 6))
            aoa += self.aoa_step
        return values


class MeshConfig(BaseModel):
    farfield_radius: float
    wake_length: float
    surface_points: int
    boundary_layer_first_height: float
    boundary_layer_growth: float
    boundary_layer_layers: int
    y_plus_target: float = 0.8
    min_boundary_layer_layers: int = 30
    max_boundary_layer_layers: int = 50
    coarse_layer_min: int = 20
    coarse_layer_max: int = 60


class SolverConfig(BaseModel):
    su2_cfd_bin: str
    gmsh_bin: str
    n_cores: int = 0
    use_mpi: bool = False
    mpiexec_bin: str = "mpiexec"
    mpi_ranks_per_case: int = 1
    omp_threads_per_rank: int = 0
    case_timeout_seconds: int = 0
    prefer_gpu: bool = False
    gpu_backend: str = "auto"
    gpu_device_id: int = 0
    transition_model: bool = True
    keep_local_if_upload_fails: bool = True
    startup_iterations: int = 600
    startup_cfl: float = 1.5
    startup_use_transition: bool = False
    startup_muscl: bool = False
    startup_cfl_adapt: bool = False
    stage1_iter: int = 500
    stage2_iter: int = 1500
    stage3_iter: int = 3000
    stage1_cfl: float = 5.0
    stage2_cfl: float = 3.0
    stage3_cfl: float = 2.0
    convergence_window: int = 100
    convergence_cl_cd_tol: float = 0.005
    convergence_residual_drop: float = 6.0
    stage1_coarse_factor: float = 2.0
    stage2_coarse_factor: float = 1.4
    stage3_coarse_factor: float = 1.0
    stage3_turbulence_intensity: float = 0.001
    stage3_turb_viscosity_ratio: float = 5.0
    mesh: MeshConfig


class OptimizationConfig(BaseModel):
    iterations: int
    batch_size: int
    candidate_pool: int
    exploitation_fraction: float
    random_injection_fraction: float
    exploration_fraction: float
    initial_random_samples: int
    duplicate_rounding: int = 5


class ScoringConfig(BaseModel):
    w1: float
    w2: float
    w3: float
    w4: float
    w5: float
    cruise_aoa: float
    w_bubble: float = 1.0
    w_peak: float = 1.0
    w_physics_violation: float = 1.0


class StorageConfig(BaseModel):
    provider: str = "local_manifest"
    local_manifest_base_url: str = "file://remote-manifest"
    supabase_url: str = ""
    supabase_api_key: str = ""
    supabase_bucket: str = ""
    firebase_bucket: str = ""
    firebase_bearer_token: str = ""


class MLConfig(BaseModel):
    model_type: str = "random_forest"
    n_estimators: int = 400
    min_samples_leaf: int = 2
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.05


class VerificationConfig(BaseModel):
    """Configuration for verification systems."""
    
    # GCI parameters
    gci_safety_factor: float = 1.25
    gci_theoretical_order: float = 2.0
    gci_asymptotic_range_lower: float = 0.5
    gci_asymptotic_range_upper: float = 1.5
    
    # Convergence parameters
    residual_threshold: float = 1e-6
    stagnation_threshold: float = 1e-3
    stagnation_iterations: int = 50
    min_iterations: int = 100
    force_stabilization_threshold: float = 0.001
    force_oscillation_threshold: float = 0.005
    force_drift_threshold: float = 0.001
    stabilization_window: int = 50
    
    # Gradient verification parameters
    fd_tolerance: float = 0.05
    fd_step_size: float = 1e-6
    directional_tolerance: float = 0.10
    cosine_threshold: float = 0.95
    variance_threshold: float = 0.5
    adaptive_fd_verification: bool = True
    
    # Mesh verification parameters
    max_y_plus: float = 1.0
    skewness_threshold: float = 0.85
    aspect_ratio_threshold: float = 1000.0
    orthogonality_threshold: float = 0.1
    le_min_cells_per_chord: int = 200
    wake_min_cells_per_chord: int = 100
    transition_min_cells_per_chord: int = 150
    curvature_min_cells_per_degree: int = 10
    
    # Numerical dissipation parameters
    artificial_viscosity_threshold: float = 0.01
    limiter_saturation_threshold: float = 0.9
    tv_growth_threshold: float = 0.1
    flux_dissipation_threshold: float = 0.1
    
    # Governance parameters
    require_convergence: bool = True
    require_gradient_validity: bool = True
    require_mesh_validity: bool = True
    require_transition_validity: bool = True
    require_physical_plausibility: bool = True
    allow_numerical_dissipation_warning: bool = False


class ValidationConfig(BaseModel):
    """Configuration for validation against literature benchmarks."""
    
    # Benchmark parameters
    cl_tolerance: float = 0.10
    cd_tolerance: float = 0.20
    transition_tolerance: float = 0.10
    min_correlation: float = 0.90
    
    # Benchmark directory
    benchmark_dir: str = "src/airfoil_discovery/validation/literature_benchmarks"
    
    # Required benchmarks
    required_benchmarks: list[str] = [
        "eppler387",
        "sd7003",
        "s1223",
        "naca4412_lowre"
    ]


class ReproducibilityConfig(BaseModel):
    """Configuration for reproducibility infrastructure."""
    
    # Seed management
    master_seed: int = 42
    deterministic_random: bool = True
    
    # Hashing
    hash_configs: bool = True
    hash_meshes: bool = True
    hash_binaries: bool = True
    
    # Serialization
    save_runtime_state: bool = True
    save_environment_snapshot: bool = True
    
    # Archival
    archive_outputs: bool = True
    compress_archives: bool = True
    keep_archives_count: int = 10


class GovernanceConfig(BaseModel):
    """Configuration for CFD governance and failure policies."""
    
    # Failure policy
    stop_on_critical: bool = True
    stop_on_severe: bool = True
    stop_on_moderate: bool = False
    
    # Crash preservation
    preserve_crash_state: bool = True
    crash_dir: str = "data/crash_states"
    
    # Wording checks
    enable_wording_checks: bool = True
    check_reports: bool = True
    check_generated_text: bool = True


class Settings(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    geometry: GeometryConfig
    flow: FlowConfig
    solver: SolverConfig
    optimization: OptimizationConfig
    scoring: ScoringConfig
    storage: StorageConfig
    ml: MLConfig
    verification: VerificationConfig = VerificationConfig()
    validation: ValidationConfig = ValidationConfig()
    reproducibility: ReproducibilityConfig = ReproducibilityConfig()
    governance: GovernanceConfig = GovernanceConfig()

    def ensure_directories(self) -> None:
        self.paths.work_root.mkdir(parents=True, exist_ok=True)
        self.paths.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.plots_dir.mkdir(parents=True, exist_ok=True)
        self.paths.remote_manifest_dir.mkdir(parents=True, exist_ok=True)
        
        # Create verification and governance directories
        Path(self.governance.crash_dir).mkdir(parents=True, exist_ok=True)
        Path(self.validation.benchmark_dir).parent.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path)
    _load_dotenv(path.parent.parent / ".env")
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    settings = Settings.model_validate(raw)
    _apply_env_overrides(settings)
    settings.ensure_directories()
    return settings


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _apply_env_overrides(settings: Settings) -> None:
    if value := os.getenv("SU2_CFD_BIN"):
        settings.solver.su2_cfd_bin = value
    if value := os.getenv("GMSH_BIN"):
        settings.solver.gmsh_bin = value
    if value := os.getenv("MPIEXEC_BIN"):
        settings.solver.mpiexec_bin = value
    if value := os.getenv("SU2_USE_MPI"):
        settings.solver.use_mpi = value.strip().lower() in {"1", "true", "yes", "on"}
    if value := os.getenv("SU2_MPI_RANKS"):
        settings.solver.mpi_ranks_per_case = max(1, int(value))
    if value := os.getenv("SU2_N_CORES"):
        settings.solver.n_cores = int(value)
    if value := os.getenv("SU2_OMP_THREADS"):
        settings.solver.omp_threads_per_rank = max(0, int(value))
    if value := os.getenv("SU2_CASE_TIMEOUT_SECONDS"):
        settings.solver.case_timeout_seconds = max(0, int(value))
    if value := os.getenv("SU2_PREFER_GPU"):
        settings.solver.prefer_gpu = value.strip().lower() in {"1", "true", "yes", "on"}
    if value := os.getenv("SU2_GPU_BACKEND"):
        settings.solver.gpu_backend = value.strip().lower()
    if value := os.getenv("SU2_GPU_DEVICE_ID"):
        settings.solver.gpu_device_id = max(0, int(value))
    if value := os.getenv("SUPABASE_URL"):
        settings.storage.supabase_url = value
    if value := os.getenv("SUPABASE_API_KEY"):
        settings.storage.supabase_api_key = value
    if value := os.getenv("SUPABASE_BUCKET"):
        settings.storage.supabase_bucket = value
    if value := os.getenv("FIREBASE_BUCKET"):
        settings.storage.firebase_bucket = value
    if value := os.getenv("FIREBASE_BEARER_TOKEN"):
        settings.storage.firebase_bearer_token = value
