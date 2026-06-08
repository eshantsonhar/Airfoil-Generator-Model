from airfoil_discovery.ui.app import JobConfig, _compute_limits, _sanitize_job_config


def test_runtime_limits_recommend_conservative_defaults() -> None:
    limits = _compute_limits()
    assert limits["recommended_iterations"] == 2
    assert limits["recommended_batch_size"] == 1
    assert limits["recommended_use_mpi"] is False
    assert 1 <= limits["recommended_n_cores"] <= 4


def test_job_config_is_clamped_to_safe_values() -> None:
    safe = _sanitize_job_config(
        JobConfig(
            iterations=99,
            batch_size=99,
            n_cores=99,
            use_mpi=True,
            mpi_ranks_per_case=32,
            omp_threads_per_rank=32,
            prefer_gpu=True,
        )
    )
    assert safe["iterations"] == 12
    assert safe["batch_size"] <= safe["limits"]["max_safe_batch_size"]
    assert safe["n_cores"] == safe["limits"]["recommended_n_cores"]
    assert safe["mpi_ranks_per_case"] <= safe["n_cores"]
    assert safe["omp_threads_per_rank"] >= 1
