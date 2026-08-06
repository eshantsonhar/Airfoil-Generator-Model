"""Regression tests for error propagation / non-silent failure handling."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from airfoil_discovery.cfd.su2 import SU2ExecutionError, SU2Runner
from airfoil_discovery.pipeline import ASOPipeline, RuntimeTracker
from airfoil_discovery.ui.optimization_routes import (
    MetricExtractionError,
    extract_metrics,
)


def _runner() -> SU2Runner:
    return SU2Runner.__new__(SU2Runner)


def test_adjoint_extraction_reports_missing_files(tmp_path: Path) -> None:
    grad_cd, grad_cl, diagnostic = _runner()._extract_adjoint_gradients(tmp_path)
    assert np.allclose(grad_cd, 0.0) and np.allclose(grad_cl, 0.0)
    assert diagnostic is not None and "No adjoint files" in diagnostic


def test_adjoint_extraction_raises_on_unparsable_file(tmp_path: Path) -> None:
    (tmp_path / "surface_adjoint.csv").write_text("x,y,dx,dy\nnot,a,number,here\n")
    with pytest.raises(SU2ExecutionError) as excinfo:
        _runner()._extract_adjoint_gradients(tmp_path)
    assert excinfo.value.stage == "ADJOINT_EXTRACTION"


def test_adjoint_extraction_raises_on_wrong_shape(tmp_path: Path) -> None:
    (tmp_path / "surface_adjoint.csv").write_text("x,y\n0.0,1.0\n0.5,2.0\n")
    with pytest.raises(SU2ExecutionError):
        _runner()._extract_adjoint_gradients(tmp_path)


def test_adjoint_extraction_reports_degenerate_sensitivities(tmp_path: Path) -> None:
    rows = "\n".join(f"{i / 10.0},0.0,0.0,0.0" for i in range(10))
    (tmp_path / "surface_adjoint.csv").write_text("x,y,dx,dy\n" + rows + "\n")
    _, _, diagnostic = _runner()._extract_adjoint_gradients(tmp_path)
    assert diagnostic is not None and "degenerate" in diagnostic


def test_extract_metrics_raises_on_malformed_history(tmp_path: Path) -> None:
    (tmp_path / "history.csv").write_text('"CL","CD"\nnan-value,0.01\n')
    with pytest.raises(MetricExtractionError):
        extract_metrics(tmp_path)


def test_extract_metrics_logs_lsb_failure_but_keeps_forces(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "history.csv").write_text('"CL","CD"\n0.8,0.01\n')
    # Malformed surface file: large enough to be read, impossible to parse.
    (tmp_path / "surface_flow.csv").write_text('"x","y","cf"\n' + "garbage,row\n" * 20)
    with caplog.at_level(logging.WARNING):
        cl, cd, lsb = extract_metrics(tmp_path)
    assert (cl, cd, lsb) == (0.8, 0.01, None)
    assert "LSB extraction failed" in caplog.text


class _StubTelemetry:
    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []

    def failure(self, category: str, message: str, **details: object) -> None:
        self.failures.append((category, message))


def test_pipeline_run_records_and_reraises_unexpected_errors(tmp_path: Path) -> None:
    pipeline = ASOPipeline.__new__(ASOPipeline)
    pipeline.tracker = RuntimeTracker(tmp_path / "latest_runtime.json")
    pipeline.telemetry = _StubTelemetry()

    def _boom(**_kwargs: object) -> str:
        raise RuntimeError("solver exploded")

    pipeline._run = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="solver exploded"):
        pipeline.run()

    assert pipeline.tracker.status == "failed"
    assert pipeline.telemetry.failures == [
        ("pipeline_exception", "RuntimeError: solver exploded")
    ]
    persisted = json.loads((tmp_path / "latest_runtime.json").read_text())
    assert persisted["status"] == "failed"
