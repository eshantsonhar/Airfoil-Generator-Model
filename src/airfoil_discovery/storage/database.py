from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, func, insert, select, text

from airfoil_discovery.schemas import SimulationResult


class ExperimentDatabase:
    def __init__(self, db_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        self.metadata = MetaData()
        self.airfoils = Table(
            "airfoils",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("case_key", String, unique=True, nullable=False),
            Column("signature", String, nullable=False),
            Column("reynolds", Float, nullable=False),
            Column("upper_0", Float, nullable=False),
            Column("upper_1", Float, nullable=False),
            Column("upper_2", Float, nullable=False),
            Column("upper_3", Float, nullable=False),
            Column("lower_0", Float, nullable=False),
            Column("lower_1", Float, nullable=False),
            Column("lower_2", Float, nullable=False),
            Column("lower_3", Float, nullable=False),
            Column("te_thickness", Float, nullable=False),
            Column("prior_score", Float, nullable=False),
            Column("score", Float, nullable=False),
            Column("stall_angle_deg", Float, nullable=False),
            Column("cd_at_cruise", Float, nullable=False),
            Column("separation_penalty", Float, nullable=False),
            Column("instability_penalty", Float, nullable=False),
            Column("archive_url", String, nullable=True),
        )
        self.polar = Table(
            "polar_points",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("case_key", String, nullable=False),
            Column("signature", String, nullable=False),
            Column("reynolds", Float, nullable=False),
            Column("aoa_deg", Float, nullable=False),
            Column("cl", Float, nullable=False),
            Column("cd", Float, nullable=False),
            Column("efficiency", Float, nullable=False),
        )
        self.transition_points = Table(
            "transition_points",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("case_key", String, nullable=False),
            Column("aoa_deg", Float, nullable=False),
            Column("x_tr", Float, nullable=True),
            Column("x_sep", Float, nullable=True),
            Column("x_reat", Float, nullable=True),
            Column("bubble_length", Float, nullable=False),
            Column("cp_min", Float, nullable=False),
            Column("x_cp_min", Float, nullable=False),
            Column("lsb_detected", Integer, nullable=False),
            Column("flags", String, nullable=False),
        )
        self.uq_runs = Table(
            "uq_runs",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("parent_case_key", String, nullable=False),
            Column("perturbation", String, nullable=False),
            Column("cl", Float, nullable=False),
            Column("cd", Float, nullable=False),
            Column("x_tr", Float, nullable=True),
            Column("bubble_length", Float, nullable=True),
            Column("cv_cl", Float, nullable=False),
            Column("cv_cd", Float, nullable=False),
            Column("numerically_sensitive", Integer, nullable=False),
        )
        self.mis_results = Table(
            "mis_results",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("case_key", String, nullable=False),
            Column("mesh_level", String, nullable=False),
            Column("node_count", Integer, nullable=False),
            Column("cl", Float, nullable=False),
            Column("cd", Float, nullable=False),
            Column("gci_cl", Float, nullable=False),
            Column("gci_cd", Float, nullable=False),
        )
        self.metadata.create_all(self.engine)
        self._run_migrations()

    def _run_migrations(self) -> None:
        columns = {
            "x_tr_cruise": "FLOAT",
            "x_sep_cruise": "FLOAT",
            "x_reat_cruise": "FLOAT",
            "bubble_length_cruise": "FLOAT",
            "cp_min_cruise": "FLOAT",
            "large_bubble_penalty": "FLOAT",
            "suction_peak_penalty": "FLOAT",
            "physics_violation_penalty": "FLOAT",
            "lsb_detected": "INTEGER",
            "transition_inconsistent": "INTEGER",
            "unrealistic_early_transition": "INTEGER",
            "fully_laminar": "INTEGER",
            "data_incomplete": "INTEGER",
            "mesh_dependent": "INTEGER",
            "gci_cl": "FLOAT",
            "gci_cd": "FLOAT",
            "numerically_sensitive": "INTEGER",
            "surface_file": "STRING",
            "validation_run": "INTEGER",
        }
        with self.engine.begin() as conn:
            for name, kind in columns.items():
                try:
                    conn.execute(text(f"ALTER TABLE airfoils ADD COLUMN {name} {kind} DEFAULT NULL"))
                except Exception as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise

    def make_case_key(self, signature: str, reynolds: float) -> str:
        return f"{signature}|Re={reynolds:.1f}"

    def has_design(self, signature: str, reynolds: float) -> bool:
        case_key = self.make_case_key(signature, reynolds)
        stmt = select(self.airfoils.c.case_key).where(self.airfoils.c.case_key == case_key).limit(1)
        with self.engine.begin() as conn:
            return conn.execute(stmt).first() is not None

    def insert_result(self, result: SimulationResult, signature_decimals: int = 5) -> None:
        signature = result.candidate.params.rounded_signature(decimals=signature_decimals)
        case_key = self.make_case_key(signature, result.candidate.reynolds)
        params = result.candidate.params
        metrics = result.candidate.geometry_metrics
        if metrics is None:
            raise ValueError("Geometry metrics must be present before insertion.")
        extra = result.extra or {}
        verification = extra.get("verification", {})
        scoring = extra.get("score_components", {})
        cruise = extra.get("cruise_physics", {})
        with self.engine.begin() as conn:
            airfoil_values = dict(
                    case_key=case_key,
                    signature=signature,
                    reynolds=result.candidate.reynolds,
                    upper_0=float(params.upper[0]),
                    upper_1=float(params.upper[1]),
                    upper_2=float(params.upper[2]),
                    upper_3=float(params.upper[3]),
                    lower_0=float(params.lower[0]),
                    lower_1=float(params.lower[1]),
                    lower_2=float(params.lower[2]),
                    lower_3=float(params.lower[3]),
                    te_thickness=params.trailing_edge_thickness,
                    prior_score=metrics.prior_score,
                    score=result.score,
                    stall_angle_deg=result.stall_angle_deg,
                    cd_at_cruise=result.cd_at_cruise,
                    separation_penalty=result.separation_penalty,
                    instability_penalty=result.instability_penalty,
                    archive_url=result.archive_url,
                    x_tr_cruise=cruise.get("x_tr"),
                    x_sep_cruise=cruise.get("x_sep"),
                    x_reat_cruise=cruise.get("x_reat"),
                    bubble_length_cruise=cruise.get("bubble_length"),
                    cp_min_cruise=cruise.get("cp_min"),
                    large_bubble_penalty=scoring.get("large_bubble_penalty"),
                    suction_peak_penalty=scoring.get("suction_peak_penalty"),
                    physics_violation_penalty=scoring.get("physics_violation_penalty"),
                    lsb_detected=int(bool(verification.get("lsb_detected", False))) if verification else None,
                    transition_inconsistent=int(bool(verification.get("transition_inconsistent", False))) if verification else None,
                    unrealistic_early_transition=int(bool(verification.get("unrealistic_early_transition", False))) if verification else None,
                    fully_laminar=int(bool(verification.get("fully_laminar", False))) if verification else None,
                    data_incomplete=int(bool(scoring.get("data_incomplete", False))),
                    mesh_dependent=int(bool(extra.get("mesh_dependent", False))) if "mesh_dependent" in extra else None,
                    gci_cl=extra.get("gci_cl"),
                    gci_cd=extra.get("gci_cd"),
                    numerically_sensitive=int(bool(extra.get("numerically_sensitive", False))) if "numerically_sensitive" in extra else None,
                    surface_file=extra.get("surface_file"),
                    validation_run=int(bool(extra.get("validation_run", False))),
                )
            allowed_airfoil_columns = {column.name for column in self.airfoils.columns}
            conn.execute(
                insert(self.airfoils).values(
                    {key: value for key, value in airfoil_values.items() if key in allowed_airfoil_columns}
                )
            )
            conn.execute(
                insert(self.polar),
                [
                    {
                        "signature": signature,
                        "case_key": case_key,
                        "reynolds": result.candidate.reynolds,
                        "aoa_deg": point.aoa_deg,
                        "cl": point.cl,
                        "cd": point.cd,
                        "efficiency": point.efficiency,
                    }
                    for point in result.polar
                ],
            )
            for row in extra.get("transition_points", []):
                conn.execute(
                    insert(self.transition_points).values(
                        case_key=case_key,
                        aoa_deg=row["aoa_deg"],
                        x_tr=row.get("x_tr"),
                        x_sep=row.get("x_sep"),
                        x_reat=row.get("x_reat"),
                        bubble_length=row["bubble_length"],
                        cp_min=row["cp_min"],
                        x_cp_min=row["x_cp_min"],
                        lsb_detected=int(bool(row.get("lsb_detected", False))),
                        flags=json.dumps(row.get("flags", [])),
                    )
                )

    def insert_mis_result(self, case_key: str, mis: Any) -> None:
        with self.engine.begin() as conn:
            for mesh_level, node_count, cl, cd in zip(mis.mesh_levels, mis.node_counts, mis.cl_values, mis.cd_values):
                conn.execute(
                    insert(self.mis_results).values(
                        case_key=case_key,
                        mesh_level=mesh_level,
                        node_count=node_count,
                        cl=cl,
                        cd=cd,
                        gci_cl=mis.gci_cl,
                        gci_cd=mis.gci_cd,
                    )
                )

    def insert_uq_result(self, case_key: str, scenarios: list[dict[str, Any]], cv_cl: float, cv_cd: float, sensitive: bool) -> None:
        with self.engine.begin() as conn:
            for row in scenarios:
                conn.execute(
                    insert(self.uq_runs).values(
                        parent_case_key=case_key,
                        perturbation=row["perturbation"],
                        cl=row["cl"],
                        cd=row["cd"],
                        x_tr=row.get("x_tr"),
                        bubble_length=row.get("bubble_length"),
                        cv_cl=cv_cl,
                        cv_cd=cv_cd,
                        numerically_sensitive=int(sensitive),
                    )
                )

    def training_frame(self) -> pd.DataFrame:
        stmt = select(
            self.polar.c.signature,
            self.polar.c.case_key,
            self.polar.c.reynolds,
            self.polar.c.aoa_deg,
            self.polar.c.cl,
            self.polar.c.cd,
            self.airfoils.c.upper_0,
            self.airfoils.c.upper_1,
            self.airfoils.c.upper_2,
            self.airfoils.c.upper_3,
            self.airfoils.c.lower_0,
            self.airfoils.c.lower_1,
            self.airfoils.c.lower_2,
            self.airfoils.c.lower_3,
            self.airfoils.c.te_thickness,
            self.airfoils.c.score,
        ).join(self.airfoils, self.airfoils.c.case_key == self.polar.c.case_key)
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return pd.DataFrame(rows)

    def best_designs(self, limit: int = 10) -> pd.DataFrame:
        stmt = select(self.airfoils).order_by(self.airfoils.c.score.desc()).limit(limit)
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return pd.DataFrame(rows)

    def transition_points_for_case(self, case_key: str) -> pd.DataFrame:
        stmt = select(self.transition_points).where(self.transition_points.c.case_key == case_key).order_by(self.transition_points.c.aoa_deg)
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return pd.DataFrame(rows)

    def total_cases(self) -> int:
        stmt = select(func.count()).select_from(self.airfoils)
        with self.engine.begin() as conn:
            count = conn.execute(stmt).scalar_one()
        return int(count)
