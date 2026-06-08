from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from airfoil_discovery.config import FlowConfig, OptimizationConfig
from airfoil_discovery.geometry.prior import GeometryPriorFilter
from airfoil_discovery.ml.surrogate import MultiOutputAirfoilSurrogate
from airfoil_discovery.schemas import CandidateDesign


@dataclass(slots=True)
class CandidateBatch:
    selected: list[CandidateDesign]
    pool_size: int


class BayesianCandidateGenerator:
    def __init__(
        self,
        geometry_filter: GeometryPriorFilter,
        flow: FlowConfig,
        optimization: OptimizationConfig,
        random_seed: int,
    ):
        self.geometry_filter = geometry_filter
        self.flow = flow
        self.optimization = optimization
        self.rng = np.random.default_rng(random_seed)

    def initial_designs(self) -> list[CandidateDesign]:
        designs: list[CandidateDesign] = []
        seen: set[str] = set()
        while len(designs) < self.optimization.initial_random_samples:
            params = self.geometry_filter.sample_parameters(self.rng)
            signature = params.rounded_signature(self.optimization.duplicate_rounding)
            if signature in seen:
                continue
            valid, metrics = self.geometry_filter.is_valid(params)
            if not valid:
                continue
            reynolds = float(self.rng.uniform(self.flow.reynolds_min, self.flow.reynolds_max))
            designs.append(CandidateDesign(params=params, reynolds=reynolds, geometry_metrics=metrics))
            seen.add(signature)
        return designs

    def propose(
        self,
        surrogate: MultiOutputAirfoilSurrogate,
        batch_size: int,
        existing_case_keys: set[str],
        signature_decimals: int,
    ) -> CandidateBatch:
        if not surrogate.is_trained:
            selected = self.initial_designs()[:batch_size]
            return CandidateBatch(selected=selected, pool_size=len(selected))

        pool = self._build_pool(existing_case_keys, signature_decimals)
        feature_rows: list[dict[str, float]] = []
        point_ranges: list[tuple[int, int]] = []
        for design in pool:
            start = len(feature_rows)
            for aoa in self.flow.aoa_values:
                vec = design.params.as_vector()
                feature_rows.append(
                    {
                        "upper_0": float(vec[0]),
                        "upper_1": float(vec[1]),
                        "upper_2": float(vec[2]),
                        "upper_3": float(vec[3]),
                        "lower_0": float(vec[4]),
                        "lower_1": float(vec[5]),
                        "lower_2": float(vec[6]),
                        "lower_3": float(vec[7]),
                        "te_thickness": float(vec[8]),
                        "reynolds": design.reynolds,
                        "aoa_deg": aoa,
                    }
                )
            point_ranges.append((start, len(feature_rows)))

        frame = pd.DataFrame(feature_rows)
        mean, std = surrogate.predict_mean_std(frame)
        acquisition: list[tuple[float, float, CandidateDesign]] = []
        for design, (start, end) in zip(pool, point_ranges):
            segment_mean = mean[start:end]
            segment_std = std[start:end]
            cl = segment_mean[:, 0]
            cd = np.maximum(segment_mean[:, 1], 1.0e-5)
            eff = cl / cd
            pred_score = float(np.max(eff) - 10.0 * np.interp(4.0, self.flow.aoa_values, cd))
            uncertainty = float(np.mean(np.linalg.norm(segment_std, axis=1)))
            design.surrogate_prediction = {"predicted_score": pred_score, "uncertainty": uncertainty}
            acquisition.append((pred_score, uncertainty, design))

        acquisition.sort(key=lambda item: item[0], reverse=True)
        n_exploit = max(1, int(round(batch_size * self.optimization.exploitation_fraction)))
        n_random = max(0, int(round(batch_size * self.optimization.random_injection_fraction)))
        n_explore = max(0, int(round(batch_size * self.optimization.exploration_fraction)))
        while n_exploit + n_explore + n_random > batch_size:
            if n_explore > 0:
                n_explore -= 1
            elif n_random > 0:
                n_random -= 1
            else:
                n_exploit -= 1

        exploit = [item[2] for item in acquisition[:n_exploit]]
        explore = [item[2] for item in sorted(acquisition, key=lambda item: item[1], reverse=True)[:n_explore]]

        used = {design.params.rounded_signature(signature_decimals) + f"|{design.reynolds:.1f}" for design in exploit + explore}
        remaining = [item[2] for item in acquisition if item[2].params.rounded_signature(signature_decimals) + f"|{item[2].reynolds:.1f}" not in used]
        random_candidates: list[CandidateDesign] = []
        if n_random:
            self.rng.shuffle(remaining)
            for design in remaining:
                key = design.params.rounded_signature(signature_decimals) + f"|{design.reynolds:.1f}"
                random_candidates.append(design)
                used.add(key)
                if len(random_candidates) >= n_random:
                    break

        selected: list[CandidateDesign] = []
        selected_keys: set[str] = set()
        for design in exploit + explore + random_candidates:
            key = design.params.rounded_signature(signature_decimals) + f"|{design.reynolds:.1f}"
            if key in selected_keys:
                continue
            selected.append(design)
            selected_keys.add(key)
            if len(selected) >= batch_size:
                break

        for design in selected:
            prediction = design.surrogate_prediction or {}
            design.acquisition_score = prediction.get("predicted_score", 0.0) + 0.25 * prediction.get("uncertainty", 0.0)
        return CandidateBatch(selected=selected, pool_size=len(pool))

    def _build_pool(self, existing_case_keys: set[str], signature_decimals: int) -> list[CandidateDesign]:
        pool: list[CandidateDesign] = []
        attempts = 0
        max_attempts = self.optimization.candidate_pool * 20
        while len(pool) < self.optimization.candidate_pool and attempts < max_attempts:
            attempts += 1
            params = self.geometry_filter.sample_parameters(self.rng)
            signature = params.rounded_signature(signature_decimals)
            valid, metrics = self.geometry_filter.is_valid(params)
            if not valid:
                continue
            reynolds = float(self.rng.uniform(self.flow.reynolds_min, self.flow.reynolds_max))
            case_key = f"{signature}|Re={reynolds:.1f}"
            if case_key in existing_case_keys:
                continue
            pool.append(CandidateDesign(params=params, reynolds=reynolds, geometry_metrics=metrics))
        return pool
