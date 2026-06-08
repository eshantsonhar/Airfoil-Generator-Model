from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from airfoil_discovery.geometry.cst import CSTAirfoil
from airfoil_discovery.schemas import CSTParameters


class AirfoilPlotter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def plot_progress(self, history: pd.DataFrame) -> Path:
        path = self.output_dir / "optimization_progress.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history["iteration"], history["best_score"], marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best score")
        ax.set_title("Optimization Progress")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_predicted_vs_actual(self, frame: pd.DataFrame) -> Path:
        path = self.output_dir / "predicted_vs_actual.png"
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(frame["predicted_score"], frame["actual_score"], alpha=0.7)
        lo = min(frame["predicted_score"].min(), frame["actual_score"].min())
        hi = max(frame["predicted_score"].max(), frame["actual_score"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Actual score")
        ax.set_title("Predicted vs Actual Performance")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_best_airfoil(self, airfoil: CSTAirfoil, params: CSTParameters) -> Path:
        path = self.output_dir / "best_airfoil.png"
        coords = airfoil.full_coordinates(params)
        fig, ax = plt.subplots(figsize=(9, 2.5))
        ax.plot(coords[:, 0], coords[:, 1], linewidth=2.0)
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title("Best Airfoil Geometry")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
