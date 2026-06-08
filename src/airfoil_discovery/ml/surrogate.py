from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from airfoil_discovery.config import MLConfig


class MultiOutputAirfoilSurrogate:
    feature_names = [
        "upper_0",
        "upper_1",
        "upper_2",
        "upper_3",
        "lower_0",
        "lower_1",
        "lower_2",
        "lower_3",
        "te_thickness",
        "reynolds",
        "aoa_deg",
    ]

    def __init__(self, config: MLConfig, random_seed: int):
        self.config = config
        self.random_seed = random_seed
        self.model = None
        self.backend = None

    def fit(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            self.model = None
            self.backend = None
            return
        x = frame[self.feature_names].to_numpy()
        y = frame[["cl", "cd"]].to_numpy()

        if self.config.model_type.lower() == "xgboost":
            try:
                from sklearn.multioutput import MultiOutputRegressor
                from xgboost import XGBRegressor

                base = XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.xgboost_max_depth,
                    learning_rate=self.config.xgboost_learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=self.random_seed,
                )
                self.model = MultiOutputRegressor(base)
                self.backend = "xgboost"
            except ImportError:
                self.model = RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.random_seed,
                    n_jobs=-1,
                )
                self.backend = "random_forest"
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.random_seed,
                n_jobs=-1,
            )
            self.backend = "random_forest"

        self.model.fit(x, y)

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict_mean_std(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Surrogate model has not been trained.")
        x = features[self.feature_names].to_numpy()
        mean = self.model.predict(x)
        if self.backend == "random_forest":
            tree_preds = np.stack([est.predict(x) for est in self.model.estimators_], axis=0)
            std = np.std(tree_preds, axis=0)
        elif self.backend == "xgboost":
            std = np.full_like(mean, 0.02)
        else:
            std = np.full_like(mean, 0.05)
        return mean, std
