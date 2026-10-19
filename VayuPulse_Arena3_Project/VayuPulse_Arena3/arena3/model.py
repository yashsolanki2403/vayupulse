from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class Detector:
    features: list
    kind: str = "isolation_forest"
    n_estimators: int = 150
    seed: int = 42
    threshold: float = 0.0

    def fit(self, healthy):
        x = healthy[self.features].to_numpy(float)
        if len(x) < 50:
            raise ValueError("At least 50 healthy training rows required.")
        self.center = np.median(x, axis=0)
        self.scale = np.maximum(np.median(np.abs(x - self.center), axis=0) * 1.4826, 1e-8)
        if self.kind == "isolation_forest":
            self.estimator = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed,
                                            contamination="auto", n_jobs=1).fit(x)
        elif self.kind != "robust_z":
            raise ValueError("Unknown detector kind.")
        return self

    def score(self, x):
        if isinstance(x, pd.DataFrame):
            x = x[self.features].to_numpy(float)
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features) or not np.isfinite(x).all():
            raise ValueError("Expected a finite 2D sensor matrix in the documented feature order.")
        if self.kind == "isolation_forest":
            return -self.estimator.score_samples(x)
        return np.max(np.abs((x - self.center) / self.scale), axis=1)

    def calibrate(self, healthy_validation, target_far=.05):
        if not 0 < target_far < 1 or healthy_validation.empty:
            raise ValueError("Need healthy validation rows and a false-alert target between 0 and 1.")
        self.threshold = float(np.quantile(self.score(healthy_validation), 1 - target_far, method="higher"))
        return self


def score_frame(model, frame, persistence=3):
    """Alert after k consecutive above-threshold observations; no look-ahead."""
    if persistence < 1:
        raise ValueError("Persistence must be positive.")
    out = frame.sort_values(["engine_id", "cycle"]).copy()
    out["anomaly_score"] = model.score(out)
    out["raw_alert"] = out.anomaly_score > model.threshold
    out["alert"] = False
    for _, g in out.groupby("engine_id"):
        run, previous = 0, None
        for idx, row in g.iterrows():
            if previous is not None and row.cycle != previous + 1:
                run = 0
            run = run + 1 if row.raw_alert else 0
            out.loc[idx, "alert"] = run >= persistence
            previous = row.cycle
    return out
