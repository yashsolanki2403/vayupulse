import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(scored):
    healthy = scored.is_fault.eq(0)
    fault = ~healthy
    tp = int((scored.alert & fault).sum())
    fp = int((scored.alert & healthy).sum())
    fn = int((~scored.alert & fault).sum())
    events = []
    for unit, g in scored.groupby("engine_id"):
        failure_cycle = float((g.cycle + g.rul).iloc[0])
        # Event metrics require complete trajectories with observed terminal failure.
        if not g.is_fault.any() or not np.isclose(g.rul.min(), 0):
            continue
        window = g[g.is_fault.eq(1) & (g.cycle < failure_cycle)]
        hits = window[window.alert]
        first = None if hits.empty else float(hits.cycle.min())
        events.append({"engine_id": unit, "failure_cycle": failure_cycle,
                       "detected_before_failure": first is not None,
                       "first_detection_cycle": first,
                       "lead_time_cycles": None if first is None else failure_cycle - first})
    leads = [e["lead_time_cycles"] for e in events if e["lead_time_cycles"] is not None]
    missed = sum(not e["detected_before_failure"] for e in events)
    both = scored.is_fault.nunique() == 2
    result = {
        "rows": len(scored), "engines": int(scored.engine_id.nunique()),
        "false_alert_rate": fp / int(healthy.sum()) if healthy.any() else None,
        "false_alerts_per_1000_healthy_cycles": 1000 * fp / int(healthy.sum()) if healthy.any() else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "failure_events": len(events), "missed_failures": missed,
        "missed_failure_rate": missed / len(events) if events else None,
        "median_lead_time_cycles_detected_only": float(np.median(leads)) if leads else None,
        "roc_auc": float(roc_auc_score(scored.is_fault, scored.anomaly_score)) if both else None,
        "average_precision": float(average_precision_score(scored.is_fault, scored.anomaly_score)) if both else None,
    }
    return result, pd.DataFrame(events)


def rul_errors(frame):
    """Evaluate Arena 1 predictions supplied by another team member; never train RUL."""
    columns = ["model", "engine_id", "cycle", "y_true", "y_pred"]
    if set(columns) - set(frame):
        raise ValueError(f"RUL CSV requires {columns}")
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError("RUL input must be nonempty and complete.")
    frame = frame.copy()
    for col in ["cycle", "y_true", "y_pred"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[["cycle", "y_true", "y_pred"]]).all().all():
        raise ValueError("RUL values must be finite.")
    if frame.duplicated(["model", "engine_id", "cycle"]).any() or (frame[["y_true", "y_pred"]] < 0).any().any():
        raise ValueError("Reject duplicate model/engine/cycle keys and negative RUL.")
    frame["error"] = frame.y_pred - frame.y_true
    rows = []
    for name, g in frame.groupby("model"):
        err = g.error.to_numpy()
        near = g[g.y_true <= 30].error
        rows.append({"model": name, "rows": len(g), "mae": float(np.abs(err).mean()),
                     "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(err.mean()),
                     "overprediction_rate": float((err > 0).mean()),
                     "near_failure_mae": None if near.empty else float(near.abs().mean())})
    return pd.DataFrame(rows), frame
