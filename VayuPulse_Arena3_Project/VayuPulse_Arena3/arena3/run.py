import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .data import DEMO_SENSORS, demo_data, load_cmapss, split_engines, validate
from .evaluate import metrics, rul_errors
from .explain import explain
from .model import Detector, score_frame


def execute(output, cmapss=None, rul_csv=None, seed=42, target_far=.05, shap_backend="builtin"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_cmapss(cmapss) if cmapss else demo_data(seed)
    candidates = [f"sensor_{i}" for i in range(1, 22)] if cmapss else DEMO_SENSORS
    data = validate(raw, candidates)
    train, val, test = split_engines(data, seed)
    # Healthy training proxy is deliberately stricter than the evaluation window.
    healthy_train = train[train.rul > 80] if cmapss else train[train.is_fault.eq(0)]
    healthy_val = val[val.is_fault.eq(0)]
    features = [c for c in candidates if healthy_train[c].std() > 1e-8]
    if not features:
        raise ValueError("No varying sensors in healthy training data.")
    comparison, fitted = [], []
    for kind, trees in [("isolation_forest", 100), ("isolation_forest", 200), ("robust_z", 0)]:
        model = Detector(features, kind, trees, seed).fit(healthy_train).calibrate(healthy_val, target_far)
        stats, _ = metrics(score_frame(model, val))
        name = f"{kind}_{trees}" if trees else kind
        comparison.append({"model": name, "threshold": model.threshold, **stats})
        fitted.append(model)
    # Compare all models; deliver IF as requested, tune its tree count on validation only.
    eligible = [i for i, m in enumerate(fitted) if m.kind == "isolation_forest"]
    def rank(i):
        m = comparison[i]
        return (m["false_alert_rate"] > target_far, m["missed_failure_rate"] if m["missed_failure_rate"] is not None else 1,
                -(m["median_lead_time_cycles_detected_only"] or 0), m["false_alert_rate"])
    selected = min(eligible, key=rank)
    model = fitted[selected]
    scored = score_frame(model, test)
    stats, events = metrics(scored)
    # One highest-scoring sample from each of the first eight held-out engines.
    selected_rows = scored.loc[scored.groupby("engine_id").anomaly_score.idxmax()].head(8)
    explanations = explain(model, healthy_train, selected_rows, backend=shap_backend)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scikit-learn", "joblib"]}
    if shap_backend == "shap":
        versions["shap"] = importlib.metadata.version("shap")
    report = {
        "project": "VayuPulse / Arena 3", "source": "NASA C-MAPSS supplied file" if cmapss else "SYNTHETIC DEMO",
        "label_definition": "RUL <= 30 cycles: proxy, not confirmed fault onset" if cmapss else "Known simulated degradation onset",
        "failure_assumption": "Final cycle is an observed failure; complete run-to-failure sequences only",
        "healthy_training_definition": "RUL > 80" if cmapss else "Simulated pre-onset observations",
        "selected_model": comparison[selected]["model"], "seed": seed, "persistence": 3,
        "shap_backend": shap_backend, "shap_permutations": 16, "shap_background_rows": 24,
        "target_validation_raw_false_alert_rate": target_far, "threshold": model.threshold,
        "features": features, "versions": versions,
        "data_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "splits": {name: sorted(part.engine_id.unique().tolist()) for name, part in
                   [("train", train), ("validation", val), ("test", test)]},
        "test_metrics": stats,
        "limitations": ["Synthetic results do not establish aircraft performance.",
                         "Scores are not calibrated failure probabilities.",
                         "SHAP describes model associations, not causal root causes.",
                         "Lead time is in observed cycles; missed failures are reported separately.",
                         "FD001 support assumes one operating regime; FD002/FD004 need regime-aware preprocessing.",
                         "No availability or downtime improvement is inferred from this experiment."],
    }
    joblib.dump(model, output / "detector.joblib")
    pd.DataFrame(comparison).to_csv(output / "validation_comparison.csv", index=False)
    scored.to_csv(output / "test_predictions.csv", index=False)
    events.to_csv(output / "failure_events.csv", index=False)
    explanations.to_csv(output / "shap_explanations.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(output / "feature_order.csv", index=False)
    if rul_csv:
        summary, errors = rul_errors(pd.read_csv(rul_csv))
        summary.to_csv(output / "rul_comparison.csv", index=False)
        errors.to_csv(output / "rul_errors.csv", index=False)
        report["rul_evaluation"] = "Externally supplied Arena 1 predictions; confirm identical held-out keys before model comparisons."
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# VayuPulse Arena 3 - Evaluation report", "", f"Source: **{report['source']}**", "",
             f"Label: {report['label_definition']}", "", f"Selected detector: {report['selected_model']}", "",
             "## Held-out test results", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in stats.items():
        text = "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {text} |")
    lines += ["", "## Interpretation", "", "Threshold and detector choice use validation engines only. Test engines are untouched until final evaluation.",
              "False-alert rate = alerted healthy rows / all healthy rows after persistence. Raw threshold calibration targets a separate 5% rate by default.",
              "A failure is detected only if a confirmed alert occurs inside its labelled degradation window and strictly before failure.",
              "Median lead time includes detected events only; read it alongside missed failures.", "", "## Limits", ""]
    lines += [f"- {x}" for x in report["limitations"]]
    (output / "EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    g = scored[scored.engine_id.eq(scored.engine_id.iloc[0])]
    ax.plot(g.cycle, g.anomaly_score, color="#127c96", label="Anomaly score")
    ax.axhline(model.threshold, color="#d45339", linestyle="--", label="Frozen threshold")
    ax.fill_between(g.cycle, 0, 1, where=g.is_fault.eq(1), transform=ax.get_xaxis_transform(),
                    alpha=.12, color="#d45339", label="Labelled degradation")
    ax.set(title=f"{report['source']} | Held-out engine {g.engine_id.iloc[0]}", xlabel="Cycle", ylabel="Score (higher = more abnormal)")
    ax.legend(loc="upper left")
    fig.savefig(output / "detection_timeline.png", dpi=160)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description="Arena 3 reproducible training, SHAP and validation")
    parser.add_argument("--output", default="artifacts/demo")
    parser.add_argument("--cmapss", help="Path to NASA train_FD001.txt (complete trajectories)")
    parser.add_argument("--rul-csv", help="Optional externally supplied Arena 1 RUL predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-far", type=float, default=.05)
    parser.add_argument("--shap-backend", choices=["builtin", "shap"], default="builtin")
    args = parser.parse_args()
    report = execute(args.output, args.cmapss, args.rul_csv, args.seed, args.target_far, args.shap_backend)
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
