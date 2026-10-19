"""Read-only local results dashboard; run with python -m streamlit run app.py."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VayuPulse | Arena 3", page_icon="✈", layout="wide")
st.markdown("<style>.stApp{background:#0c1424}h1,h2,h3{color:#e7f1ff}div[data-testid='stMetric']{background:#16243a;padding:18px;border-radius:12px}</style>", unsafe_allow_html=True)
st.caption("VAYUPULSE / ENGINE HEALTH INTELLIGENCE")
st.title("Anomaly detection. Explained.")
st.write("Arena 3 · Detection, explainability and validation")
directory = Path(st.sidebar.text_input("Artifacts directory", "artifacts/demo"))
try:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    scored = pd.read_csv(directory / "test_predictions.csv")
    comparison = pd.read_csv(directory / "validation_comparison.csv")
    explanations = pd.read_csv(directory / "shap_explanations.csv")
    events = pd.read_csv(directory / "failure_events.csv")
except (OSError, ValueError) as exc:
    st.info("Generate results first: python -m arena3.run")
    st.error(str(exc))
    st.stop()
st.info(f"{report['source']} · {report['label_definition']}")
m = report["test_metrics"]
cols = st.columns(4)
cols[0].metric("Healthy-cycle false alerts", "N/A" if m["false_alert_rate"] is None else f"{m['false_alert_rate']:.2%}")
cols[1].metric("Missed failure events", f"{m['missed_failures']} / {m['failure_events']}")
cols[2].metric("Median warning lead time", "N/A" if m["median_lead_time_cycles_detected_only"] is None else f"{m['median_lead_time_cycles_detected_only']:.0f} cycles")
cols[3].metric("Fault-row recall", "N/A" if m["recall"] is None else f"{m['recall']:.1%}")
st.caption("Lead time includes detected failures only. All metrics above use held-out engines.")
tab1, tab2, tab3, tab4 = st.tabs(["Engine investigation", "SHAP evidence", "Model validation", "Reports"])
with tab1:
    unit = st.selectbox("Held-out engine", sorted(scored.engine_id.unique()))
    g = scored[scored.engine_id.eq(unit)].copy()
    g["threshold"] = report["threshold"]
    st.line_chart(g.set_index("cycle")[["anomaly_score", "threshold"]], color=["#45c9dd", "#ff8568"])
    st.caption("Higher scores indicate greater abnormality. Alert requires three consecutive threshold exceedances.")
    sensor = st.selectbox("Sensor", report["features"])
    st.line_chart(g.set_index("cycle")[[sensor]])
    st.dataframe(g[["cycle", "rul", "is_fault", "anomaly_score", "raw_alert", "alert"]], hide_index=True)
with tab2:
    st.write("SHAP contributions to the actual anomaly score: positive values increase abnormality.")
    units = sorted(explanations.engine_id.unique())
    explanation_unit = st.selectbox("Explained engine (highest-scoring observation)", units)
    e = explanations[explanations.engine_id.eq(explanation_unit)]
    st.caption(f"Cycle {e.cycle.iloc[0]:.0f} · background score {e.base_score.iloc[0]:.4f} · actual score {e.anomaly_score.iloc[0]:.4f}")
    st.bar_chart(e.set_index("feature")[["shap_value"]], horizontal=True, color="#45c9dd")
    st.dataframe(e[["feature", "value", "shap_value"]], hide_index=True)
    st.caption("Background uses healthy training rows. Attributions describe model behaviour; they do not confirm a physical root cause.")
with tab3:
    st.subheader("Validation comparison")
    st.dataframe(comparison, hide_index=True)
    st.write(f"Frozen selection: **{report['selected_model']}**. Robust Z-score is a reference baseline.")
    st.subheader("Held-out failure events")
    st.dataframe(events, hide_index=True)
    st.subheader("Prediction reliability")
    st.json(m)
    if (directory / "rul_comparison.csv").exists():
        st.subheader("Externally supplied RUL prediction errors")
        st.dataframe(pd.read_csv(directory / "rul_comparison.csv"), hide_index=True)
with tab4:
    st.json({"split_engines": report["splits"], "features": report["features"], "versions": report["versions"]})
    for message in report["limitations"]:
        st.write(f"• {message}")
    for name in ["EVALUATION_REPORT.md", "report.json", "test_predictions.csv", "shap_explanations.csv", "failure_events.csv"]:
        st.download_button(f"Download {name}", (directory / name).read_bytes(), file_name=name)
