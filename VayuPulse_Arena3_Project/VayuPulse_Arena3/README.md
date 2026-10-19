# VayuPulse — Arena 3

**Anomaly Detection, Explainability & Validation**

A standalone Python implementation of Arena 3 from the VayuPulse aircraft predictive-maintenance project. It trains an Isolation Forest, explains anomaly scores using SHAP, measures false alarms and missed failure events, compares validation results, and evaluates externally supplied RUL predictions. A Streamlit dashboard presents the results locally.

## Quick start on Windows / VS Code

1. Extract the ZIP and open the `VayuPulse_Arena3` folder in VS Code.
2. Install Python 3.11 or 3.12 for a broadly supported environment.
3. Open a terminal in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m arena3.run
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`). In VS Code, select `.venv` using **Python: Select Interpreter**. No activation-policy change is needed when using the explicit interpreter path above.

The archive includes generated results in `artifacts/demo`, so the dashboard can also display these immediately after dependency installation. Regenerate artifacts to train a model compatible with your installed scikit-learn version. Serialized model files are for trusted local use only; the dashboard reads CSV/JSON reports and never loads an uploaded pickle.

## What is included

| Arena 3 responsibility | Implementation |
| --- | --- |
| Isolation Forest anomaly detection | `arena3/model.py`; healthy-only training and frozen validation threshold |
| False alerts, missed failures, lead time | `arena3/evaluate.py`; row-level and event-level metrics |
| SHAP explanations | `arena3/explain.py`; additive contributions to actual anomaly scores |
| Prediction errors and reliability | Anomaly precision/recall, ROC-AUC, average precision; optional RUL MAE/RMSE/bias |
| Evaluation reports and comparisons | Markdown/JSON report, validation comparison CSV, failure-event table |
| Demonstration | `app.py`; engine timelines, SHAP bars, results and downloads |

The project supplies a minimal synthetic dataset generator and FD001 loader for standalone operation. It does not train an RUL model, implement the fleet backend, schedule maintenance, or replace the other arenas.

## Workflow and evaluation design

1. Validate finite readings, unique engine/cycle keys, cycle numbering, terminal fault labels and consistent RUL.
2. Split by **engine ID**, 60% training / 20% validation / 20% test. No engine appears in multiple splits.
3. Fit two Isolation Forest configurations (100/200 trees) and a robust Z-score comparison baseline on healthy training observations.
4. Calibrate each threshold using healthy validation scores at the requested raw false-alert target (default 5%). Higher scores mean more abnormal.
5. Require three consecutive above-threshold readings. Gaps and engine changes reset persistence. This is causal and uses no future readings.
6. Select an Isolation Forest using validation results only, prioritising meeting the false-alert target, fewer missed failures, then longer lead time and fewer false alerts. The Z-score model is a comparison baseline.
7. Freeze configuration and evaluate once on held-out engines. Export score explanations for the highest-scoring sample from each of the first eight test engines.

`rul` and `is_fault` are offline evaluation labels and are never sensor features. Training uses known synthetic healthy observations; C-MAPSS uses RUL > 80 as a healthy proxy. This retrospective label assumption must be replaced by an appropriate known-healthy period for operational use.

## Metric definitions

| Metric | Definition |
| --- | --- |
| False-alert rate | Confirmed alert rows labelled healthy / all healthy rows |
| Fault-row recall | Confirmed alert rows in labelled degradation / all degradation rows |
| Precision | Fault-labelled confirmed alerts / all confirmed alerts |
| Missed failure | Complete failure trajectory with no confirmed alert in its labelled degradation window strictly before failure |
| Detection lead time | Failure cycle minus first confirmed alert cycle in that window |
| Median lead time | Median over detected failures only; always read alongside missed failures |
| ROC-AUC / average precision | Discrimination of continuous anomaly scores against row labels |
| RUL prediction error | Predicted RUL minus true RUL; positive means remaining life was overestimated |

Raw threshold false-alert calibration and post-persistence false-alert metrics are different. A 5% validation target is not a guarantee on new engines. No probability calibration or downtime/availability improvement is claimed. Incomplete trajectories are excluded from failure-event metrics.

## SHAP explanations

The default `builtin` backend implements empirical interventional **permutation SHAP**, using 24 healthy training background rows and 16 feature permutations arranged in forward/reverse pairs. For each feature order, it replaces background values with the explained observation and averages marginal changes in the actual anomaly score. This estimates Shapley values; it is not a sensor-deviation heuristic. A fixed seed makes results reproducible.

For every explained row:

```text
healthy_background_mean_score + sum(feature_SHAP_values) = anomaly_score
```

The pipeline checks this identity within 1e-6. Finite permutations introduce attribution sampling error, even when the sum is exact. More permutations improve stability. Interventional masking can create unrealistic combinations for correlated sensors. SHAP values explain the model; they do not establish physical causes.

To use the optional SHAP library backend instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-shap.txt
.\.venv\Scripts\python.exe -m arena3.run --shap-backend shap --output artifacts/shap_library
```

Change the dashboard's artifact directory to `artifacts/shap_library`. The built-in backend and its additivity checks were executed in the delivered environment. The optional library backend could not be executed because the package download was unavailable.

## Using NASA C-MAPSS

Obtain the C-MAPSS turbofan degradation dataset from NASA's Prognostics Data Repository. Extract `train_FD001.txt` into `data/`, then run:

```powershell
.\.venv\Scripts\python.exe -m arena3.run --cmapss data/train_FD001.txt --output artifacts/cmapss_fd001
```

Set the dashboard directory to `artifacts/cmapss_fd001`.

The loader expects the original whitespace-separated 26 columns: engine ID, cycle, 3 operating settings, 21 sensors. Sensors retain NASA's original scales; this project does not invent units for them. Constant sensors are selected out using **healthy training data only**. Operating settings are excluded from detection features. FD001 assumes one operating regime; FD002/FD004 require regime-aware preprocessing and are outside this implementation's validated scope.

Use the **training file's complete run-to-failure trajectories**; a seeded engine split produces the internal held-out test set. Do not substitute truncated `test_FD001.txt`. The loader infers RUL from each engine's last cycle. It labels RUL <= 30 as a degradation proxy, which is not a supplied confirmed anomaly label. Synthetic metrics are included; NASA dataset files and real-data metrics are not included or claimed.

## Evaluating Arena 1 RUL predictions

Ask Arena 1 for a CSV with this schema:

```csv
model,engine_id,cycle,y_true,y_pred
xgboost,81,120,25,28
lstm,81,120,25,23
```

Evaluate with:

```powershell
.\.venv\Scripts\python.exe -m arena3.run --rul-csv data/arena1_predictions.csv --output artifacts/with_rul
```

This exports `rul_comparison.csv` and `rul_errors.csv`. The input must contain genuine held-out predictions with the same RUL units. Compare models on identical engine/cycle keys and identical ground truth; the script reports each model's available rows, and the operator must verify this alignment before ranking them. This feature evaluates predictions supplied by Arena 1 and never trains an RUL predictor. No example RUL scores are presented as a real model result.

## Integration hand-off to Arena 2

```python
import joblib
import pandas as pd
model = joblib.load("artifacts/demo/detector.joblib")  # trusted local artifact only
readings = pd.DataFrame([{name: 0.1 for name in model.features}])
scores = model.score(readings)
raw_alerts = scores > model.threshold
```

Supply all sensor columns from `feature_order.csv`. The same preprocessing, units and operating regime as training are required. Arena 2 must retain a separate consecutive-reading counter per engine, reset on non-alerts or missing cycles, and only emit a confirmed alert after three consecutive raw alerts. `score_frame` supplies the equivalent batch reference implementation. Scores are not probabilities.

## Outputs

- `detector.joblib`: trained Isolation Forest, feature names, threshold, healthy reference statistics.
- `report.json`: data fingerprint, seeds, dependency versions, split membership, configuration and test metrics.
- `EVALUATION_REPORT.md`: readable results and limitations.
- `validation_comparison.csv`: both Isolation Forest variants and robust Z-score baseline.
- `test_predictions.csv`: held-out sensor rows, scores and alert decisions.
- `failure_events.csv`: detection result and warning lead time per complete failure.
- `shap_explanations.csv`: feature contribution, background score, actual score and additivity residual.
- `detection_timeline.png`: one held-out engine's score trajectory.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover engine split isolation, invalid inputs, causal alert persistence, cycle-gap resets, event misses, failure-time exclusion, incomplete trajectories, exact SHAP contributions for a linear model, nonlinear additivity, RUL error direction and detector reproducibility.

## References

- Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- SHAP API: https://shap.readthedocs.io/en/stable/api.html
- SHAP paper: https://arxiv.org/abs/1705.07874
- NASA prognostics repository: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

This is an educational advisory prototype for SIH demonstrations. The attached VayuPulse presentation provides project context; it is not modified by this deliverable.
