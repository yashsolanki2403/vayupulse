# VayuPulse Arena 3 - Evaluation report

Source: **SYNTHETIC DEMO**

Label: Known simulated degradation onset

Selected detector: isolation_forest_200

## Held-out test results

| Metric | Value |
| --- | --- |
| rows | 3370 |
| engines | 18 |
| false_alert_rate | 0.0000 |
| false_alerts_per_1000_healthy_cycles | 0.0000 |
| precision | 1.0000 |
| recall | 0.7358 |
| true_positives | 518 |
| false_positives | 0 |
| false_negatives | 186 |
| failure_events | 18 |
| missed_failures | 0 |
| missed_failure_rate | 0.0000 |
| median_lead_time_cycles_detected_only | 27.0000 |
| roc_auc | 0.9211 |
| average_precision | 0.8908 |

## Interpretation

Threshold and detector choice use validation engines only. Test engines are untouched until final evaluation.
False-alert rate = alerted healthy rows / all healthy rows after persistence. Raw threshold calibration targets a separate 5% rate by default.
A failure is detected only if a confirmed alert occurs inside its labelled degradation window and strictly before failure.
Median lead time includes detected events only; read it alongside missed failures.

## Limits

- Synthetic results do not establish aircraft performance.
- Scores are not calibrated failure probabilities.
- SHAP describes model associations, not causal root causes.
- Lead time is in observed cycles; missed failures are reported separately.
- FD001 support assumes one operating regime; FD002/FD004 need regime-aware preprocessing.
- No availability or downtime improvement is inferred from this experiment.