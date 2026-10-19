# Arena 3 demo — approximately 4 minutes

1. **Scope (20 seconds).** “Arena 3 checks unusual engine sensor behaviour, explains alerts and measures whether warnings arrive before failures.”
2. **Data honesty (20 seconds).** Point to the synthetic demo label. Explain that these are simulated run-to-failure engines, with C-MAPSS FD001 supported separately.
3. **Detection (60 seconds).** Select a held-out engine in the investigation tab. Show the anomaly score crossing the frozen threshold and the three-reading confirmation rule. Scroll to the alert rows. Explain the false-alert rate and missed-event count.
4. **Explanation (45 seconds).** Open SHAP evidence and select an explained engine. Show the positive and negative sensor contributions. Explain that contributions add to the actual score, while they do not prove a failed physical component.
5. **Validation (45 seconds).** Open model validation. Show the 100-tree and 200-tree Isolation Forest comparison and robust Z-score baseline. Explain that selection uses validation engines, while headline metrics use a different test-engine set.
6. **Lead time (30 seconds).** Show per-engine failure events. Explain that warning lead time is in cycles and only includes detected events; missed failures are shown separately.
7. **Hand-off (20 seconds).** Show report downloads. Arena 2 receives feature order, scores, a frozen threshold and the persistence policy. Arena 1 can supply RUL predictions for error evaluation.

## Likely questions

**Does anomaly score mean probability of failure?** No. It ranks abnormality relative to the healthy training distribution.

**Where do the fault labels come from?** Synthetic onset is known by construction. C-MAPSS uses a documented last-30-cycle proxy, not an official anomaly label.

**Have you proven higher aircraft availability?** No. This project evaluates detection on synthetic data. Operational availability needs a separate real-world evaluation.

**Can it explain every point?** The explanation function supports any valid sensor rows. The included report precomputes selected high-score observations to keep the demo fast.

**What if a reading is missing?** The input validator rejects missing sensor values. In an online integration, mark the data unavailable and reset the persistence counter for a cycle gap; do not silently issue a healthy decision.
