import numpy as np
import pandas as pd
import pytest
from arena3.data import DEMO_SENSORS, demo_data, load_cmapss, split_engines, validate
from arena3.evaluate import metrics, rul_errors
from arena3.explain import permutation_shap
from arena3.model import Detector, score_frame


def test_split_and_validation():
    df = validate(demo_data(engines=12), DEMO_SENSORS)
    train, val, test = split_engines(df)
    sets = [set(g.engine_id) for g in [train, val, test]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([df, df.iloc[[0]]]), DEMO_SENSORS)
    df.loc[0, DEMO_SENSORS[0]] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate(df, DEMO_SENSORS)


def test_persistence_resets_at_engine_and_cycle_gaps():
    class Fixed:
        threshold = .5
        def score(self, frame):
            return frame["value"].to_numpy()
    frame = pd.DataFrame({"engine_id": [1]*6 + [2]*2, "cycle": [1,2,3,5,6,7,1,2], "value": [1]*8})
    out = score_frame(Fixed(), frame)
    assert out.alert.tolist() == [False, False, True, False, False, True, False, False]
    # Prefix predictions cannot change when later records arrive.
    assert score_frame(Fixed(), frame.iloc[:4]).alert.tolist() == out.alert.iloc[:4].tolist()


def test_events_exclude_post_failure_and_healthy_alerts():
    frame = pd.DataFrame({"engine_id": [1]*5 + [2]*5, "cycle": [1,2,3,4,5]*2,
                          "rul": [4,3,2,1,0]*2, "is_fault": [0,0,1,1,1]*2,
                          "alert": [True,False,False,True,True,False,False,False,False,True],
                          "anomaly_score": [.5]*10})
    m, events = metrics(frame)
    assert m["false_alert_rate"] == .25
    assert m["missed_failures"] == 1
    assert m["median_lead_time_cycles_detected_only"] == 1
    assert events.detected_before_failure.tolist() == [True, False]


def test_incomplete_failure_is_not_counted():
    frame = pd.DataFrame({"engine_id": [1,1], "cycle": [1,2], "rul": [10,9],
                          "is_fault": [1,1], "alert": [False,True], "anomaly_score": [.4,.8]})
    m, _ = metrics(frame)
    assert m["failure_events"] == 0 and m["missed_failure_rate"] is None


def test_shap_linear_and_interaction_additivity():
    bg = np.array([[0.,1.], [2.,3.]])
    x = np.array([[4.,5.]])
    w = np.array([2.,-3.])
    base, values = permutation_shap(lambda a: a @ w + 7, bg, x)
    assert np.allclose(values, (x - bg.mean(axis=0)) * w)
    f = lambda a: a[:,0] * a[:,1] + a[:,0] ** 2
    base, values = permutation_shap(f, bg, x)
    assert np.allclose(base + values.sum(axis=1), f(x))


def test_rul_error_direction_and_invalid_values():
    df = pd.DataFrame({"model": ["a","a"], "engine_id": [1,1], "cycle": [1,2],
                       "y_true": [10,8], "y_pred": [12.,6.]})
    summary, _ = rul_errors(df)
    assert summary.mae.iloc[0] == 2 and summary.bias.iloc[0] == 0
    assert summary.overprediction_rate.iloc[0] == .5
    df.loc[0, "y_pred"] = np.inf
    with pytest.raises(ValueError):
        rul_errors(df)


def test_isolation_forest_scores_and_reproducibility():
    rng = np.random.default_rng(42)
    healthy = pd.DataFrame({"s": rng.normal(0,.1,200)})
    model = Detector(["s"], n_estimators=40).fit(healthy).calibrate(healthy)
    assert model.score(np.array([[4.]]))[0] > np.median(model.score(healthy))
    twin = Detector(["s"], n_estimators=40).fit(healthy).calibrate(healthy)
    assert np.array_equal(model.score(healthy), twin.score(healthy))


def test_cmapss_loader_and_proxy_labels():
    from io import StringIO
    # Generated fixture tests format handling; this is not NASA data.
    rows = []
    for unit in [1, 2]:
        for cycle in range(1, 41):
            rows.append([unit, cycle, 0, 0, 100, *([cycle * .01] * 21)])
    text = pd.DataFrame(rows).to_csv(sep=" ", index=False, header=False)
    result = load_cmapss(StringIO(text))
    assert result.loc[0, "rul"] == 39
    assert result.loc[0, "is_fault"] == 0
    assert result.loc[9, "is_fault"] == 1
    assert result.groupby("engine_id").rul.min().eq(0).all()


def test_invalid_cycles_and_fault_recovery_are_rejected():
    data = demo_data(engines=10)
    data["cycle"] = data.cycle.astype(float)
    data.loc[0, "cycle"] = .5
    with pytest.raises(ValueError, match="positive integers"):
        validate(data, DEMO_SENSORS)
    data = demo_data(engines=10)
    data.loc[0, "is_fault"] = 1
    with pytest.raises(ValueError, match="terminal fault"):
        validate(data, DEMO_SENSORS)
