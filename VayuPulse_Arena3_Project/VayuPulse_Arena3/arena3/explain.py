import numpy as np
import pandas as pd


def permutation_shap(score, background, rows, permutations=16, seed=42):
    """Empirical interventional permutation SHAP with shared background rows.

    Each ordering adds features from one healthy-background observation to x.
    Averaging marginal score changes estimates the Shapley value. Reverse-order
    pairs reduce variance. Contributions telescope to score(x)-mean(score(bg)).
    """
    if permutations < 2:
        raise ValueError("Use at least two permutations.")
    rng = np.random.default_rng(seed)
    bg, x = np.asarray(background, float), np.asarray(rows, float)
    if bg.ndim != 2 or x.ndim != 2 or not len(bg) or bg.shape[1] != x.shape[1]:
        raise ValueError("Background and rows require matching nonempty sensor matrices.")
    dimensions = x.shape[1]
    orders = []
    while len(orders) < permutations:
        order = rng.permutation(dimensions)
        orders.extend([order, order[::-1]])
    orders = orders[:permutations]
    base = float(np.mean(score(bg)))
    values = np.zeros_like(x)
    for i, row in enumerate(x):
        paths = []
        for order in orders:
            state = bg.copy()
            path = [state.copy()]
            for j in order:
                state[:, j] = row[j]
                path.append(state.copy())
            paths.append(np.stack(path))
        scores = score(np.stack(paths).reshape(-1, dimensions)).reshape(permutations, dimensions + 1, len(bg)).mean(axis=2)
        deltas = np.diff(scores, axis=1)
        for p, order in enumerate(orders):
            values[i, order] += deltas[p] / permutations
    return np.full(len(x), base), values


def explain(model, background, rows, permutations=16, backend="builtin"):
    """Explain the actual high-is-abnormal score, not IF's internal path length."""
    bg = background[model.features].sample(min(24, len(background)), random_state=42)
    x = rows[model.features]
    if backend == "shap":
        import shap
        explainer = shap.Explainer(model.score, bg.to_numpy(), algorithm="permutation",
                                   feature_names=model.features, seed=42)
        explanation = explainer(x.to_numpy(), max_evals=(2 * len(model.features) + 1) * permutations)
        base, values = explanation.base_values, explanation.values
    elif backend == "builtin":
        base, values = permutation_shap(model.score, bg.to_numpy(), x.to_numpy(), permutations)
    else:
        raise ValueError("SHAP backend must be builtin or shap.")
    residual = model.score(x) - (base + values.sum(axis=1))
    if not np.allclose(residual, 0, atol=1e-6):
        raise RuntimeError("SHAP additivity validation failed.")
    records = []
    for i, (_, row) in enumerate(rows.iterrows()):
        for j, feature in enumerate(model.features):
            records.append({"engine_id": row.engine_id, "cycle": row.cycle, "feature": feature,
                            "value": float(row[feature]), "shap_value": float(values[i, j]),
                            "base_score": float(base[i]), "backend": backend,
                            "anomaly_score": float(model.score(x.iloc[[i]])[0]),
                            "additivity_residual": float(residual[i])})
    return pd.DataFrame(records)
