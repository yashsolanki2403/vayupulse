from pathlib import Path
import numpy as np
import pandas as pd

DEMO_SENSORS = ["temperature", "vibration", "pressure", "rpm", "fuel_flow", "current"]


def demo_data(seed=42, engines=90):
    """Dimensionless, simulated run-to-failure data; not aircraft measurements."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, engines + 1):
        end = int(rng.integers(160, 221))
        onset = end - int(rng.integers(30, 51))
        bias = rng.normal(0, 0.14, 6)
        fault = rng.choice([0, 1, 2])
        direction = [[2.8, 3.4, -2.5, 0.4, 1.7, 2.0],
                     [1.3, 1.0, -3.2, -2.3, 3.1, 1.1],
                     [2.2, 3.8, -0.8, -1.6, 0.8, 3.0]][fault]
        for cycle in range(1, end + 1):
            ramp = max(0, (cycle - onset) / (end - onset))
            values = bias + rng.normal(0, 0.32, 6) + np.asarray(direction) * ramp
            rows.append([unit, cycle, end - cycle, int(cycle >= onset), *values])
    return pd.DataFrame(rows, columns=["engine_id", "cycle", "rul", "is_fault", *DEMO_SENSORS])


def load_cmapss(path):
    """Read a NASA train_FD001.txt run-to-failure file (26 columns)."""
    columns = ["engine_id", "cycle", *[f"setting_{i}" for i in range(1, 4)],
               *[f"sensor_{i}" for i in range(1, 22)]]
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != 26:
        raise ValueError("Expected 26 columns from train_FD001.txt; truncated test files are unsupported.")
    raw.columns = columns
    raw["rul"] = raw.groupby("engine_id")["cycle"].transform("max") - raw.cycle
    raw["is_fault"] = (raw.rul <= 30).astype(int)
    return raw


def validate(df, features):
    required = ["engine_id", "cycle", "rul", "is_fault", *features]
    if set(required) - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(df.columns))}")
    out = df[required].copy()
    if out.empty or out.engine_id.isna().any():
        raise ValueError("Data must contain records with non-null engine IDs.")
    for col in required[1:]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if not np.isfinite(out[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite readings are rejected; impute upstream using training-only statistics.")
    if out.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle rows are not allowed.")
    if not out.is_fault.isin([0, 1]).all() or (out.rul < 0).any():
        raise ValueError("is_fault must be 0/1 and RUL must be nonnegative.")
    if (out.cycle < 1).any() or (out.cycle % 1 != 0).any():
        raise ValueError("Cycles must be positive integers.")
    out = out.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    for _, g in out.groupby("engine_id"):
        if g.is_fault.diff().dropna().lt(0).any():
            raise ValueError("Only one terminal fault episode per engine is supported.")
        if not np.allclose(g.cycle + g.rul, (g.cycle + g.rul).iloc[0]):
            raise ValueError("RUL and cycle must imply a constant failure cycle per engine.")
    return out


def split_engines(df, seed=42):
    ids = np.array(sorted(df.engine_id.unique()))
    if len(ids) < 10:
        raise ValueError("At least 10 engines are needed for disjoint train/validation/test splits.")
    np.random.default_rng(seed).shuffle(ids)
    a, b = int(len(ids) * .6), int(len(ids) * .8)
    return tuple(df[df.engine_id.isin(part)].copy() for part in (ids[:a], ids[a:b], ids[b:]))
