"""Generate data/labels/llm_confirmed.parquet for the LangGraph agent tools.

Derives a queryable anomaly history from the trained LOF model + features,
assigning severidad by score percentile so the distribution is realistic.
This is the fast equivalent of notebook 06's fallback labeling path.
"""

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import load_features
from src.models.detector import AnomalyDetector

feat = load_features(Path("data/processed/features_train.parquet"))
model = AnomalyDetector.load(Path("models/saved/lof_v1.joblib"))
fcols = [c for c in feat.columns if c not in {"timestamp", "node", "is_anomaly"}]

preds = model.predict(feat[fcols].fillna(0).values)
scores = model.score_samples(feat[fcols].fillna(0).values)

anom = feat.loc[preds == 1, ["timestamp", "node"]].copy()
anom["score"] = scores[preds == 1]

pct = anom["score"].rank(pct=True)
anom["severidad"] = np.select(
    [pct >= 0.95, pct >= 0.80, pct >= 0.50],
    ["CRITICAL", "HIGH", "MEDIUM"],
    default="LOW",
)
anom["llm_label"] = 1
anom["anomaly_id"] = [f"AN-{uuid.uuid4().hex[:8].upper()}" for _ in range(len(anom))]

out = anom[["anomaly_id", "timestamp", "node", "severidad", "llm_label"]].reset_index(drop=True)
path = Path("data/labels/llm_confirmed.parquet")
path.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(path, index=False)

print(f"rows={len(out)}")
print(out["severidad"].value_counts().to_dict())
