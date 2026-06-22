"""Generate a small, self-contained sample for the public Streamlit Cloud demo.

The cloud deploy has no BGL.log (709MB), no trained model (63MB) and no Ollama.
This script reproduces EXACTLY what the local dashboard computes (the temporal
holdout where LOF scores F1=0.947) and writes small parquet files committed to
the repo:

  data/sample/predictions.parquet  model predictions for the holdout (all data tabs)
  data/sample/features.parquet     scaled holdout features for the drift expander
  data/sample/raw_logs.parquet     raw events around the top anomalies (LLM windows)
"""

from pathlib import Path

import pandas as pd

from src.data.loader import load_bgl_logs
from src.data.preprocessor import add_severity_score, train_test_split_temporal
from src.features.engineering import load_features
from src.models.detector import AnomalyDetector

OUT = Path("data/sample")
OUT.mkdir(parents=True, exist_ok=True)

# Mirror dashboard.get_predictions: split the scaled features, predict on holdout.
feat = load_features(Path("data/processed/features_train.parquet"))
feature_cols = [c for c in feat.columns if c not in {"timestamp", "node", "is_anomaly"}]
_, test_df = train_test_split_temporal(feat, test_fraction=0.2)

model = AnomalyDetector.load(Path("models/saved/lof_v1.joblib"))
X = test_df[feature_cols].fillna(0).values
scores = model.score_samples(X)
preds = model.predict(X)
scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

pred_df = test_df[["timestamp", "node", "is_anomaly"]].copy()
pred_df["anomaly_score"] = scores_norm
pred_df["is_predicted_anomaly"] = preds.astype(bool)

# Raw events around the top anomalies — only what the LLM ±5min windows need.
top = pred_df[pred_df["is_predicted_anomaly"]].nlargest(60, "anomaly_score")
raw = add_severity_score(load_bgl_logs(Path("data/raw/BGL.log"), nrows=500_000))
masks = [
    (raw["timestamp"] >= ts - pd.Timedelta(minutes=6)) & (raw["timestamp"] <= ts + pd.Timedelta(minutes=6))
    for ts in top["timestamp"]
]
keep = masks[0]
for m in masks[1:]:
    keep |= m
raw_df = raw[keep][
    ["timestamp", "node", "level", "component", "content", "severity_score", "is_anomaly"]
].drop_duplicates().reset_index(drop=True)

pred_df.to_parquet(OUT / "predictions.parquet", index=False)
test_df.to_parquet(OUT / "features.parquet", index=False)
raw_df.to_parquet(OUT / "raw_logs.parquet", index=False)

tp = int((pred_df.is_anomaly & pred_df.is_predicted_anomaly).sum())
fp = int((~pred_df.is_anomaly & pred_df.is_predicted_anomaly).sum())
fn = int((pred_df.is_anomaly & ~pred_df.is_predicted_anomaly).sum())
prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
print(f"holdout rows={len(pred_df):,} | anomalías reales={pred_df['is_anomaly'].mean():.1%}")
print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
print(f"raw_logs (ventanas top-60)={len(raw_df):,}")
for f in ["predictions.parquet", "features.parquet", "raw_logs.parquet"]:
    print(f"  {f}: {(OUT / f).stat().st_size / 1e6:.1f} MB")
