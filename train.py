"""Train the tuned Module 8 network and write the deployable artifacts.

Configuration (from the Module 8 tuning report):
  MLP 128-64-32, ReLU, Adam, alpha=0.1, learning_rate_init=0.003, batch_size=128,
  early stopping on a 15% validation split with patience 10, decision threshold 0.28.
"""
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, RAW_COLUMNS, TARGET, build_features

SEED = 20
THRESHOLD = 0.28
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "credit_default.csv"
OUT = ROOT / "model"


def load_split():
    df = pd.read_csv(DATA)
    X, y = df[RAW_COLUMNS], df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                              solver="adam", alpha=0.1, learning_rate_init=0.003,
                              batch_size=128, max_iter=500, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=10,
                              random_state=SEED)),
    ])


def evaluate(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"threshold": thr,
            "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}


def main():
    X_tr, X_te, y_tr, y_te = load_split()
    model = make_pipeline()
    t0 = time.time()
    model.fit(build_features(X_tr), y_tr)
    fit_s = time.time() - t0
    p = model.predict_proba(build_features(X_te))[:, 1]

    metrics = {"tuned_threshold": evaluate(y_te, p, THRESHOLD),
               "default_threshold": evaluate(y_te, p, 0.5)}
    meta = {
        "model": "MLPClassifier 128-64-32 (Module 8 tuned configuration)",
        "threshold": THRESHOLD, "seed": SEED, "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "test_defaulters": int(y_te.sum()), "epochs": int(model[-1].n_iter_),
        "fit_seconds": round(fit_s, 2),
        "metrics": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in metrics.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["metrics"], indent=2), f"\nepochs={meta['epochs']} fit={fit_s:.1f}s")


if __name__ == "__main__":
    main()
