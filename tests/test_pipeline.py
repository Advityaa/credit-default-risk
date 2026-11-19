"""Quality gates run by CI before anything is deployed."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import FEATURE_NAMES, RAW_COLUMNS, build_features  # noqa: E402
from train import load_split  # noqa: E402

MODEL_PATH = ROOT / "model" / "model.joblib"
META = json.loads((ROOT / "model" / "metadata.json").read_text())


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def test_feature_contract():
    df = pd.read_csv(ROOT / "data" / "credit_default.csv").head(50)
    X = build_features(df)
    assert list(X.columns) == FEATURE_NAMES and X.shape == (50, 29)
    assert not X.isna().any().any()


def test_undocumented_codes_map_to_other():
    row = pd.DataFrame([dict.fromkeys(RAW_COLUMNS, 0)], columns=RAW_COLUMNS)
    row[["SEX", "EDUCATION", "MARRIAGE"]] = [1, 6, 0]
    X = build_features(row)
    assert X["EDUCATION_4"].iloc[0] == 1 and X["MARRIAGE_3"].iloc[0] == 1


def test_metrics_gate(model):
    """Block deployment if the model regresses below agreed floors."""
    _, X_te, _, y_te = load_split()
    p = model.predict_proba(build_features(X_te))[:, 1]
    yhat = (p >= META["threshold"]).astype(int)
    tp = int(((yhat == 1) & (y_te == 1)).sum())
    precision, recall = tp / yhat.sum(), tp / y_te.sum()
    f1 = 2 * precision * recall / (precision + recall)
    assert f1 >= 0.52, f"F1 {f1:.3f} below floor"
    assert recall >= 0.45, f"recall {recall:.3f} below floor"


def test_prediction_is_probability_and_monotone_in_delinquency(model):
    base = dict.fromkeys(RAW_COLUMNS, 0)
    base.update(LIMIT_BAL=50000, SEX=2, EDUCATION=2, MARRIAGE=2, AGE=35)
    good = dict(base, **{c: -1 for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]})
    bad = dict(base, **{c: 3 for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]})
    p = model.predict_proba(build_features(pd.DataFrame([good, bad], columns=RAW_COLUMNS)))[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert p[1] > p[0], "three-months-late history should score riskier than paid-duly"


def test_app_scores_end_to_end():
    import app
    probs, text = app.score(50000, "Female", "University", "Single", 35,
                            *["Paid duly (-1)"] * 6, *[20000] * 6, *[2000] * 6)
    assert 0 <= probs["Default"] <= 1 and "threshold" in text
