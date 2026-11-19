"""Feature engineering shared by training and serving.

Keeping this in one module guarantees the deployed app transforms raw inputs
exactly as the training pipeline did (no training/serving skew).
"""
import pandas as pd

RAW_NUMERIC = (
    ["LIMIT_BAL", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
RAW_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE"] + RAW_NUMERIC[1:]
TARGET = "default"

# Fixed category levels so a single-row request yields the same 29 columns.
CATEGORY_LEVELS = {
    "SEX": [1, 2],                 # 1 = male, 2 = female
    "EDUCATION": [1, 2, 3, 4],     # graduate school, university, high school, other
    "MARRIAGE": [1, 2, 3],         # married, single, other
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Merge undocumented codes into the 'other' level (as in Module 8)."""
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].isin([1, 2, 3, 4]), 4)
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].isin([1, 2, 3]), 3)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw 23 columns -> 29 model features (20 numeric + 9 one-hot)."""
    df = clean(df[RAW_COLUMNS])
    parts = [df[RAW_NUMERIC].astype(float)]
    for col, levels in CATEGORY_LEVELS.items():
        for lvl in levels:
            parts.append((df[col] == lvl).astype(float).rename(f"{col}_{lvl}"))
    return pd.concat(parts, axis=1)


FEATURE_NAMES = RAW_NUMERIC + [f"{c}_{l}" for c, lv in CATEGORY_LEVELS.items() for l in lv]
