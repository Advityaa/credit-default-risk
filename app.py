"""Gradio front end for the credit-default model, served on Hugging Face Spaces."""
import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from features import RAW_COLUMNS, build_features

ROOT = Path(__file__).parent
MODEL = joblib.load(ROOT / "model" / "model.joblib")
META = json.loads((ROOT / "model" / "metadata.json").read_text())
THRESHOLD = META["threshold"]

SEX = {"Male": 1, "Female": 2}
EDUCATION = {"Graduate school": 1, "University": 2, "High school": 3, "Other": 4}
MARRIAGE = {"Married": 1, "Single": 2, "Other": 3}
PAY_STATUS = {
    "Paid in full / no consumption (-2)": -2, "Paid duly (-1)": -1,
    "Revolving credit, minimum paid (0)": 0,
    **{f"{k} month{'s' if k > 1 else ''} late ({k})": k for k in range(1, 9)},
}
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]  # PAY_0, PAY_2..PAY_6 / AMT1..6


def predict_proba(record: dict) -> float:
    X = build_features(pd.DataFrame([record], columns=RAW_COLUMNS))
    return float(MODEL.predict_proba(X)[0, 1])


def score(limit_bal, sex, education, marriage, age, *rest):
    pays, bills, pamts = rest[:6], rest[6:12], rest[12:18]
    rec = {"LIMIT_BAL": limit_bal, "SEX": SEX[sex], "EDUCATION": EDUCATION[education],
           "MARRIAGE": MARRIAGE[marriage], "AGE": age}
    for name, v in zip(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], pays):
        rec[name] = PAY_STATUS[v]
    for i, (b, p) in enumerate(zip(bills, pamts), start=1):
        rec[f"BILL_AMT{i}"], rec[f"PAY_AMT{i}"] = b, p
    prob = predict_proba(rec)
    flagged = prob >= THRESHOLD
    verdict = ("**HIGH RISK — flag for review**" if flagged else "**Low risk**")
    detail = (f"{verdict}\n\nEstimated probability of default next month: **{prob:.1%}** "
              f"(decision threshold {THRESHOLD:.2f}, tuned in Module 8 to maximize F1).")
    return {"Default": prob, "No default": 1 - prob}, detail


m = META["metrics"]["tuned_threshold"]
ABOUT = f"""
**Model:** {META['model']} · scikit-learn {META['versions']['sklearn']} · trained on
{META['n_train']:,} clients of the UCI *Default of Credit Card Clients* dataset (Yeh & Lien, 2009).

**Held-out test set ({META['n_test']:,} clients, threshold {THRESHOLD}):** precision {m['precision']:.3f} ·
recall {m['recall']:.3f} · F1 {m['f1']:.3f} · accuracy {m['accuracy']:.3f} · ROC-AUC {m['roc_auc']:.3f}.

*Course demonstration only (AAI 6600, Northeastern University). Not for real credit decisions:
the model uses sex and marital status and has not been audited for fairness.*
"""

with gr.Blocks(title="Credit Default Risk") as demo:
    gr.Markdown("# Credit Card Default Risk\nEnter a client's profile and last six months "
                "of repayment history (amounts in NT dollars).")
    with gr.Row():
        limit = gr.Number(label="Credit limit", value=50000, minimum=0)
        sex = gr.Dropdown(list(SEX), value="Female", label="Sex")
        edu = gr.Dropdown(list(EDUCATION), value="University", label="Education")
        mar = gr.Dropdown(list(MARRIAGE), value="Single", label="Marital status")
        age = gr.Number(label="Age", value=35, minimum=18, maximum=100, precision=0)
    pays, bills, pamts = [], [], []
    with gr.Accordion("Repayment status (Sep → Apr 2005)", open=True):
        with gr.Row():
            for mo in MONTHS:
                pays.append(gr.Dropdown(list(PAY_STATUS), value="Paid duly (-1)", label=mo))
    with gr.Accordion("Bill statement amounts", open=False):
        with gr.Row():
            for mo in MONTHS:
                bills.append(gr.Number(label=mo, value=20000))
    with gr.Accordion("Payment amounts", open=False):
        with gr.Row():
            for mo in MONTHS:
                pamts.append(gr.Number(label=mo, value=2000, minimum=0))
    btn = gr.Button("Assess risk", variant="primary")
    with gr.Row():
        label = gr.Label(label="Predicted probability", num_top_classes=2)
        text = gr.Markdown()
    btn.click(score, [limit, sex, edu, mar, age, *pays, *bills, *pamts], [label, text],
              api_name="predict")
    gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
