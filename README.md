---
title: Credit Default Risk
emoji: 💳
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Tuned MLP predicting credit card default (AAI 6600)
---

# Credit Card Default Risk

Gradio demo of the tuned neural network from the AAI 6600 Module 8 assignment
(MLP 128-64-32, alpha 0.1, learning rate 0.003, batch size 128, decision threshold 0.28),
trained on the UCI *Default of Credit Card Clients* dataset (Yeh & Lien, 2009).

| Held-out test set (6,000 clients) | Precision | Recall | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Threshold 0.28 | .570 | .524 | .546 | .806 | .776 |

## Repository layout
- `features.py` – feature engineering shared by training and serving
- `train.py` – reproducible training (seed 20) → `model/model.joblib`, `model/metadata.json`
- `app.py` – Gradio interface and `/predict` API
- `tests/` – feature contract, metric floors, sanity and end-to-end tests
- `.github/workflows/ci-cd.yml` – test → train → gate → deploy → smoke test
- `scripts/` – deployment and post-deployment smoke test

## Local use
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python train.py && .venv/bin/pytest -q && .venv/bin/python app.py
```

*Course demonstration only; not for real credit decisions.*
