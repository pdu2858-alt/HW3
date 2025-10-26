# Spam Classification (Phase 1)

This folder contains a minimal reproducible pipeline for Phase 1 (baseline SVM) of the spam classification OpenSpec change.

Files
- `requirements.txt` - Python dependencies
- `train.py` - Train baseline LinearSVC with TF-IDF, save model and evaluation report

Quick start

1. Create a Python virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training (will download dataset automatically):

```bash
python train.py
```

Artifacts will be written to `artifacts/` (model: `svm_baseline.joblib`, report: `eval_baseline.json`).
