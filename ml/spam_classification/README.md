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

Streamlit demo (Phase 2)

1. Install the Streamlit dependencies (already included in `requirements.txt`):

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app from the `ml/spam_classification` directory:

```bash
streamlit run app.py
```

The app will show evaluation metrics, a confusion matrix, and allow single-message prediction using the saved model artifact. If no model exists it can trigger training (which will run `train.py`).

Notes
- The Streamlit app is a demo wrapper around the Phase 1 pipeline and is implemented as an independent solution (inspired by the referenced example repository but not copied).
- For CI or production usage, consider running training outside the Streamlit process and serving a lightweight prediction API instead.
