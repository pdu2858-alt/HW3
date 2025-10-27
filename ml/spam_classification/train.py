#!/usr/bin/env python3
"""Train a baseline SVM spam classifier (Phase 1).

Downloads dataset, preprocesses text with TF-IDF, trains LinearSVC,
and writes model and evaluation report to artifacts/.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


DEFAULT_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download_if_missing(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Using existing dataset at {dest}")
        return dest
    print(f"Downloading dataset from {url} -> {dest}")
    r = requests.get(url)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_dataset(path: Path) -> pd.DataFrame:
    # Try reading with header: many processed outputs include a header row
    df = pd.read_csv(path, encoding="utf-8")
    # If header contains label/text columns, use them
    cols = [c.lower() for c in df.columns]
    if "label" in cols or "text" in cols or "text_clean" in cols:
        # prefer text_clean, then text
        if "text_clean" in cols:
            text_col = df.columns[cols.index("text_clean")]
        elif "text" in cols:
            text_col = df.columns[cols.index("text")]
        else:
            # fallback to second column
            text_col = df.columns[1]

        if "label" in cols:
            label_col = df.columns[cols.index("label")]
        else:
            label_col = df.columns[0]

        out = pd.DataFrame()
        out["label"] = df[label_col].astype(str).str.strip().str.lower()
        out["text"] = df[text_col].astype(str)
        return out

    # fallback: dataset has no header; expected two columns: label, text
    df = pd.read_csv(path, header=None, encoding="utf-8")
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (label, text)")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    # normalize labels (spam/ham)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def train(args):
    data_path = Path(args.data_path)
    # Prefer repository processed dataset if present and if user didn't explicitly pass an existing path
    repo_root = Path(__file__).resolve().parents[2]
    processed_candidate = repo_root / "datasets" / "processed" / "sms_spam_clean.csv"
    if (not data_path.exists()) and processed_candidate.exists():
        print(f"Using processed dataset at {processed_candidate} instead of {data_path}")
        data_path = processed_candidate

    # If the chosen data_path doesn't exist, try downloading the raw CSV to the provided path
    if not data_path.exists():
        download_if_missing(args.data_url, data_path)
    else:
        print(f"Using dataset at {data_path}")
    df = load_dataset(data_path)
    X = df["text"].fillna("")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Select model
    if args.model == "svm":
        clf = LinearSVC(random_state=42, max_iter=10000)
    elif args.model == "logreg":
        # use saga solver to handle larger feature spaces; max_iter increased
        clf = LogisticRegression(random_state=42, max_iter=2000, solver="saga")
    else:
        raise ValueError("Unsupported model type: choose 'svm' or 'logreg'")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", clf),
    ])

    print("Training baseline LinearSVC...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    # try to obtain continuous scores for ROC/PR if available
    y_score = None
    try:
        if hasattr(pipeline, 'predict_proba'):
            y_score = pipeline.predict_proba(X_test)[:, -1]
        elif hasattr(pipeline.named_steps['clf'], 'predict_proba'):
            y_score = pipeline.named_steps['clf'].predict_proba(pipeline.named_steps['tfidf'].transform(X_test))[:, -1]
        elif hasattr(pipeline.named_steps['clf'], 'decision_function'):
            y_score = pipeline.named_steps['clf'].decision_function(pipeline.named_steps['tfidf'].transform(X_test))
    except Exception:
        y_score = None

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # adjust default model/report names for logistic regression if user did not override
    model_name = args.model_name
    report_name = args.report_name
    if args.model == "logreg":
        if model_name == "svm_baseline.joblib":
            model_name = "logreg_baseline.joblib"
        if report_name == "eval_baseline.json":
            report_name = "eval_logreg.json"

    model_path = artifacts_dir / model_name
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    eval = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
        "confusion_matrix": cm,
        # include raw arrays to enable ROC / PR plotting in Streamlit
        "y_true": list(y_test.astype(str).apply(lambda x: 1 if str(x).strip().lower() == 'spam' else 0)),
        "y_score": (list(map(float, y_score)) if y_score is not None else None),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    report_path = artifacts_dir / report_name
    report_path.write_text(json.dumps(eval, indent=2))
    print(f"Wrote evaluation report to {report_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-url", default=DEFAULT_URL)
    p.add_argument("--data-path", default="data/sms_spam_no_header.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--model-name", default="svm_baseline.joblib")
    p.add_argument("--report-name", default="eval_baseline.json")
    p.add_argument("--model", choices=["svm", "logreg"], default="svm")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
