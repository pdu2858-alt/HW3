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
    # dataset has no header; expected two columns: label, text
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
    download_if_missing(args.data_url, data_path)
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
