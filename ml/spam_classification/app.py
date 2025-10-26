"""Streamlit app for Spam Classification (Phase 2)

This app shows the baseline evaluation results, allows single-text
prediction using the saved model artifact, and can trigger training
by calling the Phase 1 `train.py` script when no model exists.

Note: This app is inspired by the referenced project but is an
independent implementation that uses the project's own pipeline
and artifacts.
"""
from pathlib import Path
import io
import json
import subprocess

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer


BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
MODEL_PATH = ARTIFACTS / "svm_baseline.joblib"
EVAL_PATH = ARTIFACTS / "eval_baseline.json"


def ensure_artifacts():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def run_training():
    # Call train.py to (re)train the baseline model
    cmd = ["python3", str(BASE / "train.py"), "--data-path", str(BASE / "data" / "sms_spam_no_header.csv"), "--artifacts-dir", str(ARTIFACTS)]
    st.info("Running training — this may take a while. Output will appear below.")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        st.write(line.rstrip())
    proc.wait()
    return proc.returncode


def load_model():
    return joblib.load(MODEL_PATH)


def load_eval():
    if not EVAL_PATH.exists():
        return None
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def plot_label_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = df["label"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    return fig


def plot_length_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    lengths = df["text"].fillna("").str.len()
    sns.histplot(lengths, bins=30, ax=ax)
    ax.set_xlabel("Message length")
    return fig


def top_n_words(df: pd.DataFrame, n=20):
    texts = df["text"].fillna("")
    cv = CountVectorizer(stop_words="english", max_features=10000)
    X = cv.fit_transform(texts)
    sums = X.sum(axis=0).A1
    terms = cv.get_feature_names_out()
    top_idx = sums.argsort()[::-1][:n]
    return [(terms[i], int(sums[i])) for i in top_idx]


def main():
    st.title("Spam Classification — Demo (Phase 2)")
    st.write("A simple Streamlit interface for the spam classification pipeline.")

    ensure_artifacts()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Evaluation")
        eval_data = load_eval()
        if eval_data is None:
            st.warning("No evaluation report found. You can run training to produce model and report.")
        else:
            st.metric("Accuracy", f"{eval_data.get('accuracy'):.4f}")
            st.metric("F1 (weighted)", f"{eval_data.get('f1_weighted'):.4f}")
            st.write("\n**Classification report**")
            st.text(eval_data.get("classification_report", ""))
            cm = eval_data.get("confusion_matrix")
            if cm:
                fig = plot_confusion(cm, labels=["ham", "spam"])
                st.pyplot(fig)

    # Data / batch upload area
    st.header("Data & Batch Prediction")
    uploaded = st.file_uploader("Upload CSV for batch prediction (must contain a 'text' column)", type=["csv"]) 
    sample_df = None
    if uploaded is not None:
        try:
            sample_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            sample_df = None

    # If no upload, try load local dataset (if present) for visualizations
    local_data = None
    data_path = BASE / "data" / "sms_spam_no_header.csv"
    if sample_df is None and data_path.exists():
        try:
            # load without header expecting label,text
            df_local = pd.read_csv(data_path, header=None, encoding="utf-8")
            df_local = df_local.iloc[:, :2]
            df_local.columns = ["label", "text"]
            df_local["label"] = df_local["label"].astype(str).str.strip().str.lower()
            local_data = df_local
        except Exception:
            local_data = None

    df_for_vis = sample_df if sample_df is not None else local_data
    if df_for_vis is not None:
        st.subheader("Dataset preview")
        st.dataframe(df_for_vis.head(10))
        st.subheader("Label distribution")
        st.pyplot(plot_label_distribution(df_for_vis))
        st.subheader("Message length distribution")
        st.pyplot(plot_length_histogram(df_for_vis))
        st.subheader("Top words (approx.)")
        top_words = top_n_words(df_for_vis, n=20)
        st.table(pd.DataFrame(top_words, columns=["word", "count"]))

    with col2:
        st.header("Model")

    with col2:
        st.header("Model")
        if MODEL_PATH.exists():
            st.success(f"Loaded model: {MODEL_PATH.name}")
            model_info = load_model()
            st.write("Model pipeline steps:")
            try:
                st.write([type(step).__name__ for step in model_info.steps])
            except Exception:
                st.write("(unable to introspect model steps)")
        else:
            st.error("No model artifact found.")
            if st.button("Train baseline model now"):
                code = run_training()
                if code == 0:
                    st.success("Training finished — model and evaluation saved.")
                else:
                    st.error(f"Training failed (exit code {code}). Check logs above.")

    st.header("Try a sample message")
    user_text = st.text_area("Enter SMS / email text to classify", value="Free entry: claim your prize now!")
    if st.button("Predict"):
        if not MODEL_PATH.exists():
            st.error("No trained model found. Please run training first.")
        else:
            model = load_model()
            pred = model.predict([user_text])[0]
            st.write(f"Predicted label: **{pred}**")
            # LinearSVC doesn't provide probabilities; show decision function instead if available
            try:
                score = model.decision_function([user_text])
                st.write(f"Decision function score: {score[0]:.4f}")
            except Exception:
                pass

    # Batch prediction: if uploaded CSV with 'text' column
    if sample_df is not None:
        if "text" not in sample_df.columns:
            st.error("Uploaded CSV must have a 'text' column for predictions.")
        else:
            st.subheader("Batch prediction")
            if MODEL_PATH.exists():
                model = load_model()
                texts = sample_df["text"].fillna("").astype(str).tolist()
                preds = model.predict(texts)
                out_df = sample_df.copy()
                out_df["prediction"] = preds
                st.dataframe(out_df.head(20))
                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions (CSV)", csv, file_name="predictions.csv", mime="text/csv")
            else:
                st.error("No model found — train the baseline model first to run batch predictions.")

    st.markdown("---")
    st.caption("Implementation: Phase 2 Streamlit demo — independent implementation inspired by external example.")


if __name__ == "__main__":
    main()
