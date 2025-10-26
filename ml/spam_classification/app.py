"""Streamlit app for Spam Classification (Phase 2)

This app shows the baseline evaluation results, allows single-text
prediction using the saved model artifact, and can trigger training
by calling the Phase 1 `train.py` script when no model exists.

Note: This app is inspired by the referenced project but is an
independent implementation that uses the project's own pipeline
and artifacts.
"""
from pathlib import Path
import json
import subprocess

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


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

    st.markdown("---")
    st.caption("Implementation: Phase 2 Streamlit demo — independent implementation inspired by external example.")


if __name__ == "__main__":
    main()
