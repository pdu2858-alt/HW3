"""Streamlit app for Spam Classification (Phase 2)

Compact Streamlit demo for inspecting the processed SMS spam dataset and
running live inference with saved model artifacts.
"""

from pathlib import Path
import json
import re

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import joblib


APP_DIR = Path(__file__).resolve().parent
ARTIFACTS = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "svm_baseline.joblib"


def ensure_artifacts():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_eval(path: Path = None):
    p = path or (ARTIFACTS / "eval_baseline.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_model(path: Path = None):
    p = path or MODEL_PATH
    if not p.exists():
        return None
    return joblib.load(p)


def plot_label_distribution(df):
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.countplot(x='label', data=df, order=sorted(df['label'].unique()), ax=ax)
    plt.tight_layout()
    return fig


def top_n_words(df, text_col='text', n=20):
    texts = df[text_col].fillna("").astype(str)
    cv = CountVectorizer(stop_words='english', max_features=2000)
    X = cv.fit_transform(texts)
    sums = np.array(X.sum(axis=0)).ravel()
    terms = cv.get_feature_names_out()
    top_idx = sums.argsort()[::-1][:n]
    return [(terms[i], int(sums[i])) for i in top_idx]


def clean_steps_for_display(text: str):
    t = text
    t = re.sub(r'https?://\S+|www\.\S+', '<URL>', t)
    t = re.sub(r'\S+@\S+', '<EMAIL>', t)
    t = re.sub(r'\d+', '<NUM>', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return {'original': text, 'cleaned_preview': t}


def main():
    st.title("Spam/Ham Classifier — Visualizations & Demo")
    st.write("Interactive demo: data overview, visualizations, and live inference.")

    ensure_artifacts()

    # Sidebar
    st.sidebar.header("Inputs & Options")
    use_default = st.sidebar.checkbox("Use processed dataset in repo", value=True)
    uploaded = st.sidebar.file_uploader("Or upload CSV", type=["csv"])    
    models_dir = st.sidebar.text_input("Models dir", value=str(ARTIFACTS))
    model_files = sorted([p.name for p in Path(models_dir).glob("*.joblib")]) if Path(models_dir).exists() else []
    model_choice = st.sidebar.selectbox("Model artifact", options=model_files)
    label_col = st.sidebar.text_input("Label column name", value="label")
    text_col = st.sidebar.text_input("Text column name", value="text_clean")

    # Load data
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Loaded uploaded CSV")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            df = None
    elif use_default:
        proc = Path(__file__).resolve().parents[2] / "datasets" / "processed" / "sms_spam_clean.csv"
        if proc.exists():
            df = pd.read_csv(proc)
            st.sidebar.write(f"Using processed dataset: {proc.name}")
        else:
            st.sidebar.warning("Processed dataset not found in repo. Upload a CSV or disable the checkbox.")

    left, right = st.columns([2, 1])

    with left:
        st.header("Data overview")
        if df is None:
            st.info("No dataset loaded. Upload a CSV or enable the repo processed dataset.")
        else:
            st.dataframe(df.head(10))
            if text_col in df.columns and label_col in df.columns:
                st.pyplot(plot_label_distribution(pd.DataFrame({'label': df[label_col]})))
                st.write(f"Rows: {len(df):,}")
                st.table(pd.DataFrame(top_n_words(df, text_col=text_col, n=20), columns=["token", "count"]))

    with right:
        st.header("Model & Live Inference")
        if model_choice:
            selected = Path(models_dir) / model_choice
            st.write(f"Selected model: {model_choice}")
        else:
            selected = None
            st.info("No model artifact selected")

        eval_data = load_eval()
        if eval_data:
            st.metric("Saved accuracy", f"{eval_data.get('accuracy'):.4f}")
        st.subheader("Live inference")
        text = st.text_area("Enter text to classify")
        if st.button("Predict live"):
            if selected is None or not selected.exists():
                st.error("No model selected")
            else:
                m = load_model(selected)
                try:
                    if hasattr(m, 'predict_proba'):
                        score = m.predict_proba([text])[0][1]
                        st.write(f"Score (spam): {score:.4f}")
                        st.write("Label:", 'spam' if score >= 0.5 else 'ham')
                    else:
                        pred = m.predict([text])[0]
                        st.write("Label:", pred)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


if __name__ == '__main__':
    main()

