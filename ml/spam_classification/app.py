"""Streamlit app for Spam Classification (Phase 2)

Compact Streamlit demo for inspecting the processed SMS spam dataset and
running live inference with saved model artifacts.
"""

from pathlib import Path
import json
import re
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import joblib

# UI color palette / simple theme
PALETTE = {
    'primary': '#2b8cbe',
    'accent': '#fdae6b',
    'danger': '#b30000',
    'ok': '#0b6b0b',
    'muted': '#6c757d'
}


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
    # Page config and styles
    st.set_page_config(page_title="Spam Classifier — Demo", layout="wide", page_icon="📧")
    st.markdown(
        """
        <style>
        .stApp { background-color: #f7f9fb; }
        .title {font-size:32px; font-weight:700;}
        .metric-box {background: white; padding: 12px; border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);} 
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# 📧 Spam / Ham Classifier — Visualizations & Live Demo")
    st.markdown("Live demo using the processed SMS spam dataset. Upload your own CSV or use the repo data and a saved model artifact.")

    ensure_artifacts()

    # Sidebar: input controls and examples
    st.sidebar.header("Inputs & Options")
    use_default = st.sidebar.checkbox("Use processed dataset in repo", value=True)
    uploaded = st.sidebar.file_uploader("Or upload CSV", type=["csv"])    
    models_dir = st.sidebar.text_input("Models dir", value=str(ARTIFACTS))
    model_files = sorted([p.name for p in Path(models_dir).glob("*.joblib")]) if Path(models_dir).exists() else []
    model_choice = st.sidebar.selectbox("Model artifact", options=[""] + model_files, index=0)
    label_col = st.sidebar.text_input("Label column name", value="label")
    text_col = st.sidebar.text_input("Text column name", value="text_clean")
    spam_threshold = st.sidebar.slider("Spam threshold", min_value=0.0, max_value=1.0, value=0.5)

    # load sample messages (packaged)
    sample_path = APP_DIR / "samples" / "example_messages.csv"
    sample_df = None
    if sample_path.exists():
        try:
            sample_df = pd.read_csv(sample_path)
        except Exception:
            sample_df = None

    if sample_df is not None and 'text' in sample_df.columns:
        # show short sample list with index for convenience
        sample_list = sample_df['text'].head(20).tolist()
        sample_choice = st.sidebar.selectbox("Try a sample message", options=[""] + sample_list)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Packaged samples (first 20):**")
        for i, s in enumerate(sample_list[:8]):
            st.sidebar.write(f"{i+1}. {s[:80]}{'...' if len(s)>80 else ''}")
    else:
        sample_choice = ""

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

    # Layout
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Data overview")
        if df is None:
            st.info("No dataset loaded. Upload a CSV or enable the repo processed dataset.")
            st.write("You can also pick a sample message from the sidebar to try the classifier.")
        else:
            with st.expander("Show dataset (first 200 rows)"):
                st.dataframe(df.head(200))
            if text_col in df.columns and label_col in df.columns:
                st.markdown("**Label distribution**")
                st.pyplot(plot_label_distribution(pd.DataFrame({'label': df[label_col]})))
                st.markdown(f"**Rows:** {len(df):,}")

                # top words bar chart
                top = top_n_words(df, text_col=text_col, n=20)
                tokens, counts = zip(*top) if top else ([], [])
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(tokens[::-1], counts[::-1], color="#2b8cbe")
                ax.set_xlabel("count")
                plt.tight_layout()
                st.pyplot(fig)

                # quick density by length
                st.markdown("**Message length distribution**")
                lengths = df[text_col].fillna("").astype(str).map(len)
                fig2, ax2 = plt.subplots(figsize=(6, 2))
                ax2.hist(lengths, bins=30, color=PALETTE['accent'])
                ax2.set_xlabel('chars')
                plt.tight_layout()
                st.pyplot(fig2)

    with right:
        st.subheader("Model & Live Inference")
        if model_choice:
            selected = Path(models_dir) / model_choice
            st.markdown(f"**Selected model:** `{model_choice}`")
        else:
            selected = None
            st.info("No model artifact selected in sidebar")

        # helper to render small metric card
        def _metric_card(title, value, color=None):
            color = color or PALETTE['primary']
            st.markdown(
                f"<div style='background: white; padding:10px; border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>"
                f"<div style='font-size:12px;color:{PALETTE['muted']}'>{title}</div>"
                f"<div style='font-size:20px;font-weight:700;color:{color}'>{value}</div>"
                f"</div>", unsafe_allow_html=True
            )

        # show saved eval if present
        eval_data = load_eval()
        if eval_data:
            acc = eval_data.get('accuracy')
            prec = eval_data.get('precision')
            rec = eval_data.get('recall')
            with st.container():
                c1, c2, c3 = st.columns(3)
                with c1:
                    _metric_card('Accuracy', f"{acc:.4f}" if acc is not None else "N/A", color=PALETTE['primary'])
                with c2:
                    _metric_card('Precision', f"{prec:.4f}" if prec is not None else "N/A", color=PALETTE['accent'])
                with c3:
                    _metric_card('Recall', f"{rec:.4f}" if rec is not None else "N/A", color=PALETTE['ok'])

            # if eval_data contains arrays we can plot ROC / confusion
            y_true = eval_data.get('y_true')
            y_score = eval_data.get('y_score') or eval_data.get('y_pred')
            if y_true is not None and y_score is not None:
                try:
                    y_true = np.array(y_true)
                    y_score = np.array(y_score)
                    fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
                    y_pred_bin = (y_score >= spam_threshold).astype(int)
                    cm = metrics.confusion_matrix(y_true, y_pred_bin)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel('pred')
                    ax_cm.set_ylabel('true')
                    st.pyplot(fig_cm)

                    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
                    fig_roc, ax_roc = plt.subplots(figsize=(3, 3))
                    ax_roc.plot(fpr, tpr, label=f"AUC={metrics.auc(fpr,tpr):.3f}", color=PALETTE['primary'])
                    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    ax_roc.set_xlabel('FPR')
                    ax_roc.set_ylabel('TPR')
                    ax_roc.legend()
                    st.pyplot(fig_roc)

                    # precision-recall
                    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
                    ap = metrics.average_precision_score(y_true, y_score)
                    fig_pr, ax_pr = plt.subplots(figsize=(3, 3))
                    ax_pr.plot(recall, precision, color=PALETTE['accent'], label=f"AP={ap:.3f}")
                    ax_pr.set_xlabel('Recall')
                    ax_pr.set_ylabel('Precision')
                    ax_pr.legend()
                    st.pyplot(fig_pr)
                except Exception:
                    pass

        st.markdown("---")
        st.subheader("Live inference")
        # text input area and sample quick-fill
        if sample_choice:
            text = st.text_area("Enter text to classify", value=sample_choice, height=120)
        else:
            text = st.text_area("Enter text to classify", height=120)

        if st.button("Predict live"):
            if selected is None or not selected.exists():
                st.error("No model selected — pick a .joblib artifact in the sidebar")
            else:
                m = load_model(selected)
                try:
                    # try predict_proba first
                    if hasattr(m, 'predict_proba'):
                        score = float(m.predict_proba([text])[0][-1])
                        label = 'spam' if score >= spam_threshold else 'ham'
                        color = PALETTE['danger'] if label == 'spam' else PALETTE['ok']
                        st.markdown(f"**Predicted:** <span style='font-weight:700;color:{color}'>{label}</span> (score={score:.3f})", unsafe_allow_html=True)
                    else:
                        pred = m.predict([text])[0]
                        color = PALETTE['ok'] if str(pred).lower() != 'spam' else PALETTE['danger']
                        st.markdown(f"**Predicted:** <span style='font-weight:700;color:{color}'>{pred}</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # allow running model over packaged sample set and download
        if sample_df is not None and selected is not None and selected.exists():
            if st.button("Run model on packaged samples and download CSV"):
                try:
                    m = load_model(selected)
                    texts = sample_df['text'].astype(str).tolist()
                    if hasattr(m, 'predict_proba'):
                        probs = [float(p[-1]) for p in m.predict_proba(texts)]
                        labels = ['spam' if p >= spam_threshold else 'ham' for p in probs]
                    else:
                        labels = m.predict(texts)
                        probs = [None] * len(labels)
                    out = sample_df.copy()
                    out['pred_label'] = labels
                    out['pred_score'] = probs
                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button("Download predictions (CSV)", data=buf.getvalue(), file_name="sample_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

        # small footer
        st.markdown("---")
        st.markdown("Made with ❤️ — improve visuals or wire your own model artifacts in `ml/spam_classification/artifacts/`.")


if __name__ == '__main__':
    main()

