"""Streamlit app for Spam Classification (Phase 2)

Compact Streamlit demo for inspecting the processed SMS spam dataset and
running live inference with saved model artifacts.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import joblib

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


def top_n_words_by_label(df, label_col='label', text_col='text', n=20):
    res = {}
    for lbl in sorted(df[label_col].dropna().unique()):
        sub = df[df[label_col] == lbl]
        res[lbl] = top_n_words(sub, text_col=text_col, n=n)
    return res


def threshold_sweep_metrics(y_true, y_score, thresholds=None):
    thresholds = thresholds or np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p = metrics.precision_score(y_true, y_pred, zero_division=0)
        r = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        rows.append({'threshold': float(t), 'precision': float(p), 'recall': float(r), 'f1': float(f1)})
    return pd.DataFrame(rows)


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
    # Improved CSS for cleaner, consistent presentation 
    st.markdown(f""" 
        <style> 
        /* page & container */ 
        .stApp {{ background-color: #f5f7fa; color: #0f1724; }} 
        .css-1d391kg {{ padding-top: 1rem; }} 
 
        /* Buttons */ 
        .stButton>button, button[kind='primary']{{ 
            background: {PALETTE['primary']} !important; 
            color: white !important; 
            border-radius: 8px !important; 
            padding: 6px 12px !important; 
            box-shadow: none !important; 
        }} 
 
        /* cards / metric boxes */ 
        .card {{ background: white; padding:12px; border-radius:10px; box-shadow: 0 4px 10px rgba(16,24,40,0.06); }} 
        .section-head {{font-size:18px; font-weight:700; margin-bottom:6px; color:#0b2545}} 
        .muted {{color: {PALETTE['muted']};}} 
 
        /* badges for prediction */ 
        .badge {{ display:inline-block; padding:6px 10px; border-radius:999px; color: white; font-weight:700; }} 
        .badge-spam {{ background: {PALETTE['danger']}; }} 
        .badge-ham  {{ background: {PALETTE['ok']}; }} 
 
        /* smaller sidebar sample text */ 
        .sidebar .stText {{ font-size: 13px; color: #123; }} 
 
        /* dataframe tweaks */ 
        .stDataFrame table {{ border-radius:8px; overflow:hidden; }} 
        </style> 
        """, unsafe_allow_html=True) 

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

    # Auto-detect best text and label columns when possible
    effective_text_col = None
    effective_label_col = None
    if df is not None:
        # prefer user-specified sidebar values, then common names
        candidates_text = [text_col, 'text_clean', 'text', 'message', 'body']
        for c in candidates_text:
            if c in df.columns:
                effective_text_col = c
                break
        if effective_text_col is None:
            # fallback to first string-like column
            for c in df.columns:
                if pd.api.types.is_string_dtype(df[c]):
                    effective_text_col = c
                    break
        if effective_text_col is None:
            effective_text_col = df.columns[0]

        candidates_label = [label_col, 'label', 'target', 'y']
        for c in candidates_label:
            if c in df.columns:
                effective_label_col = c
                break
        # if no label col found, leave None (some demos may be unlabeled)
        if effective_label_col is None:
            effective_label_col = None

        # Surface the detected choices in the sidebar for clarity
        st.sidebar.markdown(f"**Detected text column:** `{effective_text_col}`")
        st.sidebar.markdown(f"**Detected label column:** `{effective_label_col if effective_label_col is not None else 'none'}`")
        # replace local pointers for downstream code
        text_col = effective_text_col
        label_col = effective_label_col or label_col

    # Layout: tabs to surface key sections on the homepage
    tab_overview, tab_model, tab_infer = st.tabs(["Overview", "Model", "Live Inference"])

    # Overview tab: dataset preview, class distribution, top tokens, token replacements
    with tab_overview:
        st.subheader("Data overview")
        if df is None:
            st.info("No dataset loaded. Upload a CSV or enable the repo processed dataset.")
            st.write("You can also pick a sample message from the sidebar to try the classifier.")
        else:
            with st.expander("Show dataset (first 200 rows)"):
                st.dataframe(df.head(200))

            if text_col in df.columns and label_col in df.columns:
                st.markdown("### Class distribution")
                st.pyplot(plot_label_distribution(pd.DataFrame({'label': df[label_col]})))
                st.markdown(f"**Rows:** {len(df):,}")

                # top words bar chart
                top = top_n_words(df, text_col=text_col, n=20)
                tokens, counts = zip(*top) if top else ([], [])
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(tokens[::-1], counts[::-1], color=PALETTE['primary'])
                ax.set_xlabel("count")
                plt.tight_layout()
                st.pyplot(fig)

                # quick density by length
                st.markdown("### Message length distribution")
                lengths = df[text_col].fillna("").astype(str).map(len)
                fig2, ax2 = plt.subplots(figsize=(6, 2))
                ax2.hist(lengths, bins=30, color=PALETTE['accent'])
                ax2.set_xlabel('chars')
                plt.tight_layout()
                st.pyplot(fig2)

                # Token replacements (approximate) — show as cards for readability
                st.markdown("### Token replacements in cleaned text (preview)")
                sample_rows = df[text_col].dropna().astype(str).head(8).tolist()
                cleaned = [clean_steps_for_display(t) for t in sample_rows]
                if cleaned:
                    cols = st.columns(2)
                    for i, item in enumerate(cleaned):
                        with cols[i % 2]:
                            st.markdown(
                                """
                                <div class='card'>
                                  <div style='font-size:12px;color: #6c757d;'>Original</div>
                                  <div style='margin-top:6px;color:#0b1724'>{orig}</div>
                                  <hr style='opacity:0.06' />
                                  <div style='font-size:12px;color: #6c757d;margin-top:6px;'>Cleaned preview</div>
                                  <div style='margin-top:6px;font-weight:700;color:#0b2545'>{clean}</div>
                                </div>
                                """.format(orig=item['original'][:600].replace('\n',' '), clean=item['cleaned_preview']),
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No text rows available to preview replacements.")

                # Top tokens by class (or overall if no label)
                st.markdown("### Top tokens by class")
                try:
                    if label_col is None or label_col not in df.columns:
                        st.info("No label column detected — showing overall top tokens.")
                        top_all = top_n_words(df, text_col=text_col, n=20)
                        if top_all:
                            tdf = pd.DataFrame(top_all, columns=['token', 'count'])
                            st.table(tdf)
                        else:
                            st.info("No tokens to display.")
                    else:
                        by_label = top_n_words_by_label(df, label_col=label_col, text_col=text_col, n=15)
                        for lbl, toks in by_label.items():
                            st.markdown(f"**{lbl}**")
                            tdf = pd.DataFrame(toks, columns=['token', 'count'])
                            st.table(tdf)
                except Exception:
                    st.info("Failed to compute top tokens by class.")

    # Model tab: metrics, confusion, ROC/PR, threshold sweep, classification report
    with tab_model:
        st.subheader("Model Performance & Evaluation")
        selected = Path(models_dir) / model_choice if model_choice else None
        if selected is None or not selected.exists():
            st.info("No model artifact selected or found. Select a .joblib model in the sidebar to see saved evaluation.")

        eval_data = load_eval()
        if not eval_data:
            st.info("No saved evaluation found (eval_baseline.json). Train a model or place an evaluation file in artifacts/.")
            # If no saved eval but a model artifact exists and labeled dataset is loaded,
            # attempt a small on-the-fly evaluation on a sample to surface ROC/PR/thresholds.
            if selected is not None and selected.exists() and df is not None and label_col in df.columns:
                st.info("Saved eval not found — running on-the-fly evaluation using the selected model and dataset sample. This may take a moment.")
                try:
                    m = load_model(selected)
                    texts = df[text_col].fillna("").astype(str).tolist()
                    sample_n = min(len(texts), 500)
                    texts_sample = texts[:sample_n]
                    y_true_raw = df[label_col].iloc[:sample_n]
                    # normalize y_true to binary 0/1 if possible
                    try:
                        # common cases: 'spam'/'ham' or 1/0
                        if pd.api.types.is_string_dtype(y_true_raw):
                            y_true = (y_true_raw.str.lower() == 'spam').astype(int).to_numpy()
                        else:
                            y_true = y_true_raw.astype(int).to_numpy()
                    except Exception:
                        # fallback: map unique highest-value to spam
                        y_true = (y_true_raw != y_true_raw.iloc[0]).astype(int).to_numpy()

                    y_score = None
                    probs = None
                    if hasattr(m, 'predict_proba'):
                        probs = [float(p[-1]) for p in m.predict_proba(texts_sample)]
                        y_score = np.array(probs)
                    elif hasattr(m, 'decision_function'):
                        # decision_function returns scores; some metrics accept these
                        try:
                            ds = m.decision_function(texts_sample)
                            y_score = np.array(ds)
                        except Exception:
                            y_score = None

                    # show confusion / ROC / PR depending on available scores
                    if y_score is not None:
                        try:
                            st.markdown("### On-the-fly Confusion matrix (sample)")
                            y_pred_bin = (y_score >= spam_threshold).astype(int)
                            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                            cm = metrics.confusion_matrix(y_true, y_pred_bin)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                            ax_cm.set_xlabel('pred')
                            ax_cm.set_ylabel('true')
                            st.pyplot(fig_cm)

                            st.markdown("### On-the-fly ROC & Precision-Recall (sample)")
                            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
                            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                            ax_roc.plot(fpr, tpr, label=f"AUC={metrics.auc(fpr,tpr):.3f}", color=PALETTE['primary'])
                            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax_roc.set_xlabel('FPR')
                            ax_roc.set_ylabel('TPR')
                            ax_roc.legend()
                            st.pyplot(fig_roc)

                            precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
                            ap = metrics.average_precision_score(y_true, y_score)
                            fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
                            ax_pr.plot(recall, precision, color=PALETTE['accent'], label=f"AP={ap:.3f}")
                            ax_pr.set_xlabel('Recall')
                            ax_pr.set_ylabel('Precision')
                            ax_pr.legend()
                            st.pyplot(fig_pr)

                            # threshold sweep
                            thresh_df = threshold_sweep_metrics(y_true, y_score)
                            st.markdown("### Threshold sweep (precision / recall / f1) — sample")
                            fig_ts, ax_ts = plt.subplots(figsize=(6, 3))
                            ax_ts.plot(thresh_df['threshold'], thresh_df['precision'], label='precision', color=PALETTE['primary'])
                            ax_ts.plot(thresh_df['threshold'], thresh_df['recall'], label='recall', color=PALETTE['accent'])
                            ax_ts.plot(thresh_df['threshold'], thresh_df['f1'], label='f1', color=PALETTE['danger'])
                            ax_ts.set_xlabel('threshold')
                            ax_ts.set_ylabel('score')
                            ax_ts.legend()
                            st.pyplot(fig_ts)
                            best = thresh_df.loc[thresh_df['f1'].idxmax()]
                            st.markdown(f"**Best F1 (sample):** {best['f1']:.3f} at threshold {best['threshold']:.3f}")
                        except Exception as e:
                            st.info(f"On-the-fly evaluation failed: {e}")
                    else:
                        # try to at least show classification report using predict
                        try:
                            y_pred = m.predict(texts_sample)
                            # normalize labels as above
                            if pd.api.types.is_string_dtype(y_true_raw):
                                # ensure y_pred comparable
                                y_pred_bin = (pd.Series(y_pred).str.lower() == 'spam').astype(int).to_numpy()
                            else:
                                y_pred_bin = pd.Series(y_pred).astype(int).to_numpy()
                            report = metrics.classification_report(y_true, y_pred_bin, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.markdown("### Classification report (sample)")
                            st.dataframe(report_df)
                        except Exception as e:
                            st.info(f"Unable to run on-the-fly evaluation with selected model: {e}")
                except Exception as e:
                    st.info(f"On-the-fly evaluation error: {e}")
        else:
            acc = eval_data.get('accuracy')
            prec = eval_data.get('precision')
            rec = eval_data.get('recall')
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Accuracy", value=f"{acc:.3f}" if acc is not None else "N/A")
            with c2:
                st.metric(label="Precision", value=f"{prec:.3f}" if prec is not None else "N/A")
            with c3:
                st.metric(label="Recall", value=f"{rec:.3f}" if rec is not None else "N/A")

            y_true = eval_data.get('y_true')
            y_score = eval_data.get('y_score') or eval_data.get('y_pred')
            if y_true is not None and y_score is not None:
                try:
                    y_true = np.array(y_true)
                    y_score = np.array(y_score)
                    st.markdown("### Confusion matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    y_pred_bin = (y_score >= spam_threshold).astype(int)
                    cm = metrics.confusion_matrix(y_true, y_pred_bin)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel('pred')
                    ax_cm.set_ylabel('true')
                    st.pyplot(fig_cm)

                    st.markdown("### ROC & Precision-Recall")
                    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
                    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                    ax_roc.plot(fpr, tpr, label=f"AUC={metrics.auc(fpr,tpr):.3f}", color=PALETTE['primary'])
                    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    ax_roc.set_xlabel('FPR')
                    ax_roc.set_ylabel('TPR')
                    ax_roc.legend()
                    st.pyplot(fig_roc)

                    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
                    ap = metrics.average_precision_score(y_true, y_score)
                    fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
                    ax_pr.plot(recall, precision, color=PALETTE['accent'], label=f"AP={ap:.3f}")
                    ax_pr.set_xlabel('Recall')
                    ax_pr.set_ylabel('Precision')
                    ax_pr.legend()
                    st.pyplot(fig_pr)

                    st.markdown("### Model Performance (Test)")
                    try:
                        report = metrics.classification_report(y_true, (y_score >= spam_threshold).astype(int), output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)
                    except Exception:
                        st.info("Unable to build classification report.")

                    try:
                        thresh_df = threshold_sweep_metrics(y_true, y_score)
                        st.markdown("### Threshold sweep (precision / recall / f1)")
                        fig_ts, ax_ts = plt.subplots(figsize=(6, 3))
                        ax_ts.plot(thresh_df['threshold'], thresh_df['precision'], label='precision', color=PALETTE['primary'])
                        ax_ts.plot(thresh_df['threshold'], thresh_df['recall'], label='recall', color=PALETTE['accent'])
                        ax_ts.plot(thresh_df['threshold'], thresh_df['f1'], label='f1', color=PALETTE['danger'])
                        ax_ts.set_xlabel('threshold')
                        ax_ts.set_ylabel('score')
                        ax_ts.legend()
                        st.pyplot(fig_ts)

                        best = thresh_df.loc[thresh_df['f1'].idxmax()]
                        st.markdown(f"**Best F1:** {best['f1']:.3f} at threshold {best['threshold']:.3f}")
                        with st.expander('Threshold sweep table'):
                            st.dataframe(thresh_df.head(200))
                    except Exception:
                        st.info("Unable to compute threshold sweep.")
                except Exception:
                    st.info("Error plotting evaluation arrays.")

    # Live inference tab: text input, predict button, batch run
    with tab_infer:
        st.subheader("Live inference")
        selected = Path(models_dir) / model_choice if model_choice else None
        if selected is None or not selected.exists():
            st.info("No model selected — pick a .joblib artifact in the sidebar to enable live inference")

        if sample_choice:
            text = st.text_area("Enter text to classify", value=sample_choice, height=140)
        else:
            text = st.text_area("Enter text to classify", height=140)

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
                        badge_class = 'badge-spam' if label == 'spam' else 'badge-ham'
                        st.markdown(f"**Predicted:** <span class='badge {badge_class}'>{label.upper()}</span>  &nbsp; (score={score:.3f})", unsafe_allow_html=True)
                    else:
                        pred = str(m.predict([text])[0])
                        is_spam = pred.lower() == 'spam'
                        badge_class = 'badge-spam' if is_spam else 'badge-ham'
                        st.markdown(f"**Predicted:** <span class='badge {badge_class}'>{pred.upper()}</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

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

        st.markdown("---")
        st.markdown("Made with ❤️ — improve visuals or wire your own model artifacts in `ml/spam_classification/artifacts/`.")


if __name__ == '__main__':
    main()

