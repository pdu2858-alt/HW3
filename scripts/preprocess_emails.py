#!/usr/bin/env python3
"""Preprocess SMS/email dataset into cleaned CSV and optional per-step outputs.

Usage example:
python scripts/preprocess_emails.py \
  --input datasets/sms_spam_no_header.csv \
  --output datasets/processed/sms_spam_clean.csv \
  --no-header --label-col-index 0 --text-col-index 1 \
  --output-text-col text_clean \
  --save-step-columns \
  --steps-out-dir datasets/processed/steps

The script performs simple cleaning steps and saves intermediate step files
if requested. The final CSV will contain at least the label column and the
cleaned text column specified by --output-text-col.
"""
import argparse
import os
import re
from pathlib import Path

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+")
PUNCT_RE = re.compile(r"[\r\n]+|[^\w\s]|_", flags=re.UNICODE)


def clean_lower(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_non_printable(text: str) -> str:
    # remove control characters
    return "".join(ch for ch in text if ch.isprintable())


def remove_punctuation(text: str) -> str:
    # keep alphanumeric and whitespace
    return PUNCT_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def run(args):
    inp = Path(args.input)
    out = Path(args.output)
    steps_dir = Path(args.steps_out_dir) if args.steps_out_dir else None
    if steps_dir:
        steps_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # read CSV
    if args.no_header:
        df = pd.read_csv(inp, header=None, encoding="utf-8", dtype=str)
    else:
        df = pd.read_csv(inp, encoding="utf-8", dtype=str)

    # extract label and text columns
    label_idx = args.label_col_index
    text_idx = args.text_col_index

    df = df.fillna("")
    df_columns = df.columns.tolist()
    # safe access by position
    df_work = pd.DataFrame()
    df_work["label_orig"] = df.iloc[:, label_idx].astype(str)
    df_work["text_orig"] = df.iloc[:, text_idx].astype(str)

    # Step 1: lowercasing
    df_work["step_1_lower"] = df_work["text_orig"].apply(clean_lower)
    if steps_dir:
        (steps_dir / "step_1_lower.csv").write_text(df_work[["label_orig", "step_1_lower"]].to_csv(index=False))

    # Step 2: remove urls
    df_work["step_2_no_urls"] = df_work["step_1_lower"].apply(remove_urls)
    if steps_dir:
        (steps_dir / "step_2_no_urls.csv").write_text(df_work[["label_orig", "step_2_no_urls"]].to_csv(index=False))

    # Step 3: remove non-printable
    df_work["step_3_printable"] = df_work["step_2_no_urls"].apply(remove_non_printable)
    if steps_dir:
        (steps_dir / "step_3_printable.csv").write_text(df_work[["label_orig", "step_3_printable"]].to_csv(index=False))

    # Step 4: remove punctuation
    df_work["step_4_no_punct"] = df_work["step_3_printable"].apply(remove_punctuation)
    if steps_dir:
        (steps_dir / "step_4_no_punct.csv").write_text(df_work[["label_orig", "step_4_no_punct"]].to_csv(index=False))

    # Step 5: normalize whitespace
    df_work["step_5_norm"] = df_work["step_4_no_punct"].apply(normalize_whitespace)
    if steps_dir:
        (steps_dir / "step_5_norm.csv").write_text(df_work[["label_orig", "step_5_norm"]].to_csv(index=False))

    # final column name
    out_text_col = args.output_text_col or "text_clean"
    df_out = pd.DataFrame()
    df_out[args.label_col_name or "label"] = df_work["label_orig"].astype(str).str.strip()
    df_out[out_text_col] = df_work["step_5_norm"]

    # optionally include step columns
    if args.save_step_columns:
        for c in ["step_1_lower", "step_2_no_urls", "step_3_printable", "step_4_no_punct", "step_5_norm"]:
            df_out[c] = df_work[c]

    # save final output
    df_out.to_csv(out, index=False)
    print(f"Wrote processed output to {out}")

    # also save steps as individual files if not already saved
    if steps_dir:
        # already saved above, but ensure index-free csvs exist for full rows
        for i, col in enumerate(["step_1_lower", "step_2_no_urls", "step_3_printable", "step_4_no_punct", "step_5_norm"], start=1):
            path = steps_dir / f"step_{i:02d}_{col}.csv"
            df_step = pd.DataFrame({args.label_col_name or "label": df_work["label_orig"], col: df_work[col]})
            df_step.to_csv(path, index=False)
        print(f"Wrote step outputs to {steps_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-header", action="store_true", dest="no_header")
    p.add_argument("--label-col-index", type=int, default=0)
    p.add_argument("--text-col-index", type=int, default=1)
    p.add_argument("--output-text-col", default="text_clean")
    p.add_argument("--save-step-columns", action="store_true", dest="save_step_columns")
    p.add_argument("--steps-out-dir", default=None)
    p.add_argument("--label-col-name", default=None, dest="label_col_name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
