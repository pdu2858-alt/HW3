## Why
Spam (unsolicited) email detection remains an important baseline capability for many systems. We want a lightweight, reproducible machine learning pipeline that can classify SMS/email messages as spam or ham for research and demonstration purposes.

This proposal scopes an initial, spec-driven implementation of a spam classification pipeline: a Phase 1 baseline using a simple SVM model trained on an openly-available dataset, followed by a Phase 2 logistic regression implementation and comparison.

## What Changes
- Add a new capability `ml/spam-classification` with a baseline implementation and specs.
- Create reproducible data download and preprocessing steps using the dataset at: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Phase 1: Train and evaluate a baseline SVM classifier and report metrics (accuracy, precision, recall, F1, confusion matrix). This provides a baseline result.
- Phase 2: Implement logistic regression classifier (the intended target model), train, evaluate, and compare against the baseline.
- Add tests and CI checks to ensure the pipeline runs and produces expected artifacts (model file, evaluation report).

**BREAKING**: None.

## Impact
- Affected specs: new `ml/spam-classification` capability under `openspec/specs/` (ADDED). Existing systems are not impacted.
- Affected code: new scripts, small training pipeline, test harness, and CI workflow changes to run training/validation jobs in CI or as part of reproducibility checks.

## Data
- Primary dataset (CSV): https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Expected format: CSV rows with label and text (confirm header/columns during implementation). We'll include a small data validation step to assert schema.

## Rollout
1. Add spec deltas and tasks (this proposal).  
2. Implement Phase 1 (baseline SVM): data download, preprocessing, feature extraction (e.g., TF-IDF), train SVM, save model, evaluate and record metrics.  
3. Open PR with results and request review.  
4. After approval, implement Phase 2 logistic regression and compare to baseline.  
5. Archive change when complete and update canonical `openspec/specs/` if the capability should remain.

## Acceptance Criteria
- A reproducible script or notebook that downloads the dataset and produces a trained baseline SVM model.  
- An evaluation report with accuracy, precision, recall, F1, and confusion matrix committed as an artifact or as part of the PR.  
- Phase 2 logistic regression implemented and evaluated; comparison summary included in PR.