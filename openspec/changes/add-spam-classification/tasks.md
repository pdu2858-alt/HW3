## 1. Implementation - Phase 1 (Baseline)
- [ ] 1.1 Create `openspec/changes/add-spam-classification/` scaffold (proposal, tasks, specs)
- [ ] 1.2 Add data download script that fetches the CSV from the provided URL and validates schema
- [ ] 1.3 Implement preprocessing and feature extraction (tokenization, TF-IDF)
- [ ] 1.4 Train baseline SVM classifier, save model artifact, and produce evaluation report (accuracy, precision, recall, F1, confusion matrix)
- [ ] 1.5 Add unit/integration tests and a small example notebook or script demonstrating training + inference

## 2. Implementation - Phase 2 (Logistic Regression)
- [ ] 2.1 Implement logistic regression pipeline using the same preprocessing and evaluation suite
- [ ] 2.2 Compare results vs baseline and document findings
- [ ] 2.3 Add model selection or hyperparameter tuning notes (optional)

## 3. CI / Docs
- [ ] 3.1 Add CI job or workflow to validate pipeline runs on a small sample (or use cached artifact)
- [ ] 3.2 Document usage and how to reproduce experiments in README or docs
- [ ] 3.3 Request review & approval from maintainers
