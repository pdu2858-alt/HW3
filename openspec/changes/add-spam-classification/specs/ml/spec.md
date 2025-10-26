## ADDED Requirements
### Requirement: Spam/ham classification capability
The system SHALL provide a reproducible pipeline to classify SMS/email messages as spam or ham. The pipeline SHALL support data download, preprocessing, model training, evaluation, and artifact export (trained model and evaluation report).

#### Scenario: Baseline SVM training and evaluation
- **WHEN** a developer runs the baseline pipeline for Phase 1
- **THEN** the system SHALL download the dataset from the specified URL, validate its schema, preprocess text, extract features (e.g., TF-IDF), train an SVM classifier, and produce an evaluation report including accuracy, precision, recall, F1, and confusion matrix.

#### Scenario: Logistic regression implementation (Phase 2)
- **WHEN** Phase 2 is executed
- **THEN** the system SHALL train a logistic regression classifier using the same preprocessing and evaluation suite and produce a comparison report versus the baseline.
