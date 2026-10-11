# IS507: Predicting Death in Heart Failure Patients

**Team:** Yosef Stifanos, Juan Lopez, Edwin Joseph

## Overview
We predict whether a heart failure patient dies during follow-up
(4 to 285 days, about 10 months) from clinical measurements. We
compare three models: Logistic Regression, Random Forest, and XGBoost.

## Data
UCI Heart Failure Clinical Records (Chicco & Jurman, 2020).
- 299 patients (one per row), 12 clinical features, 1 outcome (`DEATH_EVENT`)
- 96 of 299 patients died (32%), so the classes are imbalanced
- No missing values and no duplicate rows

Source: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

## Key idea
Follow-up `time` ends when a patient dies, so it would not be known when
predicting for a new patient and can leak the outcome. Every model is
evaluated both with and without `time`. The with-time results are
treated as an upper bound.

## Approach
- Stratified 80/20 split (239 training, 60 test patients)
- Repeated 5-fold cross-validation on the training set only
- Majority-class baseline: 0.678 accuracy, catches 0 deaths
- Class weights to handle the imbalance
- Hyperparameter tuning scored by ROC-AUC (training data only)
- Comparison of real-only vs. real + synthetic training data
- Final evaluation on the held-out test set, run once

## Preliminary results (cross-validation, training set, untuned)

| Features | Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|---|
| With time | Random Forest | 0.842 | 0.777 | 0.916 |
| Without time | Random Forest | 0.747 | 0.614 | 0.775 |
| Baseline | Always "survived" | 0.678 | 0.000 | 0.500 |

The 80% accuracy goal is reached only when `time` is included.

## Repository contents
- `IS507_PROJECT.ipynb`: full analysis notebook
- `data/`: dataset
- `docs/`: project proposal and presentation slides
- `requirements.txt`: Python packages

## How to run
1. Install packages: `pip install -r requirements.txt`
2. Open `IS507_PROJECT.ipynb` in Jupyter.
3. Run all cells from top to bottom.

## Status
The proposal and slides in `docs/` cover the preliminary stage. The
notebook goes further, through tuning, the synthetic-data comparison,
feature importance, and the final test-set evaluation.

## Limitations
Only 299 patients from one study, so scores are noisy. Synthetic
records are noisy resamples of training patients, not new patients.
These results are an initial analysis and are not ready for clinical use.

## Reference
Chicco, D., & Jurman, G. (2020). Machine learning can predict survival
of patients with heart failure from serum
