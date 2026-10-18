# IS507: Predicting Death in Heart Failure Patients

**Team:** Yosef Stifanos, Juan Lopez, Edwin Joseph

## Overview
We predict whether a heart failure patient dies during follow-up
(4 to 285 days, about 10 months) from clinical measurements. We
compare three models: Logistic Regression, Random Forest, and XGBoost.

Three questions guide the analysis:
1. Do synthetic (artificial) training records help?
2. How well do the models work **without** `time`?
3. Does handling the class imbalance help catch deaths?

## Data
UCI Heart Failure Clinical Records (Chicco & Jurman, 2020).
- 299 patients (one per row), 12 clinical features, 1 outcome (`DEATH_EVENT`)
- 96 of 299 patients died (32%), so the classes are imbalanced
- No missing values and no duplicate rows

Source: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

## Key idea
Follow-up `time` ends when a patient dies, so it would not be known when
predicting for a new patient and can leak the outcome. Every model is
evaluated both with and without `time`. With-time results are an upper
bound; without-time results are the realistic ones.

## Approach
- Stratified 80/20 split (239 training, 60 test patients)
- Repeated 5-fold cross-validation (5x10) on the training set only
- Majority-class baseline: 0.678 CV accuracy, catches 0 deaths
- Hyperparameter tuning scored by ROC-AUC (training data only)
- Imbalance comparison: no handling, class weights, random oversampling,
  synthetic minority records, and threshold tuning
- Real-only vs. real + synthetic training data (synthetic records made from
  training data only, inside each fold, and never used for testing)
- Uncertainty: bootstrap 95% ranges on the 60-patient test set, and
  paired-difference ranges for repeated cross-validation
- Feature importance: permutation importance and logistic-regression odds ratios
- Final evaluation on the held-out test set. It is scored in Steps 13 and
  16-17 of the notebook, but no model or setting was chosen from it.

## Results

**Cross-validation accuracy (tuned, training set, mean ± SD)**

| Model | With time | Without time |
|---|---|---|
| Baseline (always "survived") | 0.678 | 0.678 |
| Logistic Regression | 0.806 ± 0.042 | 0.745 ± 0.060 |
| Random Forest | 0.841 ± 0.043 | 0.763 ± 0.049 |
| XGBoost | 0.840 ± 0.047 | 0.752 ± 0.046 |

**Test set, without time, class weights (60 patients, 19 deaths; 95% range)**

| Model | Accuracy | ROC-AUC | Deaths caught |
|---|---|---|---|
| Baseline | 0.683 | 0.50 | 0 / 19 |
| Logistic Regression | 0.683 (0.57-0.80) | 0.734 (0.59-0.87) | 10 / 19 |
| Random Forest | 0.717 (0.60-0.82) | 0.801 (0.67-0.90) | 12 / 19 |
| XGBoost | 0.767 (0.65-0.87) | 0.791 (0.67-0.89) | 13 / 19 |

**Findings**
- The 80% accuracy goal is reached only when `time` is included. Without
  it, the best cross-validation accuracy is 0.763 and the best test accuracy is 0.767.
- Handling imbalance raises deaths caught from about 40-46% to 63-74% in
  cross-validation, with no loss in accuracy. The fixes perform about equally.
- Synthetic records gave no reliable gain: every paired difference had a
  range that included zero. The synthetic records are near-copies of
  training patients.
- Without `time`, serum creatinine and ejection fraction matter most.

## Repository contents
- `IS507_PROJECT.ipynb`: full analysis notebook (Steps 1-19)
- `data/`: dataset
- `docs/`: project proposal, midterm slides, and final slides
- `requirements.txt`: Python packages

## How to run
1. Install packages: `pip install -r requirements.txt`
2. Open `IS507_PROJECT.ipynb` in Jupyter.
3. Run all cells from top to bottom. Random seeds are fixed
   (`RANDOM_STATE = 42`), so results should reproduce.

## Limitations
Only 299 patients from one study, and 60 test patients (19 deaths), so one
patient moves test accuracy by about 1.7 points and the ranges are wide.
Synthetic records are noisy resamples of training patients, not new
patients. This is a course project and is not ready for clinical use.

## Reference
Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of
patients with heart failure from serum creatinine and ejection fraction
alone. *BMC Medical Informatics and Decision Making*, 20, 16.
