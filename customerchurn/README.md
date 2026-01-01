 # Customer Churn

This repository represents an example of comparing Machine Learning models using the [Kaggle Customer Churn database](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset). The project uses [`uv`](https://docs.astral.sh/uv/) for provicing virtual environments.

The models tested were

* Logistic Regression
* Random Forest
* XG Boost

SVM was evaluated but ultimately excluded from the final ensemble. Despite feature scaling and synthetic oversampling (SMOTE), the model converged to a degenerate state (predicting the majority class). Given the high dimensionality introduced by One-Hot Encoding, the computational cost and sensitivity of the SVM kernel did not provide a lift over gradient-boosted trees (XGBoost).

Various hyperparameters were tested for some of the models.

The best model was XG Boost.

## Quickstart

```shells
uv install
cd src
uv run main.py
```

## Technical Challenges & Solutions

* Challenge: Standard Scikit-Learn pipelines failed when integrating SMOTE.

Solution: Implemented imblearn.pipeline to correctly sequence over-sampling within the cross-validation loop, preventing data leakage.

* Challenge: SVM model initially converged to a degenerate state (predicting only the majority class).

Solution: Refactored the ColumnTransformer to ensure uniform scaling across all feature types and adjusted class weights to handle the remaining imbalance.

## Contents

| Item | Description |
-------------------
| data | Folder containing the data used for the model testing |
| docs | Further documentation on the project and its findings |
| logs | Output of the testing in `csv` format |
| notebooks | A preliminary test of the various models in Jupyter Notebook. Ended up with mostly the same conclusion |
| src | The source code for the testing |
| tests | Potential tests for the code |
| pyproject.toml | The requirements for this Project |
| README.md | This file |