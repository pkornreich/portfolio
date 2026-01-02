 # Customer Churn

This repository represents an example of comparing Machine Learning models using the [Kaggle Customer Churn database](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset). The project uses [`uv`](https://docs.astral.sh/uv/) for provicing virtual environments.

The models tested were

* Logistic Regression
* Random Forest
* XG Boost

SVM was evaluated but ultimately excluded from the final ensemble. Despite feature scaling and synthetic oversampling (SMOTE), the model converged to a degenerate state (predicting the majority class). Given the high dimensionality introduced by One-Hot Encoding, the computational cost and sensitivity of the SVM kernel did not provide a lift over gradient-boosted trees (XGBoost).

Various hyperparameters were tested for some of the models.

🔬 Model Selection & Evaluation

A rigorous evaluation across multiple scenarios (with/without Feature Engineering, with/without SMOTE). While tree-based models were considered, the final selection was based on unseen test data performance.

🏆 Logistic Regression (Selected Scenario): Combined with specialized feature engineering and SMOTE, this model achieved the best generalization. Its linear nature provides high interpretability for business stakeholders.

XGBoost/Random Forest: While high-performing on training data, they showed diminishing returns on the test set compared to the regularized LR model.

SVM: Deprecated due to poor convergence on the encoded feature space.


## 🎯 Objective
A production-grade classification system designed to predict customer churn. This project demonstrates the transition from exploratory data science to **Machine Learning Engineering**, emphasizing modularity, testability, and automated pipelines.

## Quickstart

```shells
uv install
cd src
# Does the experiment on the models, determines the best model, saves repots, default data, and the best Pipeline
uv run main.py -e

# Predict test data
uv run main.py --input <JSON file>
```

## Testing

```shells
uv run pytest
```

## 🏗 Architecture & Engineering Rigor
The core of this project is a unified **Pipeline-first architecture** using `imblearn`.

* **Atomic Artifacts:** The entire transformation lifecycle—including scaling, one-hot encoding, and synthetic oversampling (SMOTE)—is encapsulated in a single serialized Pipeline. This ensures 1:1 parity between training and inference environments.
* **Leakage Prevention:** SMOTE and feature scaling are applied strictly within cross-validation folds, preventing "data leakage" and ensuring realistic performance metrics.
* **Type Safety:** Implements custom `DataClasses` and strict Python type hinting to enforce interface contracts between data loading, training, and evaluation modules.

### ✅ Quality Assurance
* **Unit Tests:** Built with `pytest` to validate pipeline consistency and prevent "Model Degeneracy" (ensuring the model predicts more than one class).
* **Schema Validation:** (Optional/Upcoming) Prepared for raw JSON input validation to handle real-world API requests.

## Technical Challenges & Solutions

* Challenge: Standard Scikit-Learn pipelines failed when integrating SMOTE.

Solution: Implemented imblearn.pipeline to correctly sequence over-sampling within the cross-validation loop, preventing data leakage.

* Challenge: SVM model initially converged to a degenerate state (predicting only the majority class).

Solution: Refactored the ColumnTransformer to ensure uniform scaling across all feature types and adjusted class weights to handle the remaining imbalance.

## Contents

| Item | Description |
|------|  ---------- |
| data | Folder containing the data used for the model testing |
| docs | Further documentation on the project and its findings |
| logs | Output of the testing in `csv` format |
| notebooks | A preliminary test of the various models in Jupyter Notebook. Ended up with mostly the same conclusion |
| src | The source code for the testing |
| tests | Potential tests for the code |
| pyproject.toml | The requirements for this Project |
| README.md | This file |

