# BS"D

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from models.modelobjects import MetaModel

class ModelFactory:

    @staticmethod
    def get_models(params: list[dict]) -> list[MetaModel]:
        lr_params = params[0]
        c = lr_params["Logistic Regression__C"]
        max_iter = lr_params["Logistic Regression__max_iter"]
        solver = lr_params["Logistic Regression__solver"]
        nf_params = params[1]
        estimators = nf_params["Random Forest__n_estimators"]
        xgb_params = params[2]
        lr = xgb_params["XG Boost__learning_rate"]
        depth = xgb_params[ "XG Boost__max_depth"]
        models = [
            MetaModel(LogisticRegression(solver=solver, max_iter=max_iter, C=c),'Logistic Regression', {}),
            # MetaModel(SVC(),'Support Vector Classification'),
            MetaModel(RandomForestClassifier(class_weight='balanced',
                n_estimators=estimators),'Random Forest', {}),
            MetaModel(xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42, max_depth=depth, learning_rate=lr
            ),'XG Boost', {})
        ]
        return models

    @staticmethod
    def get_logistic_regression_test() -> list[MetaModel]:
        max_iters = np.arange(5000, 15500, 500).tolist()
        params_grid: map = {
            'Logistic Regression__solver': ['lbfgs', 'liblinear', 'newton-cholesky'],
            'Logistic Regression__max_iter': max_iters,
            'Logistic Regression__C': [0.1, 1.0, 10.0]
            # "penalty is model dependent"
        }
        meta_model = MetaModel(LogisticRegression(), 'Logistic Regression', params_grid)
        return [meta_model]

    @staticmethod
    def get_random_forest_test() -> list[MetaModel]:
        estimators = np.arange(50,1050,50).tolist()
        params_grid: map = {'Random Forest__n_estimators': estimators}
        meta_model = MetaModel(RandomForestClassifier(class_weight='balanced'),'Random Forest', params_grid)
        return [meta_model]

    @staticmethod
    def get_xgboost_test() -> list[MetaModel]:
        learning_rate = np.arange(0.002, .101, .002).tolist()
        max_depth = np.arange(2, 11, 1).tolist()
        params_grid: map = {'XG Boost__learning_rate': learning_rate, 'XG Boost__max_depth': max_depth}
        meta_model = MetaModel(xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='logloss', random_state=42),
                'XG Boost', params_grid)
        return [meta_model]
