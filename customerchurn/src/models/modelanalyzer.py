# BS"D

import logging

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import set_config

from util.logging import Logger
from models.modelobjects import MetaModel
from models.modelobjects import ModelData
from models.modelobjects import ModelResult

from data.datapipelineprovider import DataPipelineProvider

class ModelAnalyzer:

    @staticmethod
    def get_scores(model_meta: MetaModel, dpp: DataPipelineProvider, use_smote: bool=False) -> ModelResult:
        set_config(transform_output="pandas")

        data: ModelData = dpp.get_input_output()
        X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=42, stratify=data.y)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        model = model_meta.model
        pipeline = Pipeline([
            ('scaler', dpp.get_scaler()),
            (model_meta.name, model)
        ])


        if use_smote:
            pipeline = Pipeline([
                ('scaler', dpp.get_scaler()),
                ('smote', SMOTE(random_state=42)),
                (model_meta.name, model)
            ])
        
        # precision_scores = cross_val_score(
        #     pipeline,
        #     X_train,
        #     y_train,
        #     cv=cv,
        #     scoring='precision',  # Or 'accuracy', 'recall', 'f1'
        #     n_jobs=-1
        # )
        grid_search = GridSearchCV(estimator=pipeline, param_grid=model_meta.param_grid, cv=5,
                scoring='f1', verbose=1, n_jobs=-1)
        
        # model.fit(X_train, y_train)
        grid_search.fit(X_train, y_train)
        # logger.log(logging.INFO, ("Results: ")
        # logger.log(logging.INFO, (grid_search.cv_results_)
        # logger.log(logging.INFO, (model_meta.name)
        # logger.log(logging.INFO, ("Best score: %f" % grid_search.best_score_)
        # logger.log(logging.INFO, ("Best parameters:", grid_search.best_params_)

        # The best estimator is available as an attribute and can be used for predictions
        best_model = grid_search.best_estimator_
        train_results: dict = grid_search.cv_results_
        for prop in train_results:
            if isinstance(train_results[prop], np.ndarray):
                train_results[prop] = train_results[prop].tolist()


        f1_train = grid_search.best_score_
        # best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        result = ModelResult(model_meta.name, data.data_description, best_model, 
            grid_search.best_params_, f1_train, accuracy,
            precision, recall, f1, cm, train_results, use_smote)
        
        return result

