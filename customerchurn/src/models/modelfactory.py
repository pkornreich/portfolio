# BS"D

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from models.modelobjects import MetaModel

class ModelFactory:

    @staticmethod
    def get_models(lr: float = 0.05, depth: int = 5, estimators: int = 100,
                solver: str = 'lbfgs', max_iter = 100) -> list[MetaModel]:
        models = [
            MetaModel(LogisticRegression(solver=solver, max_iter=max_iter),'Logistic Regression')
            ,MetaModel(SVC(),'Support Vector Classification')
            , MetaModel(RandomForestClassifier(class_weight='balanced',
                n_estimators=estimators),'Random Forest')
            , MetaModel(xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42, max_depth=depth, learning_rate=lr
            ),'XG Boost')
        ]
        return models

    @staticmethod
    def get_vary_max_iters() -> list[MetaModel]:
        models: list[MetaModel] = []
        for solver in ['lbfgs', 'liblinear', 'newton-cholesky']:
            iters = 100
            while iters < 525:
                models.append(MetaModel(LogisticRegression(solver=solver, max_iter=iters),
                    f'Logistic Regression - {solver} with {iters} iterations'))
                iters = iters + 50
        return models

    @staticmethod
    def get_vary_estimators() -> list[MetaModel]:
        models: list[MetaModel] = []
        estimators = 25
        while estimators < 525:
            models.append(MetaModel(RandomForestClassifier(class_weight='balanced',
                n_estimators=estimators),f'Random Forest - Estimators={estimators}'))
            estimators = estimators + 25
        return models

    @staticmethod
    def get_vary_lr_models() -> list[MetaModel]:
        learning_rate = 0.01
        models: list[MetaModel] = []
        while learning_rate < 0.101:
            models.append(MetaModel(xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42, max_depth=5, learning_rate=learning_rate),
                f'XG Boost - Learning Rate =  {learning_rate}'))
            learning_rate = learning_rate + 0.005
        return models

    @staticmethod
    def get_vary_depth_models(lr: float) -> list[MetaModel]:
        models: list[MetaModel] = []
        depth = 2
        while depth < 11:
            models.append(MetaModel(xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42, max_depth=depth, learning_rate=lr),
                f'XG Boost  - Depth = {depth}'))
            depth = depth + 1
        return models

