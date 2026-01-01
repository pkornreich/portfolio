# BS"D

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

class ModelData:

    def __init__(self, X: DataFrame, y: Series, data_description: str):
        self.X: DataFrame = X
        self.y: Series = y
        self.data_description = data_description

class MetaModel:

    def __init__(self, model: BaseEstimator, name: str, param_grid: map=None):
        self.model = model
        self.name = name
        self.param_grid = param_grid

class ModelResult:

    CSV_FIELDS = ['model_name', 'data_description', 'smote', 'accuracy',
            'precision_pos', 'recall_pos', 'f1_pos', 'avg']

    def __init__(self, model_name: str, data_description: str, best_model: BaseEstimator,
                 best_parameters: dict, f1_train: float,
                 accuracy: float, precision_pos: float, recall_pos: float,
                 f1_pos: float, cm, train_results: dict = None, smote: bool = False):
        self.model_name = model_name
        # self.best_model = best_model
        self.data_description = data_description
        self.best_model = best_model
        self.best_parameters = best_parameters
        self.f1_train = f1_train
        self.accuracy = accuracy
        self.precision_pos = precision_pos
        self.recall_pos = recall_pos
        self.f1_pos = f1_pos
        self.cm = cm
        self.train_results = train_results
        self.smote = smote

        self.avg = self.f1_pos + self.recall_pos + self.precision_pos
        self.avg = self.avg/3.
