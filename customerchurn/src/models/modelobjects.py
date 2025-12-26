# BS"D

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

class ModelData:

    def __init__(self, X: DataFrame, y: Series, data_description: str):
        self.X: DataFrame = X
        self.y: Series = y
        self.data_description = data_description

class MetaModel:

    def __init__(self, model: BaseEstimator, name: str):
        self.model = model
        self.name = name

class ModelResult:

    CSV_FIELDS = ['model_name', 'data_description', 'smote', 'precision_cv', 'accuracy',
            'precision_pos', 'recall_pos', 'f1_pos', 'avg']

    def __init__(self, model_name: str, data_description: str, precision_cv: float,
                 accuracy: float, precision_pos: float, recall_pos: float,
                 f1_pos: float, cm, smote: bool = False):
        self.model_name = model_name
        self.data_description = data_description
        self.precision_cv = precision_cv
        self.accuracy = accuracy
        self.precision_pos = precision_pos
        self.recall_pos = recall_pos
        self.f1_pos = f1_pos
        self.cm = cm
        self.smote = smote

        self.avg = self.precision_cv + self.recall_pos + self.f1_pos
        self.avg = self.avg/3.
