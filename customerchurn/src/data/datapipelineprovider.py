# BS"D

from abc import ABC, abstractmethod
from pandas import DataFrame, Series
from sklearn.base import TransformerMixin
from models.modelobjects import ModelData

class DataPipelineProvider(ABC):

    @abstractmethod
    def get_scaler(self) -> TransformerMixin:
        pass

    @abstractmethod
    def get_input_output(self) -> ModelData:
        pass

