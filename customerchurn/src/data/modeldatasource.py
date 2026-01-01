# BS"D

import pickle
from pathlib import Path

from sklearn.base import BaseEstimator

from constants import Constants
from data.datamanager import DataManager

class ModelDataSource():

    @staticmethod
    def read_model() -> BaseEstimator:
        model_file = Constants.MODEL_PATH + Constants.MODEL_FILE_NAME
        if DataManager.is_running_from_src():
            model_file = Constants.PATH_TO_ROOT + model_file
        file = open(model_file, 'rb')
        model: BaseEstimator = pickle.load(file)
        return model