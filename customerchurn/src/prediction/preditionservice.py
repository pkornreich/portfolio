# BS"D

from pandas import DataFrame

from sklearn.base import BaseEstimator

from data.datasource import DataService
from data.modeldatasource import ModelDataSource

class PredictionService:

    @staticmethod
    def find_and_fix(ds: DataService):
        df: DataFrame = ds.get_raw()
        dv: DataFrame = ds.read_default_values()
        column_names = list(dv.columns)
        for name in column_names:
            if (not name in list(df.columns)) or (df[name] is None) or (len(str(df[name])) == 0):
                df[name] = dv[name]
        ds.set_raw(df)

    @staticmethod
    def predict(file_name: str) -> str:
        ds: DataService = DataService(file_name)
        PredictionService.find_and_fix(ds)

        ds.cleanse_data()
        ds.engineer_features()
        input = ds.get_cleansed_input()
        model: BaseEstimator = ModelDataSource.read_model()
        raw_result = model.predict(input)
        result = ds.get_churn_translation()[raw_result[0]]
        return result
