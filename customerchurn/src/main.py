# BS"D
import sys
import argparse
import logging

from rich import print

from util.logging import Logger
from data.datasource import DataService
from models.modelobjects import ModelData
from models.modelobjects import MetaModel
from models.modelfactory import ModelFactory
from models.modelcomparator import ModelComparator
from models.modelobjects import ModelResult
from data.resultmnanager import ResultManager

from prediction.preditionservice import PredictionService


def find_best_model():
    ds = DataService()
    result_log = ResultManager()
    ds.cleanse_data()

    models: list[MetaModel] = ModelFactory.get_logistic_regression_test()
    results: list[ModelResult] = ModelComparator.compare_models(models, ds)
    result: ModelResult = results[0]
    result_log.save_results(result)
    best_params_lr: dict = result.best_parameters
    
    models: list[MetaModel] = ModelFactory.get_random_forest_test()
    results: list[ModelResult] = ModelComparator.compare_models(models, ds)
    result: ModelResult = results[0]
    result_log.save_results(result)
    best_params_rf: dict = result.best_parameters

    models: list[MetaModel] = ModelFactory.get_xgboost_test()
    results: list[ModelResult] = ModelComparator.compare_models(models, ds)
    result: ModelResult = results[0]
    result_log.save_results(result)
    best_params_xgb: dict = result.best_parameters

    best_params = [best_params_lr, best_params_rf, best_params_xgb]
    result_log.save_best_params(best_params)

    all_results: list[ModelResult] = []
    models: list[MetaModel] = ModelFactory.get_models(best_params)
    all_results = all_results + ModelComparator.compare_models(models, ds)
    # Use SMOTE
    models: list[MetaModel] = ModelFactory.get_models(best_params)
    all_results = all_results + ModelComparator.compare_models(models, ds)

     # Feature Engineering
    ds.engineer_features()
    models: list[MetaModel] = ModelFactory.get_models(best_params)
    all_results = all_results + ModelComparator.compare_models(models, ds)

    ds.remove_low_impact_features()
    models: list[MetaModel] = ModelFactory.get_models(best_params)
    all_results = all_results + ModelComparator.compare_models(models, ds)
    result_log.save_model_comparison(all_results)

    best_index: int = ModelComparator.get_best_result(all_results)
    best_result: ModelResult = all_results[best_index]
    logger = Logger.get_logger()
    logger.log(logging.INFO, f'Model: {best_result.model_name}, Data: {best_result.data_description}')
    result_log.save_best_model(best_result)

def predict(file_name: str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='Customer Churn',
            description='Finds the best model for a Customer Churn dataset and makes predictions based on it',
            epilog='Choose either to experiment and find the best model or make a prediction')
    parser.add_argument('--input', type=str, dest='file_name', help='Name of file with row of customer churn data in JSON format on which to make a prediction')
    parser.add_argument('-e', action='store_true', dest='is_experiment', help='Perform experiments to find the best model')
    args = parser.parse_args()
    if args.is_experiment:
        if (args.file_name is not None) and len(args.file_name) > 0:
            print('Usage: Choose either experiment or prediction')
            sys.exit()
        find_best_model()
    else:
        if args.file_name is not None and len(args.file_name) > 0:
            result: str = PredictionService.predict(args.file_name)
            print(result)
        else:
            print('Usage: Choose either experiment or prediction')

        
