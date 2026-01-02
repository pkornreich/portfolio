# BS"D

import logging

import csv
import json
import pickle
from pathlib import Path

from constants import Constants
from util.logging import Logger
from models.modelobjects import ModelResult
from data.datamanager import DataManager

class ResultManager:

    def __init__(self):
        results_file = Constants.LOG_PATH + Constants.RESULTS_FILE_NAME
        if DataManager.is_running_from_src():
            results_file = Constants.PATH_TO_ROOT + results_file
        file = open(results_file, 'w', newline='')
        self.writer = csv.DictWriter(file, fieldnames=ModelResult.CSV_FIELDS)
        self.writer.writeheader()
    
    @staticmethod
    def display_result(result: ModelResult, data_description: str):
        logger = Logger.get_logger()
        logger.log(logging.INFO, '-----------------------------------------')
        logger.log(logging.INFO, 'Results for ' + result.model_name)
        logger.log(logging.INFO, '------------------------------------------')
        
        logger.log(logging.INFO, f"Average Precision with CV: {result.precision_cv:.4f}")
        logger.log(logging.INFO, '------------------------------------------')
        logger.log(logging.INFO, 'Confusion Matrix')
        logger.log(logging.INFO, result.cm)

        logger.log(logging.INFO, f"Accuracy: {result.accuracy}")
        logger.log(logging.INFO, f"Precision (positive class): {result.precision_pos}")
        logger.log(logging.INFO, f"Recall (positive class): {result.recall_pos}")
        logger.log(logging.INFO, f"F1-Score (positive class): {result.f1_pos}")

    def save_results(self, results: ModelResult):
        results_file = Constants.LOG_PATH + results.model_name.replace(' ', '') + '.json'
        if DataManager.is_running_from_src():
            results_file = Constants.PATH_TO_ROOT + results_file
        file = open(results_file, 'w', newline='')
        json.dump(results.train_results, file, indent=4)

    def save_best_params(self, params: list[dict]):
        results_file = Constants.LOG_PATH + 'params.json'
        if DataManager.is_running_from_src():
            results_file = Constants.PATH_TO_ROOT + results_file
        file = open(results_file, 'w', newline='')
        json.dump(params, file, indent=4)


    def save_model_comparison(self, results: list[ModelResult]):
        for result in results:
            self.writer.writerow({field: getattr(result, field) for field in ModelResult.CSV_FIELDS})

            
    def save_best_model(self, result: ModelResult):
        results_file = Constants.MODEL_PATH + Constants.MODEL_FILE_NAME
        if DataManager.is_running_from_src():
            results_file = Constants.PATH_TO_ROOT + results_file
        file_path = Path(results_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file = open(results_file, 'wb')
        pickle.dump(result.best_model, file)
        


