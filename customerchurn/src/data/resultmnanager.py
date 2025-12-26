# BS"D

import csv

from constants import Constants
from models.modelobjects import ModelResult

class ResultManager:

    def __init__(self):
        results_file = Constants.LOG_PATH + Constants.RESULTS_FILE_NAME
        file = open(results_file, 'w', newline='')
        self.writer = csv.DictWriter(file, fieldnames=ModelResult.CSV_FIELDS)
        self.writer.writeheader()

    @staticmethod
    def display_result(result: ModelResult, data_description: str):
        print('-----------------------------------------')
        print('Results for ' + result.model_name)
        print('------------------------------------------')
        
        print(f"Average Precision with CV: {result.precision_cv:.4f}")
        print('------------------------------------------')
        print('Confusion Matrix')
        print(result.cm)

        print(f"Accuracy: {result.accuracy}")
        print(f"Precision (positive class): {result.precision_pos}")
        print(f"Recall (positive class): {result.recall_pos}")
        print(f"F1-Score (positive class): {result.f1_pos}")

    def save_results(self, results: list[ModelResult]):
        for result in results:
            self.writer.writerow({field: getattr(result, field) for field in ModelResult.CSV_FIELDS})

            



