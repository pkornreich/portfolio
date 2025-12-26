# BS"D

from models.modelobjects import MetaModel
from models.modelanalyzer import ModelAnalyzer
from models.modelobjects import ModelData
from models.modelobjects import ModelResult

class ModelComparator:

    @staticmethod
    def compare_models(models: list[MetaModel], data: ModelData,
            use_smote: bool = False) -> list[ModelResult]:
        results: list[ModelResult] = []
        for model in models:
            results.append(ModelAnalyzer.get_scores(model, data, use_smote))
        return results
    
    @staticmethod
    def get_best_result(results: list[ModelResult]) -> int:
        highest_avg = 0
        hi_index = 0
        index = 0
        for result in results:
            if result.avg > highest_avg:
                highest_avg = result.avg
                hi_index = index
            index = index + 1
        return hi_index

