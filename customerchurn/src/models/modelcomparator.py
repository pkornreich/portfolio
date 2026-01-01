# BS"D

from models.modelobjects import MetaModel
from models.modelanalyzer import ModelAnalyzer
from models.modelobjects import ModelResult

from data.datapipelineprovider import DataPipelineProvider

class ModelComparator:

    @staticmethod
    def compare_models(models: list[MetaModel], dpp: DataPipelineProvider,
            use_smote: bool = False) -> list[ModelResult]:
        results: list[ModelResult] = []
        for model in models:
            results.append(ModelAnalyzer.get_scores(model, dpp, use_smote))
        return results
    
    @staticmethod
    def get_best_result(results: list[ModelResult]) -> int:
        highest_f1 = 0
        hi_index = 0
        index = 0
        for result in results:
            if result.f1_pos > highest_f1:
                highest_f1 = result.f1_pos
                hi_index = index
            index = index + 1
        return hi_index

