# BS"D

from rich import print

from data.datasource import DataService
from models.modelobjects import ModelData
from models.modelobjects import MetaModel
from models.modelfactory import ModelFactory
from models.modelcomparator import ModelComparator
from models.modelobjects import ModelResult
from data.resultmnanager import ResultManager


def main():
    ds = DataService()
    result_log = ResultManager()
    ds.cleanse_data()
    X, y = ds.get_input_output()
    data: ModelData = ModelData(X,y,'Base Data')

    models: list[MetaModel] = ModelFactory.get_vary_max_iters()
    results: list[ModelResult] = ModelComparator.compare_models(models, data)
    result_log.save_results(results)
    index = ModelComparator.get_best_result(results)
    model = models[index].model
    max_iter = model.max_iter
    solver = model.solver
    
    models: list[MetaModel] = ModelFactory.get_vary_estimators()
    results = ModelComparator.compare_models(models, data)
    result_log.save_results(results)
    estimators = 25 + 25*ModelComparator.get_best_result(results)
    print(f'Best estimators for Random Forest - {estimators}')
    models: list[MetaModel] = ModelFactory.get_vary_lr_models()
    results = ModelComparator.compare_models(models, data)
    result_log.save_results(results)
    lr = 0.01 + 0.005*ModelComparator.get_best_result(results)
    models: list[MetaModel] = ModelFactory.get_vary_depth_models(lr)  
    results = ModelComparator.compare_models(models, data)
    depth = 2 + ModelComparator.get_best_result(results)
    print(f'Best Learning Rate: {lr}')
    print(f'Best Depth: {depth}')
    models: list[MetaModel] = ModelFactory.get_models(lr, depth, estimators, solver, max_iter)
    results = ModelComparator.compare_models(models, data)
    result_log.save_results(results)
    # Use SMOTE
    models: list[MetaModel] = ModelFactory.get_models(lr, depth, estimators, solver, max_iter)
    results = ModelComparator.compare_models(models, data, True)
    result_log.save_results(results)
     # Feature Engineering
    ds.engineer_features()
    X, y = ds.get_input_output()
    data =  ModelData(X,y,'Feature Engineered')
    models: list[MetaModel] = ModelFactory.get_models(lr, depth, estimators, solver, max_iter)
    results = ModelComparator.compare_models(models, data)
    result_log.save_results(results)

    ds.remove_low_impact_features()
    X, y = ds.get_input_output()
    data =  ModelData(X,y,'Low Impact Features Removed')
    models: list[MetaModel] = ModelFactory.get_models(lr, depth, estimators, solver, max_iter)
    results = ModelComparator.compare_models(models, data)
    result_log.save_results(results)



if __name__ == "__main__":
    main()
