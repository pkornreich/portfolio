# BS"D

import pandas as pd
from sklearn.model_selection import train_test_split
from models.modelfactory import ModelFactory

def test_model_prediction_range(dataset):
    model_data = dataset.get_input_output()
    X_train, X_test, y_train, y_test = train_test_split(model_data.X, model_data.y, test_size=0.2, random_state=42, stratify=model_data.y)
    models = ModelFactory.get_models()
    for meta_model in models:
        model = meta_model.model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        s = pd.Series(y_pred)
        assert s.between(0,1).all()
